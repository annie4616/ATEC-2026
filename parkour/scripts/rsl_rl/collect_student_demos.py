"""Collect teacher-climb demos for the Task D SUBMISSION student.

Runs the parkour STEP_COLLECT env (0.6 m up-step + a submission-matched head_camera).
The parkour TEACHER (model_18100) drives the legs from its full 753-D obs (accurate
scandots/goals), so it climbs reliably. Each step we record what the SUBMISSION student
will see + must output:

  head_rgb (480x640x3 uint8)         # submission head_camera view
  proprio  (submission 53-ish layout, 20-joint with arm=0)   # student state input
  action   (20-D env action; legs = teacher 12 mapped to submission scale, arm = 0)

Only episodes where the robot actually CLIMBS the step (advances past it, stays upright)
are kept. Output: an hdf5 of trajectories for ACT training.

Run (parkour_il env):
  CUDA_VISIBLE_DEVICES=N TMPDIR=/hdd0/sohyun/tmp \
  PYTHONPATH=/hdd0/sohyun/atec_2026_tesk_e/Isaaclab_Parkour python \
  scripts/rsl_rl/collect_student_demos.py --num_demos 60 --enable_cameras --headless \
  --load_run 2026-06-25_05-32-19 --checkpoint model_18100.pt
"""
import argparse
import os
from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str,
                    default="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-StepCollect-v0")
parser.add_argument("--num_demos", type=int, default=60)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--max_steps", type=int, default=400, help="steps per episode")
parser.add_argument("--output_dir", type=str,
                    default="/hdd0/sohyun/atec_2026_tesk_e/datasets/taskd_student_step")
parser.add_argument("--min_advance", type=float, default=1.0,
                    help="min +x advance (m) in env-local for a 'climbed' episode")
cli_args.add_rsl_rl_args(parser)        # adds --load_run --checkpoint --resume --seed ...
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.load_run is None:
    args.load_run = "2026-06-25_05-32-19"
if args.checkpoint is None:
    args.checkpoint = "model_18100.pt"
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import time
import numpy as np
import torch
import h5py
import gymnasium as gym

from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import ParkourRslRlOnPolicyRunnerCfg  # noqa: F401 register
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# ---- submission proprio/action constants (must match solution.py) ----
# submission proprio layout: lin_vel(3) ang_vel(3) vel_cmd(3) gravity(3) joint_pos(20)
#   joint_vel(20) actions(20) = 72.  Legs are the first 12 joints; arm (12:20) = 0.
LEG_DEFAULT = [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5]
TEACHER_TO_ENV_SCALE = 0.5     # teacher action scale 0.25 -> env leg scale 0.5


def main():
    os.makedirs(args.output_dir, exist_ok=True)

    agent_cfg: ParkourRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    estimator = runner.get_estimator_inference_policy(device=env.unwrapped.device)
    paras = agent_cfg.to_dict()["estimator"]
    num_prop, num_scan, num_priv = paras["num_prop"], paras["num_scan"], paras["num_priv_explicit"]

    dev = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    cam = env.unwrapped.scene["head_camera"]
    lidar = env.unwrapped.scene["lidar_sensor"]
    parkour = env.unwrapped.parkour_manager.get_term("base_parkour")
    leg_default = torch.tensor(LEG_DEFAULT, device=dev)
    arm_default = torch.zeros(8, device=dev)

    N = args.num_envs
    obs, extras = env.get_observations()

    # per-env rolling buffers
    buf_rgb = [[] for _ in range(N)]
    buf_depth = [[] for _ in range(N)]
    buf_lidar = [[] for _ in range(N)]
    buf_prop = [[] for _ in range(N)]
    buf_act = [[] for _ in range(N)]
    start_x = (robot.data.root_pos_w[:, 0] - env.unwrapped.scene.env_origins[:, 0]).clone()
    # Track the MAX forward progress + max height reached DURING the episode. IsaacLab
    # auto-resets done envs inside env.step(), so reading root_pos_w AFTER the step gives
    # the POST-RESET position (~-7m) for done envs — using that as the episode end made
    # advance spuriously negative and rejected every climb after the first batch. Instead
    # accumulate the per-step max while the episode runs, evaluated at done.
    ep_max_adv = torch.full((N,), -1e9, device=dev)
    ep_max_rz = torch.zeros(N, device=dev)
    ep_step = torch.zeros(N, dtype=torch.long, device=dev)

    demos = []   # list of dicts {rgb, proprio, action}
    print(f"[collect] target {args.num_demos} demos, {N} envs", flush=True)

    while len(demos) < args.num_demos:
        with torch.inference_mode():
            obs[:, num_prop + num_scan:num_prop + num_scan + num_priv] = estimator.inference(obs[:, :num_prop])
            actions = policy(obs, hist_encoding=True)        # (N, NJ) teacher action
        # B2Piper teacher outputs 20 (12 legs + 8 arm); B2 teacher outputs 12. The env is
        # stepped with the FULL teacher action (env_action = actions). For the DEMO we
        # record only the 12 LEG actions (arm = 0) to match deployment (arm action = 0).
        leg_actions = actions[:, :12]                        # (N,12) leg slice

        # --- build submission-format proprio (N, 72) from robot data ---
        jp = robot.data.joint_pos[:, :12]                    # 12 leg joints (native order)
        jv = robot.data.joint_vel[:, :12]
        lin_vel_b = robot.data.root_lin_vel_b               # (N,3)
        ang_vel_b = robot.data.root_ang_vel_b
        grav = robot.data.projected_gravity_b
        # vel_cmd placeholder (submission walking-cmd slot; teacher ignores) -> forward
        vel_cmd = torch.zeros(N, 3, device=dev); vel_cmd[:, 0] = 1.0
        jp20 = torch.cat([jp - leg_default, torch.zeros(N, 8, device=dev)], dim=1)
        jv20 = torch.cat([jv, torch.zeros(N, 8, device=dev)], dim=1)
        # env action (submission): leg = teacher_leg*scale, arm = 0
        act20 = torch.cat([leg_actions * TEACHER_TO_ENV_SCALE, torch.zeros(N, 8, device=dev)], dim=1)
        # ACT STATE: absolute 20-D joint_pos (legs from robot, arm=0) — same as the
        # submission's ACT state (joint_pos_all), so submission inference is unchanged.
        obs20 = torch.cat([jp, torch.zeros(N, 8, device=dev)], dim=1)

        rgb = cam.data.output["rgb"][..., :3]                # (N,H,W,3) uint8
        depth = cam.data.output["depth"]                     # (N,H,W) or (N,H,W,1) float (m)
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        if depth.ndim == 4:
            depth = depth[..., 0]
        # lidar height_scan, exactly as the submission obs: sensor_z - hit_z - 0.5
        lz = lidar.data.pos_w[:, 2].unsqueeze(1)
        lidar_scan = torch.nan_to_num(lz - lidar.data.ray_hits_w[..., 2] - 0.5,
                                      nan=0.0, posinf=0.0, neginf=0.0)   # (N, num_rays)

        for i in range(N):
            buf_rgb[i].append(rgb[i].detach().to(torch.uint8).cpu().numpy())
            buf_depth[i].append(depth[i].detach().cpu().numpy().astype(np.float16))
            buf_lidar[i].append(lidar_scan[i].detach().cpu().numpy().astype(np.float32))
            buf_prop[i].append(obs20[i].detach().cpu().numpy().astype(np.float32))
            buf_act[i].append(act20[i].detach().cpu().numpy().astype(np.float32))

        # PRE-STEP position: update the episode's running max progress/height now, while the
        # robot is still at its in-episode pose (before env.step() may auto-reset it).
        pre_x = robot.data.root_pos_w[:, 0] - env.unwrapped.scene.env_origins[:, 0]
        pre_rz = robot.data.root_pos_w[:, 2]
        ep_max_adv = torch.maximum(ep_max_adv, pre_x - start_x)
        ep_max_rz = torch.maximum(ep_max_rz, pre_rz)

        obs, _, dones, extras = env.step(actions)
        ep_step += 1

        timeout = ep_step >= args.max_steps
        done_mask = dones.bool() | timeout
        _done_idx = torch.nonzero(done_mask).squeeze(-1).tolist()
        if _done_idx:
            _adv = [float(ep_max_adv[i]) for i in _done_idx]
            _rz = [float(ep_max_rz[i]) for i in _done_idx]
            print(f"[diag] {len(_done_idx)} done | advance max={max(_adv):.2f} "
                  f"mean={sum(_adv)/len(_adv):.2f} | rz max={max(_rz):.2f}", flush=True)
        for i in _done_idx:
            advance = float(ep_max_adv[i])
            climbed = advance >= args.min_advance and float(ep_max_rz[i]) > 0.2
            if climbed and len(buf_rgb[i]) > 20:
                demos.append({
                    "rgb": np.stack(buf_rgb[i]),
                    "depth": np.stack(buf_depth[i]),
                    "lidar": np.stack(buf_lidar[i]),
                    "proprio": np.stack(buf_prop[i]),
                    "action": np.stack(buf_act[i]),
                })
                print(f"[collect] demo {len(demos)}/{args.num_demos} "
                      f"(env {i}, len {len(buf_rgb[i])}, advance {advance:.2f})", flush=True)
            buf_rgb[i].clear(); buf_depth[i].clear(); buf_lidar[i].clear()
            buf_prop[i].clear(); buf_act[i].clear()
            ep_step[i] = 0
            # new episode starts from the (post-reset) position; reset running maxes.
            start_x[i] = robot.data.root_pos_w[i, 0] - env.unwrapped.scene.env_origins[i, 0]
            ep_max_adv[i] = -1e9
            ep_max_rz[i] = 0.0

    # ---- save hdf5 (keys match DemoDataset_ACT: obs / actions / images/{rgb,depth}) ----
    out = os.path.join(args.output_dir, "trajectory.hdf5")
    with h5py.File(out, "w") as f:
        for k, d in enumerate(demos):
            g = f.create_group(f"traj_{k}")
            g.create_dataset("obs", data=d["proprio"])               # (T,20) abs joint_pos
            g.create_dataset("actions", data=d["action"])            # (T,20) env action
            g.create_dataset("lidar", data=d["lidar"], compression="gzip")  # (T,num_rays)
            ig = g.create_group("images")
            ig.create_dataset("rgb", data=d["rgb"], compression="gzip")     # (T,H,W,3) uint8
            ig.create_dataset("depth", data=d["depth"], compression="gzip") # (T,H,W) f16
    print(f"[collect] saved {len(demos)} demos -> {out}", flush=True)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
