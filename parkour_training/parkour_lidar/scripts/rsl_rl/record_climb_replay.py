"""Record the parkour teacher's box->platform CLIMB as an OPEN-LOOP action sequence.

The learned ACT student destabilizes the robot, but the teacher itself climbs reliably.
So instead of imitating, just RECORD the teacher's 20-D env-action sequence over the
climb and REPLAY it open-loop at deployment (after walkclimb reaches the wedged pose).

Runs the box+platform StepCollect env (deterministic start), drives the teacher, records
per-step: action20 + root (x,z,yaw) for env 0. Saves the action sequence + the climb
segment (frames where z rises from ~0 to platform top) to an npz for the submission.

Run (parkour_il):
  CUDA_VISIBLE_DEVICES=2 TMPDIR=/hdd0/sohyun/tmp \
  PYTHONPATH=/hdd0/sohyun/atec_2026_tesk_e/Isaaclab_Parkour python \
  scripts/rsl_rl/record_climb_replay.py --enable_cameras --headless \
  --load_run 2026-06-25_05-32-19 --checkpoint model_18100.pt
"""
import argparse
import os
import math
from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str,
                    default="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-StepCollect-v0")
parser.add_argument("--max_steps", type=int, default=320)
parser.add_argument("--out", type=str,
                    default="/hdd0/sohyun/atec_2026_tesk_e/ATEC2026_Simulation_Challenge/submission_taskd_teacher/climb_replay.npz")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.load_run is None:
    args.load_run = "2026-06-25_05-32-19"
if args.checkpoint is None:
    args.checkpoint = "model_18100.pt"
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import ParkourRslRlOnPolicyRunnerCfg  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

TEACHER_TO_ENV_SCALE = 0.5


def yaw_of(q):
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    # deterministic start: disable the start randomization for a clean repeatable climb
    try:
        from parkour_isaaclab.envs.mdp import events as _ev
        env_cfg.events.reset_root_state.func = _ev.reset_root_state
        env_cfg.events.reset_root_state.params = {'offset': 3.0}
    except Exception as e:
        print("[warn] could not reset to deterministic start:", e)

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    estimator = runner.get_estimator_inference_policy(device=env.unwrapped.device)
    paras = agent_cfg.to_dict()["estimator"]
    np_, ns_, npe_ = paras["num_prop"], paras["num_scan"], paras["num_priv_explicit"]

    dev = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    obs, _ = env.get_observations()

    acts, zs, xs, yaws, ys = [], [], [], [], []
    org_x = float(env.unwrapped.scene.env_origins[0, 0])
    org_y = float(env.unwrapped.scene.env_origins[0, 1])
    for t in range(args.max_steps):
        with torch.inference_mode():
            obs[:, np_ + ns_:np_ + ns_ + npe_] = estimator.inference(obs[:, :np_])
            a = policy(obs, hist_encoding=True)               # (1,12)
        act20 = torch.cat([a * TEACHER_TO_ENV_SCALE, torch.zeros(1, 8, device=dev)], dim=1)
        rp = robot.data.root_pos_w[0]
        acts.append(act20[0].cpu().numpy().astype(np.float32))
        xs.append(float(rp[0]) - org_x)
        ys.append(float(rp[1]) - org_y)
        zs.append(float(rp[2]))
        yaws.append(yaw_of(robot.data.root_quat_w[0]))
        obs, _, dones, _ = env.step(a)
        if bool(dones[0]):
            break

    acts = np.stack(acts); zs = np.array(zs); xs = np.array(xs); yaws = np.array(yaws); ys = np.array(ys)
    print(f"[rec] steps={len(acts)} z range {zs.min():.2f}..{zs.max():.2f} x range {xs.min():.2f}..{xs.max():.2f}")
    # climb segment: from last time z<0.45 (on ground/box-front) until z first exceeds 1.0
    # (on platform). Print so we can pick the chunk.
    on_ground = np.where(zs < 0.5)[0]
    on_plat = np.where(zs > 1.1)[0]
    seg0 = int(on_ground[-1]) if len(on_ground) else 0
    seg1 = int(on_plat[0]) if len(on_plat) else len(acts) - 1
    print(f"[rec] climb segment guess: t={seg0}..{seg1} (z {zs[seg0]:.2f}->{zs[min(seg1,len(zs)-1)]:.2f})")
    for t in range(0, len(acts), 10):
        print(f"  t={t:3d} z={zs[t]:+.2f} x={xs[t]:+.2f} y={ys[t]:+.2f} yaw={yaws[t]:+.2f}")
    np.savez(args.out, actions=acts, z=zs, x=xs, y=ys, yaw=yaws, seg0=seg0, seg1=seg1)
    print(f"[rec] saved -> {args.out}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
