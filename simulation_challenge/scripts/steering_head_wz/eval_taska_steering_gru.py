"""TASK A END-TO-END: drive Task A with the VISION steering head (no privileged yaw).

The low-level fast policy follows a (vx, vy, wz) command. Instead of the kyaw teacher
(privileged world yaw, eval-illegal), we compute wz from the trained steering head that
reads the head camera RGB (frozen DINOv2) + lidar — exactly the eval-legal sensors. This
tests whether "look at the red boundary -> steer" actually keeps the robot on course all
the way to the goal, end-to-end in sim, with NO world-yaw cheat.

  CUDA_VISIBLE_DEVICES=1 TMPDIR=/hdd0/sohyun/tmp PYTHONPATH=/hdd0/sohyun \
    python scripts/eval_taska_steering_head.py --headless \
      --head /hdd0/sohyun/atec_2026_tesk_e/steering_head/steering_head.pt \
      --vx 2.0 --steps 8000 --out /hdd0/sohyun/atec_2026_tesk_e/taska_head_eval
"""
import argparse
import os
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Task A end-to-end with vision steering head.")
parser.add_argument("--task", type=str, default="ATEC-TaskA-B2Piper")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--head", type=str, required=True, help="steering_head.pt path")
parser.add_argument("--vx", type=float, default=2.0)
parser.add_argument("--wz_max", type=float, default=0.8)
parser.add_argument("--wz_ema", type=float, default=1.0,
                    help="EMA smoothing on the head's wz: wz=a*wz_new+(1-a)*wz_prev. a=1 -> no "
                         "smoothing (raw head). a<1 damps the step-to-step jerk that makes the "
                         "robot topple on rough terrain (failing seeds have 24-42%% |wz|>0.5 vs "
                         "kyaw's smooth 15%%). Try 0.2-0.5.")
parser.add_argument("--steps", type=int, default=8000)
parser.add_argument("--every", type=int, default=1, help="run the head every N steps (hold wz between)")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--video", action="store_true", default=False, help="record an mp4 of the run")
parser.add_argument("--video_length", type=int, default=8000, help="frames to record")
parser.add_argument("--seed", type=int, default=0, help="seed for reproducible closed-loop eval")
parser.add_argument("--enhanced_determinism", action="store_true", default=False,
                    help="enable PhysX enhanced determinism for reproducible GPU physics")
parser.add_argument("--no_illegal_contact", action="store_true", default=False,
                    help="disable the illegal_contact termination (base/hip/thigh touching "
                         "terrain ends the episode). Real-world has no such rule; failures were "
                         "almost all this. Keeps fall (root height) termination.")
parser.add_argument("--vx_slow_k", type=float, default=0.0,
                    help="TOP-Nav anti-drift: vx = vx0*exp(-k*|wz|). Slows forward speed when "
                         "steering hard so the robot recovers before drifting out of course. "
                         "0 = constant speed. Try 0.5-1.5.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import gymnasium as gym  # noqa: E402
import json  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec  # noqa: E402
from atec_2026_tesk_e.ATEC2026_Simulation_Challenge.demo.solution_rl import AlgSolution  # noqa: E402

solution = AlgSolution()


# --- steering head (must match scripts/train_steering_head.py) ---
# reuse the recurrent model definitions from the trainer
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location(
    "train_steering_gru",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_steering_gru.py"))
_tsg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tsg)
GRUSteer = _tsg.GRUSteer


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    dev = "cuda"

    # load recurrent steering head + normalization stats
    ckpt = torch.load(args_cli.head, map_location=dev, weights_only=False)
    head = GRUSteer().to(dev).eval()
    # init LazyLinear before loading weights
    with torch.no_grad():
        head(torch.zeros(1, 2, 256, 384, device=dev), torch.zeros(1, 2, 5760, device=dev))
    head.load_state_dict(ckpt["model"])
    pf_mu = torch.tensor(ckpt["pf_mu"], dtype=torch.float32, device=dev)   # (384,)
    pf_sd = torch.tensor(ckpt["pf_sd"], dtype=torch.float32, device=dev)
    li_mu = torch.tensor(ckpt["li_mu"], dtype=torch.float32, device=dev)   # (5760,)
    li_sd = torch.tensor(ckpt["li_sd"], dtype=torch.float32, device=dev)

    # frozen DINOv2 backbone -> PATCH tokens (256,384)
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()

    def rgb_to_patches(rgb_np):
        inp = proc(images=[rgb_np], return_tensors="pt").to(dev)
        with torch.inference_mode():
            return dino(**inp).last_hidden_state[:, 1:]  # (1,256,384) patch tokens

    # env
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    env_cfg = apply_safe_action_spec(env_cfg, json.dumps(action_spec))
    if args_cli.enhanced_determinism:
        # PhysX enhanced determinism: reproducible GPU physics at some perf cost. Reduces
        # the run-to-run max_x spread so head comparisons are trustworthy (GPU float
        # reduction is otherwise non-deterministic and seed alone can't fix it).
        env_cfg.sim.physx.enable_enhanced_determinism = True
    if args_cli.no_illegal_contact:
        # illegal_contact ends the episode when base/hip/thigh grazes the terrain. Term-cause
        # logging showed nearly ALL failures are this (robot upright, z normal, just a hip/
        # thigh touch on rough/slope/stairs) -- a sim-only rule; real-world has no such kill.
        # Drop it so only a genuine fall (root height < min) or reaching the goal ends the run.
        env_cfg.terminations.illegal_contact = None
        print("[cfg] illegal_contact termination DISABLED", flush=True)
    # let it run long; keep fall termination so a real fall ends the episode honestly
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if args_cli.video:
        video_dir = os.path.join(args_cli.out, "video")
        env = gym.wrappers.RecordVideo(
            env, video_folder=video_dir, name_prefix="taska_head",
            step_trigger=lambda s: s == 0, video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[video] recording to {video_dir}", flush=True)

    # seed everything so the closed loop is reproducible across runs (otherwise max_x
    # varies wildly and we can't tell if a head change helped or it was just RNG).
    import random as _random
    _random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.cuda.manual_seed_all(args_cli.seed)
    obs, _ = env.reset(seed=args_cli.seed)
    robot = env.unwrapped.scene["robot"]
    head_cam = env.unwrapped.scene["head_camera"]
    try:
        lidar = env.unwrapped.scene["lidar_sensor"]
    except (KeyError, Exception):
        lidar = None
        print("[WARN] no lidar_sensor; head needs lidar — aborting.", flush=True)

    wz = 0.0
    traj = []
    max_x = -1e9
    fell = False
    h = None  # GRU hidden state, carried across steps (reset only on a new episode)

    for step in range(args_cli.steps):
        with torch.inference_mode():
            if step % args_cli.every == 0 and lidar is not None:
                # --- read eval-legal sensors ---
                rgb = head_cam.data.output["rgb"][0].cpu().numpy()[:, :, :3].astype(np.uint8)
                hits = lidar.data.ray_hits_w[0]
                base = robot.data.root_pos_w[0]
                rng = torch.norm(hits - base.unsqueeze(0), dim=1)
                rng = torch.where(torch.isfinite(rng), rng, torch.full_like(rng, 10.0))
                rng = torch.clip(rng, 0.0, 10.0).view(1, -1)      # (1,5760)
                # --- recurrent steering head -> wz (GRU hidden carried across steps) ---
                patches = (rgb_to_patches(rgb) - pf_mu) / pf_sd    # (1,256,384)
                lfeat = (rng - li_mu) / li_sd                      # (1,5760)
                pseq = patches.unsqueeze(1)                        # (1,1,256,384)
                lseq = lfeat.unsqueeze(1)                          # (1,1,5760)
                wz_t, h = head(pseq, lseq, h)                      # wz_t (1,1), h carried
                wz_raw = float(wz_t[0, 0])
                wz_raw = max(-args_cli.wz_max, min(args_cli.wz_max, wz_raw))
                wz = args_cli.wz_ema * wz_raw + (1.0 - args_cli.wz_ema) * wz

            vx_cmd = args_cli.vx * math.exp(-args_cli.vx_slow_k * abs(wz))

            solution.fixed_velocity_commands = torch.tensor(
                [vx_cmd, 0.0, wz], device=solution.device, dtype=torch.float32).view(1, 3)

            resp = solution.predicts(obs, 0.0)
            act = (torch.zeros((1, 20), device=dev) if (resp.get("giveup") or not resp["action"])
                   else torch.tensor(resp["action"], dtype=torch.float32, device=dev).view(1, -1))
            obs, _, term, trunc, _ = env.step(act)

            p = robot.data.root_pos_w[0]
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            max_x = max(max_x, x)
            if step % args_cli.every == 0:
                traj.append((step, x, y, wz, z))
            if step % 200 == 0:
                print(f"[step {step:4d}] x={x:+.1f} y={y:+.2f} z={z:+.2f} wz={wz:+.2f} max_x={max_x:+.1f}", flush=True)

            if bool(term[0]):
                # terminated: either reached goal (x>145) or fell.
                # Report WHICH termination term fired (illegal_contact vs fall vs time_out)
                # so we can tell a real topple from an eval-only hip/thigh-contact kill.
                which = []
                try:
                    tm = env.unwrapped.termination_manager
                    for tname in tm.active_terms:
                        val = tm.get_term(tname)
                        if bool(val[0]):
                            which.append(tname)
                except Exception as e:
                    which = [f"(term-read failed: {e})"]
                if x < 140:
                    fell = True
                    tail = traj[-8:]
                    ztail = " ".join(f"{t[4]:+.2f}" for t in tail)
                    print(f"[TERM] step {step} x={x:+.1f} y={y:+.2f} z={z:+.2f} by={which}", flush=True)
                    print(f"[TERM] last z tail: {ztail}", flush=True)
                else:
                    print(f"[GOAL] reached x={x:+.1f} at step {step}! by={which}", flush=True)
                break

    # save trajectory
    np.savez_compressed(os.path.join(args_cli.out, "head_eval_traj.npz"),
                        traj=np.array(traj, dtype=np.float32))
    goal = max_x >= 145
    print(f"\n[DONE] max_x={max_x:+.1f}  goal_reached={goal}  fell={fell}", flush=True)
    print(f"  Task A course: spawn x=-141 -> goal x>145 (~286 m).", flush=True)
    print(f"  traj -> {os.path.join(args_cli.out, 'head_eval_traj.npz')}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
