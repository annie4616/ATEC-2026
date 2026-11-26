"""Record the A2 + Piper X RMA blind locomotion policy on the Task A (Hiking) straight course.

Blind proprioceptive RMA walker: no lidar, no cameras. The solution (solution_taska_a2_rma.py)
builds a 45-dim proprio, runs the phase-2 adaptation module to estimate the env latent, and the
phase-1 actor outputs 12 leg targets (arm held at default). A forward velocity command with a
dead-reckoned yaw-hold drives it straight down +x.

Run (GPU 2, MPS bypass required on this box):
  CUDA_VISIBLE_DEVICES=2 CUDA_MPS_PIPE_DIRECTORY=/tmp/nonexistent_x TMPDIR=/hdd0/sohyun/tmp \
  python scripts/steering_ytrack/play_taska_a2_rma.py \
    --task ATEC-TaskA-A2Piper --video --enable_cameras --no_terminate --video_length 1500 \
    --video_dir /hdd0/sohyun/task_a_real_exp/a2_rma
"""

import argparse
import os
import time
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record the A2 RMA blind walker on Task A.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=1500)
parser.add_argument("--video_dir", type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="ATEC-TaskA-A2Piper")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--no_terminate", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# My A2 Task A env disables the head/ee cameras in-cfg, leaving no RTX sensors -> --enable_cameras
# does NOT flip the sim into a rendering mode, so RecordVideo's render() would raise
# NO_GUI_OR_RENDERING. Force offscreen rendering on for video runs.
if args_cli.video:
    import carb  # noqa: E402
    carb.settings.get_settings().set("/isaaclab/render/offscreen", True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

import sys  # noqa: E402
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../scripts
_REPO_DIR = os.path.dirname(_SCRIPTS_DIR)                                    # repo root
for _p in (os.path.join(_SCRIPTS_DIR, "common"), os.path.join(_REPO_DIR, "demo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rl_utils import camera_follow  # noqa: E402  (scripts/common/rl_utils.py)
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec  # noqa: E402

from solution_taska_a2_rma import AlgSolution  # noqa: E402  (demo/)

solution = AlgSolution()


def play() -> tuple[float, float]:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device,
        num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric,
    )
    if hasattr(env_cfg.sim, "save_logs_to_file"):
        env_cfg.sim.save_logs_to_file = False
    if hasattr(env_cfg, "wait_for_textures"):
        env_cfg.wait_for_textures = False

    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    env_cfg = apply_safe_action_spec(env_cfg, json.dumps(action_spec))

    if args_cli.no_terminate:
        for _name in ("fall", "illegal_contact", "base_contact"):
            if hasattr(env_cfg.terminations, _name):
                setattr(env_cfg.terminations, _name, None)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_dir = args_cli.video_dir or os.path.abspath(
            os.path.join("logs", "videos", args_cli.task, "play"))
        os.makedirs(video_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": "taska_a2_rma",
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    obs, _ = env.reset()
    if hasattr(solution, "reset"):
        solution.reset()

    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
    timestep = 0
    total_episode_reward = 0.0
    total_elapsed_time = 0.0

    while simulation_app.is_running():
        with torch.inference_mode():
            start_time = time.time()
            resp = solution.predicts(obs, total_episode_reward)
            if resp["giveup"]:
                break
            actions = torch.tensor(resp["action"], dtype=torch.float32, device="cuda").view(1, -1)
            obs, reward, terminated, truncated, info = env.step(actions)
            if args_cli.video:
                try:
                    camera_follow(env)
                except Exception:
                    pass

            if timestep % 50 == 0:
                _rob = env.unwrapped.scene["robot"]
                _p = _rob.data.root_pos_w[0]
                _q = _rob.data.root_quat_w[0]
                import math as _m
                _yaw = _m.atan2(2*(float(_q[0])*float(_q[3]) + float(_q[1])*float(_q[2])),
                                1 - 2*(float(_q[2])**2 + float(_q[3])**2))
                print(f"[TRAJ t={timestep:4d}] x={float(_p[0]):+.2f} y={float(_p[1]):+.2f} "
                      f"z={float(_p[2]):+.2f} yaw={_yaw:+.2f}rad", flush=True)

            sim_dt = info.get("Step_dt", dt or 0.02)
            r = reward.mean().item() if isinstance(reward, torch.Tensor) else float(reward)
            total_episode_reward += r / sim_dt if sim_dt else r
            if isinstance(info, dict) and "Elapsed_Time" in info:
                elapsed = info["Elapsed_Time"]
                total_elapsed_time = elapsed.item() if hasattr(elapsed, "item") else float(elapsed)
            elif dt is not None:
                total_elapsed_time += dt

            done = (terminated.item() or truncated.item())
            if done and not args_cli.no_terminate:
                print(f"[INFO] episode done at step {timestep}.", flush=True)
                break

            timestep += 1
            if args_cli.video and timestep >= args_cli.video_length:
                break
            if args_cli.real_time and dt is not None:
                st = dt - (time.time() - start_time)
                if st > 0:
                    time.sleep(st)

    print(f"[DONE] steps={timestep} total_reward={total_episode_reward:.1f} "
          f"elapsed={total_elapsed_time:.1f}s", flush=True)
    env.close()
    return total_episode_reward, total_elapsed_time


if __name__ == "__main__":
    play()
    simulation_app.close()
