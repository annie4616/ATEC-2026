"""Render a top-down (and an angled) SNAPSHOT of a Task A env so we can SEE the course
layout -- walls, red strips, terrain. Saves PNG(s). No head needed; the robot just stands.

  CUDA_VISIBLE_DEVICES=6 TMPDIR=/hdd0/sohyun/tmp PYTHONPATH=/hdd0/sohyun \
    python scripts/steering_head_wz/snapshot_taska_env.py --headless \
      --task ATEC-TaskA-B2Piper-Maze --out .../maze_snap
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="ATEC-TaskA-B2Piper-Maze")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--cam_eye", type=float, nargs=3, default=[-110.0, 6.0, 55.0],
                    help="viewer camera world position (default high above the maze)")
parser.add_argument("--cam_lookat", type=float, nargs=3, default=[-110.0, 6.0, 0.0],
                    help="viewer camera look-at (default straight down)")
parser.add_argument("--settle", type=int, default=40, help="steps to let the robot settle")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import json  # noqa: E402
import imageio  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec  # noqa: E402
from atec_2026_tesk_e.ATEC2026_Simulation_Challenge.demo.solution_rl import AlgSolution  # noqa: E402

solution = AlgSolution()


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    dev = "cuda"
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    env_cfg = apply_safe_action_spec(env_cfg, json.dumps(action_spec))
    env_cfg.terminations.illegal_contact = None

    # FIXED viewer camera (not following the robot) so we see the whole course layout.
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye = tuple(args_cli.cam_eye)
    env_cfg.viewer.lookat = tuple(args_cli.cam_lookat)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    obs, _ = env.reset(seed=0)

    # let the robot settle and render a few frames (first frame is often black)
    frames = []
    for step in range(args_cli.settle):
        solution.fixed_velocity_commands = torch.tensor(
            [0.0, 0.0, 0.0], device=solution.device, dtype=torch.float32).view(1, 3)
        resp = solution.predicts(obs, 0.0)
        act = (torch.zeros((1, 20), device=dev)
               if (resp.get("giveup") or not resp["action"])
               else torch.tensor(resp["action"], device=dev, dtype=torch.float32).view(1, -1))
        obs, _, term, trunc, _ = env.step(act)
        img = env.render()
        if img is not None and step >= args_cli.settle - 3:
            frames.append(np.asarray(img))

    if frames:
        out_png = os.path.join(args_cli.out, "snapshot.png")
        imageio.imwrite(out_png, frames[-1])
        print(f"[snapshot] saved {out_png}  shape={frames[-1].shape}", flush=True)
    else:
        print("[WARN] no frames captured", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
