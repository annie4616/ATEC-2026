"""Smoke-test + video of the competition course env (zero actions, wide static cam).

Run (only when a GPU slot is free):
  cd real_world_preliminary/ATEC2026_Simulation_Challenge
  CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/isaaclab/bin/python \
      course_sim/scripts/view_course.py --headless --device cuda:0 \
      --task ATEC-Course-A2Piper --out_dir course_sim/videos
"""

import argparse
import os
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
RW = _SCRIPT.parents[2]
for _p in (str(RW / "source" / "atec_rl_lab"), str(RW)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="View the competition course test env.")
parser.add_argument("--task", type=str, default="ATEC-Course-A2Piper")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--out_dir", type=str, default=str(RW / "course_sim" / "videos"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import course_sim  # noqa: F401, E402  (registers ATEC-Course-*)


def main():
    device = args_cli.device if getattr(args_cli, "device", None) else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=1)
    env_cfg.viewer.eye = (-27.0, -14.0, 10.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    os.makedirs(args_cli.out_dir, exist_ok=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env, video_folder=args_cli.out_dir, name_prefix="course",
        step_trigger=lambda step: step == 0, video_length=args_cli.steps, disable_logger=True,
    )
    env.reset()
    uenv = env.unwrapped
    n_act = uenv.action_manager.total_action_dim
    zeros = torch.zeros(1, n_act, device=uenv.device)
    for i in range(args_cli.steps):
        env.step(zeros)
    # report scene composition for verification
    robot = uenv.scene["robot"]
    print(f"[COURSE] robot={args_cli.task} root z={robot.data.root_pos_w[0, 2].item():.3f}")
    n_obj = 0
    for name in list(uenv.scene.rigid_objects.keys()):
        n_obj += 1
    print(f"[COURSE] rigid objects in scene: {n_obj}")
    print("[COURSE] OK")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
