"""Hand-scripted PLATFORM strategy experiment (Task D).

New idea: instead of dropping the box into the pit, push the box up against the
platform's -y wall and CLIMB +y onto the platform top, then walk +x off the far
side toward x > 3.5. See task_d/push_controller_platform.py for the geometry.

This script:
  1. resets the env (box + robot to init), settles a few steps,
  2. PROBES and prints the real world-frame geometry (robot/box/platform/pit z),
     so the controller's climb_x / wall_y / platform_top_z can be corrected,
  3. drives the PushToPlatformController and records a third-person video,
  4. reports robot max x / max z and whether it crossed (x > 3.5).

Usage:
  python scripts/act/experiment_platform.py --max_steps 2500 --headless --enable_cameras
  # tune the parking/climb x after watching the video:
  python scripts/act/experiment_platform.py --climb_x 1.0 --platform_top_z 1.45 ...
  # probe geometry only (no controller, just print + short video):
  python scripts/act/experiment_platform.py --probe_only --max_steps 200 ...
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Platform-climb experiment for Task D.")
parser.add_argument(
    "--policy_path",
    type=str,
    default="logs/rsl_rl/unitree_b2_rough/2026-06-18_04-37-10/exported/policy.pt",
)
parser.add_argument("--output_dir", type=str, default="./datasets/atec_task_d_platform")
parser.add_argument("--run_tag", type=str, default="")
parser.add_argument("--max_steps", type=int, default=2500)
parser.add_argument("--park_x", type=float, default=0.0,
                    help="box center x to park at (its +x face ~ +0.4 = platform front)")
parser.add_argument("--climb_y", type=float, default=1.6,
                    help="y-line to push/climb along (box start y, inside platform band)")
parser.add_argument("--platform_top_z", type=float, default=1.4)
parser.add_argument("--platform_front_x", type=float, default=0.4)
parser.add_argument("--probe_only", action="store_true", default=False)
parser.add_argument("--diag", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
import numpy as np
import torch
import imageio

from isaaclab.envs import ManagerBasedRLEnv
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg

from task_d.walking_oracle import WalkingOracle
from task_d.collector import _reset_asset_root
from task_d.push_controller_platform import PushToPlatformController

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_utils import camera_follow  # noqa: E402

SUCCESS_X = 3.5


def probe_platform_z(env, dev, xs, ys):
    """Drop a tiny marker is overkill; instead read terrain by sampling the box.

    We can't easily raycast here, so we just report the configured derivation and
    let the printed box/robot poses + video confirm. Returns nothing; prints.
    """
    pass


def main() -> None:
    os.makedirs(args_cli.output_dir, exist_ok=True)

    cfg = TaskDEnvB2Cfg()
    cfg.seed = int(time.time_ns() % (2**31))
    cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg, render_mode="rgb_array")
    dev = env.unwrapped.device

    if not os.path.isfile(args_cli.policy_path):
        raise FileNotFoundError(f"Walking policy not found: {args_cli.policy_path}")

    robot = env.unwrapped.scene["robot"]
    box = env.unwrapped.scene["box"]
    oracle = WalkingOracle(args_cli.policy_path, robot, dev)

    ctrl = PushToPlatformController(
        park_x=args_cli.park_x,
        climb_y=args_cli.climb_y,
        platform_top_z=args_cli.platform_top_z,
        platform_front_x=args_cli.platform_front_x,
    )

    env.reset()
    _reset_asset_root(robot)
    _reset_asset_root(box)
    oracle.reset()
    ctrl.reset()

    # settle on the terrain (zero command)
    oracle.set_velocity_command(0.0, 0.0, 0.0)
    for _ in range(20):
        env.step(oracle.compute_action())

    org = env.unwrapped.scene.env_origins[0]
    rp = robot.data.root_pos_w[0]
    bp = box.data.root_pos_w[0]
    print("\n==================== GEOMETRY PROBE ====================")
    print(f"env_origin       = ({org[0].item():+.3f}, {org[1].item():+.3f}, {org[2].item():+.3f})")
    print(f"robot root (world)= ({rp[0].item():+.3f}, {rp[1].item():+.3f}, {rp[2].item():+.3f})")
    print(f"box   root (world)= ({bp[0].item():+.3f}, {bp[1].item():+.3f}, {bp[2].item():+.3f})")
    print(f"controller targets: park_x={args_cli.park_x} climb_y={args_cli.climb_y} "
          f"platform_top_z={args_cli.platform_top_z} platform_front_x={args_cli.platform_front_x}")
    print("Derived platform (from terrain.py): world x in [+0.4,+2.0], "
          "y in [+0.5,+3.5], top z ~ +1.4..1.5; pit on -y side, floor z=-1.0")
    print("========================================================\n")

    frames = []
    max_x, max_z = -99.0, -99.0
    crossed = False

    for step in range(args_cli.max_steps):
        rp = robot.data.root_pos_w[0]
        bp = box.data.root_pos_w[0]

        if args_cli.probe_only:
            oracle.set_velocity_command(0.0, 0.0, 0.0)
        else:
            vx, vy, wz = ctrl.compute_velocity_command(
                robot.data.root_pos_w[0],
                box.data.root_pos_w[0],
                robot.data.root_quat_w[0],
            )
            oracle.set_velocity_command(vx, vy, wz)

        action = oracle.compute_action()
        camera_follow(env)
        frame = env.render()
        if frame is not None:
            frames.append(frame[:, :, :3])
        env.step(action)

        rx, ry, rz = rp[0].item(), rp[1].item(), rp[2].item()
        max_x = max(max_x, rx)
        max_z = max(max_z, rz)

        if args_cli.diag and step % 25 == 0:
            st = "probe" if args_cli.probe_only else ctrl.state
            print(f"[step {step:4d}] robot=({rx:+.2f},{ry:+.2f},{rz:+.2f}) "
                  f"box=({bp[0].item():+.2f},{bp[1].item():+.2f},{bp[2].item():+.2f}) state={st}")

        # Real success = reached x>3.5 while UPRIGHT (not free-falling off an edge).
        # try2 logged a false crossed=True because the robot's x drifted past 3.5
        # mid-fall (z went to -50). Require z above the fall threshold.
        if rx > SUCCESS_X and rz > -0.4:
            crossed = True
            break
        # Abort early if the robot has clearly fallen off into the void.
        if rz < -2.0:
            print(f"[INFO] robot fell into void at step {step} (z={rz:.1f}) -> abort")
            break

    tag = f"_{args_cli.run_tag}" if args_cli.run_tag else ""
    vp = os.path.join(args_cli.output_dir, f"platform{tag}.mp4")
    if frames:
        imageio.mimwrite(vp, frames, fps=50, quality=7)
    print(f"\n[RESULT] max_x={max_x:+.2f} max_z={max_z:+.2f} crossed={crossed} "
          f"final_state={'probe' if args_cli.probe_only else ctrl.state} -> {vp}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
