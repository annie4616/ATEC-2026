"""Sweep the CROSS speed and record a third-person video per speed.

Idea: after the box is pushed in and settles as a stepping stone, drive the robot
across the pit. Raise the crossing speed to the highest value where the robot rides
the box over the 0.4 m step and reaches x > 3.5 WITHOUT toppling. Each speed gets one
attempt and one MP4 in --output_dir, plus a printed summary (max robot x, fell?, crossed?).

Usage
-----
python scripts/act/experiment_cross_speed.py \
    --policy_path <walking policy.pt> \
    --speeds 1.3 1.8 2.3 2.8 \
    --output_dir ./datasets/atec_task_d_test \
    --max_steps 1300 --headless --enable_cameras
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CROSS-speed sweep for Task D.")
parser.add_argument("--policy_path", type=str, required=True)
parser.add_argument("--speeds", type=float, nargs="+",
                    default=[1.3, 1.8, 2.3, 2.8],
                    help="CROSS speeds to try (also used as the post-box sprint speed).")
parser.add_argument("--output_dir", type=str, default="./datasets/atec_task_d_test")
parser.add_argument("--run_tag", type=str, default="",
                    help="Suffix appended to each video filename so runs don't overwrite "
                         "(e.g. a timestamp passed from the shell).")
parser.add_argument("--max_steps", type=int, default=1300)
parser.add_argument("--diag", action="store_true", default=False)
parser.add_argument("--no_rotate", action="store_true", default=False,
                    help="Skip the 90-deg box rotation; push the box in spawn orientation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# A camera/third-person viewport is needed for the video.
args_cli.enable_cameras = True

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
import numpy as np
import imageio

from isaaclab.envs import ManagerBasedRLEnv
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg

from task_d.walking_oracle import WalkingOracle
from task_d.collector import _reset_asset_root
from task_d.push_controller_cross import PushIntoPitCrossController

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_utils import camera_follow  # noqa: E402

SUCCESS_X = 3.5
FALLEN_Z  = 0.30   # robot base z below this for a sustained stretch == fallen


def build_env():
    cfg = TaskDEnvB2Cfg()
    cfg.seed           = int(time.time_ns() % (2**31))
    cfg.scene.num_envs = 1
    return ManagerBasedRLEnv(cfg, render_mode="rgb_array")


def run_one(env, oracle, robot, box, dev, v_cross, max_steps, diag):
    """One crossing attempt at the given speed; returns (frames, summary dict)."""
    env.reset()
    _reset_asset_root(robot)
    _reset_asset_root(box)
    oracle.reset()
    push = PushIntoPitCrossController(v_cross=v_cross, v_forward_cross=v_cross,
                                      enable_rotate=not args_cli.no_rotate)
    push.reset()

    # Warm-up (let the robot settle).
    for _ in range(10):
        env.step(oracle.compute_action())

    frames = []
    max_x = -99.0
    fallen_steps = 0
    fell = False
    crossed = False
    for step in range(max_steps):
        vx, vy, wz = push.compute_velocity_command(
            robot.data.root_pos_w[0], box.data.root_pos_w[0],
            robot.data.root_quat_w[0], box.data.root_quat_w[0],
        )
        oracle.set_velocity_command(vx, vy, wz)
        action = oracle.compute_action()

        camera_follow(env)
        frame = env.render()
        if frame is not None:
            frames.append(frame[:, :, :3])

        env.step(action)

        rx = robot.data.root_pos_w[0, 0].item()
        rz = robot.data.root_pos_w[0, 2].item()
        max_x = max(max_x, rx)
        if rz < FALLEN_Z:
            fallen_steps += 1
        else:
            fallen_steps = 0

        if diag and step % 50 == 0:
            ry = robot.data.root_pos_w[0, 1].item()
            bx = box.data.root_pos_w[0, 0].item()
            by = box.data.root_pos_w[0, 1].item()
            bz = box.data.root_pos_w[0, 2].item()
            byaw = push._yaw_from_quat(box.data.root_quat_w[0])
            print(f"    [v={v_cross:.1f} step {step:4d}] robot=({rx:+.2f},{ry:+.2f},{rz:+.2f}) "
                  f"box=({bx:+.2f},{by:+.2f},{bz:+.2f}) box_yaw={byaw:+.2f} state={push.state}")

        if rx > SUCCESS_X:
            crossed = True
            break
        if fallen_steps >= 80:
            fell = True
            break

    return frames, {"v_cross": v_cross, "max_x": max_x, "crossed": crossed,
                    "fell": fell, "steps": step + 1}


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)
    env   = build_env()
    dev   = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    box   = env.unwrapped.scene["box"]
    oracle = WalkingOracle(args_cli.policy_path, robot, dev)

    summaries = []
    for v in args_cli.speeds:
        print(f"\n[INFO] === CROSS speed v={v:.2f} ===")
        frames, summ = run_one(env, oracle, robot, box, dev, v,
                               args_cli.max_steps, args_cli.diag)
        tag = f"_{args_cli.run_tag}" if args_cli.run_tag else ""
        vp = os.path.join(args_cli.output_dir, f"cross_v{v:.2f}{tag}.mp4")
        if frames:
            imageio.mimwrite(vp, frames, fps=50, quality=7)
            summ["video"] = vp
        print(f"[INFO] v={v:.2f}: max_x={summ['max_x']:+.2f} crossed={summ['crossed']} "
              f"fell={summ['fell']} steps={summ['steps']} -> {vp}")
        summaries.append(summ)

    print("\n[RESULT] CROSS-speed sweep:")
    for s in summaries:
        tag = "CROSSED" if s["crossed"] else ("FELL" if s["fell"] else "stalled")
        print(f"  v={s['v_cross']:.2f}  max_x={s['max_x']:+.2f}  {tag}  ({s['steps']} steps)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
