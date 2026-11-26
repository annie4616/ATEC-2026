"""Quick scripted test: can B2wPiper ROLL its wheels up the Task D box?

No learning — just drive the 4 foot-wheels forward (velocity) with the legs held at a
neutral crouch, point the robot at the box, and watch (video) whether the wheels carry it
up the 0.6 m box / platform. This validates the 'roll-up' hypothesis before investing in
a full walking+climb policy for B2wPiper.

Action layout (B2wPiper, 24-D): [leg_pos(12), wheel_vel(4), arm_pos(8)].
Run (env_isaaclab):
  CUDA_VISIBLE_DEVICES=2 TMPDIR=/hdd0/sohyun/tmp PYTHONPATH=/hdd0/sohyun \
  python scripts/test_b2w_wheel_climb.py --task ATEC-TaskD-B2wPiper --num_envs 1 \
  --headless --enable_cameras --video --video_length 500
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=500)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="ATEC-TaskD-B2wPiper")
parser.add_argument("--wheel_vel", type=float, default=8.0, help="forward wheel velocity (rad/s)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import torch
import gymnasium as gym
import atec_rl_lab.tasks  # noqa: F401 register
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(env, video_folder=os.path.abspath(
            os.path.join("logs", "videos", args_cli.task, "wheeltest")),
            step_trigger=lambda s: s == 0, video_length=args_cli.video_length, disable_logger=True)
    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    # neutral leg crouch (B2 default leg pose: hip ~0, thigh ~0.8, calf ~-1.5)
    # action layout = [leg_pos(12), wheel_vel(4), arm_pos(8)]
    leg_default = torch.tensor([-0.1, 0.8, -1.5, 0.1, 0.8, -1.5,
                                -0.1, 1.0, -1.5, 0.1, 1.0, -1.5], device="cuda")
    t = 0
    box0 = env.unwrapped.scene["box"].data.root_pos_w[0].clone()
    while simulation_app.is_running():
        with torch.inference_mode():
            act = torch.zeros((1, 24), device="cuda")
            act[0, 0:12] = leg_default                  # legs hold crouch (position target)
            act[0, 12:16] = args_cli.wheel_vel          # 4 wheels roll forward (velocity)
            act[0, 16:24] = 0.0                         # arm neutral
            obs, reward, term, trunc, info = env.step(act)
            if t % 25 == 0:
                r = robot.data.root_pos_w[0]
                b = env.unwrapped.scene["box"].data.root_pos_w[0]
                print(f"[t={t:4d}] robot=({r[0]:+.2f},{r[1]:+.2f},{r[2]:+.2f}) "
                      f"box=({b[0]:+.2f},{b[1]:+.2f},{b[2]:+.2f}) "
                      f"box_dx={float(b[0]-box0[0]):+.2f} robot_z={float(r[2]):.2f}", flush=True)
            t += 1
            if (term.item() or trunc.item()) or (args_cli.video and t >= args_cli.video_length):
                break
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
