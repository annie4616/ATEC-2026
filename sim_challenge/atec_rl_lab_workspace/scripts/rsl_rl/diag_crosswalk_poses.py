"""Diagnostic: print the ACTUAL world poses of robot/box/pit after reset in the
CrossWalk env, to verify the demo-matched geometry (box half-sunk in pit center,
robot on the -y lane in front). Reuses train.py's exact import order so rsl_rl
loads cleanly.

Run (env_isaaclab):
  CUDA_VISIBLE_DEVICES=2 TMPDIR=/hdd0/sohyun/tmp PYTHONPATH=/hdd0/sohyun \
  python scripts/rsl_rl/diag_crosswalk_poses.py --headless
"""
import argparse
import atec_2026_tesk_e.ATEC2026_Simulation_Challenge.scripts.rsl_rl.cli_args as cli_args  # noqa  (same as train.py line 12)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="ATEC-TaskD-RL-CrossWalk-Unitree-B2-v0")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import atec_rl_lab.train  # noqa: F401  register RL tasks (after app + cli_args, like train.py)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    act = torch.zeros((args_cli.num_envs, env.action_space.shape[1]), device=args_cli.device)
    # settle a few steps
    for _ in range(20):
        env.step(act)
    u = env.unwrapped
    robot = u.scene["robot"]; box = u.scene["box"]
    origins = u.scene.env_origins
    rp = robot.data.root_pos_w
    bp = box.data.root_pos_w
    print("\n========== CROSSWALK POSE DIAG (world frame) ==========", flush=True)
    for i in range(args_cli.num_envs):
        o = origins[i].cpu().numpy()
        r = (rp[i] - origins[i]).cpu().numpy()   # env-local
        b = (bp[i] - origins[i]).cpu().numpy()
        rw = rp[i].cpu().numpy(); bw = bp[i].cpu().numpy()
        print(f"env{i}: origin=({o[0]:+.2f},{o[1]:+.2f},{o[2]:+.2f})", flush=True)
        print(f"   robot WORLD=({rw[0]:+.2f},{rw[1]:+.2f},{rw[2]:+.2f})  env-local=({r[0]:+.2f},{r[1]:+.2f},{r[2]:+.2f})", flush=True)
        print(f"   box   WORLD=({bw[0]:+.2f},{bw[1]:+.2f},{bw[2]:+.2f})  env-local=({b[0]:+.2f},{b[1]:+.2f},{b[2]:+.2f})", flush=True)
    print("\nDEMO TARGET (world): robot~(-0.6,-0.45,0.55)  box~(-0.25,-0.45,-0.7, top -0.4)", flush=True)
    print("pit (world): x in [-0.67,+0.67], full y; platform on +y half only; floor z=-1.0", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
