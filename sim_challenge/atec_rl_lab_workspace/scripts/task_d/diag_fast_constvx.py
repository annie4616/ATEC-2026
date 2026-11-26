"""Controlled test: does the fast (vx<=3.5) policy actually WALK at a constant
high forward command, or does it stall? Force vel_command_b = [VX, 0, 0] every
step (bypassing the random resampler + the 0.2 threshold-to-zero), run the trained
policy, and log commanded-vs-actual base velocity and x-displacement.

Run:
  CUDA_VISIBLE_DEVICES=5 TMPDIR=/hdd0/sohyun/tmp PYTHONPATH=/hdd0/sohyun \
    python scripts/diag_fast_constvx.py --headless --enable_cameras --vx 3.5
"""
import argparse
import os as _os
import sys as _sys

# This script lives in scripts/, so Python puts scripts/ on sys.path[0] -> the
# local scripts/rsl_rl/ package SHADOWS the installed `rsl_rl` (which has the
# `rsl_rl.env` submodule that isaaclab_rl needs). Drop the scripts dir from the
# path so the real rsl_rl resolves.
_sys.path = [p for p in _sys.path if _os.path.abspath(p) != _os.path.dirname(_os.path.abspath(__file__))]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--vx", type=float, default=3.5)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--checkpoint", type=str,
                    default="logs/rsl_rl/unitree_b2_fast/2026-06-22_07-55-23/model_3998.pt")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import torch
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
import atec_rl_lab.train  # noqa: F401


def main():
    cfg = parse_env_cfg("ATEC-Isaac-Velocity-Fast-Unitree-B2-v0", num_envs=1)
    # Long episode, no early reset, so we observe sustained walking.
    cfg.episode_length_s = 60.0
    env = gym.make("ATEC-Isaac-Velocity-Fast-Unitree-B2-v0", cfg=cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    # Load the trained policy.
    from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.agents.rsl_rl_ppo_cfg import (
        UnitreeB2FastPPORunnerCfg,
    )
    agent_cfg = UnitreeB2FastPPORunnerCfg()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
    runner.load(os.path.abspath(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped
    cmd_term = u.command_manager.get_term("base_velocity")
    robot = u.scene["robot"]

    obs, _ = env.reset()
    x0 = robot.data.root_pos_w[0, 0].item()
    print(f">>> const-vx test: vx_cmd={args_cli.vx}, x0={x0:+.3f}", flush=True)
    print("step | cmd[vx,vy,wz]      | actual base_vel_b[vx,vy] | world(x,y) | dx_from_start", flush=True)
    for i in range(args_cli.steps):
        # FORCE the command every step (overwrite whatever the resampler/threshold set).
        cmd_term.vel_command_b[:, 0] = args_cli.vx
        cmd_term.vel_command_b[:, 1] = 0.0
        cmd_term.vel_command_b[:, 2] = 0.0
        with torch.inference_mode():
            act = policy(obs)
        obs, _, _, _ = env.step(act)
        # re-force after step's command update too (defensive)
        cmd_term.vel_command_b[:, 0] = args_cli.vx
        cmd_term.vel_command_b[:, 1] = 0.0
        cmd_term.vel_command_b[:, 2] = 0.0
        if i % 25 == 0 or i == args_cli.steps - 1:
            c = cmd_term.command[0].tolist()
            vb = robot.data.root_lin_vel_b[0, :2].tolist()
            wx = robot.data.root_pos_w[0, 0].item()
            wy = robot.data.root_pos_w[0, 1].item()
            print(f"{i:4d} | [{c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}] | "
                  f"[{vb[0]:+.2f},{vb[1]:+.2f}] | ({wx:+.2f},{wy:+.2f}) | {wx - x0:+.2f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
