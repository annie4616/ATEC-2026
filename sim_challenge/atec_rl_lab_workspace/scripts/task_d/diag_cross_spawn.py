import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app

import torch
import gymnasium as gym
# Mirror scripts/rsl_rl/train.py import order so the rsl_rl plumbing initializes
# before the task registry is imported.
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import atec_rl_lab.train  # noqa: F401  register train tasks


def main():
    cfg = parse_env_cfg("ATEC-TaskD-RL-Cross-Unitree-B2-v0", num_envs=1)
    env = gym.make("ATEC-TaskD-RL-Cross-Unitree-B2-v0", cfg=cfg)
    env.reset()
    u = env.unwrapped
    env.step(torch.zeros((1, u.action_manager.total_action_dim), device=u.device))
    sc = u.scene
    org = sc.env_origins[0].tolist()
    rp = sc["robot"].data.root_pos_w[0].tolist()
    bp = sc["box"].data.root_pos_w[0].tolist()
    print("CROSS_ENV_ORIGIN", org, flush=True)
    print(f"CROSS robot world: x={rp[0]:+.3f} y={rp[1]:+.3f} z={rp[2]:+.3f}", flush=True)
    print(f"CROSS box   world: x={bp[0]:+.3f} y={bp[1]:+.3f} z={bp[2]:+.3f}", flush=True)
    pit_ok = -0.4 <= bp[0] <= 0.4
    print(f"--> pit is world x in [-0.4,+0.4]; box in pit? {pit_ok} (box z should be ~-0.7)", flush=True)
    env.close()


if __name__ == "__main__":
    main(); simulation_app.close()
