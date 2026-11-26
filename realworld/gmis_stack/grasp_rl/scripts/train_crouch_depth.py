"""Depth-parameterized B2 crouch trainer (RW-portable copy for the new-depth pipeline).

Adapted from the main repo's logs/rsl_rl/260630/scripts/train_crouch_depth.py: overrides the
commanded crouch height on both height reward terms + the base_too_low termination floor,
warm-starts from a donor checkpoint (model only, optimizer reset, iteration 0), and logs
DIRECTLY under logs/rsl_rl/crouch_sweep/runs/crouch<tag>_<stamp>/ — the location the grasp
env's frozen-leg glob (crouch{DEPTH}_*/model_2999.pt) resolves. Existing runs are untouched.

Run:
  cd real_world_preliminary/ATEC2026_Simulation_Challenge
  CUDA_VISIBLE_DEVICES=5 ~/.conda/envs/isaaclab/bin/python \
      grasp_rl/scripts/train_crouch_depth.py --headless --device cuda:0 \
      --crouch_h 0.21 --donor logs/rsl_rl/final_minju/model_9998.pt
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
GRASP_RL = _SCRIPT.parents[1]
RW = _SCRIPT.parents[2]
for _p in (str(RW / "source" / "atec_rl_lab"), str(GRASP_RL)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Train a B2 crouch policy for a given target depth.")
parser.add_argument("--task", type=str, default="ATEC-Crouch-B2-v0")
parser.add_argument("--crouch_h", type=float, required=True, help="Commanded crouch base height [m].")
parser.add_argument(
    "--base_too_low", type=float, default=None,
    help="base_too_low termination height [m]; default crouch_h - 0.05 (floor 0.07).",
)
parser.add_argument("--tag", type=str, default=None,
                    help="Run dir tag: crouch<tag>_<stamp>; default = %g of crouch_h.")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--donor", type=str, required=True, help="Checkpoint to warm-start from (model only).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.base_too_low is None:
    args_cli.base_too_low = max(args_cli.crouch_h - 0.05, 0.07)
if args_cli.base_too_low >= args_cli.crouch_h:
    parser.error("--base_too_low must be below --crouch_h (else episodes terminate on success)")
if args_cli.tag is None:
    args_cli.tag = f"{args_cli.crouch_h:g}"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import crouchpkg  # noqa: E402, F401  registers ATEC-Crouch-B2-v0 from OUR copy
from crouchpkg import mdp_crouch  # noqa: E402

assert str(GRASP_RL) in mdp_crouch.__file__, f"crouchpkg resolved outside grasp_rl: {mdp_crouch.__file__}"


def main():
    device = args_cli.device if getattr(args_cli, "device", None) else "cuda:0"

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.device = device
    agent_cfg.seed = args_cli.seed
    env_cfg.seed = args_cli.seed

    env_cfg.rewards.commanded_height_l2.params["crouch_h"] = args_cli.crouch_h
    env_cfg.rewards.crouch_height.params["crouch_h"] = args_cli.crouch_h
    env_cfg.terminations.base_too_low.params["minimum_height"] = args_cli.base_too_low
    print(f"[INFO] crouch_h={args_cli.crouch_h} base_too_low={args_cli.base_too_low}", flush=True)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = str(RW / "logs" / "rsl_rl" / "crouch_sweep" / "runs" / f"crouch{args_cli.tag}_{stamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] task={args_cli.task} num_envs={env_cfg.scene.num_envs} "
          f"iters={agent_cfg.max_iterations} device={device} log_dir={log_dir}", flush=True)

    genv = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    genv = RslRlVecEnvWrapper(genv)

    runner = OnPolicyRunner(genv, agent_cfg.to_dict(), log_dir=log_dir, device=device)

    print(f"[INFO] warm-start load (model only): {args_cli.donor}", flush=True)
    runner.load(args_cli.donor, load_optimizer=False)
    runner.current_learning_iteration = 0

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    genv.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
