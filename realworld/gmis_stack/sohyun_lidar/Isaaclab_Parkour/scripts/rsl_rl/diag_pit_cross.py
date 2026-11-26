"""Diagnose whether the B2Piper teacher ACTUALLY crosses the TASKD_PIT terrain or
stalls/falls mid-pit. Logs base x (relative to env origin) and base z per step for a
few envs, then reports the final/peak x and the z-profile so we can see if the robot
reaches the far rim (z back to ~0 rim height after dipping) or gets stuck in the pit.

Run (parkour_il):
  CUDA_VISIBLE_DEVICES=2 TMPDIR=/hdd0/sohyun/tmp PYTHONPATH=/hdd0/sohyun/atec_2026_tesk_e/Isaaclab_Parkour \
  PARKOUR_ROBOT=b2piper PARKOUR_NJOINTS=20 python scripts/rsl_rl/diag_pit_cross.py \
  --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-TaskDPitPlay-v0 --num_envs 8 --headless \
  --checkpoint /hdd0/.../model_9900.pt
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import parkour_tasks  # noqa: F401
# importing the go2 config package runs its __init__ which gym.register()s the tasks
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import ParkourRslRlOnPolicyRunnerCfg  # noqa
import parkour_tasks.extreme_parkour_task.config.go2  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor  # noqa
import cli_args  # noqa


def main():
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path  # noqa

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # use the play agent cfg
    import importlib
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=args_cli.device)

    obs, _ = env.get_observations()
    robot = env.unwrapped.scene["robot"]
    org = env.unwrapped.scene.env_origins
    N = min(args_cli.num_envs, 8)

    # per-env max x advance and the z at that moment, plus min z (pit dip) and final
    start_x = (robot.data.root_pos_w[:, 0] - org[:, 0]).clone()
    max_x = torch.full((env.num_envs,), -1e9, device=args_cli.device)
    z_at_maxx = torch.zeros(env.num_envs, device=args_cli.device)
    min_z = torch.full((env.num_envs,), 1e9, device=args_cli.device)

    for t in range(args_cli.steps):
        with torch.inference_mode():
            act = policy(obs)
            obs, _, _, _ = env.step(act)
        rx = robot.data.root_pos_w[:, 0] - org[:, 0]
        rz = robot.data.root_pos_w[:, 2]
        adv = rx - start_x
        upd = adv > (max_x)
        max_x = torch.where(upd, adv, max_x)
        z_at_maxx = torch.where(upd, rz, z_at_maxx)
        min_z = torch.minimum(min_z, rz)
        if t % 40 == 0:
            line = " ".join(
                f"e{i}:x{float(adv[i]):+.2f},z{float(rz[i]):.2f}" for i in range(N))
            print(f"[t{t:3d}] {line}", flush=True)

    print("\n=== SUMMARY (per env) ===", flush=True)
    for i in range(N):
        # crossing = max advance large AND z recovered to rim (~0.4-0.6) at max-x (far side),
        # NOT stuck low in the pit.
        print(f"env{i}: max_adv={float(max_x[i]):+.2f}  z_at_maxadv={float(z_at_maxx[i]):.2f}  "
              f"min_z(pit dip)={float(min_z[i]):.2f}", flush=True)
    print("\nINTERPRET: a TRUE cross has max_adv past the far rim AND z_at_maxadv ~0.4-0.6 "
          "(back on flat far ground). If z_at_maxadv is low (~0.0-0.2) the robot stalled "
          "IN the pit (crossed partway then stuck) = 'crossed partway only'.", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
