"""Trainer for ATEC-Grasp-B2Piper-v0 (stage 1a/1b via --width_range override).

Stage 1a: narrow width 40-50 mm (default), full mass/friction randomization.
Stage 1b: --width_range 0.030,0.060 --resume <1a checkpoint>.
From scratch by design (locked: crouch checkpoints are NOT loaded as initial weights).

Run (stage 1a):
  cd real_world_preliminary/ATEC2026_Simulation_Challenge
  CUDA_VISIBLE_DEVICES=5 ~/.conda/envs/isaaclab/bin/python \
      grasp_rl/scripts/train_grasp.py --headless --device cuda:0 \
      --num_envs 2048 --max_iterations 5000 --run_name grasp1a
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

parser = argparse.ArgumentParser(description="Train the B2+Piper grasp policy.")
parser.add_argument("--task", type=str, default="ATEC-Grasp-B2Piper-v0",
                    help="ATEC-Grasp-B2Piper-v0 (stage 1, fixed base) or ATEC-Grasp-B2Piper-Stage2-v0.")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=5000)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--width_range", type=str, default="0.040,0.050",
                    help="Grasp-axis width range [m] (1a: 0.040,0.050 / 1b: 0.030,0.060; cap 0.060).")
parser.add_argument("--height_range", type=str, default="0.060,0.130")
parser.add_argument("--length_range", type=str, default=None,
                    help="Object length (y-axis) range [m]; default = width_range. "
                         "Stage 1c uses 0.030,0.250 to cover elongated task-B-like objects.")
parser.add_argument("--run_name", type=str, default="grasp")
parser.add_argument("--resume", type=str, default=None, help="Checkpoint to continue from (stage 1b).")
parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb", "neptune"])
parser.add_argument("--wandb_project", type=str, default="atec_b2piper_grasp")
parser.add_argument("--object_usd", type=str, default=None, help="Train on a USD object instead of primitives.")
parser.add_argument("--crouch_depth", type=str, default="0.2",
                    help="Crouch depth tag (crouch pose, leg-policy checkpoint, pregrasp bank).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
os.environ["GRASP_CROUCH_DEPTH"] = args_cli.crouch_depth

w_lo, w_hi = (float(s) for s in args_cli.width_range.split(","))
h_lo, h_hi = (float(s) for s in args_cli.height_range.split(","))
l_lo, l_hi = (float(s) for s in (args_cli.length_range or args_cli.width_range).split(","))
_grip_open = float(__import__("os").environ.get("GRASP_GRIP_OPEN", "0.035"))
_w_cap = 2 * _grip_open - 0.004  # PiPER 0.066, PiPER-X 0.096 (4 mm closing margin)
if w_hi > _w_cap + 1e-9:
    parser.error(f"width upper bound {w_hi} exceeds the {_w_cap:.3f} m cap (gripper opens {2*_grip_open:.3f} m)")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import atec_rl_lab  # noqa: E402
import graspenv  # noqa: F401, E402
from graspenv.agents import GraspPPORunnerCfg  # noqa: E402
from graspenv.grasp_env_cfg import _CUBE_BASE, B2PiperGraspEnvCfg, B2PiperGraspStage2EnvCfg  # noqa: E402

assert str(RW) in atec_rl_lab.__file__, f"atec_rl_lab outside repo: {atec_rl_lab.__file__}"


def main():
    device = args_cli.device if getattr(args_cli, "device", None) else "cuda:0"

    env_cfg = B2PiperGraspStage2EnvCfg() if "Stage2" in args_cli.task else B2PiperGraspEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.seed
    if args_cli.object_usd:
        import isaaclab.sim as sim_utils
        env_cfg.scene.object.spawn = sim_utils.UsdFileCfg(
            usd_path=args_cli.object_usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        )
        env_cfg.events.object_scale = None
        env_cfg.events.object_mass = None
        env_cfg.events.object_material = None
    # width/height ranges -> prestartup scale ranges on the base dims
    bx, _, bz = _CUBE_BASE
    if env_cfg.events.object_scale is not None:
      env_cfg.events.object_scale.params["scale_range"] = {
        "x": (w_lo / bx, w_hi / bx),
        "y": (l_lo / bx, l_hi / bx),
        "z": (h_lo / bz, h_hi / bz),
      }
    print(f"[TRAIN] width {w_lo * 1000:.0f}-{w_hi * 1000:.0f} mm, length {l_lo * 1000:.0f}-{l_hi * 1000:.0f} mm, "
          f"height {h_lo * 1000:.0f}-{h_hi * 1000:.0f} mm, num_envs {args_cli.num_envs}", flush=True)

    agent_cfg = GraspPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.device = device
    agent_cfg.seed = args_cli.seed
    agent_cfg.logger = args_cli.logger
    agent_cfg.wandb_project = args_cli.wandb_project

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = str(Path(os.environ.get("GRASP_RUNS_DIR", str(GRASP_RL / "runs"))) / f"{args_cli.run_name}_{stamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[TRAIN] log_dir={log_dir}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=device)
    if args_cli.resume:
        print(f"[TRAIN] resume (model only) from: {args_cli.resume}", flush=True)
        runner.load(args_cli.resume, load_optimizer=False)
        runner.current_learning_iteration = 0
        # GRASP_RESUME_STD_SCALE (opt-in): shrink the loaded action std. The egxs075
        # champ carries std [0.91, 0.01, 0.90, 1.85, 1.72, 3.65, 4.04] — deterministic
        # eval grasps cans at 15-21% while PPO rollouts under that same policy score
        # ~0.5%: the exploration noise itself destroys the delicate behavior the
        # finetune is supposed to preserve, so the success bonus never fires and
        # learning stalls (measured across the v1-v3 frozen-finetune rounds, 0907).
        _ss = float(os.environ.get("GRASP_RESUME_STD_SCALE", "0") or 0)
        if _ss > 0:
            import torch
            with torch.no_grad():
                _pol = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
                _pol.std.mul_(_ss)
                print(f"[TRAIN] resume std scaled x{_ss}: "
                      f"{[round(float(v), 3) for v in _pol.std.flatten()]}", flush=True)

    # params dump (the crouch runs lacked this; keep it for reproducibility)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
