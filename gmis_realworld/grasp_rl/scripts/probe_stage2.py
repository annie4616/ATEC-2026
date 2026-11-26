"""Diagnose stage-2 early object_away terminations: where are the feet, where do objects go.

Zero arm actions; prints foot positions in the crouch, object positions at reset, and the
object's relative position at the exact step object_away fires (ejected vs sunk).
"""

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
GRASP_RL = _SCRIPT.parents[1]
RW = _SCRIPT.parents[2]
for _p in (str(RW / "source" / "atec_rl_lab"), str(GRASP_RL)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

# Shared machine: GPUs 1-7 are other people's training runs. Pin to GPU 0 BEFORE any
# CUDA-aware import so forgetting the CUDA_VISIBLE_DEVICES prefix cannot land this on
# someone else's device.
import os as _os
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--self_collision", action="store_true")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--no_rsi", action="store_true", help="Disable the RSI reset (arm stays folded).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import graspenv  # noqa: F401, E402
from graspenv.grasp_env_cfg import B2PiperGraspStage2EnvCfg  # noqa: E402


def main():
    device = args_cli.device if getattr(args_cli, "device", None) else "cuda:0"
    cfg = B2PiperGraspStage2EnvCfg()
    if args_cli.self_collision:
        cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        print("[P2] SELF-COLLISION ENABLED", flush=True)
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = device
    if args_cli.no_rsi:
        cfg.events.rsi_reset = None

    env = gym.make("ATEC-Grasp-B2Piper-Stage2-v0", cfg=cfg)
    env.reset()
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dj = robot.data.default_joint_pos[0]
    print("[P2] cfg init joint_pos dict:", robot.cfg.init_state.joint_pos, flush=True)
    print("[P2] parsed default_joint_pos:", {n: round(float(dj[i]), 3) for i, n in enumerate(robot.joint_names)},
          flush=True)
    qq = robot.data.joint_pos[0]
    print("[P2] ACTUAL joint_pos after reset:", {n: round(float(qq[i]), 3) for i, n in enumerate(robot.joint_names)},
          flush=True)
    tt = robot.data.joint_pos_target[0]
    print("[P2] joint TARGETS after reset:", {n: round(float(tt[i]), 3) for i, n in enumerate(robot.joint_names)},
          flush=True)
    obj = uenv.scene["object"]
    feet = [i for i, n in enumerate(robot.body_names) if n.endswith("_foot")]
    foot_names = [robot.body_names[i] for i in feet]
    n_act = uenv.action_manager.total_action_dim
    zero = torch.zeros(args_cli.num_envs, n_act, device=uenv.device)

    rel0 = obj.data.root_pos_w - uenv.scene.env_origins
    print(f"[P2] object rel pos at reset: x {rel0[:,0].min():.2f}-{rel0[:,0].max():.2f} "
          f"y {rel0[:,1].min():.2f}-{rel0[:,1].max():.2f} z {rel0[:,2].min():.3f}-{rel0[:,2].max():.3f}", flush=True)

    for step in range(args_cli.steps):
        _, _, term, trunc, _ = env.step(zero)
        away = uenv.termination_manager.get_term("object_away")
        if step in (5, 50, 150):
            fp = robot.data.body_pos_w[:, feet] - uenv.scene.env_origins.unsqueeze(1)
            for j, nme in enumerate(foot_names):
                print(f"[P2] step {step} {nme}: x {fp[:,j,0].mean():.3f}±{fp[:,j,0].std():.2f} "
                      f"y {fp[:,j,1].mean():.3f}±{fp[:,j,1].std():.2f}", flush=True)
            z = robot.data.root_pos_w[:, 2] - uenv.scene.env_origins[:, 2]
            drift = (robot.data.root_pos_w[:, :2] - uenv.scene.env_origins[:, :2]).norm(dim=1)
            print(f"[P2] step {step} base z {z.mean():.3f}±{z.std():.3f} | root drift {drift.mean():.3f}±{drift.std():.3f} m",
                  flush=True)
        if away.any():
            ids = away.nonzero(as_tuple=False).flatten()
            rel = (obj.data.root_pos_w - uenv.scene.env_origins)[ids]
            for k in range(min(3, len(ids))):
                print(f"[P2] step {step} AWAY env{int(ids[k])}: rel=({rel[k,0]:.2f},{rel[k,1]:.2f},{rel[k,2]:.3f})",
                      flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
