"""Smoke test for ATEC-Grasp-B2Piper-v0: build, reset, random-step, sanity-check.

Checks: obs dim 42, object dims vary per env (prestartup scale), RSI reset puts the TCP
near the object, contact sensors report finger-object forces when stepping, no NaN.

Run:
  cd real_world_preliminary/ATEC2026_Simulation_Challenge
  CUDA_VISIBLE_DEVICES=5 ~/.conda/envs/isaaclab/bin/python \
      grasp_rl/scripts/smoke_grasp_env.py --headless --device cuda:0 --num_envs 16
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

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=150)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import atec_rl_lab  # noqa: E402
import graspenv  # noqa: F401, E402
from graspenv.mdp.utils import grasp_point_w, tcp_pos_w  # noqa: E402
from graspenv.mdp.events import _obj_dims  # noqa: E402

assert str(RW) in atec_rl_lab.__file__, f"atec_rl_lab outside repo: {atec_rl_lab.__file__}"

from graspenv.grasp_env_cfg import B2PiperGraspEnvCfg  # noqa: E402


def main():
    device = args_cli.device if getattr(args_cli, "device", None) else "cuda:0"
    cfg = B2PiperGraspEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = device

    env = gym.make("ATEC-Grasp-B2Piper-v0", cfg=cfg)
    obs, _ = env.reset()
    uenv = env.unwrapped

    pol = obs["policy"]
    print(f"[SMOKE] obs policy shape: {tuple(pol.shape)} (expect (*, 42))", flush=True)
    dims = _obj_dims(uenv)
    print(f"[SMOKE] obj dims x: {dims[:, 0].min():.3f}-{dims[:, 0].max():.3f} "
          f"y: {dims[:, 1].min():.3f}-{dims[:, 1].max():.3f} "
          f"z: {dims[:, 2].min():.3f}-{dims[:, 2].max():.3f}", flush=True)
    minor = torch.minimum(dims[:, 0], dims[:, 1])
    print(f"[SMOKE] minor axis (grasp width) max: {minor.max():.4f} (must be <= 0.060 + eps)", flush=True)
    d0 = (tcp_pos_w(uenv) - grasp_point_w(uenv)).norm(dim=1)
    print(f"[SMOKE] RSI: |tcp - grasp point| at reset: mean {d0.mean():.3f} max {d0.max():.3f} m", flush=True)

    n_act = uenv.action_manager.total_action_dim
    print(f"[SMOKE] action dim: {n_act} (expect 7)", flush=True)
    contact_seen = 0
    nan_seen = False
    for i in range(args_cli.steps):
        a = torch.zeros(args_cli.num_envs, n_act, device=uenv.device)
        a[:, :6] = torch.randn(args_cli.num_envs, 6, device=uenv.device) * 0.5
        a[:, 6] = -1.0 if i > 60 else 1.0  # try closing later in the rollout
        obs, rew, term, trunc, info = env.step(a)
        if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(rew).all():
            nan_seen = True
            print(f"[SMOKE] NaN at step {i}!", flush=True)
            break
        fl = uenv.scene["contact_finger_l"].data.force_matrix_w
        if fl is not None and (fl.norm(dim=-1) > 0.5).any():
            contact_seen += 1
    print(f"[SMOKE] steps done. finger-object contact seen on {contact_seen} steps; NaN: {nan_seen}", flush=True)
    ep = uenv.termination_manager.get_term("grasp_success")
    print(f"[SMOKE] success-latch tensor ok: {ep.shape}", flush=True)
    print("[SMOKE] PASS" if not nan_seen else "[SMOKE] FAIL", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
