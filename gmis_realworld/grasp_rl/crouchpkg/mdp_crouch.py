"""Custom MDP terms for the B2 commanded stand<->crouch task.

Self-contained — lives under logs/rsl_rl/260630/ so the base repo is never modified.

A per-env "crouch command" c in {0,1} (0 = stand, 1 = crouch) is injected into the
(otherwise unused) velocity-command obs slot so the policy can be told which posture to
hold. The command is sampled at reset AND resampled on an interval, so each episode
contains both stand->crouch and crouch->stand transitions — that is what trains a stable
recovery back to standing. Obs dimension is unchanged (still a 3-vector), so the
warm-start from the locomotion checkpoint still loads.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


# ───────────────────────── command buffer ─────────────────────────
def _get_cmd(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-env crouch command c in {0,1}; lazily created as zeros (=stand)."""
    if not hasattr(env, "_crouch_cmd"):
        env._crouch_cmd = torch.zeros(env.num_envs, device=env.device)
    return env._crouch_cmd


def set_crouch_cmd(env: ManagerBasedRLEnv, env_ids, p_crouch: float = 0.5):
    """Event term (reset AND interval): (re)sample the crouch command per env.

    With prob p_crouch the env is commanded to crouch (1), else stand (0). Used for both
    the reset event (fresh command each episode) and an interval event (mid-episode
    switches) so both transition directions are trained.
    """
    c = _get_cmd(env)
    if env_ids is None:
        env_ids = slice(None)
    n = c[env_ids].shape[0]
    c[env_ids] = (torch.rand(n, device=env.device) < p_crouch).float()


# ───────────────────────── observation ─────────────────────────
def crouch_cmd_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Command obs: [c, 0, 0] (N,3). Same width as the velocity-command slot it replaces,
    so the policy/critic obs dims (45/48) — and thus the warm-start — are preserved."""
    c = _get_cmd(env).unsqueeze(-1)              # (N,1)
    zeros = torch.zeros_like(c)
    return torch.cat([c, zeros, zeros], dim=-1)  # (N,3)


# ───────────────────────── rewards ─────────────────────────
def _target_height(env: ManagerBasedRLEnv, stand_h: float, crouch_h: float) -> torch.Tensor:
    c = _get_cmd(env)
    return torch.where(c > 0.5, torch.full_like(c, crouch_h), torch.full_like(c, stand_h))


def _base_z(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]


def commanded_height(
    env: ManagerBasedRLEnv,
    stand_h: float = 0.53,
    crouch_h: float = 0.30,
    std: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian bonus peaking (=1) when base world-Z equals the COMMANDED target height.
    Rewards standing at 0.53 when c=0 and crouching at 0.30 when c=1 — so moving toward
    the target in BOTH directions is rewarded (this is what trains the recovery)."""
    z = _base_z(env, asset_cfg)
    target = _target_height(env, stand_h, crouch_h)
    return torch.exp(-((z - target) ** 2) / (2.0 * std * std))


def commanded_height_l2(
    env: ManagerBasedRLEnv,
    stand_h: float = 0.53,
    crouch_h: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 penalty (use negative weight) on deviation from the COMMANDED target height —
    the broad pull that complements the sharp gaussian above."""
    z = _base_z(env, asset_cfg)
    target = _target_height(env, stand_h, crouch_h)
    return (z - target) ** 2


# kept for reference / backward compat (fixed-target crouch bonus; unused by the cycle cfg)
def crouch_height(
    env: ManagerBasedRLEnv,
    target_height: float = 0.30,
    std: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    z = _base_z(env, asset_cfg)
    return torch.exp(-((z - target_height) ** 2) / (2.0 * std * std))
