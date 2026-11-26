"""Event terms: object-dimension caching, grasp-parameter sampling, and the RSI reset.

RSI reset = every episode starts at a pre-grasp state: an arm configuration drawn from
grasp_rl/banks/pregrasp_bank.npz with the object placed under the fingertips (+/- noise).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

import os

GRASP_RL = Path(__file__).resolve().parents[2]         # grasp_rl/
_BANK_DIR = Path(os.environ.get("GRASP_BANK_DIR", str(GRASP_RL / "banks")))
_DEPTH = os.environ.get("GRASP_CROUCH_DEPTH", "0.2")
_SUFFIXED = _BANK_DIR / f"pregrasp_bank_{_DEPTH}.npz"
BANK_PATH = _SUFFIXED if _SUFFIXED.exists() else _BANK_DIR / "pregrasp_bank.npz"
POSES_PATH = Path(os.environ.get("GRASP_POSES_JSON", str(_BANK_DIR / "crouch_poses.json")))


# ---------------- per-env cached buffers (created lazily on the env object) ----------------
def _obj_dims(env) -> torch.Tensor:
    if not hasattr(env, "_grasp_obj_dims"):
        env._grasp_obj_dims = torch.full((env.num_envs, 3), 0.05, device=env.device)
    return env._grasp_obj_dims


def _grasp_params(env) -> torch.Tensor:
    if not hasattr(env, "_grasp_param_vec"):
        env._grasp_param_vec = torch.zeros(env.num_envs, 5, device=env.device)
    return env._grasp_param_vec


def _bank(env):
    if not hasattr(env, "_pregrasp_bank"):
        data = np.load(BANK_PATH, allow_pickle=True)
        env._pregrasp_bank = {
            "arm_q": torch.tensor(data["arm_q"], dtype=torch.float32, device=env.device),
            "ee_pos_b": torch.tensor(data["ee_pos_b"], dtype=torch.float32, device=env.device),
            "crouch_z": float(data["crouch_z"]),
        }
        print(f"[GRASP] pregrasp bank loaded: {env._pregrasp_bank['arm_q'].shape[0]} configs", flush=True)
    return env._pregrasp_bank


# ---------------- events ----------------
def cache_object_dims(env: ManagerBasedEnv, env_ids: torch.Tensor | None):
    """mode="startup": read each env's spawned Object prim AABB (pre-physics, identity yaw)
    and cache the OBB dims. Runs AFTER the prestartup scale randomization."""
    from pxr import Usd, UsdGeom

    dims = _obj_dims(env)
    stage = env.scene.stage
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for i in range(env.num_envs):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Object")
        if not prim.IsValid():
            raise RuntimeError(f"Object prim missing for env {i}")
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        size = box.GetSize()
        dims[i, 0], dims[i, 1], dims[i, 2] = float(size[0]), float(size[1]), float(size[2])
    print(f"[GRASP] object dims cached: min={dims.min(dim=0).values.tolist()} "
          f"max={dims.max(dim=0).values.tolist()}", flush=True)


def sample_grasp_params(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    width_margin: float = 0.005,
    noise: float = 0.10,
):
    """mode="reset": fill the reserved 5-dim grasp-parameter vector from object size,
    with mild randomization. Layout: [target_width(m), approach_type, close_speed,
    force_limit, lift_speed] (last three normalized nominals)."""
    dims = _obj_dims(env)[env_ids]
    params = _grasp_params(env)
    n = len(env_ids)
    minor = torch.minimum(dims[:, 0], dims[:, 1])
    base = torch.stack(
        [
            minor + width_margin,
            torch.zeros(n, device=env.device),           # approach type 0 = top-down
            torch.full((n,), 0.5, device=env.device),    # closing speed (nominal)
            torch.full((n,), 0.5, device=env.device),    # force limit (nominal)
            torch.full((n,), 0.5, device=env.device),    # lift speed (nominal)
        ],
        dim=1,
    )
    jitter = 1.0 + (torch.rand(n, 5, device=env.device) * 2 - 1) * noise
    jitter[:, 1] = 1.0  # approach type stays exact
    params[env_ids] = base * jitter


def reset_pregrasp_rsi(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    xy_noise: float = 0.05,
    ee_down_offset: float = 0.13,
    drop_height: float = 0.005,
    min_ee_x: float = 0.0,
):
    """mode="reset" (declared LAST): sample a pregrasp arm configuration from the bank and
    place the object under the fingertips with +/-xy_noise and yaw +/-180deg.

    min_ee_x restricts bank entries to EE base-frame x >= this value — stage 2 uses it to
    keep objects clear of the (now moving) front feet, which otherwise stomp them away.
    """
    robot = env.scene["robot"]
    obj = env.scene["object"]
    bank = _bank(env)
    dims = _obj_dims(env)[env_ids]
    n = len(env_ids)

    key = f"_bank_idx_x{min_ee_x:g}"
    if not hasattr(env, key):
        mask = bank["ee_pos_b"][:, 0] >= min_ee_x
        # opt-in (GRASP_BANK_KOFILTER=1): also drop entries whose gripper_base starts inside
        # the margin-inflated camera keep-out box — those resets camera_hit-fail before the
        # policy can act, putting an unlearnable floor under the termination rate
        if os.environ.get("GRASP_BANK_KOFILTER") == "1":
            from .utils import KEEPOUT_HI, KEEPOUT_LO

            ee = bank["ee_pos_b"]
            lo = torch.tensor(KEEPOUT_LO, device=ee.device) - 0.035
            hi = torch.tensor(KEEPOUT_HI, device=ee.device) + 0.035
            clear = ~((ee > lo) & (ee < hi)).all(dim=1)
            print(f"[GRASP] RSI bank keep-out filter: dropped {(~clear).sum().item()} configs", flush=True)
            mask &= clear
        idx = mask.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            raise RuntimeError(f"pregrasp bank has no entries with ee x >= {min_ee_x}")
        setattr(env, key, idx)
        print(f"[GRASP] RSI bank filter x>={min_ee_x:g}: {idx.numel()} configs", flush=True)
    idx = getattr(env, key)
    pick = idx[torch.randint(idx.numel(), (n,), device=env.device)]
    arm_q = bank["arm_q"][pick]                     # [n, 6]
    ee_b = bank["ee_pos_b"][pick]                   # [n, 3] gripper_base pos in base frame

    # arm + open fingers; legs stay at their default (crouch) angles
    q = robot.data.default_joint_pos[env_ids].clone()
    names = robot.joint_names
    arm_ids = [names.index(f"arm_joint{i}") for i in range(1, 7)]
    q[:, arm_ids] = arm_q
    _go = float(os.environ.get("GRASP_GRIP_OPEN", "0.035"))
    q[:, names.index("arm_joint7")] = _go
    q[:, names.index("arm_joint8")] = -_go
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    robot.set_joint_position_target(q, env_ids=env_ids)

    # object under the fingertips: bank poses approach top-down, so TCP is ~ee_down_offset
    # below gripper_base; place the object there in the horizontal plane (+noise)
    root_pos = robot.data.root_pos_w[env_ids]
    root_quat = robot.data.root_quat_w[env_ids]
    ee_w = root_pos + math_utils.quat_apply(root_quat, ee_b)
    obj_anchor = ee_w[:, 0:2].clone()
    # opt-in (GRASP_RSI_FAR=p): with prob p per env, anchor the object at ANOTHER reachable
    # bank spot (>= 0.15 m away) instead of under the fingertips — trains the re-approach
    # phase needed for sequential pickup (arm starts away from the object).
    _far_p = float(os.environ.get("GRASP_RSI_FAR", "0") or 0)
    if _far_p > 0:
        pick2 = idx[torch.randint(idx.numel(), (n,), device=env.device)]
        ee2_w = root_pos + math_utils.quat_apply(root_quat, bank["ee_pos_b"][pick2])
        use2 = (torch.rand(n, device=env.device) < _far_p) &                ((ee2_w[:, 0:2] - ee_w[:, 0:2]).norm(dim=1) >= 0.15)
        obj_anchor[use2] = ee2_w[use2, 0:2]
    pose = obj.data.default_root_state[env_ids].clone()
    pose[:, 0:2] = obj_anchor + (torch.rand(n, 2, device=env.device) * 2 - 1) * xy_noise
    pose[:, 2] = env.scene.env_origins[env_ids, 2] + dims[:, 2] / 2 + drop_height
    yaw = (torch.rand(n, device=env.device) * 2 - 1) * torch.pi
    zero = torch.zeros_like(yaw)
    # COMPOSE the random yaw with the configured default orientation (USD objects may be
    # deliberately spawned on their side; overwriting with yaw-only would undo that)
    pose[:, 3:7] = math_utils.quat_mul(math_utils.quat_from_euler_xyz(zero, zero, yaw), pose[:, 3:7])
    pose[:, 7:] = 0.0
    obj.write_root_pose_to_sim(pose[:, :7], env_ids=env_ids)
    obj.write_root_velocity_to_sim(pose[:, 7:], env_ids=env_ids)

    # per-episode buffers
    if hasattr(env, "_grasp_hold_counter"):
        env._grasp_hold_counter[env_ids] = 0
