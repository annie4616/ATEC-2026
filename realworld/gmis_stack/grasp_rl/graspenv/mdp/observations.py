"""Observation terms for the B2+Piper grasp environment.

Obs layout (42 total): ee_pose_b 7 | grasp_vec_ee 3 | axes 6 | gripper_width 1 |
last_action 7 | grasp_params 5 | obb_dims 3 | base_state 10 (constant slot in stage 1).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .events import _obj_dims, _grasp_params
from .utils import EE_BODY, TCP_OFFSET, ee_pose_w, grasp_point_w, gripper_axes_w

# ---- perception-noise model (opt-in: GRASP_PERC_NOISE=rgbl) --------------------------------
# Real-robot assumption: NO depth camera — the object estimate comes from RGB detection fused
# with sparse LiDAR returns. The policy's object observations (grasp point, OBB dims) become
# a 10 Hz-updated WORLD-frame estimate with per-episode systematic bias + per-update jitter
# and a per-episode size error; proprioceptive terms (ee_pose, axes, width) stay exact.
# Rewards/terminations keep the TRUE state: the policy must succeed on the real object while
# seeing only the noisy estimate.
_PERC = os.environ.get("GRASP_PERC_NOISE", "").lower()
# eval-time heuristics (opt-in): EMA filter over the held estimate, and a known-object
# dims prior (e.g. the competition can is spec'd 66x66x115 mm — no need to trust the
# noisy size estimate for a catalogued object)
_PN_EMA = float(os.environ.get("GRASP_PN_EMA", "0") or 0)
_DIMS_PRIOR = os.environ.get("GRASP_DIMS_PRIOR", "")
# Magnitudes model the RELATIVE gripper-object error: the head camera sees both the gripper
# and the object, so extrinsic/systematic errors largely cancel — close-range (~0.5 m)
# RGB+LiDAR fusion leaves ~1 cm relative error. (A first round with 2.5 cm bias made
# success unreachable: the policy abandoned grasping and farmed shaping reward instead.)
# GRASP_PN_BIAS / GRASP_PN_JITTER: env-var overrides for the frozen-estimate retrain
# curriculum (2026-09-07). The honest mission freezes the estimate at the pre-crouch
# detection (NEAR_CUT + arm occlusion: refresh rate is truly zero), and its error is far
# above the 12 mm this file was tuned for — stop-and-look scatter alone is ~50 mm median.
# Defaults preserve the historical values exactly.
_PN_BIAS = tuple(float(x) for x in os.environ.get(
    "GRASP_PN_BIAS", "0.012,0.012,0.010").split(","))
_PN_JITTER = float(os.environ.get("GRASP_PN_JITTER", "0.004"))
_PN_DIM_REL = 0.12                # per-episode relative OBB size error bound
# GRASP_PN_HOLD: control steps between estimate updates. 5 = the 10 Hz the champions were
# trained at. On the REAL robot the update rate during the attempt is ZERO (the head
# camera cannot see the grasp workspace: NEAR_CUT + arm occlusion), and the mission
# measured what that costs a 10 Hz-trained policy: 7/17 at 10 Hz -> 2/30 pooled below it.
# Set a huge value (e.g. 100000) to freeze the estimate at its reset value -- training
# under the sensing the robot actually has, instead of sensing it will never have.
_PN_HOLD = int(os.environ.get("GRASP_PN_HOLD", "5"))


def _pn_state(env: ManagerBasedRLEnv) -> None:
    """Resample per-episode errors on reset; refresh the held world-frame estimate at 10 Hz."""
    if not hasattr(env, "_pn_bias"):
        n, dev = env.num_envs, env.device
        env._pn_bias = torch.zeros(n, 3, device=dev)
        env._pn_dims = torch.ones(n, 3, device=dev)
        env._pn_point_w = torch.zeros(n, 3, device=dev)
        env._pn_init = torch.zeros(n, dtype=torch.bool, device=dev)
        env._pn_stamp = -1
    if env._pn_stamp == env.common_step_counter:
        return
    env._pn_stamp = env.common_step_counter
    dev = env.device
    fresh = env.episode_length_buf == 0
    if fresh.any():
        k = int(fresh.sum())
        env._pn_bias[fresh] = (torch.rand(k, 3, device=dev) * 2 - 1) * torch.tensor(_PN_BIAS, device=dev)
        env._pn_dims[fresh] = 1.0 + (torch.rand(k, 3, device=dev) * 2 - 1) * _PN_DIM_REL
        env._pn_init[fresh] = False
    upd = (env.episode_length_buf % _PN_HOLD == 0) | ~env._pn_init
    if upd.any():
        k = int(upd.sum())
        meas = grasp_point_w(env)[upd] + env._pn_bias[upd] + torch.randn(k, 3, device=dev) * _PN_JITTER
        if _PN_EMA > 0:
            prev = env._pn_point_w[upd]
            first = ~env._pn_init[upd]
            meas = torch.where(first.unsqueeze(1), meas, _PN_EMA * meas + (1 - _PN_EMA) * prev)
        env._pn_point_w[upd] = meas
        env._pn_init[upd] = True


def ee_pose_b(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End-effector (gripper_base) pose in the base frame: [pos(3), quat(4)]."""
    robot = env.scene["robot"]
    root_pos, root_quat = robot.data.root_pos_w, robot.data.root_quat_w
    pos_w, quat_w = ee_pose_w(env)
    pos_b = math_utils.quat_apply_inverse(root_quat, pos_w - root_pos)
    quat_b = math_utils.quat_mul(math_utils.quat_inv(root_quat), quat_w)
    return torch.cat([pos_b, quat_b], dim=1)


def grasp_vec_ee(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Target grasp point (on the object SURFACE) minus TCP, expressed in the EE frame.

    Under GRASP_PERC_NOISE=rgbl the point is the held noisy WORLD-frame estimate (10 Hz,
    biased) while the EE transform stays exact (proprioception)."""
    pos_w, quat_w = ee_pose_w(env)
    tcp_w = pos_w + math_utils.quat_apply(quat_w, torch.tensor([0.0, 0.0, TCP_OFFSET], device=env.device).expand_as(pos_w))
    if _PERC == "rgbl":
        _pn_state(env)
        return math_utils.quat_apply_inverse(quat_w, env._pn_point_w - tcp_w)
    return math_utils.quat_apply_inverse(quat_w, grasp_point_w(env) - tcp_w)


def approach_axes(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Target approach axis (base frame) and current gripper tool axis (base frame): 6."""
    robot = env.scene["robot"]
    root_quat = robot.data.root_quat_w
    n = env.num_envs
    target_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(n, 3)  # top-down
    tool_w, _ = gripper_axes_w(env)
    target_b = math_utils.quat_apply_inverse(root_quat, target_w)
    tool_b = math_utils.quat_apply_inverse(root_quat, tool_w)
    return torch.cat([target_b, tool_b], dim=1)


def gripper_width(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Current opening width [m] (arm_joint7 - arm_joint8, in [0, 0.07])."""
    robot = env.scene["robot"]
    names = robot.joint_names
    j7, j8 = names.index("arm_joint7"), names.index("arm_joint8")
    q = robot.data.joint_pos
    return (q[:, j7] - q[:, j8]).unsqueeze(1)


def grasp_params(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reserved 5-dim grasp parameter vector (width/approach/close-speed/force/lift-speed)."""
    return _grasp_params(env)


def obb_dims(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Object OBB axis lengths [m] (3). Under GRASP_PERC_NOISE=rgbl: scaled by the
    per-episode size-estimate error (sparse LiDAR + RGB bbox -> coarse dims).
    GRASP_DIMS_PRIOR='x,y,z' (eval heuristic) returns the catalogued dims instead."""
    if _DIMS_PRIOR:
        prior = torch.tensor([float(s) for s in _DIMS_PRIOR.split(",")], device=env.device)
        return prior.expand(env.num_envs, 3)
    if _PERC == "rgbl":
        _pn_state(env)
        return _obj_dims(env) * env._pn_dims
    return _obj_dims(env)


def base_state_const(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Base state slot [quat(4), ang_vel(3), lin_acc(3)] = 10.

    Stage 1 trains with a FIXED base, so this is a constant; the slot exists so the stage-2
    checkpoint (moving base) can reuse stage-1 weights without changing the obs layout.
    """
    n = env.num_envs
    out = torch.zeros(n, 10, device=env.device)
    out[:, 0] = 1.0  # identity quaternion w
    return out


def base_state_real(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 2: the same 10-dim slot filled with the real base state (free-floating base)."""
    robot = env.scene["robot"]
    base_idx = robot.body_names.index("base_link")
    lin_acc = robot.data.body_lin_acc_w[:, base_idx].clamp(-50.0, 50.0) * 0.1
    return torch.cat([robot.data.root_quat_w, robot.data.root_ang_vel_b * 0.25, lin_acc], dim=1)
