"""Shared helpers for the grasp MDP terms (single source of truth for frames/constants)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

EE_BODY = "gripper_base"
# gripper_base -> between-fingertips distance along tool z [m] (PiPER 0.13; PiPER-X 0.135)
TCP_OFFSET = float(os.environ.get("GRASP_TCP_OFFSET", "0.13"))
# how far below the object top face the fingers should close [m] (shallower for slope pinches)
GRIP_DEPTH = float(os.environ.get("GRASP_GRIP_DEPTH", "0.015"))

# head-camera keep-out box in the BASE frame (B2 default: camera at (0.422, 0.025, 0.062),
# pitched down 30 deg). Other robots override via env vars, e.g. the A2 camera sits at
# (0.338, 0.034, 0.053) -> same margins shifted with the camera.
import os as _os

def _env_vec(name, default):
    v = _os.environ.get(name)
    return tuple(float(s) for s in v.split(",")) if v else default

KEEPOUT_LO = _env_vec("GRASP_KEEPOUT_LO", (0.30, -0.11, -0.03))
KEEPOUT_HI = _env_vec("GRASP_KEEPOUT_HI", (0.52, 0.16, 0.17))
# kinematic chain sampled densely (origins + segment midpoints + TCP): link ORIGINS alone
# miss sweeps where the mesh between two origins passes through the box
KEEPOUT_CHAIN = ["arm_link3", "arm_link4", "arm_link5", "arm_link6", "gripper_base"]
KEEPOUT_FINGERS = ["arm_link7", "arm_link8"]


def _keepout_points_b(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sample points along the arm in the BASE frame: chain origins, segment midpoints,
    gripper_base->TCP midpoint, TCP, and both finger origins. [N, P, 3]"""
    robot = env.scene["robot"]
    if not hasattr(env, "_keepout_chain_ids"):
        env._keepout_chain_ids = [robot.body_names.index(n) for n in KEEPOUT_CHAIN]
        env._keepout_finger_ids = [robot.body_names.index(n) for n in KEEPOUT_FINGERS]
        env._keepout_lo = torch.tensor(KEEPOUT_LO, device=env.device)
        env._keepout_hi = torch.tensor(KEEPOUT_HI, device=env.device)
    chain = robot.data.body_pos_w[:, env._keepout_chain_ids]      # [N, C, 3]
    mids = (chain[:, 1:] + chain[:, :-1]) / 2                     # [N, C-1, 3]
    tcp = tcp_pos_w(env).unsqueeze(1)                             # [N, 1, 3]
    mid_tcp = (chain[:, -1:] + tcp) / 2
    fingers = robot.data.body_pos_w[:, env._keepout_finger_ids]
    pts_w = torch.cat([chain, mids, mid_tcp, tcp, fingers], dim=1)
    rel = pts_w - robot.data.root_pos_w.unsqueeze(1)
    return math_utils.quat_apply_inverse(
        robot.data.root_quat_w.unsqueeze(1).expand(-1, pts_w.shape[1], -1), rel
    )


def camera_keepout_violation(env: ManagerBasedRLEnv, margin: float = 0.0) -> torch.Tensor:
    """Per-env count of arm sample points inside the (optionally inflated) keep-out box."""
    pos_b = _keepout_points_b(env)
    inside = ((pos_b > env._keepout_lo - margin) & (pos_b < env._keepout_hi + margin)).all(dim=-1)
    return inside.float().sum(dim=1)


def camera_keepout_depth(env: ManagerBasedRLEnv, margin: float = 0.035) -> torch.Tensor:
    """Summed penetration depth [m] of the sample points into the margin-inflated box.
    Smooth (unlike the binary count), so the policy gets a push-out gradient BEFORE the
    real box is reached."""
    pos_b = _keepout_points_b(env)
    lo = env._keepout_lo - margin
    hi = env._keepout_hi + margin
    depth = torch.minimum(pos_b - lo, hi - pos_b).amin(dim=-1).clamp(min=0.0)  # [N, P]
    return depth.sum(dim=1)


def ee_pose_w(env: ManagerBasedRLEnv):
    robot = env.scene["robot"]
    idx = robot.body_names.index(EE_BODY)
    return robot.data.body_pos_w[:, idx], robot.data.body_quat_w[:, idx]


def ee_vel_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    idx = robot.body_names.index(EE_BODY)
    return robot.data.body_lin_vel_w[:, idx]


def tcp_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    pos_w, quat_w = ee_pose_w(env)
    offset = torch.tensor([0.0, 0.0, TCP_OFFSET], device=env.device).expand_as(pos_w)
    return pos_w + math_utils.quat_apply(quat_w, offset)


def gripper_axes_w(env: ManagerBasedRLEnv):
    """(tool z-axis, finger-closing x-axis) of the gripper in world frame."""
    _, quat_w = ee_pose_w(env)
    n = quat_w.shape[0]
    z = math_utils.quat_apply(quat_w, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(n, 3))
    x = math_utils.quat_apply(quat_w, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(n, 3))
    return z, x


def grasp_point_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Grasp point on the object SURFACE (top face center, GRIP_DEPTH below the top), world frame.

    Clamped to >= 10 mm above the ground so very flat objects (watermelon rind ~25 mm) do
    not reward pressing the fingertips into the floor."""
    from .events import _obj_dims  # lazy: avoid import cycle at module load

    obj = env.scene["object"]
    dims = _obj_dims(env)
    top = obj.data.root_pos_w.clone()
    if os.environ.get("GRASP_POINT_MODE") == "rim":
        # clamshell rim pinch: target the long-side seam ledge at mid-height instead of
        # the (ungraspably wide) top face — fingers must close vertically on the lip
        minor = object_minor_axis_w(env)
        top += minor * (torch.minimum(dims[:, 0], dims[:, 1]) / 2 - 0.005).unsqueeze(1)
        top[:, 2] = torch.maximum(top[:, 2], env.scene.env_origins[:, 2] + 0.015)
        return top
    top[:, 2] = torch.maximum(
        top[:, 2] + dims[:, 2] / 2 - GRIP_DEPTH, env.scene.env_origins[:, 2] + 0.010
    )
    return top


def object_minor_axis_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """World-frame direction of the object's smaller horizontal OBB axis (the grasp axis)."""
    from .events import _obj_dims

    obj = env.scene["object"]
    dims = _obj_dims(env)
    quat = obj.data.root_quat_w
    n = quat.shape[0]
    ex = math_utils.quat_apply(quat, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(n, 3))
    ey = math_utils.quat_apply(quat, torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(n, 3))
    use_x = (dims[:, 0] <= dims[:, 1]).unsqueeze(1)
    return torch.where(use_x, ex, ey)


def stable_grasp(env: ManagerBasedRLEnv, force_thresh: float = 1.0, vel_thresh: float = 0.10) -> torch.Tensor:
    """Grasp-established test per the locked design: BOTH finger-object contacts AND
    |v_object - v_ee| below the threshold (contact alone rewards dragging the object).

    Contact detection uses the filtered force matrix when it fires; some USD assets never
    register in the filtered view (colliders on nested child prims), so a fallback accepts
    per-finger NET contact force combined with a tight TCP-to-object proximity gate."""
    from .events import _obj_dims

    sl, sr = env.scene["contact_finger_l"], env.scene["contact_finger_r"]
    left = sl.data.force_matrix_w.norm(dim=-1).amax(dim=(1, 2)) > force_thresh
    right = sr.data.force_matrix_w.norm(dim=-1).amax(dim=(1, 2)) > force_thresh
    # fallback: net force on each finger + TCP within the object's graspable envelope
    obj = env.scene["object"]
    dims = _obj_dims(env)
    span = torch.minimum(dims[:, 0], dims[:, 1]) / 2 + 0.04
    near = (tcp_pos_w(env) - obj.data.root_pos_w).norm(dim=1) < span
    net_l = sl.data.net_forces_w.norm(dim=-1).amax(dim=1) > force_thresh
    net_r = sr.data.net_forces_w.norm(dim=-1).amax(dim=1) > force_thresh
    left = left | (net_l & near)
    right = right | (net_r & near)
    vel_ok = (obj.data.root_lin_vel_w - ee_vel_w(env)).norm(dim=1) < vel_thresh
    return left & right & vel_ok


def object_clearance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Height of the object's bottom face above the ground [m]."""
    from .events import _obj_dims

    obj = env.scene["object"]
    dims = _obj_dims(env)
    return obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - dims[:, 2] / 2
