"""Reward terms for the grasp+lift window (aligned with the locked design decisions)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .utils import (
    grasp_point_w,
    gripper_axes_w,
    object_clearance,
    object_minor_axis_w,
    stable_grasp,
    tcp_pos_w,
)


def reach_grasp_point(env: ManagerBasedRLEnv, std: float = 0.1) -> torch.Tensor:
    """Kernelized TCP distance to the SURFACE grasp point (not the object center)."""
    dist = (tcp_pos_w(env) - grasp_point_w(env)).norm(dim=1)
    return 1.0 - torch.tanh(dist / std)


def align_gripper(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Approach-axis alignment (tool z vs straight down) x closing-axis alignment
    (finger axis vs the object's minor horizontal axis; sign-agnostic)."""
    tool_z, finger_x = gripper_axes_w(env)
    import os as _os
    if _os.environ.get("GRASP_POINT_MODE") == "rim":
        # rim pinch: tool z HORIZONTAL (approach from the side), fingers closing VERTICALLY
        horiz = (1.0 - tool_z[:, 2].abs()).clamp(min=0.0)
        vert_fingers = finger_x[:, 2].abs()
        return horiz * (0.5 + 0.5 * vert_fingers)
    down = (-tool_z[:, 2]).clamp(min=0.0)                 # 1 = pointing straight down
    minor = object_minor_axis_w(env)
    yaw_align = (finger_x * minor).sum(dim=1).abs()       # 1 = fingers across the minor axis
    return down * (0.5 + 0.5 * yaw_align)


def grasp_established(env: ManagerBasedRLEnv, force_thresh: float = 1.0, vel_thresh: float = 0.10) -> torch.Tensor:
    """Both-finger contact AND object-velocity agreement (the locked stable_grasp test)."""
    return stable_grasp(env, force_thresh, vel_thresh).float()


def lift_progress(
    env: ManagerBasedRLEnv,
    target_clearance: float = 0.10,
    force_thresh: float = 1.0,
    vel_thresh: float = 0.10,
) -> torch.Tensor:
    """Object bottom clearance, gated on stable_grasp (no reward for flicking the object up)."""
    frac = (object_clearance(env) / target_clearance).clamp(0.0, 1.0)
    return stable_grasp(env, force_thresh, vel_thresh).float() * frac


def object_disturb(env: ManagerBasedRLEnv, lin_scale: float = 1.0, ang_scale: float = 0.1) -> torch.Tensor:
    """Penalty (use negative weight): object speed while NOT stably grasped — bumping,
    tipping and rolling the object away (the dominant cylinder failure). Zero once the
    grasp is established, so transporting the object is never penalized."""
    obj = env.scene["object"]
    v = obj.data.root_lin_vel_w.norm(dim=1) * lin_scale + obj.data.root_ang_vel_w.norm(dim=1) * ang_scale
    return v * (~stable_grasp(env)).float()


def arm_joint_vel_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    ids = [robot.joint_names.index(f"arm_joint{i}") for i in range(1, 7)]
    return robot.data.joint_vel[:, ids].square().sum(dim=1)


def camera_keepout(env: ManagerBasedRLEnv, margin: float = 0.035, depth_scale: float = 20.0) -> torch.Tensor:
    """Penalty (use negative weight): binary count of arm points inside the REAL camera box
    plus a shaped penetration-depth term on a margin-inflated box, so the gradient starts
    pushing the arm out ~margin meters before actual contact."""
    from .utils import camera_keepout_depth, camera_keepout_violation

    return camera_keepout_violation(env) + depth_scale * camera_keepout_depth(env, margin)
