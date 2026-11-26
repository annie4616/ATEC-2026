from __future__ import annotations

from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_x_greater_than(
    env: ManagerBasedEnv,
    asset_cfg,
    x_threshold: float,
) -> torch.Tensor:
    """Terminate when robot world x is greater than x_threshold."""
    robot = env.scene[asset_cfg.name]
    robot_x = robot.data.root_pos_w[:, 0]   # world x

    return robot_x > x_threshold


def robot_reached_point(
    env: ManagerBasedEnv,
    asset_cfg,
    goal_x: float,
    goal_y: float,
    radius: float = 3.0,
) -> torch.Tensor:
    """Terminate when the robot reaches within `radius` of a world (goal_x, goal_y).

    Used by the BENT-goal Task A variant: the course turns off the +x axis, so a pure
    x-threshold no longer marks the finish -- the goal sits at a point that is offset in
    y. The robot must actually follow the (curved) course to the bent endpoint.
    """
    robot = env.scene[asset_cfg.name]
    p = robot.data.root_pos_w[:, :2]        # world (x,y)
    gx = torch.full_like(p[:, 0], goal_x)
    gy = torch.full_like(p[:, 1], goal_y)
    d = torch.sqrt((p[:, 0] - gx) ** 2 + (p[:, 1] - gy) ** 2)
    return d < radius