"""Task D specific termination terms."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reach_goal_x(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_threshold: float = 3.5,
) -> torch.Tensor:
    """Terminate (success) when the robot root x crosses the goal threshold.

    Mirrors the evaluation env's `robot_x_greater_than` (tasks/task_d), so a
    trained policy's success condition matches scoring exactly.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 0] > float(x_threshold)
