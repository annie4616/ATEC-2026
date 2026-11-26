# Reference: https://github.com/fan-ziqi/robot_lab

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def grid_velocity_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    lin_reward_term: str = "track_lin_vel_xy_exp",
    ang_reward_term: str = "track_ang_vel_z_exp",
) -> torch.Tensor:
    """Drive the GridVelocityCommand's grid from per-episode tracking rewards.

    MIT "Rapid Locomotion" grid curriculum (arXiv:2205.02824): each (vx, wz) bin
    expands to its neighbors once an env assigned to it tracks its command well.
    This replaces the global-mean `command_levels_lin_vel` gate, which stalled
    because hard-command envs dragged the shared mean below threshold.

    Runs at each episode boundary (max_episode_length). For the envs that just
    reset (env_ids), it reads their finished-episode mean lin/ang tracking reward,
    normalizes to a [0,1] fraction of the term weight, and hands them to the command
    term to bump successful bins + neighbors. Returns the current top active vx for
    logging (shows up as Curriculum/grid_velocity_curriculum).
    """
    cmd = env.command_manager.get_term("base_velocity")
    # Only meaningful for the grid command; no-op otherwise.
    if not hasattr(cmd, "update_curriculum"):
        return torch.tensor(0.0, device=env.device)

    # IMPORTANT: the curriculum term is called every step with env_ids = the envs
    # that JUST reset (finished an episode) this step. With init_at_random_ep_len
    # the 4096 envs finish on staggered steps, so we must process whatever reset now
    # — NOT gate on common_step_counter % max_episode_length (that near-never lines
    # up with a reset and the grid would never update, which is the bug we hit).
    env_ids_t = torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)
    if env_ids_t.numel() == 0:
        return torch.tensor(cmd.max_active_vx, device=env.device)

    # Per-env mean tracking reward over the finished episode, as a fraction of the
    # term's max (weight * dt sum / episode_length_s ~ weight; divide it out).
    lin_cfg = env.reward_manager.get_term_cfg(lin_reward_term)
    ang_cfg = env.reward_manager.get_term_cfg(ang_reward_term)
    lin_sums = env.reward_manager._episode_sums[lin_reward_term][env_ids_t]
    ang_sums = env.reward_manager._episode_sums[ang_reward_term][env_ids_t]
    denom = env.max_episode_length_s
    lin_frac = (lin_sums / denom) / (lin_cfg.weight if lin_cfg.weight != 0 else 1.0)
    ang_frac = (ang_sums / denom) / (ang_cfg.weight if ang_cfg.weight != 0 else 1.0)

    cmd.update_curriculum(env_ids_t, lin_frac, ang_frac)
    return torch.tensor(cmd.max_active_vx, device=env.device)


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_lin_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_ang_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original angular velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])

            # Update ranges
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)
