"""Termination terms: sustained-hold success (N steps, not lift height) and blow-up guard."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .utils import object_clearance, stable_grasp


def _hold_counter(env) -> torch.Tensor:
    if not hasattr(env, "_grasp_hold_counter"):
        env._grasp_hold_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return env._grasp_hold_counter


def grasp_success(
    env: ManagerBasedRLEnv,
    hold_steps: int = 25,
    clearance: float = 0.05,
    force_thresh: float = 1.0,
    vel_thresh: float = 0.10,
) -> torch.Tensor:
    """Success when [stable_grasp AND object lifted >= clearance] holds for hold_steps
    CONSECUTIVE control steps (the locked N-step criterion, not a height snapshot)."""
    counter = _hold_counter(env)
    ok = stable_grasp(env, force_thresh, vel_thresh) & (object_clearance(env) >= clearance)
    counter[:] = torch.where(ok, counter + 1, torch.zeros_like(counter))
    return counter >= hold_steps


def object_out_of_zone(env: ManagerBasedRLEnv, max_dist: float = 1.0) -> torch.Tensor:
    """Guard: the object left the workspace (thrown/exploded) -> end the episode.

    Measured relative to the ROBOT BASE (not the env origin) so a drifting free base in
    stage 2 does not spuriously terminate episodes; height uses the env origin (ground)."""
    obj = env.scene["object"]
    robot = env.scene["robot"]
    rel_base = obj.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (rel_base.norm(dim=1) > max_dist) | (z > 1.0) | (z < -0.2)


def base_fell(env: ManagerBasedRLEnv, tilt_gravity_z: float = -0.7, min_height: float = 0.12) -> torch.Tensor:
    """Stage 2 guard: crouching base tipped over or collapsed (free-floating base only)."""
    robot = env.scene["robot"]
    z = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (robot.data.projected_gravity_b[:, 2] > tilt_gravity_z) | (z < min_height)


def camera_hit(env: ManagerBasedRLEnv, hold_steps: int = 5) -> torch.Tensor:
    """FAIL the episode when the wrist/gripper stays inside the head-camera keep-out box
    for hold_steps consecutive control steps (real-robot camera collision = mission failure).

    GRASP_CAMHIT_GRACE=<N> (env var, opt-in) suppresses the termination during the first N
    steps of an episode: RSI resets can start with wrist points already inside the box, and
    failing before the policy can move puts an unlearnable floor under the hit rate. The
    keep-out penalty stays active during the grace window, so escaping fast is still learned."""
    from .utils import camera_keepout_violation

    if not hasattr(env, "_camera_hit_counter"):
        env._camera_hit_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._camera_hit_grace = int(os.environ.get("GRASP_CAMHIT_GRACE", "0"))
    c = env._camera_hit_counter
    inside = camera_keepout_violation(env) > 0
    c[:] = torch.where(inside, c + 1, torch.zeros_like(c))
    hit = c >= hold_steps
    if env._camera_hit_grace > 0:
        hit &= env.episode_length_buf > env._camera_hit_grace
    return hit
