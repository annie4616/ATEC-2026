"""Task D specific reward terms (MVP: forward progress + goal bonus).

Also includes JUMP / dynamic-climb reward terms (ported from the parkour RL
literature — extreme-parkour, robot-parkour-learning — into IsaacLab's
manager-based form). The crossing task needs the robot to mount a ~0.4 m step
(box top -> platform top), which the walking prior alone never achieves (it
shuffles in place; see memory atec2026-taskd-climb-findings). A jump requires a
true *flight phase* (all four feet off the ground) plus an explosive vertical
takeoff right in front of the step. These terms shape exactly that:

  takeoff_impulse        reward +z base velocity while in the takeoff x-band
  flight_phase           reward all-feet-airborne (the defining feature of a jump)
  base_height_target_band reward reaching a target base height over the step band
  foot_clearance_band    reward lifting the feet high while approaching the step
  hard_landing_penalty   penalize large vertical contact impulses on landing

All x-bands are in WORLD frame (matching the milestone thresholds in
taskd_cross_env_cfg.py: near-rim -0.15, enter 0.0, far-rim 0.8).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def progress_x(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense reward for moving the robot base forward (+x) this step.

    Returns the per-step change in world-frame root x (Δx). Positive when the
    robot advances toward the goal line, negative when it slips back. This is the
    core driving signal that pushes the policy to cross the pit toward x > 3.5.
    """
    asset = env.scene[asset_cfg.name]
    cur_x = asset.data.root_pos_w[:, 0]

    # Per-env cache of last x. Initialized lazily on the env instance so it
    # survives across steps and is reset on episode reset (see reset hook below).
    if not hasattr(env, "_taskd_prev_x") or env._taskd_prev_x.shape[0] != cur_x.shape[0]:
        env._taskd_prev_x = cur_x.clone()

    delta_x = cur_x - env._taskd_prev_x
    env._taskd_prev_x = cur_x.clone()

    # Guard against teleport-sized jumps at reset (a fresh episode starts back at
    # x≈-3, which would otherwise register as a huge negative Δx).
    delta_x = torch.clamp(delta_x, min=-1.0, max=1.0)
    return delta_x


def reach_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_threshold: float = 3.5,
) -> torch.Tensor:
    """Sparse terminal-style bonus: 1.0 on the step the robot crosses x > 3.5."""
    asset = env.scene[asset_cfg.name]
    return (asset.data.root_pos_w[:, 0] > float(x_threshold)).float()


def reach_x_milestone(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_threshold: float = 0.8,
    key: str = "m",
    rearm_x: float = -0.25,
) -> torch.Tensor:
    """One-time bonus the FIRST step the robot's base passes x_threshold this episode.

    Bridges the gap between the dense `progress_x` (which goes ~0 at the step the robot
    can't climb) and the far `reach_goal` (x>3.5, never reached): place milestones at
    the near rim / settled box / FAR rim so the policy gets a staged carrot for actually
    climbing the 0.4 m step instead of marching in place. Returns 1.0 exactly once per
    episode per env when x first exceeds the threshold, 0 otherwise. Each milestone uses
    a distinct `key` so multiple `reach_x_milestone` reward terms don't share the latch.
    """
    asset = env.scene[asset_cfg.name]
    cur_x = asset.data.root_pos_w[:, 0]
    n = cur_x.shape[0]

    attr = f"_taskd_milestone_{key}"
    flags = getattr(env, attr, None)
    if flags is None or flags.shape[0] != n:
        flags = torch.zeros(n, dtype=torch.bool, device=cur_x.device)
        setattr(env, attr, flags)

    # Reset the latch for envs that JUST respawned. Prefer the robust signal:
    # episode_length_buf <= 1 marks an env that was reset this step (works no matter
    # where the start pose is). Fall back to the x-based re-arm (cur_x < rearm_x) if
    # the buffer isn't present, for envs that wander back behind the start.
    elb = getattr(env, "episode_length_buf", None)
    if elb is not None and elb.shape[0] == n:
        respawned = (elb <= 1) | (cur_x < float(rearm_x))
    else:
        respawned = cur_x < float(rearm_x)
    flags = flags & ~respawned

    newly = (cur_x > float(x_threshold)) & (~flags)
    flags = flags | newly
    setattr(env, attr, flags)
    return newly.float()


def farrim_climb_band(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_lo: float = 0.40,
    x_hi: float = 1.10,
    z_lo: float = 0.16,
    z_hi: float = 0.55,
) -> torch.Tensor:
    """Dense reward for RAISING the base while in the far-rim x-band (the +20 blocker).

    The decisive part of the WALK-ACROSS crossing is climbing the ~0.4 m far rim
    back onto flat ground. Demo z-profile (root_pos_seq.npy) over the far rim:
    in world x in [0.40, 1.10] the base rises from z ~ 0.16 (lowest, pressed
    against the far wall on the box top) to z ~ 0.55 (back on flat ground).

    The sparse `reach_x_milestone` far-rim bonus gives NO gradient until the robot
    is already up, and `progress_x` goes ~0 against the vertical step — so the
    policy plateaus on the box top (farrim milestone stayed 0 across every prior
    run). This term supplies the missing dense vertical gradient: a normalized
    height fraction in [0,1] that pays MORE the higher the base climbs, but only
    inside the far-rim x-band (so it doesn't bribe the robot to stand tall on the
    flat approach or pronk across the platform). Combined with the dense
    `progress_x`, it shapes the actual step-up motion the demo performs.
    """
    asset = env.scene[asset_cfg.name]
    x = asset.data.root_pos_w[:, 0]
    z = asset.data.root_pos_w[:, 2]
    frac = torch.clamp((z - float(z_lo)) / (float(z_hi) - float(z_lo)), 0.0, 1.0)
    return frac * _in_x_band(x, x_lo, x_hi).float()


def box_progress_x(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Dense reward for pushing the BOX forward (+x) this step.

    Mirror of `progress_x` but tracking the box root instead of the robot, with a
    SEPARATE per-env cache key so the two do not clobber each other. This is the
    core box-pushing signal: positive Δx whenever the box advances toward the
    target x-range (the evaluation env scores the box ending near x≈0, see
    tasks/task_d RewardBoxXInRange with x in [-0.7, 0.7]). The box starts beside
    the robot at x=-3, so pushing it forward is rewarded all the way to the goal.
    """
    asset = env.scene[asset_cfg.name]
    cur_x = asset.data.root_pos_w[:, 0]

    if not hasattr(env, "_taskd_box_prev_x") or env._taskd_box_prev_x.shape[0] != cur_x.shape[0]:
        env._taskd_box_prev_x = cur_x.clone()

    delta_x = cur_x - env._taskd_box_prev_x
    env._taskd_box_prev_x = cur_x.clone()

    # Guard against teleport-sized jumps at episode reset (box respawns at x≈-3).
    delta_x = torch.clamp(delta_x, min=-1.0, max=1.0)
    return delta_x


def box_in_target(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    x_min: float = -0.7,
    x_max: float = 0.7,
) -> torch.Tensor:
    """Sparse bonus: 1.0 while the box x is inside the scoring target range.

    Matches the evaluation env's success band (RewardBoxXInRange center range
    [-0.7, 0.7]). Given as a per-step bonus (not one-time) so the policy is
    rewarded for keeping the box parked in the target rather than pushing it
    straight through. Combined with `box_progress_x` (which drives it there).
    """
    box_x = env.scene[asset_cfg.name].data.root_pos_w[:, 0]
    return ((box_x >= float(x_min)) & (box_x <= float(x_max))).float()


# ===========================================================================
# JUMP / dynamic-climb rewards (parkour-style). These add the explosive
# takeoff + flight-phase shaping that a pure walking prior cannot discover on
# its own. They are gated to the world-x band right in front of the step so the
# robot is only paid to jump where a jump is actually needed (and isn't bribed
# to pronk across the whole platform).
# ===========================================================================


def _in_x_band(x: torch.Tensor, x_lo: float, x_hi: float) -> torch.Tensor:
    """Boolean mask: world-x within [x_lo, x_hi]."""
    return (x >= float(x_lo)) & (x <= float(x_hi))


def takeoff_impulse(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_lo: float = -0.4,
    x_hi: float = 0.4,
    vz_cap: float = 2.5,
) -> torch.Tensor:
    """Reward upward base velocity (+z) while in the takeoff x-band.

    This is the core "jump now" signal: right in front of / on top of the box
    (world x in [x_lo, x_hi]) the policy is rewarded for the explosive vertical
    velocity that launches the body up onto the higher step. Positive vz only
    (downward motion isn't a takeoff) and capped so a single lucky spike can't
    dominate. Outside the band the reward is 0 — we only want a jump at the step,
    not random hopping across the flat platform.
    """
    asset = env.scene[asset_cfg.name]
    x = asset.data.root_pos_w[:, 0]
    vz = asset.data.root_lin_vel_w[:, 2]
    vz = torch.clamp(vz, min=0.0, max=float(vz_cap))
    return vz * _in_x_band(x, x_lo, x_hi).float()


def flight_phase(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_lo: float = -0.4,
    x_hi: float = 0.9,
) -> torch.Tensor:
    """Reward an all-four-feet-airborne flight phase in the step x-band.

    A jump (as opposed to a walk/step-up) is *defined* by a moment where every
    foot is simultaneously off the ground. We detect it from the contact sensor:
    a foot is airborne when its current air time > 0. Reward 1.0 on steps where
    ALL configured feet are airborne and the base is in the step band. Gating to
    the band prevents the policy from collecting this by simply leaping across
    the flat platform forever.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    all_airborne = (air_time > 0.0).all(dim=1)

    asset = env.scene[asset_cfg.name]
    x = asset.data.root_pos_w[:, 0]
    return (all_airborne & _in_x_band(x, x_lo, x_hi)).float()


def base_height_target_band(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 1.0,
    std: float = 0.3,
    x_lo: float = -0.2,
    x_hi: float = 0.9,
) -> torch.Tensor:
    """Reward reaching a target base HEIGHT (world z) over the step x-band.

    The robot must lift its body from the box top (~z 0.0-0.1 standing) to the
    platform top (~z 0.4 of clearance + standing height). A Gaussian around a
    target world-z pulls the base up to platform-mounting height while it is in
    the step band. This is the parkour "base height tracking" term that, more
    than anything, gets the body airborne to the right altitude. Outside the band
    it's 0 so we don't fight the flat-ground base-height term elsewhere.
    """
    asset = env.scene[asset_cfg.name]
    x = asset.data.root_pos_w[:, 0]
    z = asset.data.root_pos_w[:, 2]
    rew = torch.exp(-torch.square(z - float(target_height)) / (2.0 * float(std) ** 2))
    return rew * _in_x_band(x, x_lo, x_hi).float()


def foot_clearance_band(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_clearance: float = 0.45,
    x_lo: float = -0.5,
    x_hi: float = 0.4,
) -> torch.Tensor:
    """Reward lifting the feet to a target world-z height in the approach band.

    Approaching the step the robot must pick its feet up high enough to land them
    ON the step (not stub them into the vertical face). Reward the max foot height
    (world z) of the configured feet, saturated at `target_clearance`, while the
    base is in the approach x-band. Saturating (clamp to target) avoids paying for
    absurdly high kicks.

    `asset_cfg` must select the FOOT bodies (e.g. body_names=".*_foot"); foot
    world positions come from the robot asset's `body_pos_w` (the contact sensor
    is not configured with track_pose), same source as velocity.feet_height.
    """
    from isaaclab.assets import RigidObject

    asset: RigidObject = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    clearance = torch.clamp(feet_z, max=float(target_clearance)).amax(dim=1)

    x = asset.data.root_pos_w[:, 0]
    return clearance * _in_x_band(x, x_lo, x_hi).float()


def hard_landing_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 400.0,
) -> torch.Tensor:
    """Penalize large vertical foot-contact forces (hard landings).

    A jump that ends in a slamming landing is unsafe and unstable. Penalize the
    summed vertical contact force above a threshold across the configured feet,
    so the policy learns to absorb the landing (flex the legs) rather than crash
    down. Returned as a positive magnitude; apply with a NEGATIVE weight.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w: (num_envs, num_bodies, 3); vertical component on the feet.
    fz = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].abs()
    excess = torch.clamp(fz - float(force_threshold), min=0.0)
    return excess.sum(dim=1)
