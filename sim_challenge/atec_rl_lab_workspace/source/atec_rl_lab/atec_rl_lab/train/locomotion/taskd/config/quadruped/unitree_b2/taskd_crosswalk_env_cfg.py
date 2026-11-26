"""Task D RL: WALK-ACROSS crossing (the REAL 36-point path).

Derived from analyzing the 88-demo 36-point trajectory (submission_mem/root_pos_seq.npy):
the robot does NOT jump or climb a 1 m platform. It crosses the pit on the -y lane
(where there is NO platform) by walking over the box that fills the pit:

  * box settles HALF-SUNK in the pit center at WORLD x ~= -0.25, z ~= -0.7
    (box top z ~= -0.4), filling the 1.0 m pit down to a 0.4 m residual step.
  * robot approaches the near rim at z ~= 0.52 (standing on flat -y ground),
  * STEPS DOWN onto the box top: base z dips to 0.16-0.37 (NOT a climb up!),
  * walks across the box top (x from -0.6 to +0.6, base z stays ~0.2-0.37),
  * STEPS UP the 0.4 m far rim back onto flat ground: base z returns to ~0.55,
  * continues +x to x > 3.5.

  Measured demo z-profile over the crossing (x in [-0.6, 2.0]):
    z MAX over the WHOLE episode = 0.669  (never climbs above standing height)
    z MIN over the crossing     = 0.163  (lowest, on the box top in the pit)
    crossing y                  = -0.59 .. -0.03  (the platform-free -y lane)

This is the OPPOSITE of taskd_cross_env_cfg.py (UnitreeB2TaskDCrossEnvCfg), which
shaped a JUMP onto a 0.95 m platform (takeoff_impulse / flight_phase /
base_height target 0.95) and placed the box against the far wall at x=1.5 — neither
matches the demo. That env's milestones stayed at 0 because it taught the robot to
do something the winning trajectory never does. This env instead rewards the demo's
actual WALK-OVER motion: dense +x progress + staged x-milestones across the box, NO
jump terms, NO high base-height target, box at the demo's settle pose.

Warm-starts from the flat / integrated walking checkpoint (same obs/action interface).
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.velocity.mdp as vel_mdp
import atec_rl_lab.train.locomotion.taskd.mdp as taskd_mdp
from atec_rl_lab.train.locomotion.taskd.config.quadruped.unitree_b2.taskd_env_cfg import (
    UnitreeB2TaskDEnvCfg,
)


# ==========================================================================
# FRAME RULES (verified empirically with diag_crosswalk_poses.py, 2026-06-28):
#   * env_origin = (-4.2, 0, 0). world = env_local + env_origin.
#   * reset_root_state_uniform (velocity-lineage events.py:256) sets
#       world_pos = default_root_state.pos + env_origin + reset_sample
#     so the reset pose_range values are a DELTA added to the asset's DEFAULT
#     init_state.pos, NOT an absolute pose. To land an asset at a desired WORLD
#     pose we must pass delta = world_target - env_origin - default_init_pos.
#   * Reward terms read world-frame root_pos_w directly (milestones in WORLD x).
_ENV_ORIGIN = (-4.2, 0.0, 0.0)

# Asset DEFAULT init positions (env-local, from the cfg init_state.pos):
_BOX_DEFAULT = (-3.0, 1.6, 0.5)      # taskd_env_cfg box init_state
_ROBOT_DEFAULT = (-3.0, 0.0, 0.8)    # UnitreeB2TaskDEnvCfg robot init_state

# ---- Demo-matched WORLD poses (from root_pos_seq.npy analysis) --------------
# Box settles half-sunk in the pit CENTER (demo: box x=-0.25, z=-0.50 surface ->
# center z ~ -0.7 for a 0.6 m-tall box with top at -0.4). The robot walks OVER it.
_BOX_WORLD = (-0.25, -0.45, -0.7)
# Robot starts on the flat -y lane BEHIND the near rim (the pit spans world x in
# [-0.67,+0.67], so start at x=-1.1 — on solid flat ground, NOT in the pit mouth —
# facing +x with a short run-up to the rim). z=0.55 = standing on flat. -y lane =
# platform-free side. (Earlier x=-0.6 put the robot AT the rim -> it spawned in the
# pit at z~0.05 and fell, diag confirmed.)
_ROBOT_WORLD = (-1.1, -0.45, 0.55)


def _reset_delta(world_target, default_local):
    """Reset pose_range center = world_target - env_origin - default_init_pos."""
    return tuple(world_target[i] - _ENV_ORIGIN[i] - default_local[i] for i in range(3))


_BOX_D = _reset_delta(_BOX_WORLD, _BOX_DEFAULT)        # (1.95, -2.05, -1.2)
_ROBOT_D = _reset_delta(_ROBOT_WORLD, _ROBOT_DEFAULT)  # (1.80, -0.45, -0.25)

# ---- Demo-matched crossing milestones (WORLD x) ----------------------------
# From the demo: near rim ~ -0.5, robot down on the box top through x ~ +0.6,
# far rim (back onto flat) ~ +0.67, then clear advance.
_M_NEARRIM = -0.30   # committed to the pit edge (about to step onto the box)
_M_ONBOX = 0.10      # walking across the box top inside the pit
_M_FARRIM = 0.75     # climbed the 0.4 m far rim back onto flat ground (the hard part)
_M_ADVANCE = 1.6     # clear of the obstacle on the far flat ground


@configclass
class UnitreeB2TaskDCrossWalkEnvCfg(UnitreeB2TaskDEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ------------------------------ Observations ------------------------
        # Match the rough-walking checkpoint layout (actor 45-dim: no height_scan;
        # critic restricted to the 12 leg joints) so it warm-starts cleanly. Same
        # rationale as UnitreeB2TaskDCrossEnvCfg.
        self.observations.policy.height_scan = None
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------ Reset randomization -----------------
        # pose_range is a DELTA added to the asset's DEFAULT init pos + env_origin
        # (see FRAME RULES above). _BOX_D / _ROBOT_D are the deltas that land each
        # asset at its demo-matched WORLD pose. Small jitter for robustness to the
        # deploy-time push controller's varying final box/robot pose.
        # Box: settled half-sunk in the pit center (WORLD ~ (-0.25,-0.45,-0.7)).
        self.events.reset_box = EventTerm(
            func=vel_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("box"),
                "pose_range": {
                    "x": (_BOX_D[0] - 0.15, _BOX_D[0] + 0.15),
                    "y": (_BOX_D[1] - 0.20, _BOX_D[1] + 0.20),
                    "z": (_BOX_D[2], _BOX_D[2]),
                    "yaw": (-0.1, 0.1),
                },
                "velocity_range": {},
            },
        )
        # Robot: on the flat -y lane just before the near rim (WORLD ~ (-0.6,-0.45,0.55)), facing +x.
        self.events.reset_robot_front = EventTerm(
            func=vel_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pose_range": {
                    "x": (_ROBOT_D[0] - 0.15, _ROBOT_D[0] + 0.15),
                    "y": (_ROBOT_D[1] - 0.15, _ROBOT_D[1] + 0.15),
                    "z": (_ROBOT_D[2], _ROBOT_D[2]),
                    "yaw": (-0.15, 0.15),
                },
                "velocity_range": {},
            },
        )
        self.events.randomize_reset_base = None

        # ------------------------------ Rewards -----------------------------
        # No box-pushing (box is placed); the robot only walks across.
        self.rewards.taskd_box_progress_x.weight = 0.0
        self.rewards.taskd_box_in_target.weight = 0.0

        # Remove the terms that fight a WALKING step-down/up over the box:
        #   - track_lin/ang_vel: command-mime, pays for marching in place (no command).
        #   - lin_vel_z_l2 (-2.0): penalizes the vertical velocity of stepping down/up.
        #   - feet_height_body (-5.0): penalizes lifting feet to step onto the box.
        #   - upward (3.0): pays for standing upright -> stand-in-front-of-pit optimum.
        #   - base_height_l2: would pull base toward the flat 0.53 target while the
        #     robot legitimately dips to z~0.2 on the box top -> penalizes the crossing.
        self.rewards.upward.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0
        if hasattr(self.rewards, "base_height_l2"):
            self.rewards.base_height_l2.weight = 0.0

        # Core driver: dense +x progress, hard, + stand-still penalty.
        self.rewards.taskd_progress_x.weight = 60.0
        self.rewards.stand_still.weight = -5.0

        # Staged WALK-ACROSS milestones (demo-matched world x). The far rim is the
        # decisive one (climbing the 0.4 m step back onto flat ground), so it pays
        # the most — and far more than any fall cost, so the policy commits to the
        # crossing instead of standing safe on the near side.
        # rearm_x = -1.4 is BELOW the episode start x (~-1.1 +/- 0.15) so no milestone
        # (lowest threshold -0.30) re-arms mid-episode and pays unboundedly. (Episode
        # reset is handled robustly by episode_length_buf<=1 in reach_x_milestone.)
        _rearm = -1.4
        self.rewards.taskd_m_nearrim = RewTerm(
            func=taskd_mdp.reach_x_milestone, weight=80.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "x_threshold": _M_NEARRIM, "key": "nearrim", "rearm_x": _rearm},
        )
        self.rewards.taskd_m_onbox = RewTerm(
            func=taskd_mdp.reach_x_milestone, weight=150.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "x_threshold": _M_ONBOX, "key": "onbox", "rearm_x": _rearm},
        )
        self.rewards.taskd_m_farrim = RewTerm(
            func=taskd_mdp.reach_x_milestone, weight=300.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "x_threshold": _M_FARRIM, "key": "farrim", "rearm_x": _rearm},
        )
        self.rewards.taskd_m_advance = RewTerm(
            func=taskd_mdp.reach_x_milestone, weight=100.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "x_threshold": _M_ADVANCE, "key": "advance", "rearm_x": _rearm},
        )

        # NO jump shaping. The demo never jumps (z never exceeds standing height) and
        # never reaches platform altitude. If a UnitreeB2TaskDCrossEnvCfg attribute
        # for a jump term exists from the parent chain, ensure it is absent here.
        for _name in ("taskd_takeoff", "taskd_flight", "taskd_base_height",
                      "taskd_foot_clear", "taskd_hard_landing"):
            if hasattr(self.rewards, _name):
                setattr(self.rewards, _name, None)

        # ------------------------------ Terminations ------------------------
        # Match the eval env: fall only when base z < -0.4 (the crossing legitimately
        # dips the base to ~0.16 on the box top; a true fall to the pit floor is far
        # lower). illegal_contact OFF (the eval env disables it; during the step the
        # base/legs brush the box and far rim).
        self.terminations.fall.params["minimum_height"] = -0.4
        self.terminations.illegal_contact = None

        # ------------------------------ Curriculum --------------------------
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # ------------------------------ Episode -----------------------------
        # Crossing from in front of the pit is quick.
        self.episode_length_s = 8.0

        self.disable_zero_weight_rewards()
