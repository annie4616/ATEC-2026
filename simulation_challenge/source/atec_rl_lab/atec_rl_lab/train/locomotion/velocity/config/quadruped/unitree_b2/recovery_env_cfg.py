# Fall-recovery / get-up variant of the B2 velocity policy.
#
# Goal: from an arbitrary FALLEN or FLIPPED pose the robot should roll upright and
# push up to a nominal standing height, then hold it. This is the classic quadruped
# fall-recovery task (Isaac Gym Go1/ANYmal recovery controllers; "Learning Complex
# Motor Skills for Legged Robot Fall Recovery", RA-L 2023; adaptive fall-recovery
# arXiv:2412.16924). There is no widely-used drop-in *Isaac Lab manager-based*
# recovery repo, so this ports the community-standard recipe onto our rough env:
#
#   1. Reset the base to fully random orientation (roll/pitch/yaw over the full
#      circle) *dropped* from a small height, so ~every episode starts fallen —
#      the base rough env already samples roll/pitch in (-pi, pi) but we drop the
#      base and zero the reset velocity so it settles into a genuine fallen pose.
#   2. Never terminate on body contact (the robot IS lying on its body) — inherited
#      from the rough env (illegal_contact = None).
#   3. Zero velocity command: recovery is a stand-still-and-get-up task, not walking,
#      so the tracking rewards target v=0 (i.e. "come to rest standing").
#   4. Dense get-up rewards: exp uprightness (from any flipped pose) + height-tracking
#      gated on being upright, so the ordering is: flip upright THEN stand.
#
# Warm-starts from the rough checkpoint (identical robot / obs / action interface),
# so the recovered policy keeps the same 45-dim obs and 12-joint action layout and
# can be swapped in / blended with the walker at deploy time.

import math

from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.velocity.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.rough_env_cfg import (
    UnitreeB2RoughEnvCfg,
)


@configclass
class UnitreeB2RecoveryEnvCfg(UnitreeB2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ------------------------------ Commands ------------------------------
        # Recovery = get up and STAND (no locomotion). Command velocity ~0 so the
        # velocity-tracking rewards reward coming to rest while upright, and the
        # policy is never asked to walk away while still recovering. A tiny range
        # keeps the command obs non-degenerate.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0  # all envs are "standing" commands
        self.commands.base_velocity.heading_command = False

        # ------------------------------ Events --------------------------------
        # Start every episode FALLEN: full random orientation, dropped from a small
        # height, zero initial velocity so it settles onto its body/side/back.
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (0.25, 0.45),          # drop from above so it lands fallen
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {             # no initial push -> genuine fallen settle
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        # Joints start scattered around default so the policy can't memorize one pose.
        self.events.randomize_reset_joints.params = {
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        }
        # Keep periodic pushes off during recovery (we want it to settle & get up,
        # not be knocked around mid-stand). Interval pushes stay but weakened.
        if getattr(self.events, "randomize_push_robot", None) is not None:
            self.events.randomize_push_robot.params["velocity_range"] = {
                "x": (-0.3, 0.3), "y": (-0.3, 0.3),
            }

        # ------------------------------ Rewards -------------------------------
        # Dense get-up signal (added mdp terms). Uprightness dominates so the robot
        # first flips over; height (gated on upright) then makes it push to a stand.
        self.rewards.upright_orientation = RewTerm(
            func=mdp.upright_orientation_exp,
            weight=5.0,
            params={"std": 0.6, "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"])},
        )
        self.rewards.stand_height = RewTerm(
            func=mdp.stand_height_when_upright,
            weight=5.0,
            params={
                "target_height": 0.53,
                "std": 0.2,
                "upright_threshold": 0.5,
                "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
                "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            },
        )
        # Keep the coarse L2 upward term as extra pull toward vertical.
        self.rewards.upward.weight = 2.0
        # Turn ON base-height L2 (rough had it at 0) as a mild standing anchor.
        self.rewards.base_height_l2.weight = -1.0
        # Recovery has no gait; drop walking-specific shaping that fights lying still.
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.stand_still.weight = 0.0           # don't punish "not walking"
        self.rewards.joint_pos_penalty.weight = -0.5    # softer: big pose changes needed to get up
        # Encourage stillness once up: penalize residual joint velocity a bit.
        self.rewards.joint_vel_l2.weight = -1e-4
        # Keep tracking (targets v=0) so the standing pose is quiet, not twitchy.
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5

        # ------------------------------ Curriculum ----------------------------
        # Recovery is learned on flat/rough terrain difficulty, not a locomotion
        # course. Keep terrain_levels but no command curriculum (command is fixed 0).
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # Longer episodes: getting up + settling can take a few seconds.
        self.episode_length_s = 8.0

        # Re-run zero-weight pruning for THIS leaf class.
        if self.__class__.__name__ == "UnitreeB2RecoveryEnvCfg":
            self.disable_zero_weight_rewards()
