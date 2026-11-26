# High-speed rough-walking variant of the B2 velocity policy — PROGRESSIVE (curriculum).
#
# The base rough policy (UnitreeB2RoughEnvCfg) only sees forward commands in
# lin_vel_x = (-1.0, 1.0). We want fast forward walking (vx up to 3.5 m/s). A first
# attempt simply widened the range to (0.0, 3.5) all at once and warm-started — that
# FAILED: the policy learned to STAND STILL at high commands (verified: at a constant
# vx=3.5 command the robot's actual velocity was ~0 and it moved <0.05 m in 10 s,
# while at vx=1.0 it walked fine). Jumping straight to 3.5 made tracking so hard that
# "stand still and eat the tracking penalty" beat trying to run.
#
# FIX: ramp the command range UP GRADUALLY with the reward-gated velocity curriculum
# (mdp.command_levels_lin_vel). It starts the range at 10% of the target and only
# widens it (+0.1 m/s) once the tracking reward exceeds 80% of its max at the current
# speed — i.e. the robot must actually walk well at the current speed before it is
# asked to go faster. So vx ramps 0.35 -> ... -> 3.5 only as fast as the robot can
# keep up. Warm-starts from the rough checkpoint (same robot/obs/action interface).

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.velocity.mdp as mdp
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.rough_env_cfg import (
    UnitreeB2RoughEnvCfg,
)


@configclass
class UnitreeB2FastEnvCfg(UnitreeB2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ------------------------------ Commands ------------------------------
        # FINAL (target) command ranges. The curriculum below starts at 10% of these
        # and grows toward them. Forward bias only (lower bound 0 -> no backward), with
        # modest lateral/turn so the policy can still correct heading at speed (the OOD
        # failure mode was a +y drift, so we keep some lateral/yaw samples).
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 3.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)

        # ------------------------------ Curriculum ----------------------------
        # Re-enable the reward-gated velocity curriculum (the rough base disabled it).
        # range_multiplier=(0.1, 1.0): start the x/y ranges at 10% of the configured
        # target and widen to 100% as tracking quality allows. Result: vx ramps
        # 0.35 -> 3.5 progressively, never faster than the robot can actually track.
        self.curriculum.command_levels_lin_vel = CurrTerm(
            func=mdp.command_levels_lin_vel,
            params={"reward_term_name": "track_lin_vel_xy_exp", "range_multiplier": (0.1, 1.0)},
        )
        # Keep the angular curriculum too so turn commands ramp alongside.
        self.curriculum.command_levels_ang_vel = CurrTerm(
            func=mdp.command_levels_ang_vel,
            params={"reward_term_name": "track_ang_vel_z_exp", "range_multiplier": (0.1, 1.0)},
        )

        # Re-run the zero-weight reward pruning for THIS class (the parent only prunes
        # when it is the leaf class).
        if self.__class__.__name__ == "UnitreeB2FastEnvCfg":
            self.disable_zero_weight_rewards()
