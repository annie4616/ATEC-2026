# Grid-adaptive fast walker — MIT "Rapid Locomotion" (arXiv:2205.02824) curriculum.
#
# WHY: both the plain fast env and its continued/heading-hardened variants stalled
# at vx≈2.63 (command curriculum stuck at multiplier 0.75). Root cause is the global
# curriculum gate: the shared vx range only widens when the MEAN tracking reward over
# all envs clears 80%, so the envs handed the fastest commands drag the mean down and
# freeze everyone. Confirmed empirically — vx_curr sat at 0.75 for the whole run.
#
# FIX: replace the global gate with a GRID curriculum. (vx, wz) is discretized into
# bins each with a sampling weight; commands are drawn only from active bins, per-env;
# after each episode a bin whose env tracked well (and its neighbors) gains weight.
# Mastery of a speed spreads outward on its own, and a slow bin can't block a fast one
# — the robot walks its own dynamic limit out instead of hitting a shared wall.
#
# Inherits the heading-hardened reward shaping (strong yaw tracking) from FastSym so
# the known leftward yaw drift stays suppressed while speed grows. Also turns on a mild
# base-height reward (MIT note: robots "sink at high speed") to keep posture at speed.
# Warm-starts from the finished 2.63 policy.

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.velocity.mdp as mdp
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.fast_sym_env_cfg import (
    UnitreeB2FastSymEnvCfg,
)


@configclass
class UnitreeB2FastGridEnvCfg(UnitreeB2FastSymEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---------------------- Swap in the grid command ----------------------
        # Rebuild the base_velocity command as a GridVelocityCommand, carrying over
        # the tuning that matters (heading command, standing ratio) from the parent.
        old = self.commands.base_velocity
        self.commands.base_velocity = mdp.GridVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=old.resampling_time_range,
            rel_standing_envs=old.rel_standing_envs,
            rel_heading_envs=old.rel_heading_envs,
            heading_command=old.heading_command,
            heading_control_stiffness=old.heading_control_stiffness,
            debug_vis=old.debug_vis,
            ranges=old.ranges,  # vy / heading still sampled from these
            # Grid over vx in [0, 4] (0.25 m/s bins) and wz in [-1.5, 1.5] (0.5 rad/s).
            grid_vx_range=(0.0, 4.0),
            grid_vx_bins=17,
            grid_wz_range=(-1.5, 1.5),
            grid_wz_bins=7,
            # Seed the WHOLE range the warm-start policy already walks (up to ~2.5),
            # so mean xy_err starts low (no struggling to also crawl very slowly) and
            # the grid only has to grow ABOVE 2.5 toward the true dynamic limit.
            grid_seed_vx=2.5,
            grid_seed_frac=0.15,      # (unused when grid_seed_vx>0)
            # HIGH success bars: a bin only unlocks the next speed when the robot
            # tracks it *well*, so a warm-start policy coasting through easy low
            # speeds can't cascade the whole grid open (that happened at 0.6/0.5:
            # grid shot 0.5->3.5 in ~20 iters while xy_err stayed ~0.6).
            grid_lin_threshold=0.8,   # per-step lin-track frac counted as success
            grid_ang_threshold=0.7,
        )

        # ---------------------- Swap in the grid curriculum -------------------
        # Drop the global-mean command curricula; drive the grid instead.
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        self.curriculum.grid_velocity = CurrTerm(
            func=mdp.grid_velocity_curriculum,
            params={
                "lin_reward_term": "track_lin_vel_xy_exp",
                "ang_reward_term": "track_ang_vel_z_exp",
            },
        )

        # ---------------------- Posture at speed ------------------------------
        # MIT: robots sink their body at high speed. Keep a mild height anchor so the
        # gait stays tall enough to keep pushing faster (rough had this at 0).
        self.rewards.base_height_l2.weight = -1.0

        # Re-run zero-weight pruning for THIS leaf class.
        if self.__class__.__name__ == "UnitreeB2FastGridEnvCfg":
            self.disable_zero_weight_rewards()
