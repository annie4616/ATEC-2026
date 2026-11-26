"""Flat-ground walking PRETRAIN for the Task D B2+Piper policy.

The Task D rough env (pit/platform terrain + box, spawned on rough ground) is too
hard to learn a B2+Piper walking gait from scratch: the policy falls within ~0.3 s
every episode and never accumulates forward progress (progress_x stays ~0).

This env keeps EVERYTHING that defines the policy interface identical to
`UnitreeB2TaskDEnvCfg` -- same robot (20-DoF B2+Piper, 12-DoF leg action), same
observation terms INCLUDING the height scan (so the obs dimension matches and the
trained checkpoint loads straight into the Task D env) -- but:

  * swaps the terrain to a flat plane,
  * removes the box and the Task-D objective reward/termination terms, and
  * restores full velocity-command ranges,

so the policy first learns a robust omnidirectional walking gait. That checkpoint
is then used to warm-start Task D training (`--resume`/load_run), where only the
pit/platform traversal + box pushing remain to be learned.
"""

from isaaclab.utils import configclass

from atec_rl_lab.train.locomotion.taskd.config.quadruped.unitree_b2.taskd_env_cfg import (
    UnitreeB2TaskDEnvCfg,
)


@configclass
class UnitreeB2FlatPretrainEnvCfg(UnitreeB2TaskDEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ------------------------------ Terrain -----------------------------
        # Flat ground. Keep the height scanner (so obs dim is unchanged and the
        # checkpoint transfers to Task D) -- on a plane it just reads ~constant.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # The project's custom terrain importer (terrain_base.py:import_ground_plane)
        # references an undefined `logger` on the "visual_material set but no diffuse
        # color" branch; clearing visual_material skips that branch (pre-existing bug).
        self.scene.terrain.visual_material = None
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.mesh_prim_paths = ["/World/ground"]
        if self.scene.height_scanner_base is not None:
            self.scene.height_scanner_base.mesh_prim_paths = ["/World/ground"]

        # ------------------------------ Scene -------------------------------
        # No box for the walking pretrain; spawn at the flat origin (z=0.8 keeps the
        # same settle margin as Task D so the standing pose matches).
        self.scene.box = None
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.8)

        # ------------------------------ Commands ----------------------------
        # Robust omnidirectional walking: restore the full velocity command ranges
        # (Task D had biased them to +x only). The leg gait learned here generalizes.
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)

        # ------------------------------ Rewards -----------------------------
        # Pure walking: drop the Task-D objective terms. Velocity tracking + the
        # inherited walking-quality rewards carry the gait. Zero the weights (not
        # None) so disable_zero_weight_rewards() below removes them cleanly.
        self.rewards.taskd_progress_x.weight = 0.0
        self.rewards.taskd_reach_goal.weight = 0.0
        # The box is removed on flat ground, so its push rewards (inherited from
        # TaskDRoughEnvCfg) reference a non-existent 'box' entity -> zero them so
        # disable_zero_weight_rewards() strips them before the scene is built.
        self.rewards.taskd_box_progress_x.weight = 0.0
        self.rewards.taskd_box_in_target.weight = 0.0
        # Make velocity tracking the dominant signal so the robot actually moves.
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        # ------------------------------ Terminations ------------------------
        # No Task-D goal termination on flat ground (there is no x=3.5 goal here).
        self.terminations.taskd_reach_goal = None

        # ------------------------------ Episode -----------------------------
        # Standard walking episode length is plenty for gait learning.
        self.episode_length_s = 20.0

        # Drop any now-zero-weight rewards (clears unused attrs).
        if self.__class__.__name__ == "UnitreeB2FlatPretrainEnvCfg":
            self.disable_zero_weight_rewards()
