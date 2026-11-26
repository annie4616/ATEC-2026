"""Task D RL environment (MVP).

Built on top of the locomotion velocity-tracking env so all walking-quality
rewards, observations, events and PPO plumbing are inherited. We then swap in
the Task D pit/platform terrain, add the pushable box, and add the minimal
Task-D objective: a dense forward-progress reward + a goal bonus/termination at
x > 3.5. Box-aware observations and box-pushing rewards are intentionally left
for the next iteration (see RL_POLICY_DESIGN.md).
"""

from __future__ import annotations

import copy

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.taskd.mdp as taskd_mdp
import atec_rl_lab.train.locomotion.velocity.mdp as vel_mdp
from atec_rl_lab.train.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

# Reuse the evaluation env's terrain so training matches scoring geometry.
from atec_rl_lab.tasks.task_d.terrain import TASK_D_TERRAIN_CFG, PitAndPlatformTerrainCfg


@configclass
class TaskDRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Task D RL env, robot-agnostic base. Robot binding is done in subclasses.

    Scene geometry (terrain difficulty, box) is kept identical to the evaluation
    env `tasks/task_d/env_cfg.py:TaskDEnvCfg` so a trained policy transfers to
    scoring without a domain gap.
    """

    # Match the evaluation env's pit/platform difficulty exactly (TaskDEnvCfg).
    pit_width_range: tuple[float, float] = (1.3, 1.4)
    platform_height_range: tuple[float, float] = (1.0, 1.2)

    def _build_terrain_cfg(self):
        """Same as TaskDEnvCfg._build_terrain_cfg: apply pit/platform difficulty."""
        terrain_cfg = copy.deepcopy(TASK_D_TERRAIN_CFG)
        pit_cfg = terrain_cfg.terrain_generator.sub_terrains.get("pit_and_platform")
        if isinstance(pit_cfg, PitAndPlatformTerrainCfg):
            pit_cfg.pit_width_range = self.pit_width_range
            pit_cfg.platform_height_range = self.platform_height_range
        return terrain_cfg

    def __post_init__(self):
        super().__post_init__()

        # ---------------------------- Scene: terrain ----------------------------
        # Task D pit/platform terrain with the SAME difficulty as the eval env.
        self.scene.terrain = self._build_terrain_cfg()
        self.sim.physics_material = self.scene.terrain.physics_material
        # Height scan reads the ground mesh; keep it pointed at /World/ground.
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.mesh_prim_paths = ["/World/ground"]
        if self.scene.height_scanner_base is not None:
            self.scene.height_scanner_base.mesh_prim_paths = ["/World/ground"]

        # ---------------------------- Scene: box --------------------------------
        # Same spec as the evaluation env (tasks/task_d/env_cfg.py): 0.8x1.0x0.6,
        # 8 kg, high friction, spawned beside the robot at (-3, 1.6, 0.5).
        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.CuboidCfg(
                size=(0.8, 1.0, 0.6),
                # KINEMATIC: the box kept falling through the pit floor in this env
                # (collision filtering differs from the eval env), so it never stayed as a
                # stepping stone and the robot had nothing to climb. For crossing-only RL
                # the box doesn't need to move — pin it as a fixed/kinematic stepping
                # stone so the robot can reliably step onto it.
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True, kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.9, dynamic_friction=0.8, restitution=0.0,
                ),
                # Visual material so the box RENDERS (without it the box is invisible in
                # videos — it physically exists/collides, but isn't drawn). Blue so it's
                # easy to see the robot step onto it in the pit.
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 0.9)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.0, 1.6, 0.5)),
        )

        # ---------------------------- Commands ----------------------------------
        # Task D is "traverse forward", not arbitrary velocity tracking. Bias the
        # command toward +x and drop lateral/turn ranges so the walking prior keeps
        # the robot heading toward the goal. (Box-pushing emerges from rewards.)
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = True

        # ---------------------------- Rewards (MVP) -----------------------------
        # Keep inherited walking-quality rewards as-is; ADD Task D objective terms.
        self.rewards.taskd_progress_x = RewTerm(
            func=taskd_mdp.progress_x,
            weight=10.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.taskd_reach_goal = RewTerm(
            func=taskd_mdp.reach_goal,
            weight=200.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "x_threshold": 3.5},
        )

        # Box-pushing objective (obs unchanged at 232 -> the flat-walking checkpoint
        # still warm-starts). The box is observed only implicitly via dynamics; these
        # reward terms shape the policy to push it across the pit into the scoring
        # band. box_progress_x drives the box forward; box_in_target parks it in the
        # evaluation success range x in [-0.7, 0.7] (tasks/task_d RewardBoxXInRange).
        self.rewards.taskd_box_progress_x = RewTerm(
            func=taskd_mdp.box_progress_x,
            weight=10.0,
            params={"asset_cfg": SceneEntityCfg("box")},
        )
        self.rewards.taskd_box_in_target = RewTerm(
            func=taskd_mdp.box_in_target,
            weight=50.0,
            params={"asset_cfg": SceneEntityCfg("box"), "x_min": -0.7, "x_max": 0.7},
        )

        # ---------------------------- Terminations ------------------------------
        self.terminations.taskd_reach_goal = DoneTerm(
            func=taskd_mdp.reach_goal_x,
            params={"asset_cfg": SceneEntityCfg("robot"), "x_threshold": 3.5},
            time_out=False,
        )
        # Box contact is expected, so do not terminate on generic illegal contact
        # of the whole body; keep base-only illegal contact (subclass sets bodies).
        #
        # Fall termination, matching the evaluation env (TaskDEnvCfg): end the
        # episode if the base drops below 0.25 m. The velocity lineage has no
        # `fall` term by default, so we add it here. illegal_contact stays on
        # (base-only) for training stability per the agreed plan.
        self.terminations.fall = DoneTerm(
            func=vel_mdp.root_height_below_minimum,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "minimum_height": 0.25,
            },
            time_out=False,
        )
        # Disable terrain_out_of_bounds (inherited from the velocity lineage).
        # It terminates when |world x| > 0.5*map_width - distance_buffer, but Task D
        # spawns the robot at world x=-3 on a single small fixed map, so it fires on
        # the very first step (Mean episode length collapses to 1). The evaluation
        # env (TaskDEnvCfg) has no such term, so we drop it to match.
        self.terminations.terrain_out_of_bounds = None

        # ---------------------------- Episode length ----------------------------
        # Pushing the box across the pit takes a while at flat-policy speeds.
        self.episode_length_s = 40.0

        # ---------------------------- Curriculum --------------------------------
        # Single fixed Task D map -> no terrain-level curriculum for the MVP.
        self.curriculum.terrain_levels = None
