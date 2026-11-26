
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
##
# Pre-defined configs
##
from parkour_isaaclab.terrains.extreme_parkour.config.parkour import EXTREME_PARKOUR_TERRAINS_CFG, TASKD_PIT_TERRAINS_CFG, STEP_04_TERRAINS_CFG, TASKD_PIT_REAL_TERRAINS_CFG, STAIRCASE_TERRAINS_CFG  # isort: skip
from parkour_isaaclab.envs import ParkourManagerBasedRLEnvCfg
from .parkour_mdp_cfg import * 
from parkour_tasks.default_cfg import ParkourDefaultSceneCfg, VIEWER

@configclass
class ParkourTeacherSceneCfg(ParkourDefaultSceneCfg):
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",  # B2Piper base body (was Go2 'base')
        offset=RayCasterCfg.OffsetCfg(pos=(0.375, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=[1.65, 1.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", 
                                      history_length=2, 
                                      track_air_time=True, 
                                      debug_vis= False,
                                      force_threshold=1.
                                      )
    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator = EXTREME_PARKOUR_TERRAINS_CFG
        
@configclass
class UnitreeGo2TeacherParkourEnvCfg(ParkourManagerBasedRLEnvCfg):
    scene: ParkourTeacherSceneCfg = ParkourTeacherSceneCfg(num_envs=6144, env_spacing=1.)
    # Basic settings
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: TeacherRewardsCfg = TeacherRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    parkours: ParkourEventsCfg = ParkourEventsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**18
        # update sensor update periods
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        self.scene.terrain.terrain_generator.curriculum = True
        self.actions.joint_pos.use_delay = False
        self.actions.joint_pos.history_length = 1
        self.events.random_camera_position = None

@configclass
class UnitreeGo2TeacherParkourEnvCfg_EVAL(UnitreeGo2TeacherParkourEnvCfg):
    viewer = VIEWER 

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 256
        self.episode_length_s = 20.
        self.parkours.base_parkour.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.random_difficulty = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0,1.0)
        self.events.randomize_rigid_body_com = None
        self.events.randomize_rigid_body_mass = None
        self.events.push_by_setting_velocity.interval_range_s = (6.,6.)
        self.commands.base_velocity.resampling_time_range = (60.,60.)
        for key, sub_terrain in self.scene.terrain.terrain_generator.sub_terrains.items():
            if key ==['parkour','parkour_hurdle','parkour_step','parkour_gap']:
                sub_terrain.noise_range = (0.02, 0.02)
                sub_terrain.proportion = 0.25
                
@configclass
class UnitreeGo2TeacherParkourEnvCfg_PLAY(UnitreeGo2TeacherParkourEnvCfg_EVAL):
    viewer = VIEWER

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 60.
        self.scene.num_envs = 16
        self.parkours.base_parkour.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.7,1.0)
        self.events.push_by_setting_velocity = None
        for key, sub_terrain in self.scene.terrain.terrain_generator.sub_terrains.items():
            if key =='parkour_flat':
                sub_terrain.proportion = 0.0
            else:
                sub_terrain.proportion = 0.2
                sub_terrain.noise_range = (0.02, 0.02)


# ---------------------------------------------------------------------------
# TASK D PIT: train/play the teacher on the "box-as-stepping-stone pit" — flat ->
# 1.0m-deep pit with a 0.45m-below-rim stone -> flat. This is the REAL Task D
# walk-across (drop into the box-filled pit, then climb the far rim back to flat),
# which the STEP-only teacher cannot do (it only learned to climb UP a single step,
# not the down-then-up pit). We RL-train this with the verified parkour rewards,
# only swapping the terrain to TASKD_PIT and keeping curriculum on so the goal
# waypoints (near rim -> stone -> far rim) steer the down-then-up crossing.
@configclass
class UnitreeGo2TeacherParkourEnvCfg_TASKDPIT_TRAIN(UnitreeGo2TeacherParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = TASKD_PIT_TERRAINS_CFG
        # TASKD_PIT pit geometry is fixed (difficulty scales only surface roughness);
        # curriculum on so the policy is steered through the near-rim -> stone -> far-rim
        # goal waypoints. Warm-start from the STEP teacher (model_9900) carries the
        # walking + step-up prior into the harder down-then-up pit crossing.
        self.scene.terrain.terrain_generator.curriculum = True


@configclass
class UnitreeGo2TeacherParkourEnvCfg_TASKDPIT_PLAY(UnitreeGo2TeacherParkourEnvCfg_PLAY):
    """Play/eval on the fixed TASKD_PIT (down-then-up pit cross), to check whether the
    teacher actually reaches the far rim (z back to ~0.5) rather than stalling in the pit."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = TASKD_PIT_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.difficulty_range = (0.5, 0.5)  # roughness only
        for key, sub_terrain in self.scene.terrain.terrain_generator.sub_terrains.items():
            sub_terrain.noise_range = (0.02, 0.02)


@configclass
class UnitreeGo2TeacherParkourEnvCfg_STEP_PLAY(UnitreeGo2TeacherParkourEnvCfg_PLAY):
    """Play on the STEP-only terrain (single up-step) to isolate the teacher's step
    climbing. step_height = 0.1 + 0.5*difficulty; difficulty pinned via PARKOUR_PLAY_DIFF
    (default 1.0 = 0.6m; 0.6 = 0.4m, 0.72 = 0.46m the teacher mastered)."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = STEP_04_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        import os as _os_sp
        _pd = float(_os_sp.environ.get('PARKOUR_PLAY_DIFF', '1.0'))
        self.scene.terrain.terrain_generator.difficulty_range = (_pd, _pd)
        for key, sub_terrain in self.scene.terrain.terrain_generator.sub_terrains.items():
            sub_terrain.noise_range = (0.02, 0.02)


@configclass
class UnitreeGo2TeacherParkourEnvCfg_STAIRCASE_PLAY(UnitreeGo2TeacherParkourEnvCfg_PLAY):
    """Play on the STAIRCASE-only terrain (5 consecutive up-steps, <=0.17 m each = the
    competition course staircase) to inspect the teacher climbing a full 5-step staircase.
    difficulty pinned via PARKOUR_PLAY_DIFF (default 1.0 = 0.17 m steps; 0.5 = 0.12 m)."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = STAIRCASE_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        import os as _os_sc
        _pd = float(_os_sc.environ.get('PARKOUR_PLAY_DIFF', '1.0'))
        self.scene.terrain.terrain_generator.difficulty_range = (_pd, _pd)
        for key, sub_terrain in self.scene.terrain.terrain_generator.sub_terrains.items():
            sub_terrain.noise_range = (0.02, 0.02)


# TASKD_PIT matched to the REAL Task D geometry (pit_width 1.35, box-stone len 0.8) —
# the prior TASKD_PIT (0.8 pit / 0.5 stone) was too narrow/easy and didn't match deploy.
@configclass
class UnitreeGo2TeacherParkourEnvCfg_TASKDPITREAL_TRAIN(UnitreeGo2TeacherParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = TASKD_PIT_REAL_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = True


@configclass
class UnitreeGo2TeacherParkourEnvCfg_TASKDPITREAL_PLAY(UnitreeGo2TeacherParkourEnvCfg_PLAY):
    """Play/eval on the REAL-geometry Task D pit (pit 1.35, box-stone 0.8)."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = TASKD_PIT_REAL_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.difficulty_range = (0.5, 0.5)  # roughness only
        for key, sub_terrain in self.scene.terrain.terrain_generator.sub_terrains.items():
            sub_terrain.noise_range = (0.02, 0.02)


