from isaaclab.utils import configclass
##
# Pre-defined configs
##
from parkour_isaaclab.terrains.extreme_parkour.extreme_parkour_terrains_cfg import ExtremeParkourRoughTerrainCfg
# isort: skip
from parkour_isaaclab.terrains.extreme_parkour.config.parkour import STEP_04_TERRAINS_CFG  # isort: skip
from parkour_isaaclab.envs import ParkourManagerBasedRLEnvCfg
from .parkour_mdp_cfg import *
from parkour_tasks.default_cfg import  CAMERA_USD_CFG, CAMERA_CFG, VIEWER
from .parkour_teacher_cfg import ParkourTeacherSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns

# Deploy sensor is the challenge A2's 16-channel 360-deg LiDAR (16 x 360 = 5760 rays),
# matching ATEC2026 task_base envs_base_cfg.lidar_sensor. The student distills the teacher's
# privileged scandots from THIS lidar instead of a depth camera, so the learned policy runs
# on the real robot's actual exteroception (and avoids noisy depth in sim2real).
#
# MOUNT: the real A2 Pro has its LiDAR at front_lidar_link, NOT base_link. From the A2 URDF
# (a2.urdf front_lidar_joint): xyz=(0.33767, 0.0, 0.08134) w.r.t. base_link (fwd 33.8cm, centered,
# up 8.1cm). We keep prim_path on base_link (front_lidar_link is an empty frame) and reproduce the
# real mount via offset.pos so the sim scan originates where the physical sensor does -> sim2real
# position match. rot left identity (URDF rpy is a ROS-optical convention, not a physical tilt;
# LidarPattern already sweeps 360deg horizontally). ray_alignment kept default ("base") to match
# the prior lidar training -- only the MOUNT POSITION changes this run.
# LIDAR_MOUNT env selects the mount: "front" = real A2 front_lidar_link offset (default),
# "base" = base_link origin (matches the ORIGINAL lidar students lo31k/final68 which trained with
# no offset). Lets us run controlled A/B on mount position without editing the cfg.
import os as _os
_MOUNT = _os.environ.get("LIDAR_MOUNT", "front")
_LIDAR_OFFSET = (0.33767, 0.0, 0.08134) if _MOUNT == "front" else (0.0, 0.0, 0.0)
LIDAR_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link",
    offset=RayCasterCfg.OffsetCfg(pos=_LIDAR_OFFSET),
    update_period=0.1,
    pattern_cfg=patterns.LidarPatternCfg(
        vertical_fov_range=(-20.0, 20.0),
        horizontal_fov_range=(-180.0, 180.0),
        horizontal_res=1.0,
        channels=16,
    ),
    max_distance=10.0,
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)


@configclass
class ParkourStudentSceneCfg(ParkourTeacherSceneCfg):
    # depth_camera kept for backward-compat but no longer read by the obs (see StudentObservationsCfg).
    # (Removing it gave no speedup -- the bottleneck is num_steps_per_env, not this sensor -- so it's
    # left in place to avoid the event-manager term-validation error its removal triggered.)
    depth_camera = CAMERA_CFG
    depth_camera_usd = None
    # LiDAR the student actually consumes.
    lidar_sensor = LIDAR_CFG
    
    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator.num_rows = 10
        self.terrain.terrain_generator.num_cols = 20
        self.terrain.terrain_generator.horizontal_scale = 0.1
        for key, sub_terrain in self.terrain.terrain_generator.sub_terrains.items():
            sub_terrain: ExtremeParkourRoughTerrainCfg
            sub_terrain.use_simplified = True 
            sub_terrain.horizontal_scale = 0.1
            if key == 'parkour_demo':
                sub_terrain.proportion = 0.15

            elif key =='parkour_flat':
                sub_terrain.proportion = 0.05

            else:
                sub_terrain.proportion = 0.2
                if key is not 'parkour':
                    sub_terrain.y_range = (-0.1, 0.1)



@configclass
class UnitreeGo2StudentParkourEnvCfg(ParkourManagerBasedRLEnvCfg):
    scene: ParkourStudentSceneCfg = ParkourStudentSceneCfg(num_envs=192, env_spacing=1.)
    # Basic settings
    observations: StudentObservationsCfg = StudentObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: StudentRewardsCfg = StudentRewardsCfg()
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
        self.scene.depth_camera.update_period = self.sim.dt * self.decimation
        self.scene.lidar_sensor.update_period = self.sim.dt * self.decimation
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        self.scene.terrain.terrain_generator.curriculum = True
        self.actions.joint_pos.use_delay = True
        self.actions.joint_pos.history_length = 8



@configclass
class UnitreeGo2StudentParkourEnvCfg_EVAL(UnitreeGo2StudentParkourEnvCfg):
    viewer = VIEWER 
    rewards: TeacherRewardsCfg = TeacherRewardsCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 256
        self.episode_length_s = 20.
        self.commands.base_velocity.debug_vis = True

        self.scene.depth_camera_usd = CAMERA_USD_CFG
        self.scene.terrain.max_init_terrain_level = None

        # (lidar obs term has no debug_vis param; depth-grid cv2 view removed)

        self.commands.base_velocity.resampling_time_range = (60.,60.)
        self.commands.base_velocity.debug_vis = True

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.random_difficulty = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0,1.0)
        self.events.randomize_rigid_body_com = None
        self.events.randomize_rigid_body_mass = None
        self.events.push_by_setting_velocity.interval_range_s = (6.,6.)
        self.events.random_camera_position.params['rot_noise_range'] = {'pitch':(0, 1)}
        
        for key, sub_terrain in self.scene.terrain.terrain_generator.sub_terrains.items():
            if key in ['parkour_flat', 'parkour_demo']:
                sub_terrain.proportion = 0.0
            else:
                sub_terrain.proportion = 0.25
                sub_terrain.noise_range = (0.02, 0.02)

@configclass
class UnitreeGo2StudentParkourEnvCfg_PLAY(UnitreeGo2StudentParkourEnvCfg_EVAL):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.num_envs = 16
        self.episode_length_s = 60.

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.7,1.0)
        self.events.push_by_setting_velocity = None
        for key, sub_terrain in self.scene.terrain.terrain_generator.sub_terrains.items():
            if key =='parkour_flat':
                sub_terrain.proportion = 0.0
            else:
                sub_terrain.proportion = 0.25
                sub_terrain.noise_range = (0.02, 0.02)


@configclass
class UnitreeGo2StudentParkourEnvCfg_STEP06_TRAIN(UnitreeGo2StudentParkourEnvCfg):
    """Distill the STEP06 0.6 m-step teacher (model_18100) into a depth-camera student
    on the SAME step-only terrain the teacher trained on (STEP_04_TERRAINS_CFG, step
    height 0.1+0.5*difficulty, curriculum). Keeps the depth camera + parkour goal signals
    so the student learns the teacher's 0.6 m climbing from depth instead of scandots."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = STEP_04_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.num_rows = 10
        self.scene.terrain.terrain_generator.num_cols = 20
        # headless training: the depth-cam debug viz uses cv2.imshow (needs a display),
        # which crashes under --headless. Turn it off.
        # (lidar obs term has no debug_vis param)

