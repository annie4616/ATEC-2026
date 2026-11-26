from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
# B2: use the ATEC project's verified B2 articulation cfg (legs-only 12-DoF, base
# body = "base_link"). The parkour port was Go2 (base body "base"); we swap to B2
# and update the prim paths that hardcoded /Robot/base -> /Robot/base_link below.
from atec_rl_lab.assets.robots.b2 import UNITREE_B2_CFG  # isort: skip
# B2Piper: deployment task D uses B2 + Piper arm (20 DoF). The B2 teacher fails to
# balance the heavier B2Piper (embodiment gap), so we retrain the teacher natively on
# B2Piper. Same base_link, 12 legs + 8 arm joints. Arm rides at default (passive) —
# we keep all-joint action so dims stay self-consistent; collection records leg actions.
from atec_rl_lab.assets.robots.b2 import UNITREE_B2_PIPER_CFG  # isort: skip
# A2 Pro + Piper X: the deploy robot changed from B2+Piper to A2 Pro + Piper X (2026-08).
# Same 12 legs (FL/FR/RL/RR x hip/thigh/calf, base_link) + 8 arm joints, so the parkour
# pipeline just needs the CFG swap. A2 is much lighter than B2 (~19.6 kg base vs ~60 kg),
# so leg efforts come down (URDF: hip/thigh 120, calf 180) vs B2's 200/320.
from atec_rl_lab.assets.robots.a2 import UNITREE_A2_CFG, UNITREE_A2_PIPER_CFG  # isort: skip
import os as _os
# PARKOUR_ROBOT selects the embodiment: a2piper (default now) | a2 | b2piper | b2.
_PARKOUR_ROBOT = _os.environ.get("PARKOUR_ROBOT", "a2piper").lower()
USE_B2PIPER = _PARKOUR_ROBOT == "b2piper"
_ROBOT_CFG = {
    "a2piper": UNITREE_A2_PIPER_CFG,
    "a2": UNITREE_A2_CFG,
    "b2piper": UNITREE_B2_PIPER_CFG,
    "b2": UNITREE_B2_CFG,
}[_PARKOUR_ROBOT]
# A2 is lighter -> use A2's native leg PD/effort (URDF hip/thigh 120, calf 180). B2 keeps
# its heavier 200/320 gains (set in __post_init__ below only for the B2 embodiments).
_IS_A2 = _PARKOUR_ROBOT in ("a2piper", "a2")
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from parkour_isaaclab.terrains.parkour_terrain_importer import ParkourTerrainImporter
from parkour_tasks.extreme_parkour_task.config.go2 import agents 
from isaaclab.sensors import RayCasterCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.envs import ViewerCfg
import os, torch 
from parkour_isaaclab.actuators.parkour_actuator_cfg import ParkourDCMotorCfg

def quat_from_euler_xyz_tuple(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> tuple:
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    # compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    convert = torch.stack([qw, qx, qy, qz], dim=-1) * torch.tensor([1.,1.,1.,-1])
    return tuple(convert.numpy().tolist())

@configclass
class ParkourDefaultSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = _ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot")
    
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    terrain = TerrainImporterCfg(
        class_type= ParkourTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=None,
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    def __post_init__(self):
        self.robot.spawn.articulation_props.enabled_self_collisions = True
        # B2 is ~4x heavier than Go2; the original Go2 ParkourDCMotor gains
        # (stiffness 40 / damping 1, effort 35-40) cannot hold the B2 up — it sagged
        # and dragged base/thighs on the ground (reward_collision pinned ~-10,
        # current_goal_idx stuck ~0 through iter ~1500). Use B2's native PD/effort
        # from the project cfg (atec_rl_lab b2.py): stiffness 160, damping 5,
        # effort hip/thigh 200, calf 320; velocity limits 23/14 rad/s.
        # Leg effort/velocity limits per embodiment: A2 (light, ~19.6 kg) uses its URDF
        # limits (hip/thigh 120, calf 180; vel 22/14.6). B2 (heavy, ~60 kg) keeps 200/320.
        if _IS_A2:
            _eff = {'.*_hip_joint': 120.0, '.*_thigh_joint': 120.0, '.*_calf_joint': 180.0}
            _vel = {'.*_hip_joint': 22.0, '.*_thigh_joint': 22.0, '.*_calf_joint': 14.6}
        else:
            _eff = {'.*_hip_joint': 200.0, '.*_thigh_joint': 200.0, '.*_calf_joint': 320.0}
            _vel = {'.*_hip_joint': 23.0, '.*_thigh_joint': 23.0, '.*_calf_joint': 14.0}
        self.robot.actuators['base_legs'] = ParkourDCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=dict(_eff),
            saturation_effort=dict(_eff),
            velocity_limit=dict(_vel),
            stiffness=160.0,
            damping=5.0,
            friction=0.0,
        )

## we are now using a raycaster based camera, not a pinhole camera. see tail issue https://github.com/isaac-sim/IsaacLab/issues/719
CAMERA_CFG = RayCasterCameraCfg( 
    prim_path= '{ENV_REGEX_NS}/Robot/base_link',
    data_types=["distance_to_camera"],
    offset=RayCasterCameraCfg.OffsetCfg(
        # pos matched to the A2 head_camera mount (a2.py head_camera_offset pos). rot kept in the
        # raycast camera's ros convention (euler 180,70,-90) -- a forward+down view; the 70deg here
        # approximates the A2 camera's 30deg-down pitch in this convention. Tune after a render check.
        pos=(0.4216, 0.025, 0.0619),
        rot=quat_from_euler_xyz_tuple(*tuple(torch.deg2rad(torch.tensor([180,70,-90])))),
        convention="ros"
        ),
    depth_clipping_behavior = 'max',
    pattern_cfg = PinholeCameraPatternCfg(
        focal_length=11.041, 
        horizontal_aperture=20.955,
        vertical_aperture = 12.240,
        height=60,
        width=106,
    ),
    mesh_prim_paths=["/World/ground"],
    max_distance = 2.,
)

CAMERA_USD_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link/d435",
    spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(agents.__path__[0],'d435.usd')),
    init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.33, 0.0, 0.08), 
            rot=quat_from_euler_xyz_tuple(*tuple(torch.deg2rad(torch.tensor([180,90,-90]))))
    )
)
VIEWER = ViewerCfg(
    # Follow the robot, framed for B2 (bigger than Go2) on the raised parkour terrain.
    # The original (0, 2.6, 1.6) sat too close/low and the recorded frame showed only
    # ground texture with no robot. Pull the eye back (-x behind), out (+y side) and up
    # (+z), and aim the lookat slightly ahead/up of the robot root so the body is in view.
    eye=(-3.0, 3.0, 2.2),
    lookat=(0.0, 0.0, 0.4),
    asset_name = "robot",
    origin_type = 'asset_root',
)
