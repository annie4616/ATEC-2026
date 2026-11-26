# Created by skywoodsz on 2026/02/06.

"""
Implementation of Task A environment configuration with different robots.
"""

import math

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from atec_rl_lab.tasks.task_base import TerminationsCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
import isaaclab.sim as sim_utils

from atec_rl_lab.tasks.task_base import BaseEnvCfg
from .terrain import TASK_A_TERRAIN_CFG, TASK_A_CORRIDOR_TERRAIN_CFG
import atec_rl_lab.tasks.task_a.mdp as atec_mdp

# ---------------------------------------------------------------------------
# Red road-boundary strips (real-world guideline III-3: "A red 40 cm isolation
# strip is installed along the course as a road-boundary marker"). The sim course
# runs along +x (x=-140..+150) with the robot near y=0; add two long red visual
# strips at y=+/-COURSE_HALF_W so the head camera actually sees the steering cue
# the kyaw teacher uses (fallacy-1: sim otherwise has NO visual heading cue).
# Purely visual (no collision), thin box of width 0.40 m per the spec.
# ---------------------------------------------------------------------------
_COURSE_HALF_W = 4.0          # course half-width: strips at y = +/-4.0 m
_STRIP_W = 0.40               # red strip width (guideline: 40 cm)
_STRIP_LEN = 300.0            # spans the whole ~290 m course
_STRIP_X = 5.0                # centered so it covers x in [-145, +155]
_STRIP_Z = 0.02               # sit just above ground, laid flat


def _red_strip_cfg(prim_path: str, y: float) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=(_STRIP_LEN, _STRIP_W, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.05, 0.05), roughness=0.8
            ),
            collision_props=None,   # visual only, no physics
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_STRIP_X, y, _STRIP_Z)),
    )

@configclass
class TaskATerminationsCfg(TerminationsCfg):
    reach_goal_x = DoneTerm(
        func=atec_mdp.robot_x_greater_than,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "x_threshold": 145.0,
        },
        time_out=False,
    )

@configclass
class RewardsCfg:
    progress_reward = RewTerm(
        func=atec_mdp.CrossXMulti,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "x_thresholds": [-115.0, -35.0, 45.0, 125.0, 140.0],
            "rewards": [2.0, 4.0, 8.0, 8.0, 4.0],
            "debug": False,
            "visual_assets": True,
        },
        weight=1.0,
    )


@configclass
class TaskAEnvCfg(BaseEnvCfg):
    """Environment base configuration for Task A."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain = TASK_A_TERRAIN_CFG
        self.sim.physics_material = self.scene.terrain.physics_material
        self.rewards = RewardsCfg()
        self.terminations = TaskATerminationsCfg()

        # Red road-boundary strips (real-world guideline III-3) so the camera has a
        # visual heading cue. NOTE: these flat CuboidCfg strips sit at a FIXED z and
        # therefore float / sink where the terrain height varies (rough / slopes /
        # stairs). For rendering (and any camera-based use over non-flat terrain) the
        # strips must HUG the terrain surface instead — see scripts/render_taska_redstrip.py
        # which ray-casts the terrain mesh and lays ground-hugging segments. Kept here
        # (flat approximation) only for the earlier flat-course data collection.
        self.scene.red_strip_left = _red_strip_cfg("/World/red_strip_left", +_COURSE_HALF_W)
        self.scene.red_strip_right = _red_strip_cfg("/World/red_strip_right", -_COURSE_HALF_W)

        # Turn off the DR and noise
        self.observations.proprio.enable_corruption = False
        self.observations.extero.enable_corruption = False
        self.observations.image.enable_corruption = False
        self.events.physics_material = None
        self.events.base_external_force_torque = None
        # self.events.reset_robot_joints = None

        self.terminations.fall.params["minimum_height"] = -20.0



@configclass
class TaskAEnvG1Cfg(TaskAEnvCfg):
    """Environment configuration for Task A with Unitree g1."""
    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_G1_29DOF_DEX1_CFG

        self.scene.robot = UNITREE_G1_29DOF_DEX1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_G1_29DOF_DEX1_CFG.init_state.replace(
                pos=(-141, 0, 0.8),
            )
        )
        super().__post_init__()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            UNITREE_G1_29DOF_DEX1_CFG.base_link_name,
            ".*_hip_(pitch|roll|yaw)_link"
        ]

        joint_names = UNITREE_G1_29DOF_DEX1_CFG.joint_names
        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names
        self.actions.joint_leg.joint_names = joint_names
        self.actions.joint_wheel = None
        self.actions.joint_arm = None


@configclass
class TaskAEnvTron1Cfg(TaskAEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import TRON1A_PIPER_CFG

        self.scene.robot = TRON1A_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=TRON1A_PIPER_CFG.init_state.replace(
                pos=(-141, 0, 0.8 + 0.166),
            )
        )
        super().__post_init__()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            TRON1A_PIPER_CFG.base_link_name,
            "abad_[LR]_Link"
        ]

        joint_names = TRON1A_PIPER_CFG.joint_names
        leg_joint_names = TRON1A_PIPER_CFG.leg_joint_names
        wheel_joint_names = TRON1A_PIPER_CFG.wheel_joint_names
        arm_joint_names = TRON1A_PIPER_CFG.arm_joint_names

        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names

        self.actions.joint_leg.joint_names = leg_joint_names
        self.actions.joint_wheel.joint_names = wheel_joint_names
        self.actions.joint_arm.joint_names = arm_joint_names


@configclass
class TaskAEnvTron2ALeggedCfg(TaskAEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import TRON2A_LEGGED_CFG

        self.scene.robot = TRON2A_LEGGED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=TRON2A_LEGGED_CFG.init_state.replace(
                pos=(-141, 0, 0.8 + 0.166),
            )
        )
        super().__post_init__()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            TRON2A_LEGGED_CFG.base_link_name,
            "proximal_.*_[LR]_Link",
        ]

        joint_names = TRON2A_LEGGED_CFG.joint_names
        leg_joint_names = TRON2A_LEGGED_CFG.leg_joint_names
        arm_joint_names = TRON2A_LEGGED_CFG.arm_joint_names

        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names

        self.actions.joint_leg.joint_names = leg_joint_names
        self.actions.joint_wheel = None
        self.actions.joint_arm.joint_names = arm_joint_names


@configclass
class TaskAEnvTron2AWheelCfg(TaskAEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import TRON2A_WHEEL_CFG

        self.scene.robot = TRON2A_WHEEL_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=TRON2A_WHEEL_CFG.init_state.replace(
                pos=(-141, 0, 0.8 + 0.166),
            )
        )
        super().__post_init__()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            TRON2A_WHEEL_CFG.base_link_name,
            "proximal_.*_[LR]_Link",
        ]

        joint_names = TRON2A_WHEEL_CFG.joint_names
        leg_joint_names = TRON2A_WHEEL_CFG.leg_joint_names
        wheel_joint_names = TRON2A_WHEEL_CFG.wheel_joint_names
        arm_joint_names = TRON2A_WHEEL_CFG.arm_joint_names

        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names

        self.actions.joint_leg.joint_names = leg_joint_names
        self.actions.joint_wheel.joint_names = wheel_joint_names
        self.actions.joint_arm.joint_names = arm_joint_names


@configclass
class TaskAEnvB2Cfg(TaskAEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_B2_PIPER_CFG

        self.scene.robot = UNITREE_B2_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_B2_PIPER_CFG.init_state.replace(
                pos=(-141, 0, 0.58),
            )
        )
        super().__post_init__()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            UNITREE_B2_PIPER_CFG.base_link_name,
            ".*_hip",
            ".*_thigh"
        ]

        joint_names = UNITREE_B2_PIPER_CFG.joint_names
        leg_joint_names = UNITREE_B2_PIPER_CFG.leg_joint_names
        arm_joint_names = UNITREE_B2_PIPER_CFG.arm_joint_names

        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names

        self.actions.joint_leg.joint_names = leg_joint_names
        self.actions.joint_arm.joint_names = arm_joint_names
        self.actions.joint_wheel = None

@configclass
class TaskAEnvA2Cfg(TaskAEnvCfg):
    """Task A with the A2 Pro + Piper X (20 DoF) — the deploy embodiment the parkour lidar
    student was trained on. Mirrors TaskAEnvB2Cfg; A2 shares B2's body/joint naming
    (base_link, .*_hip/.*_thigh/.*_calf/.*_foot, per-leg FR/FL/RR/RL + joint1..8) so only the
    robot CFG and the spawn height change. A2 stands lower/lighter than B2, so spawn z=0.55
    (folded-leg init clears the ground; the robot settles to base z~0.38)."""

    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_A2_PIPER_CFG

        self.scene.robot = UNITREE_A2_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_A2_PIPER_CFG.init_state.replace(
                pos=(-141, 0, 0.55),
            )
        )
        super().__post_init__()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            UNITREE_A2_PIPER_CFG.base_link_name,
            ".*_hip",
            ".*_thigh"
        ]

        joint_names = UNITREE_A2_PIPER_CFG.joint_names
        leg_joint_names = UNITREE_A2_PIPER_CFG.leg_joint_names
        arm_joint_names = UNITREE_A2_PIPER_CFG.arm_joint_names

        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names

        self.actions.joint_leg.joint_names = leg_joint_names
        self.actions.joint_arm.joint_names = arm_joint_names
        self.actions.joint_wheel = None

        # The lidar parkour student consumes ONLY the 5760-ray lidar (extero.lidar_scan) —
        # it never reads the head/ee RGB-D cameras. Rendering those per-step camera textures
        # is the dominant cost and makes play unusably slow, so disable the cameras and their
        # image obs terms. The lidar RayCaster (extero) stays on; RecordVideo's viewport
        # capture is independent of these sensor cameras.
        self.scene.head_camera = None
        self.scene.ee_camera = None
        self.scene.ee_dual_camera = None
        self.observations.image = None


@configclass
class TaskAEnvB2WCfg(TaskAEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_B2W_PIPER_CFG

        self.scene.robot = UNITREE_B2W_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_B2W_PIPER_CFG.init_state.replace(
                pos=(-141, 0, 0.78), 
            )
        )
        super().__post_init__()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            UNITREE_B2W_PIPER_CFG.base_link_name,
            ".*_hip",
            ".*_thigh"
        ]

        joint_names = UNITREE_B2W_PIPER_CFG.joint_names
        leg_joint_names = UNITREE_B2W_PIPER_CFG.leg_joint_names
        wheel_joint_names = UNITREE_B2W_PIPER_CFG.wheel_joint_names
        arm_joint_names = UNITREE_B2W_PIPER_CFG.arm_joint_names

        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names

        self.actions.joint_leg.joint_names = leg_joint_names
        self.actions.joint_wheel.joint_names = wheel_joint_names
        self.actions.joint_arm.joint_names = arm_joint_names


# ===========================================================================
# CURVED-COURSE VARIANTS (robustness / generalization tests)
#
# The base course is a STRAIGHT +x corridor with two straight red strips at
# y=+/-4. Two variants below bend the course so we can check whether the vision
# steering head follows a NON-STRAIGHT red boundary instead of just marching +x:
#
#   1) TaskAEnvB2CurvedStripCfg : red strips follow a sine centerline; terrain and
#      the x>=145 goal are UNCHANGED. The head should steer along the curve, but a
#      pure +x goal still ends it -- isolates "does the head TRACK a curved cue".
#   2) TaskAEnvB2BentGoalCfg    : same curved strips AND the finish is a POINT that
#      sits off the +x axis (end of the curve). The x-threshold goal is replaced by
#      a (x,y) proximity goal, so the robot must genuinely follow the curve to win.
#
# Centerline: y_c(x) = amp * sin(2*pi*(x-x0)/period). Each red strip is built from
# many short cuboid segments laid along the centerline, offset by +/-half_w along the
# local NORMAL and yawed to the local tangent so the boundary stays a clean curve.
# ===========================================================================

_CURVE_AMP = 6.0        # sine amplitude (m): peak lateral excursion of the centerline
_CURVE_PERIOD = 180.0   # sine period (m): one full wiggle over ~180 m (gentle)
_CURVE_X0 = -141.0      # phase so the centerline starts at y=0 at the spawn
_SEG_LEN = 4.0          # length of each red cuboid segment (m)
_SEG_OVERLAP = 1.15     # draw segments slightly longer than the step so no gaps on the curve
_CURVE_XMIN = -145.0
_CURVE_XMAX = 155.0


def _centerline_y(x: float) -> float:
    return _CURVE_AMP * math.sin(2.0 * math.pi * (x - _CURVE_X0) / _CURVE_PERIOD)


def _centerline_tangent(x: float) -> float:
    """dy/dx of the centerline -> local tangent angle (yaw) in radians."""
    dydx = _CURVE_AMP * (2.0 * math.pi / _CURVE_PERIOD) * math.cos(
        2.0 * math.pi * (x - _CURVE_X0) / _CURVE_PERIOD)
    return math.atan2(dydx, 1.0)


def _yaw_to_quat(yaw: float):
    """(w,x,y,z) quaternion for a rotation about +z."""
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def _curved_strip_segments(scene, side_sign: float, half_w: float, name_prefix: str):
    """Attach many short red cuboid segments following the sine centerline, offset by
    side_sign*half_w along the local normal. Adds them as scene attributes."""
    n = int((_CURVE_XMAX - _CURVE_XMIN) / _SEG_LEN)
    for i in range(n):
        xc = _CURVE_XMIN + (i + 0.5) * _SEG_LEN
        yaw = _centerline_tangent(xc)
        yc = _centerline_y(xc)
        nx, ny = -math.sin(yaw), math.cos(yaw)   # local normal (left=+, right=-)
        px = xc + side_sign * half_w * nx
        py = yc + side_sign * half_w * ny
        seg = AssetBaseCfg(
            prim_path=f"/World/{name_prefix}_{i}",
            spawn=sim_utils.CuboidCfg(
                size=(_SEG_LEN * _SEG_OVERLAP, _STRIP_W, 0.02),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.05, 0.05), roughness=0.8),
                collision_props=None,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(px, py, _STRIP_Z), rot=_yaw_to_quat(yaw)),
        )
        setattr(scene, f"{name_prefix}_{i}", seg)


@configclass
class TaskAEnvB2CurvedStripCfg(TaskAEnvB2Cfg):
    """Curved red strips (sine centerline); terrain and x>=145 goal unchanged.
    Tests whether the head TRACKS a non-straight red boundary."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.red_strip_left = None
        self.scene.red_strip_right = None
        _curved_strip_segments(self.scene, +1.0, _COURSE_HALF_W, "curved_strip_left")
        _curved_strip_segments(self.scene, -1.0, _COURSE_HALF_W, "curved_strip_right")


@configclass
class TaskAEnvB2BentGoalCfg(TaskAEnvB2CurvedStripCfg):
    """Curved strips AND a bent finish: the goal is a POINT at the end of the curve
    (off the +x axis), so the robot must actually follow the curve to reach it."""
    def __post_init__(self):
        super().__post_init__()
        goal_x = 140.0
        goal_y = _centerline_y(goal_x)
        self.terminations.reach_goal_x = DoneTerm(
            func=atec_mdp.robot_reached_point,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "goal_x": goal_x,
                "goal_y": goal_y,
                "radius": 3.0,
            },
            time_out=False,
        )


# ===========================================================================
# NARROW RIGHT-ANGLE MAZE VARIANT (video LCCkxRamgWE: narrow corridors + 90-deg turns)
#
# A narrow walled corridor that turns at right angles. Walls are STATIC colliders
# (hitting them = failure via the robot toppling / getting stuck); a red boundary
# strip is painted just inside each wall as the visual steering cue. The corridor
# centerline is a polyline of waypoints; between consecutive waypoints we lay wall +
# red segments on both sides, offset by +/-corridor half-width along the local normal.
#
# The path is authored over the FLAT spawn region (x from -141) so the narrow turning
# is tested without the terrain-push confound; the finish is the last waypoint.
# ===========================================================================

_MAZE_HALF_W = 1.5      # corridor half-width (m): narrow (course is 3 m wide)
_MAZE_WALL_H = 0.8      # wall height (m)
_MAZE_WALL_T = 0.15     # wall thickness (m)

# Right-angle corridor centerline (world x,y). Starts at the spawn (-141,0), goes +x,
# turns +y, +x, -y, +x ... a zig-zag of 90-deg corners. NOTE the terrain only spans
# y in [-10,+10] (num_cols=1, size 20 m), so the y excursion is kept within +/-7 so the
# corridor (and its +/-1.5 walls) never leave the ground -- earlier +/-12 waypoints put
# the second leg over empty space (no floor).
_MAZE_WAYPOINTS = [
    (-141.0,  0.0),
    (-120.0,  0.0),   # straight
    (-120.0,  7.0),   # turn left (+y)
    (-100.0,  7.0),   # turn right (+x)
    (-100.0, -7.0),   # turn right (-y), cross the middle
    (-80.0,  -7.0),   # turn left (+x)  -> finish
]

# FULL course: the same narrow zig-zag corridor extended all the way to the Task A finish
# (x ~ +145). It rides OVER the real terrain sequence (rough @ -100..-40, slopes @ -20..40,
# stairs @ 60..120, flat goal @ 140) so the robot must do rough/slope/stairs locomotion AND
# 90-deg corridor turns at the same time -- the closest thing to the real Task A. y stays in
# +/-7 (terrain spans +/-10) so walls never leave the ground. Corners land near terrain-tile
# boundaries (multiples of ~20 m) so a turn doesn't sit mid-slope where a wall would bury.
_MAZE_FULL_WAYPOINTS = [
    (-141.0,  0.0),
    (-120.0,  0.0),   # straight over the flat spawn
    (-120.0,  7.0),   # +y
    (-100.0,  7.0),   # +x  (entering random_rough)
    (-100.0, -7.0),   # -y  cross
    (-60.0,  -7.0),   # +x  through rough
    (-60.0,   7.0),   # +y
    (-20.0,   7.0),   # +x  (entering slopes)
    (-20.0,  -7.0),   # -y
    (40.0,   -7.0),   # +x  through slopes
    (40.0,    7.0),   # +y
    (80.0,    7.0),   # +x  (through stairs)
    (80.0,   -7.0),   # -y
    (120.0,  -7.0),   # +x  through stairs
    (120.0,   0.0),   # +y  back to center
    (145.0,   0.0),   # +x  -> Task A finish
]


def _seg_normal(p0, p1):
    """unit left-normal (nx,ny) and yaw of segment p0->p1."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    yaw = math.atan2(dy, dx)
    return (-math.sin(yaw), math.cos(yaw)), yaw


def _offset_vertices(wps, side_sign, half_w):
    """Offset the centerline polyline by side_sign*half_w to one side, computing each
    interior vertex as the INTERSECTION of the two adjacent offset lines so the wall meets
    cleanly at right-angle corners (no cross-over into the path, no gap). Endpoints are
    offset along their single segment normal."""
    verts = []
    n = len(wps)
    for i in range(n):
        if i == 0:
            (nx, ny), _ = _seg_normal(wps[0], wps[1])
            verts.append((wps[0][0] + side_sign * half_w * nx,
                          wps[0][1] + side_sign * half_w * ny))
        elif i == n - 1:
            (nx, ny), _ = _seg_normal(wps[-2], wps[-1])
            verts.append((wps[-1][0] + side_sign * half_w * nx,
                          wps[-1][1] + side_sign * half_w * ny))
        else:
            # offset lines of the two adjacent segments; intersect them
            (n0x, n0y), _ = _seg_normal(wps[i - 1], wps[i])
            (n1x, n1y), _ = _seg_normal(wps[i], wps[i + 1])
            # point on each offset line = wps[i] shifted by that segment's offset
            a0 = (wps[i][0] + side_sign * half_w * n0x, wps[i][1] + side_sign * half_w * n0y)
            d0 = (wps[i][0] - wps[i - 1][0], wps[i][1] - wps[i - 1][1])   # dir seg0
            a1 = (wps[i][0] + side_sign * half_w * n1x, wps[i][1] + side_sign * half_w * n1y)
            d1 = (wps[i + 1][0] - wps[i][0], wps[i + 1][1] - wps[i][1])   # dir seg1
            # solve a0 + t*d0 = a1 + s*d1  -> intersection
            den = d0[0] * (-d1[1]) - d0[1] * (-d1[0])
            if abs(den) < 1e-9:
                verts.append(a0)
            else:
                rhsx, rhsy = a1[0] - a0[0], a1[1] - a0[1]
                t = (rhsx * (-d1[1]) - rhsy * (-d1[0])) / den
                verts.append((a0[0] + t * d0[0], a0[1] + t * d0[1]))
    return verts


# --- terrain height grid (for hugging walls to slopes/stairs) --------------------------
_TERRAIN_GRID = None
_TERRAIN_GRID_PATH = "/hdd0/sohyun/atec_2026_tesk_e/taska_terrain_grid.json"


def _set_terrain_grid_path(path):
    """Select which precomputed height grid the wall-hug uses (pyramid vs corridor-symmetric)
    and clear the cache so the next _terrain_height reloads from `path`."""
    global _TERRAIN_GRID, _TERRAIN_GRID_PATH
    _TERRAIN_GRID_PATH = path
    _TERRAIN_GRID = None


def _load_terrain_grid():
    """Lazy-load the precomputed (x,y)->height grid (from probe_terrain_grid.py). Returns
    None if the file is missing (callers then fall back to flat z=0)."""
    global _TERRAIN_GRID
    if _TERRAIN_GRID is None:
        try:
            import json
            with open(_TERRAIN_GRID_PATH) as f:
                _TERRAIN_GRID = json.load(f)
        except Exception:
            _TERRAIN_GRID = False   # sentinel: tried and failed
    return _TERRAIN_GRID or None


def _terrain_height(cx, cy):
    """Bilinear-interpolate the terrain height under (cx,cy) from the grid; 0.0 if no grid."""
    g = _load_terrain_grid()
    if g is None:
        return 0.0
    fx = (cx - g["x0"]) / g["dx"]
    fy = (cy - g["y0"]) / g["dy"]
    i0 = max(0, min(g["nx"] - 1, int(math.floor(fx))))
    j0 = max(0, min(g["ny"] - 1, int(math.floor(fy))))
    i1 = min(g["nx"] - 1, i0 + 1)
    j1 = min(g["ny"] - 1, j0 + 1)
    tx = max(0.0, min(1.0, fx - i0))
    ty = max(0.0, min(1.0, fy - j0))
    grid = g["grid"]
    h00, h10 = grid[i0][j0], grid[i1][j0]
    h01, h11 = grid[i0][j1], grid[i1][j1]
    return (h00 * (1 - tx) * (1 - ty) + h10 * tx * (1 - ty)
            + h01 * (1 - tx) * ty + h11 * tx * ty)


def _lay_polyline(scene, name, verts, height, thick, is_wall, seg_len=2.0, semantic_label=None,
                  hug_terrain=False):
    """Lay cuboids along the polyline `verts`. is_wall -> tall collider; else flat red.
    semantic_label (str) tags every segment with ("class", label) for the semantic camera.
    hug_terrain -> each segment's z is raised to the terrain height under it (so walls sit ON
    slopes/stairs instead of burying on up-slopes / floating on down-slopes). The wall is also
    made extra tall when hugging so a small height mismatch never opens a gap under it."""
    tags = [("class", semantic_label)] if semantic_label else None
    idx = 0
    for k in range(len(verts) - 1):
        x0, y0 = verts[k]
        x1, y1 = verts[k + 1]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        yaw = math.atan2(dy, dx)
        n = max(1, int(round(length / seg_len)))
        for i in range(n):
            t = (i + 0.5) / n
            cx, cy = x0 + t * dx, y0 + t * dy
            h_terr = _terrain_height(cx, cy) if hug_terrain else 0.0
            if is_wall:
                # when hugging, sink the base ~0.5 m below the terrain and make the wall taller
                # so up/down steps within a segment never leave a gap under the wall.
                wall_h = height + (1.0 if hug_terrain else 0.0)
                size = (length / n * 1.02, thick, wall_h)
                z = h_terr + wall_h / 2.0 - (0.5 if hug_terrain else 0.0)
                mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.62), roughness=0.9)
                coll = sim_utils.CollisionPropertiesCfg()
            else:
                size = (length / n * 1.05, _STRIP_W, 0.02)
                z = h_terr + _STRIP_Z
                mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.05, 0.05), roughness=0.8)
                coll = None
            seg = AssetBaseCfg(
                prim_path=f"/World/{name}_{idx}",
                spawn=sim_utils.CuboidCfg(size=size, visual_material=mat, collision_props=coll,
                                          semantic_tags=tags),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(cx, cy, z), rot=_yaw_to_quat(yaw)),
            )
            setattr(scene, f"{name}_{idx}", seg)
            idx += 1
    return idx   # number of segments laid (so callers can build the lidar mesh path list)


def _build_maze(scene, wall_label=None, red_label=None, waypoints=None, hug_terrain=False,
                half_w=None, no_walls=False):
    """Build the narrow right-angle corridor (walls + red strips) on `scene`. Optional
    semantic labels tag walls/red strips for the semantic camera (ViPlanner variant).
    `waypoints` selects the corridor centerline (defaults to the short flat-region maze).
    hug_terrain -> walls/red follow the terrain height (needed for the full course that rides
    over slopes/stairs); a shorter seg_len is used so the walls step-follow the surface.
    half_w overrides the corridor half-width (wider corridor = room for rough-terrain turns).
    no_walls -> lay ONLY the red boundary strips, no collider walls (real course has no walls;
    the robot must follow the red line alone). No wall segments are created, so the LiDAR has
    nothing to raycast -- steering must come from the red-line semantic, not wall geometry."""
    scene.red_strip_left = None
    scene.red_strip_right = None
    wps = waypoints if waypoints is not None else _MAZE_WAYPOINTS
    hw = half_w if half_w is not None else _MAZE_HALF_W
    seg = 1.0 if hug_terrain else 2.0
    left_red = _offset_vertices(wps, +1.0, hw - 0.1)
    right_red = _offset_vertices(wps, -1.0, hw - 0.1)
    if no_walls:
        n_L = n_R = 0
    else:
        left_wall = _offset_vertices(wps, +1.0, hw)
        right_wall = _offset_vertices(wps, -1.0, hw)
        n_L = _lay_polyline(scene, "maze_wall_L", left_wall, _MAZE_WALL_H, _MAZE_WALL_T, True,
                            seg_len=seg, semantic_label=wall_label, hug_terrain=hug_terrain)
        n_R = _lay_polyline(scene, "maze_wall_R", right_wall, _MAZE_WALL_H, _MAZE_WALL_T, True,
                            seg_len=seg, semantic_label=wall_label, hug_terrain=hug_terrain)
    _lay_polyline(scene, "maze_red_L", left_red, 0.0, 0.0, False,
                  seg_len=seg, semantic_label=red_label, hug_terrain=hug_terrain)
    _lay_polyline(scene, "maze_red_R", right_red, 0.0, 0.0, False,
                  seg_len=seg, semantic_label=red_label, hug_terrain=hug_terrain)
    # explicit prim paths of every wall segment, so the LiDAR RayCaster can be pointed at the
    # walls (not just /World/ground) -> the LiDAR then actually measures the corridor walls,
    # which is what makes the front/side distances mean "how close is the end wall / is the
    # side open". Real LiDAR sees walls; this matches sim to that.
    wall_prims = [f"/World/maze_wall_L_{i}" for i in range(n_L)] \
               + [f"/World/maze_wall_R_{i}" for i in range(n_R)]
    return wall_prims


def _maze_goal_term(waypoints=None):
    wps = waypoints if waypoints is not None else _MAZE_WAYPOINTS
    gx, gy = wps[-1]
    return DoneTerm(
        func=atec_mdp.robot_reached_point,
        params={"asset_cfg": SceneEntityCfg("robot"), "goal_x": gx, "goal_y": gy, "radius": 2.0},
        time_out=False,
    )


@configclass
class TaskAEnvB2MazeCfg(TaskAEnvB2Cfg):
    """Narrow walled corridor with right-angle turns over the flat region. Walls collide;
    red strips inside them are the visual cue. Goal = last waypoint (x,y)-proximity."""
    def __post_init__(self):
        super().__post_init__()
        _build_maze(self.scene)
        self.terminations.reach_goal_x = _maze_goal_term()


# ViPlanner-convention semantic colors (must match viplanner_sem_meta.py so the segmentation
# image the camera renders matches what the pretrained planner was trained on).
_VIP_FLOOR_RGBA = (0, 204, 0, 255)       # traversable_intended (loss 0)
_VIP_OBSTACLE_RGBA = (255, 0, 0, 255)    # obstacle (loss 2.0) -- walls AND the red boundary


def _setup_maze_viplanner(self, waypoints=None, hug_terrain=False, half_w=None, no_walls=False):
    """Shared setup for the ViPlanner maze variants: build the (labeled) corridor along
    `waypoints`, set its goal, point the LiDAR at the walls, add the semantic camera stream.
    no_walls -> red boundary only (real-course style); the LiDAR then has no walls to raycast."""
    _build_maze(self.scene, wall_label="obstacle", red_label="obstacle",
                waypoints=waypoints, hug_terrain=hug_terrain, half_w=half_w, no_walls=no_walls)
    self.terminations.reach_goal_x = _maze_goal_term(waypoints=waypoints)
    # Make the LiDAR see the maze WALLS, not just /World/ground. The default RayCaster
    # supports only ONE mesh prim, so with the ground-only target the LiDAR never measured
    # the corridor walls (front/side distances were just the terrain under the robot). Swap
    # it for a MultiMeshRayCaster and target the ground + every wall segment via one regex.
    # Now the LiDAR corner-turn fires on real wall geometry -- matching a real 360-deg LiDAR.
    # only add the wall raycast target when walls actually exist -- otherwise the regex
    # matches no prim and IsaacLab raises at startup (no_walls red-line course).
    base = self.scene.lidar_sensor
    if base is not None:
        mesh_targets = ["/World/ground"]
        if not no_walls:
            mesh_targets.append(MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/maze_wall_.*", track_mesh_transforms=False))
        self.scene.lidar_sensor = MultiMeshRayCasterCfg(
            prim_path=base.prim_path,
            update_period=base.update_period,
            pattern_cfg=base.pattern_cfg,
            max_distance=base.max_distance,
            debug_vis=base.debug_vis,
            mesh_prim_paths=mesh_targets,
        )
    # add a semantic segmentation stream to the head camera, colored to ViPlanner's convention
    self.scene.head_camera.data_types = ["rgb", "depth", "semantic_segmentation"]
    self.scene.head_camera.colorize_semantic_segmentation = True
    self.scene.head_camera.semantic_segmentation_mapping = {
        "class:obstacle": _VIP_OBSTACLE_RGBA,
        "class:floor": _VIP_FLOOR_RGBA,
    }


@configclass
class TaskAEnvB2MazeViPlannerCfg(TaskAEnvB2MazeCfg):
    """Maze variant that also renders a ViPlanner-convention SEMANTIC image from the head
    camera: walls + red boundary -> obstacle color, everything else stays background. Used
    to drive the pretrained ViPlanner (depth + semantic -> waypoints)."""
    def __post_init__(self):
        # rebuild the maze WITH semantic labels on walls + red strips
        super(TaskAEnvB2MazeCfg, self).__post_init__()
        _setup_maze_viplanner(self, waypoints=_MAZE_WAYPOINTS)


# ── RED-LINE-ONLY corner course (real-course study, 2026-08-03) ──────────────────────────
# Flat ground, NO walls -- just the two red boundary strips. The robot must follow the red
# line alone (real course has no walls). Corridor is NARROW (half_w 1.5 = 3 m, same as the
# walled maze) per user. Multiple 90-deg bends kept entirely within the FLAT region (x from
# the spawn -141 to ~-100, before rough starts); y within the +/-10 terrain band.
_REDLINE_CORNERS_WAYPOINTS = [
    (-141.0,  0.0),
    (-128.0,  0.0),   # straight along +x over the flat spawn
    (-128.0,  7.0),   # 90-deg LEFT  (+y)
    (-114.0,  7.0),   # 90-deg RIGHT (+x)
    (-114.0, -7.0),   # 90-deg RIGHT (-y), cross the middle
    (-102.0, -7.0),   # 90-deg LEFT  (+x) -> finish (still on flat, before rough at -100)
]


@configclass
class TaskAEnvB2RedLineCornerViPlannerCfg(TaskAEnvB2MazeCfg):
    """Flat, wall-less multi-90-deg red-line course. The red boundary is the ONLY cue
    (semantic obstacle); there are no collider walls and nothing for the LiDAR to raycast,
    so steering must come from the red-line semantic alone -- the real-course condition.
    Narrow (half_w 1.5) with 4 right-angle bends, all on the flat spawn region."""
    def __post_init__(self):
        super(TaskAEnvB2MazeCfg, self).__post_init__()
        _setup_maze_viplanner(self, waypoints=_REDLINE_CORNERS_WAYPOINTS,
                              half_w=1.5, no_walls=True)


@configclass
class TaskAEnvB2MazeFullViPlannerCfg(TaskAEnvB2MazeCfg):
    """FULL-course ViPlanner maze: the same narrow zig-zag corridor extended over the whole
    Task A terrain (rough/slope/stairs) all the way to the finish (x ~ +145). Tests rough
    locomotion AND 90-deg corridor turns together -- the closest thing to real Task A."""
    def __post_init__(self):
        super(TaskAEnvB2MazeCfg, self).__post_init__()
        _setup_maze_viplanner(self, waypoints=_MAZE_FULL_WAYPOINTS, hug_terrain=True)


# WIDE corridor over the rough section only (through x=-40): 6 m wide (half_w 3.0) so the
# robot has room to cross rough terrain AND take the 90-deg turns. Tests whether a wider
# corridor fixes the "narrow + rough + sharp turn -> topple" failure of the full course.
_MAZE_ROUGH_WAYPOINTS = [
    (-141.0,  0.0),
    (-120.0,  0.0),   # straight over flat spawn
    (-120.0,  5.0),   # +y
    (-100.0,  5.0),   # +x  (entering random_rough)
    (-100.0, -5.0),   # -y  cross
    (-60.0,  -5.0),   # +x  through rough
    (-60.0,   5.0),   # +y
    (-40.0,   5.0),   # +x  end of rough -> finish
]


@configclass
class TaskAEnvB2MazeRoughWideViPlannerCfg(TaskAEnvB2MazeCfg):
    """WIDE corridor (6 m) through the rough section only. Isolates whether widening the
    corridor lets the robot clear rough terrain + 90-deg turns (the full narrow course
    toppled at the first rough corner)."""
    def __post_init__(self):
        super(TaskAEnvB2MazeCfg, self).__post_init__()
        _setup_maze_viplanner(self, waypoints=_MAZE_ROUGH_WAYPOINTS, hug_terrain=True, half_w=3.0)


@configclass
class TaskAEnvB2MazeSymFullViPlannerCfg(TaskAEnvB2MazeCfg):
    """FULL narrow-corridor course over CORRIDOR-SYMMETRIC terrain: slopes/stairs vary only
    along the corridor (constant across its width) so the robot climbs them head-on. This is
    the design that lets a NARROW corridor survive slopes/stairs -- practising the tight turns
    (per user: real corridor may be narrow) while making the terrain corridor-traversable."""
    def __post_init__(self):
        # swap in the corridor-symmetric terrain BEFORE the base builds the scene
        super(TaskAEnvB2MazeCfg, self).__post_init__()
        self.scene.terrain = TASK_A_CORRIDOR_TERRAIN_CFG
        self.sim.physics_material = self.scene.terrain.physics_material
        # hug the walls to the SYMMETRIC terrain's height grid (not the pyramid one)
        _set_terrain_grid_path("/hdd0/sohyun/atec_2026_tesk_e/taska_terrain_grid_sym.json")
        _setup_maze_viplanner(self, waypoints=_MAZE_FULL_WAYPOINTS, hug_terrain=True)
