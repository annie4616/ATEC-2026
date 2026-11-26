"""Task E demo-collection constants.
"""
#/home/yewon/ATEC2026_Simulation_Challenge/scripts/act/task_e/config_my_mustard.py

from atec_rl_lab.tasks.task_e.env_cfg import (
    BASKET_CENTER_X, BASKET_CENTER_Y,
    TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_TOP_Z, TABLE_HALF_X,
    BASKET_EXCL_HALF_X, BASKET_EXCL_HALF_Y,
)

__all__ = [
    "BASKET_CENTER_X", "BASKET_CENTER_Y",
    "TABLE_CENTER_X", "TABLE_CENTER_Y", "TABLE_TOP_Z", "TABLE_HALF_X",
    "BASKET_EXCL_HALF_X", "BASKET_EXCL_HALF_Y",
    # robot
    "EE_BODY_NAME", "ARM_JOINT_NAMES", "GRIPPER_JOINT_NAMES",
    "GRIPPER_OPEN_POS", "GRIPPER_CLOSE_POS", "ACTION_SCALE",
    # state machine
    "STEPS", "STATE_ORDER",
    # geometry
    "PRE_GRASP_CLEARANCE", "GRASP_Z_OFFSET_PER_OBJ",
    "CARRY_Z", "PLACE_HEIGHT",
    "RETRACT_POS_X", "RETRACT_POS_Y",
    "DEFAULT_PLACE_QUAT_W",
    # success check
    "BASKET_IN_X", "BASKET_IN_Y",
    # spawn regions
    "OBJ_SPAWN_X_MIN", "OBJ_SPAWN_X_MAX", "OBJ_SPAWN_Z", "OBJ_SPAWN_Y_BANDS",
    "OBJ_HALF_EXTENTS", "OBJ_BBOX_MARGIN",
    # camera
    "CAM_POS", "CAM_ROT", "CAM_H", "CAM_W",
    # actuator
    "ACT_STIFFNESS", "ACT_DAMPING", "ACT_EFFORT_LIMIT", "ACT_VEL_LIMIT",
    # warm-up
    "WARMUP_STEPS", "SETTLE_STEPS",
]

# ------------------------------------------------------------------ #
# Robot
# ------------------------------------------------------------------ #
EE_BODY_NAME        = "gripper_base"
ARM_JOINT_NAMES     = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT_NAMES = ["joint7", "joint8"]
GRIPPER_OPEN_POS    = [0.035, -0.035]   # joint7, joint8
GRIPPER_CLOSE_POS   = [-0.008,  0.008]

# Must match ActionsCfg: scale=0.5, use_default_offset=True
#   env_action = (joint_target - default_joint_pos) / ACTION_SCALE
ACTION_SCALE = 0.5

# ------------------------------------------------------------------ #
# State-machine
# ------------------------------------------------------------------ #
STEPS: dict[str, int] = {
    "INIT":         4,
    "PRE_GRASP":    14,
    "REACH":        19,
    "CLOSE":        15,
    "LIFT":         13,
    "TRANSPORT":    21,
    "PLACE":         8,
    "OPEN":          7,
    "LIFT_RETRACT":  7,
    "RETRACT":       5,
    "KNOCK":         10,
    "WAIT":          20,
}
STATE_ORDER = ["INIT", "PRE_GRASP", "REACH", "CLOSE", "LIFT",
               "TRANSPORT", "PLACE", "OPEN", "LIFT_RETRACT", "RETRACT"]

STATE_ORDER_MUSTARD = [
    "INIT", "PRE_GRASP", "KNOCK", "WAIT",
    "REACH", "CLOSE", "LIFT",
    "TRANSPORT", "PLACE", "OPEN", "LIFT_RETRACT", "RETRACT"
]

# ------------------------------------------------------------------ #
# Geometry
# ------------------------------------------------------------------ #
PRE_GRASP_CLEARANCE = 0.10   # metres above object before descent

#GRASP_Z_OFFSET      = 0.09   # metres: gripper approach height above object centre
GRASP_Z_OFFSET_PER_OBJ = {
    1: 0.11,   # Sugar box: 조금 더 높게
    2: 0.11,   # Mustard bottle
    3: 0.09,   # Banana: 납작하니까 낮게
}

MUSTARD_UPRIGHT_Z_OFFSET = 0.11   # 세워진 병 위에서 잡기
MUSTARD_UPRIGHT_X_OFFSET = 0.0399 
MUSTARD_UPRIGHT_Y_OFFSET = -0.014 
MUSTARD_FALLEN_Z_OFFSET  = 0.04
MUSTARD_KNOCK_Z_OFFSET = 0.10   # 병 위쪽 살짝 건드리기
MUSTARD_KNOCK_Y_OFFSET = -0.03  # 병 뒤쪽에서 밀기

PLACE_X_OFFSET = 0.05   
PLACE_Y_OFFSET = 0.05  


CARRY_Z      = TABLE_TOP_Z + 0.40   # safe carry height
PLACE_HEIGHT = TABLE_TOP_Z + 0.25   # height at which to release into basket

RETRACT_POS_X = TABLE_CENTER_X + TABLE_HALF_X - 0.05
RETRACT_POS_Y = TABLE_CENTER_Y
DEFAULT_PLACE_QUAT_W = [0.0, 1.0, 0.0, 0.0]   # top-down orientation (w,x,y,z)

# ------------------------------------------------------------------ #
# Object spawn regions
#
#   object_1  Y ∈ [0.21, 0.28]   (top band)
#   object_2  Y ∈ [0.12, 0.19]   (middle band)
#   object_3  Y ∈ [0.03, 0.10]   (bottom band, closest to basket)
# ------------------------------------------------------------------ #
OBJ_SPAWN_X_MIN = TABLE_CENTER_X - 0.10
OBJ_SPAWN_X_MAX = TABLE_CENTER_X + 0.10
OBJ_SPAWN_Z     = TABLE_TOP_Z + 0.03

# Per-object Y-bands: {object_idx: (y_min, y_max)}
OBJ_SPAWN_Y_BANDS = {
    1: (TABLE_CENTER_Y + 0.25, TABLE_CENTER_Y + 0.29),  # env_cfg 기준
    2: (TABLE_CENTER_Y + 0.14, TABLE_CENTER_Y + 0.20),  # env_cfg 기준
    3: (TABLE_CENTER_Y + 0.03, TABLE_CENTER_Y + 0.10),
}

# Per-object 2-D bounding-box half-extents (metres, world XY plane, scale=1).
# Used for AABB overlap rejection during randomisation.
OBJ_HALF_EXTENTS: dict[int, tuple[float, float]] = {
    1: (0.050, 0.044),   # Sugar box   (half_x, half_y)
    2: (0.050, 0.030),   # Mustard bottle
    3: (0.100, 0.040),   # Banana
}
OBJ_BBOX_MARGIN = 0.015  # extra clearance between object bounding boxes

# ------------------------------------------------------------------ #
# Success / basket bounds
# ------------------------------------------------------------------ #
BASKET_IN_X = 0.20   # ± metres in X around BASKET_CENTER_X
BASKET_IN_Y = 0.11   # ± metres in Y around BASKET_CENTER_Y

# ------------------------------------------------------------------ #
# Camera (for --save_video / --save_images)
# ------------------------------------------------------------------ #
CAM_H, CAM_W = 480, 640
CAM_POS = (TABLE_CENTER_X - 1.2, TABLE_CENTER_Y, TABLE_TOP_Z + 0.8)
CAM_ROT = (0.957, 0.0, 0.290, 0.0)   # ~34° around +Y → looks forward-down

# ------------------------------------------------------------------ #
# Actuator overrides for data collection (high stiffness for fast tracking)
# ------------------------------------------------------------------ #
ACT_STIFFNESS    = 800.0
ACT_DAMPING      = 80.0
ACT_EFFORT_LIMIT = 100.0
ACT_VEL_LIMIT    = 100.0

# ------------------------------------------------------------------ #
# Warm-up / settle steps before recording starts
# ------------------------------------------------------------------ #
WARMUP_STEPS = 150
SETTLE_STEPS = 30
