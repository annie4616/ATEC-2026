"""Unified Task E demo-collection constants and per-object grasp config.

Place this file at:
    ATEC2026_Simulation_Challenge/scripts/act/task_e/config_all.py

Object ids:
    1: sugar
    2: mustard
    3: banana

The important design point is that gripper close values and xyz offsets are NOT
single globals anymore.  They are looked up by object id at runtime, so both
orders below work without changing code:
    --pick_objects 2 3 1
    --pick_objects 3 2 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from atec_rl_lab.tasks.task_e.env_cfg import (
    BASKET_CENTER_X,
    BASKET_CENTER_Y,
    TABLE_CENTER_X,
    TABLE_CENTER_Y,
    TABLE_TOP_Z,
    TABLE_HALF_X,
    BASKET_EXCL_HALF_X,
    BASKET_EXCL_HALF_Y,
)

__all__ = [
    # env geometry
    "BASKET_CENTER_X", "BASKET_CENTER_Y",
    "TABLE_CENTER_X", "TABLE_CENTER_Y", "TABLE_TOP_Z", "TABLE_HALF_X",
    "BASKET_EXCL_HALF_X", "BASKET_EXCL_HALF_Y",
    # robot
    "EE_BODY_NAME", "ARM_JOINT_NAMES", "GRIPPER_JOINT_NAMES",
    "GRIPPER_OPEN_POS", "GRIPPER_CLOSE_POS", "GRIPPER_CLOSE_POS_PER_OBJ",
    "ACTION_SCALE",
    # state machine
    "BANANA_STEPS", "MUSTARD_STEPS", "SUGAR_STEPS",
    "STEPS", "STATE_ORDER", "STATE_ORDER_MUSTARD",
    # object-specific config
    "ObjectPickConfig", "OBJECT_CONFIGS",
    "get_object_config", "get_step_count", "get_state_order",
    "get_gripper_close_pos", "get_gripper_pos",
    "get_pose_offset", "get_pre_grasp_xyz_offset", "get_reach_xyz_offset",
    "get_close_xyz_offset", "get_knock_xyz_offset",
    "get_pre_grasp_clearance", "get_grasp_z_offset",
    "validate_pick_sequence",
    "PICK_ORDER_MUSTARD_BANANA_SUGAR", "PICK_ORDER_BANANA_MUSTARD_SUGAR",
    # geometry
    "PRE_GRASP_CLEARANCE", "GRASP_Z_OFFSET_PER_OBJ",
    "CARRY_Z", "PLACE_HEIGHT",
    "RETRACT_POS_X", "RETRACT_POS_Y",
    "DEFAULT_PLACE_QUAT_W",
    "PLACE_X_OFFSET", "PLACE_Y_OFFSET",
    # mustard-specific constants kept for debugging/backward compatibility
    "MUSTARD_UPRIGHT_Z_OFFSET", "MUSTARD_UPRIGHT_X_OFFSET", "MUSTARD_UPRIGHT_Y_OFFSET",
    "MUSTARD_FALLEN_Z_OFFSET", "MUSTARD_KNOCK_Z_OFFSET", "MUSTARD_KNOCK_Y_OFFSET",
    # sugar-specific (provenance only)
    "SUGAR_IK_OFFSET_X", "SUGAR_IK_OFFSET_Y",
    # banana IK offset
    "BANANA_IK_OFFSET_X", "BANANA_IK_OFFSET_Y",
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
EE_BODY_NAME = "gripper_base"
ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT_NAMES = ["joint7", "joint8"]

GRIPPER_OPEN_POS = [0.035, -0.035]

# Per-object gripper close values.
#   1 sugar  : from config_sugar.py
#   2 mustard: from config_my_mustard.py
#   3 banana : from config_my_banana.py
GRIPPER_CLOSE_POS_PER_OBJ: dict[int, list[float]] = {
    1: [-0.015, 0.015], # 바나나
    2: [-0.008, 0.008], # 슈가박스
    3: [-0.005, 0.005], # 머스타드
}

# Backward-compatible fallback. New code must call get_gripper_close_pos(obj_idx).
GRIPPER_CLOSE_POS = GRIPPER_CLOSE_POS_PER_OBJ[3]

# Must match ActionsCfg: scale=0.5, use_default_offset=True
# env_action = (joint_target - default_joint_pos) / ACTION_SCALE
ACTION_SCALE = 0.5

# ------------------------------------------------------------------ #
# State-machine timing
# ------------------------------------------------------------------ #
BANANA_STEPS: dict[str, int] = {
    "INIT": 4,
    "PRE_GRASP": 8,
    "REACH": 20,
    "CLOSE": 20,
    "LIFT": 7,
    "TRANSPORT": 15,
    "PLACE": 4,
    "OPEN": 4,
    "LIFT_RETRACT": 4,
    "RETRACT": 4,
}

MUSTARD_STEPS: dict[str, int] = {
    "INIT": 4,
    "PRE_GRASP": 30,
    "REACH": 19,
    "CLOSE": 15,
    "LIFT": 13,
    "TRANSPORT": 24,
    "PLACE": 8,
    "OPEN": 7,
    "LIFT_RETRACT": 7,
    "RETRACT": 5,
    "KNOCK": 10,
    "WAIT": 20,
}

# Sugar timing from config_sugar.py.
SUGAR_STEPS: dict[str, int] = {
    "INIT": 4,
    "PRE_GRASP": 23,
    "REACH": 32,
    "CLOSE": 20,
    "LIFT": 15,
    "TRANSPORT": 28,
    "PLACE": 5,
    "OPEN": 5,
    "LIFT_RETRACT": 5,
    "RETRACT": 5,
}

STATE_ORDER = [
    "INIT", "PRE_GRASP", "REACH", "CLOSE", "LIFT",
    "TRANSPORT", "PLACE", "OPEN", "LIFT_RETRACT", "RETRACT",
]

STATE_ORDER_MUSTARD = [
    "INIT", "PRE_GRASP", "KNOCK", "WAIT",
    "REACH", "CLOSE", "LIFT",
    "TRANSPORT", "PLACE", "OPEN", "LIFT_RETRACT", "RETRACT",
]

# Backward-compatible default only. New state_machine.py uses get_step_count().
STEPS = BANANA_STEPS

# ------------------------------------------------------------------ #
# Geometry
# ------------------------------------------------------------------ #
CARRY_Z = TABLE_TOP_Z + 0.40
PLACE_HEIGHT = TABLE_TOP_Z + 0.25

RETRACT_POS_X = TABLE_CENTER_X + TABLE_HALF_X - 0.05
RETRACT_POS_Y = TABLE_CENTER_Y
DEFAULT_PLACE_QUAT_W = [0.0, 1.0, 0.0, 0.0]

PLACE_X_OFFSET = 0.05
PLACE_Y_OFFSET = 0.05

# Sugar-specific IK tracking-error compensation from config_sugar.py.
# Consumed via OBJECT_CONFIGS[1].{pre_grasp,reach,close}_xyz_offset below, which
# state_machine.py applies through get_pose_offset(1, ...) for sugar's PRE_GRASP/REACH/CLOSE.
SUGAR_IK_OFFSET_X = 0.06
SUGAR_IK_OFFSET_Y = 0.0

# Banana IK tracking-error compensation (X/Y), folded into
# OBJECT_CONFIGS[3].{reach,close}_xyz_offset below. Default 0.0 — tune here if
# the gripper lands off-centre on the banana. (Mustard uses MUSTARD_*_OFFSET.)
BANANA_IK_OFFSET_X = 0.0
BANANA_IK_OFFSET_Y = 0.02

# Mustard-specific xyz offsets from config_my_mustard.py.
MUSTARD_UPRIGHT_Z_OFFSET = 0.11
MUSTARD_UPRIGHT_X_OFFSET = 0.065
# MUSTARD_UPRIGHT_Y_OFFSET = -0.014
MUSTARD_UPRIGHT_Y_OFFSET = -0.014
MUSTARD_FALLEN_Z_OFFSET = 0.04
MUSTARD_KNOCK_Z_OFFSET = 0.10
MUSTARD_KNOCK_Y_OFFSET = -0.03


@dataclass(frozen=True)
class ObjectPickConfig:
    name: str
    steps: dict[str, int]
    state_order: tuple[str, ...]
    gripper_close_pos: tuple[float, float]
    pre_grasp_clearance: float
    grasp_z_offset: float
    # Generic offsets used by non-mustard objects.
    pre_grasp_xyz_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reach_xyz_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    close_xyz_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Mustard dynamic offsets. These are used only for object_2.
    mustard_upright_pre_grasp_xyz_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mustard_upright_reach_xyz_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mustard_upright_close_xyz_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mustard_fallen_reach_xyz_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mustard_fallen_close_xyz_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    knock_xyz_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


OBJECT_CONFIGS: dict[int, ObjectPickConfig] = {
    1: ObjectPickConfig(
        name="sugar",
        steps=SUGAR_STEPS,
        state_order=tuple(STATE_ORDER),
        gripper_close_pos=tuple(GRIPPER_CLOSE_POS_PER_OBJ[1]),
        pre_grasp_clearance=0.12,   # config_sugar.PRE_GRASP_CLEARANCE
        grasp_z_offset=0.08,        # config_sugar.GRASP_Z_OFFSET
        # X/Y = config_sugar IK tracking-error compensation; Z = grasp_z_offset.
        pre_grasp_xyz_offset=(SUGAR_IK_OFFSET_X, SUGAR_IK_OFFSET_Y, 0.0),
        reach_xyz_offset=(SUGAR_IK_OFFSET_X, SUGAR_IK_OFFSET_Y, 0.08),
        close_xyz_offset=(SUGAR_IK_OFFSET_X, SUGAR_IK_OFFSET_Y, 0.08),
    ),
    2: ObjectPickConfig(
        name="mustard",
        steps=MUSTARD_STEPS,
        state_order=tuple(STATE_ORDER),
        gripper_close_pos=tuple(GRIPPER_CLOSE_POS_PER_OBJ[2]),
        pre_grasp_clearance=0.10,
        grasp_z_offset=0.11,
        mustard_upright_pre_grasp_xyz_offset=(
            MUSTARD_UPRIGHT_X_OFFSET,
            MUSTARD_UPRIGHT_Y_OFFSET,
            0.0,
        ),
        mustard_upright_reach_xyz_offset=(
            MUSTARD_UPRIGHT_X_OFFSET,
            MUSTARD_UPRIGHT_Y_OFFSET,
            MUSTARD_UPRIGHT_Z_OFFSET,
        ),
        mustard_upright_close_xyz_offset=(
            MUSTARD_UPRIGHT_X_OFFSET,
            MUSTARD_UPRIGHT_Y_OFFSET,
            MUSTARD_UPRIGHT_Z_OFFSET,
        ),
        mustard_fallen_reach_xyz_offset=(0.0, 0.0, MUSTARD_FALLEN_Z_OFFSET),
        mustard_fallen_close_xyz_offset=(0.0, 0.0, MUSTARD_FALLEN_Z_OFFSET),
        knock_xyz_offset=(0.0, MUSTARD_KNOCK_Y_OFFSET, MUSTARD_KNOCK_Z_OFFSET),
    ),
    3: ObjectPickConfig(
        name="banana",
        steps=BANANA_STEPS,
        state_order=tuple(STATE_ORDER),
        gripper_close_pos=tuple(GRIPPER_CLOSE_POS_PER_OBJ[3]),
        pre_grasp_clearance=0.05,
        grasp_z_offset=0.07,
        # X/Y = banana IK tracking-error compensation; Z = grasp_z_offset.
        pre_grasp_xyz_offset=(BANANA_IK_OFFSET_X, BANANA_IK_OFFSET_Y, 0.0),
        reach_xyz_offset=(BANANA_IK_OFFSET_X, BANANA_IK_OFFSET_Y, 0.07),
        close_xyz_offset=(BANANA_IK_OFFSET_X, BANANA_IK_OFFSET_Y, 0.07),
    ),
}

# Backward-compatible values for older code.
PRE_GRASP_CLEARANCE = OBJECT_CONFIGS[3].pre_grasp_clearance
GRASP_Z_OFFSET_PER_OBJ = {
    obj_idx: cfg.grasp_z_offset for obj_idx, cfg in OBJECT_CONFIGS.items()
}

PICK_ORDER_MUSTARD_BANANA_SUGAR = [2, 3, 1]
PICK_ORDER_BANANA_MUSTARD_SUGAR = [3, 2, 1]
SUPPORTED_PICK_ORDERS = (
    tuple(PICK_ORDER_MUSTARD_BANANA_SUGAR),
    tuple(PICK_ORDER_BANANA_MUSTARD_SUGAR),
)


def get_object_config(obj_idx: int) -> ObjectPickConfig:
    obj_idx = int(obj_idx)
    if obj_idx not in OBJECT_CONFIGS:
        raise ValueError(
            f"Unknown Task E object id: {obj_idx}. "
            f"Expected one of {sorted(OBJECT_CONFIGS)}."
        )
    return OBJECT_CONFIGS[obj_idx]


def get_step_count(obj_idx: int, state: str) -> int:
    cfg = get_object_config(obj_idx)
    if state not in cfg.steps:
        raise ValueError(f"State {state!r} is not configured for object {obj_idx} ({cfg.name}).")
    return cfg.steps[state]


def get_state_order(obj_idx: int, mustard_needs_knock: bool = False) -> list[str]:
    obj_idx = int(obj_idx)
    if obj_idx == 2 and mustard_needs_knock:
        return list(STATE_ORDER_MUSTARD)
    return list(get_object_config(obj_idx).state_order)


def get_gripper_close_pos(obj_idx: int) -> list[float]:
    return list(get_object_config(obj_idx).gripper_close_pos)


def get_gripper_pos(obj_idx: int, command: str) -> list[float]:
    if command == "open":
        return list(GRIPPER_OPEN_POS)
    if command == "close":
        return get_gripper_close_pos(obj_idx)
    raise ValueError(f"Unknown gripper command: {command!r}")


def get_pre_grasp_clearance(obj_idx: int) -> float:
    return get_object_config(obj_idx).pre_grasp_clearance


def get_grasp_z_offset(obj_idx: int) -> float:
    return get_object_config(obj_idx).grasp_z_offset


def get_pre_grasp_xyz_offset(obj_idx: int, mustard_is_upright: bool = True) -> tuple[float, float, float]:
    cfg = get_object_config(obj_idx)
    if int(obj_idx) == 2 and mustard_is_upright:
        return cfg.mustard_upright_pre_grasp_xyz_offset
    return cfg.pre_grasp_xyz_offset


def get_reach_xyz_offset(obj_idx: int, mustard_is_upright: bool = True) -> tuple[float, float, float]:
    cfg = get_object_config(obj_idx)
    if int(obj_idx) == 2:
        return cfg.mustard_upright_reach_xyz_offset if mustard_is_upright else cfg.mustard_fallen_reach_xyz_offset
    return cfg.reach_xyz_offset


def get_close_xyz_offset(obj_idx: int, mustard_is_upright: bool = True) -> tuple[float, float, float]:
    cfg = get_object_config(obj_idx)
    if int(obj_idx) == 2:
        return cfg.mustard_upright_close_xyz_offset if mustard_is_upright else cfg.mustard_fallen_close_xyz_offset
    return cfg.close_xyz_offset


def get_knock_xyz_offset(obj_idx: int) -> tuple[float, float, float]:
    return get_object_config(obj_idx).knock_xyz_offset


def get_pose_offset(obj_idx: int, state: str, mustard_is_upright: bool = True) -> tuple[float, float, float]:
    """Return the xyz offset that should be added to object position for state.

    This is the main function used by state_machine.py.  It guarantees that the
    current object id, not the global config file that happened to be imported,
    decides the grasp target xyz.
    """
    if state == "PRE_GRASP":
        return get_pre_grasp_xyz_offset(obj_idx, mustard_is_upright)
    if state == "REACH":
        return get_reach_xyz_offset(obj_idx, mustard_is_upright)
    if state == "CLOSE":
        return get_close_xyz_offset(obj_idx, mustard_is_upright)
    if state == "KNOCK":
        return get_knock_xyz_offset(obj_idx)
    return 0.0, 0.0, 0.0


def validate_pick_sequence(pick_objects: Sequence[int]) -> list[int]:
    """Return duplicate-free pick order while preserving user-provided order.

    Do not use sorted(set(...)) for Task E data collection, because it destroys
    requested sequences such as [2, 3, 1] and [3, 2, 1].
    """
    seq: list[int] = []
    seen: set[int] = set()
    for obj_idx in pick_objects:
        obj_idx = int(obj_idx)
        get_object_config(obj_idx)
        if obj_idx not in seen:
            seq.append(obj_idx)
            seen.add(obj_idx)
    if not seq:
        raise ValueError("pick_objects is empty.")
    return seq

# ------------------------------------------------------------------ #
# Object spawn regions
# ------------------------------------------------------------------ #
OBJ_SPAWN_X_MIN = TABLE_CENTER_X - 0.10
OBJ_SPAWN_X_MAX = TABLE_CENTER_X + 0.10
OBJ_SPAWN_Z = TABLE_TOP_Z + 0.03

OBJ_SPAWN_Y_BANDS = {
    1: (TABLE_CENTER_Y + 0.25, TABLE_CENTER_Y + 0.29),
    2: (TABLE_CENTER_Y + 0.14, TABLE_CENTER_Y + 0.20),
    3: (TABLE_CENTER_Y + 0.03, TABLE_CENTER_Y + 0.10),
}

OBJ_HALF_EXTENTS: dict[int, tuple[float, float]] = {
    1: (0.050, 0.044),
    2: (0.050, 0.030),
    3: (0.100, 0.040),
}
OBJ_BBOX_MARGIN = 0.015

# ------------------------------------------------------------------ #
# Success / basket bounds
# ------------------------------------------------------------------ #
BASKET_IN_X = 0.20
BASKET_IN_Y = 0.11

# ------------------------------------------------------------------ #
# Camera
# ------------------------------------------------------------------ #
CAM_H, CAM_W = 480, 640
CAM_POS = (TABLE_CENTER_X - 1.2, TABLE_CENTER_Y, TABLE_TOP_Z + 0.8)
CAM_ROT = (0.957, 0.0, 0.290, 0.0)

# ------------------------------------------------------------------ #
# Actuator overrides
# ------------------------------------------------------------------ #
ACT_STIFFNESS = 800.0
ACT_DAMPING = 80.0
ACT_EFFORT_LIMIT = 100.0
ACT_VEL_LIMIT = 100.0

# ------------------------------------------------------------------ #
# Warm-up / settle steps
# ------------------------------------------------------------------ #
WARMUP_STEPS = 150
SETTLE_STEPS = 30
