"""Competition-course TEST env: the linear course terrain + 6 trash types x4 scattered
+ 3 tapered bins at the end zone. Built on task_b's env cfg (sensors/terminations
inherited); the 18 YCB objects of task_b are removed and replaced with the
competition set (yellow can, plasticic bottle, watermelon rind, lunchbox, red/purple
paper bags). Robots: B2+PiPER (stock) and A2+PiPER-X (competition robot).
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

_VENUE_LEGACY = os.environ.get("COURSE_VENUE", "hk").lower() == "legacy"

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from atec_rl_lab.tasks.task_b.env_cfg import TaskBEnvCfg
from course_sim.terrain import (
    COURSE_SEQUENCE, COURSE_TERRAIN_CFG, COURSE_X0, SECTION_CENTERS, SECTION_LEN,
    curve_map,
)

# Single-cell S-course terrain: env origin = cell center = world (0, 0, 0.05).
# Init poses are relative to it; course-frame (arc-x, lateral) converts via curve_map.
_ENV_ORIGIN = (0.0, 0.0, 0.05)


def _rel(x, y, z):
    return [x - _ENV_ORIGIN[0], y - _ENV_ORIGIN[1], z - _ENV_ORIGIN[2]]


def _rel_course(arc_x, lateral, z):
    X, Y = curve_map(arc_x, lateral)
    return _rel(float(X), float(Y), z)

_RW = Path(__file__).resolve().parents[1]
LUNCHBOX_USD = str(_RW / "a2pro_rl" / "assets" / "lunchbox.usd")
# Rind size, kept in step with course_mission/catalogue.py OBJ_DIMS["rind"] --
# duplicated rather than imported ON PURPOSE. course_sim is the STANDALONE course and
# must not depend on the mission built on top of it; importing course_mission.catalogue
# runs course_mission/__init__.py, which imports course_mission.env_cfg, which imports
# THIS module -> circular import, and the standalone viewer died with
# "cannot import name CourseEnvA2Cfg from partially initialized module".
# Guide 2026-08-20 line 289: 윗면 폭 5cm, 아랫면 폭 6cm, 길이 20cm.
OBJ_DIMS_RIND = ((0.090, 0.050, 0.036) if _VENUE_LEGACY else (0.200, 0.050, 0.020))

# Photo-textured props, same switch as the ground textures in terrain.py and the detector
# prototypes in course_mission/perception.py. Kept self-contained here rather than imported
# from course_mission: this module is the STANDALONE course, and it must not depend on the
# mission that sits on top of it.
#
# CourseMissionEnvCfg clears every trash_* this module spawns and re-spawns its own
# (course_mission/env_cfg.py:327-331), so nothing here reaches a mission run either way.
_PROP_USD = {c: str(_RW / "a2pro_rl" / "assets" / f"{c}.usd")
             for c in ("can", "bottle", "rind", "redbag", "purplebag", "lunchbox")}


def _photo_props():
    return (os.environ.get("MISSION_VENUE_PHOTO") == "1"
            or os.environ.get("MISSION_PROP_TEX") == "1")


def _usd_prop(name, pos, cls, mass):
    return _prim_obj(name, pos, sim_utils.UsdFileCfg(
        usd_path=_PROP_USD[cls], scale=(1.0, 1.0, 1.0),
        rigid_props=_RIGID, collision_props=_COLL,
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
    ))

_RIGID = sim_utils.RigidBodyPropertiesCfg(
    rigid_body_enabled=True, kinematic_enabled=False,
    linear_damping=0.5, angular_damping=0.8, max_depenetration_velocity=0.5,
)
_COLL = sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0)


def _prim_obj(name, pos, spawn):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def can_cfg(name, pos):  # yellow aluminium can 330 ml / 200 g
    if _photo_props():
        return _usd_prop(name, pos, "can", 0.2)
    return _prim_obj(name, pos, sim_utils.CylinderCfg(
        radius=0.033, height=0.115,
        rigid_props=_RIGID, collision_props=_COLL,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
    ))


def bottle_cfg(name, pos):  # plastic bottle 330 ml / 200 g
    if _photo_props():
        return _usd_prop(name, pos, "bottle", 0.2)
    return _prim_obj(name, pos, sim_utils.CylinderCfg(
        radius=0.030, height=0.170,
        rigid_props=_RIGID, collision_props=_COLL,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.85, 0.95)),
    ))


def rind_cfg(name, pos):  # mock watermelon rind, guide line 289: 20 cm long
    if _photo_props():
        return _usd_prop(name, pos, "rind", 0.1)
    return _prim_obj(name, pos, sim_utils.CuboidCfg(
        size=OBJ_DIMS_RIND,
        rigid_props=_RIGID, collision_props=_COLL,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.55, 0.2)),
    ))


def lunchbox_cfg(name, pos):  # eco clamshell (mass baked in the USD: 0.05 kg)
    return _prim_obj(name, pos, sim_utils.UsdFileCfg(
        usd_path=LUNCHBOX_USD, scale=(1.0, 1.0, 1.0),
        rigid_props=_RIGID, collision_props=_COLL,
    ))


def bag_cfg(name, pos, color):  # paper bag 13 x 9.5 x 27 cm / 100 g
    if _photo_props():
        cls = "redbag" if color[0] > color[2] else "purplebag"
        return _usd_prop(name, pos, cls, 0.10)
    return _prim_obj(name, pos, sim_utils.CuboidCfg(
        size=(0.13, 0.095, 0.27),
        rigid_props=_RIGID, collision_props=_COLL,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    ))


@configclass
class CourseEnvB2Cfg(TaskBEnvCfg):
    """Course test env with the B2+PiPER robot (task_b sensor wiring reused)."""

    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_B2_PIPER_CFG

        self.scene.robot = UNITREE_B2_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_B2_PIPER_CFG.init_state.replace(
                pos=tuple(_rel_course(COURSE_X0 + 1.0, 0.0, 0.73)),
            ),
        )
        super().__post_init__()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            UNITREE_B2_PIPER_CFG.base_link_name, ".*_hip", ".*_thigh",
        ]
        self._wire_joints(
            UNITREE_B2_PIPER_CFG.joint_names,
            UNITREE_B2_PIPER_CFG.leg_joint_names,
            UNITREE_B2_PIPER_CFG.arm_joint_names,
        )
        self._course_setup()

    def _wire_joints(self, joint_names, leg_joint_names, arm_joint_names):
        """task_b robot-subclass convention: fill action/obs terms per robot."""
        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names
        self.actions.joint_leg.joint_names = leg_joint_names
        self.actions.joint_arm.joint_names = arm_joint_names
        self.actions.joint_wheel = None

    def _course_setup(self):
        # course terrain replaces the task_b flat+bin terrain
        self.scene.terrain = COURSE_TERRAIN_CFG
        # drop task_b's 18 YCB objects
        for i in range(1, 19):
            setattr(self.scene, f"object_{i}", None)
        # task_b's circle-success and object rewards reference object_1..18 — all off
        if hasattr(self.terminations, "objects_in_circle_done"):
            self.terminations.objects_in_circle_done = None
        for rname in list(vars(self.rewards)):
            rterm = getattr(self.rewards, rname, None)
            fn = getattr(rterm, "func", None)
            if fn is not None and "task_b" in getattr(fn, "__module__", str(fn.__class__.__module__ if hasattr(fn, "__class__") else "")):
                setattr(self.rewards, rname, None)
        # terrain has raised sections; keep the fall check well below the course
        if hasattr(self.terminations, "fall"):
            self.terminations.fall.params["minimum_height"] = -1.0

        # Competition objects, seeded scatter on the walkable flat-ish sections, inside
        # the red strips (|y| < 1.4). The section list is now venue-dependent -- the Hong
        # Kong course has no "mixed" cell at all -- so the zones are DERIVED from whatever
        # sections exist rather than naming four of them, which raised KeyError: 'mixed'
        # the moment the course changed.
        rng = np.random.default_rng(seed=getattr(self, "seed", 0) or 0)
        _obj_ok = ("mud", "pebble", "grass", "mixed")
        zones = [(SECTION_CENTERS[s_], SECTION_LEN - 1.0)
                 for s_ in COURSE_SEQUENCE if s_ in _obj_ok]
        makers = [
            ("Can", can_cfg), ("Bottle", bottle_cfg), ("Rind", rind_cfg),
            ("Lunchbox", lunchbox_cfg),
            ("RedBag", lambda n, p: bag_cfg(n, p, (0.85, 0.1, 0.1))),
            ("PurpleBag", lambda n, p: bag_cfg(n, p, (0.55, 0.15, 0.7))),
        ]
        idx = 0
        for tname, maker in makers:
            for k in range(4):
                cx, clen = zones[idx % len(zones)]
                x = float(cx + rng.uniform(-clen / 2, clen / 2))
                w = float(rng.uniform(-1.4, 1.4))
                z = 0.25 if tname in ("RedBag", "PurpleBag") else 0.15
                setattr(self.scene, f"trash_{idx + 1}", maker(f"Trash{idx + 1}", _rel_course(x, w, z)))
                idx += 1


_LEG_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
_ARM_JOINTS = [f"arm_joint{i}" for i in range(1, 9)]


@configclass
class CourseEnvA2Cfg(CourseEnvB2Cfg):
    """Course test env with the competition robot: A2 Pro + PiPER-X.

    The grasp-lineage A2 cfg lacks the task-env wiring fields (joint_names lists,
    ee camera link), so they are filled here BEFORE the base post-init wires the
    action/sensor terms (the B2 flow in the parent is skipped entirely).
    """

    def __post_init__(self):
        _a2pro = str(_RW / "a2pro_rl")
        if _a2pro not in sys.path:
            sys.path.insert(0, _a2pro)
        from a2pkg.a2piper import UNITREE_A2_PIPER_CFG

        robot = UNITREE_A2_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_A2_PIPER_CFG.init_state.replace(
                pos=tuple(_rel_course(COURSE_X0 + 1.0, 0.0, 0.52)),
            ),
        )
        robot.ee_camera_link_name = "gripper_base"
        robot.leg_joint_names = list(_LEG_JOINTS)
        robot.arm_joint_names = list(_ARM_JOINTS)
        robot.joint_names = list(_LEG_JOINTS) + list(_ARM_JOINTS)
        self.scene.robot = robot
        # skip CourseEnvB2Cfg.__post_init__ (it would re-set the B2 robot);
        # run the TaskB base chain directly, then the shared course setup
        TaskBEnvCfg.__post_init__(self)
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            robot.base_link_name, ".*_hip", ".*_thigh",
        ]
        self._wire_joints(robot.joint_names, robot.leg_joint_names, robot.arm_joint_names)
        self._course_setup()


@configclass
class CourseEnvA2BareCfg(CourseEnvA2Cfg):
    """A/B probe: bare A2 (no arm) on the same course — isolates the arm's effect on gait."""

    def __post_init__(self):
        _a2pro = str(_RW / "a2pro_rl")
        if _a2pro not in sys.path:
            sys.path.insert(0, _a2pro)
        from a2pkg.a2 import UNITREE_A2_CFG

        robot = UNITREE_A2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_A2_CFG.init_state.replace(
                pos=tuple(_rel_course(COURSE_X0 + 1.0, 0.0, 0.52)),
            ),
        )
        robot.ee_camera_link_name = None
        robot.leg_joint_names = list(_LEG_JOINTS)
        robot.arm_joint_names = []
        robot.joint_names = list(_LEG_JOINTS)
        self.scene.robot = robot
        TaskBEnvCfg.__post_init__(self)
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            robot.base_link_name, ".*_hip", ".*_thigh",
        ]
        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = robot.joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = robot.joint_names
        self.actions.joint_leg.joint_names = list(_LEG_JOINTS)
        self.actions.joint_arm = None
        self.actions.joint_wheel = None
        # no arm -> no ee camera sensor
        if hasattr(self.scene, "ee_camera"):
            self.scene.ee_camera = None
        if hasattr(self.scene, "ee_dual_camera"):
            self.scene.ee_dual_camera = None
        self._course_setup()
