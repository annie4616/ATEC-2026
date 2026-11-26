"""Mission env: the S-course terrain + competition trash + 3 bins + the basket robot.

Built on course_sim's A2+PiPER-X course env, but re-tuned so the trained grasp policy is
valid inside it:
  - robot USD swapped for the basket-equipped variant (a2_piperx_basket.usd)
  - the finger actuator split + stiff solver + tight contact offsets from the grasp env
    (without these a closed gripper has ~2.8 N of grip and drops everything)
  - objects carry the grasp env's rigid/collision properties, not task_b's loose ones
  - finger contact sensors (net force; the multi-object filtered matrix is not needed)
  - the arm is left WITHOUT an action term on purpose: the mission driver writes arm joint
    targets directly (scripted IK trajectories + the policy's differential-IK output), the
    same pattern verified in eunha_rl/scripts/play_deposit_cycle.py

Object classes follow the 참가안내서 (2026-08-03):
  target  can       -> BLUE  (재활용)
  target  rind      -> GREEN (음식물)
  target  lunchbox  -> BLUE  (재활용)
  target  redbag    -> BLUE  (재활용)
  distractor bottle, purplebag  (하위과제에 없음 -> 집으면 0점)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from course_sim.env_cfg import CourseEnvA2Cfg, _rel_course
import course_sim.terrain as ct
from course_sim.terrain import SECTION_CENTERS, SECTION_LEN, curve_map

_RW = Path(__file__).resolve().parents[1]
LUNCHBOX_USD = str(_RW / "a2pro_rl" / "assets" / "lunchbox.usd")
# The rind is a TWO-SIDED shell, not a green block: the real prop
# (course_sim/image/watermelon_*.jpg) is green with black stripes on its convex face and
# bright RED flesh on its concave one, so which colour the camera sees depends on how it
# lands. A uniform cuboid cannot express that, and the flipped case therefore never
# existed in simulation — while the detector's redbag prototype sits at H=0 and the
# measured flesh is H~350, close enough to be confused. Built by
# a2pro_rl/scripts/build_rind_usd.py with its origin at the bbox centre.
RIND_USD = str(_RW / "a2pro_rl" / "assets" / "rind.usd")
BASKET_USD = os.environ.get(
    "MISSION_BASKET_USD",
    str(_RW / "eunha_rl" / "assets" / "a2_piperx_basket.usd"))

# Photo-textured props (a2pro_rl/scripts/build_prop_usds.py) built from the venue
# photographs in course_sim/image/. Geometry, mass and extents are identical to the
# CuboidCfg/CylinderCfg primitives they replace -- only the surface changes -- but the
# surface is exactly what the camera detector keys on, so this is OFF BY DEFAULT and
# MISSION_PROP_TEX=1 turns it on. See perception.OBJ_PROTO: the prototypes are fitted to
# the flat colours, and the real props do not match them (the bottle is clear plastic with
# a YELLOW label, not light blue; the red bag is a dusty rose, not saturated red).
PROP_USD = {c: str(_RW / "a2pro_rl" / "assets" / f"{c}.usd")
            for c in ("can", "bottle", "redbag", "purplebag")}
# MISSION_BAG_HANDLE=1 swaps in the bag WITH a carry handle (build_prop_usds.py
# handle=True) so the scripted pinch has something a gripper can actually hold.
if os.environ.get("MISSION_BAG_HANDLE") == "1":
    PROP_USD["redbag"] = str(_RW / "a2pro_rl" / "assets" / "redbag_handle.usd")


def prop_tex_on():
    """MISSION_VENUE_PHOTO=1 is the switch to use: it dresses props AND ground together.

    The two must move as a pair because perception.py carries ONE fitted prototype set per
    world, keyed off the same variable -- a half-dressed course would be scored against
    prototypes fitted to neither. MISSION_PROP_TEX stays for isolating the props in an A/B.
    """
    return (os.environ.get("MISSION_PROP_TEX") == "1"
            or os.environ.get("MISSION_VENUE_PHOTO") == "1")

# The catalogue itself lives in catalogue.py (no Isaac imports) so the offline
# detector tooling can use it without a running simulator; re-exported here because
# every existing caller imports these names from env_cfg.
from course_mission.catalogue import (  # noqa: F401
    BIN_OF, OBJ_DIMS, OBJ_RGB, OBJ_OK_SECTIONS, TERRAIN_RANK,
)

_RIGID = sim_utils.RigidBodyPropertiesCfg(
    rigid_body_enabled=True, kinematic_enabled=False,
    linear_damping=0.5, angular_damping=0.8, max_depenetration_velocity=1.0,
    solver_position_iteration_count=16, solver_velocity_iteration_count=1,
)
# grasp-env contact offsets: the fingers need tight offsets or contacts fire early/late
_COLL = sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)
_PHYS = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.8, dynamic_friction=0.7, restitution=0.0,
    friction_combine_mode="multiply", restitution_combine_mode="multiply",
)


def _obj(name, pos, spawn):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def _spawn_for(cls):
    if prop_tex_on() and cls in PROP_USD:
        # UsdFileCfg takes no physics_material, so friction comes from the per-object
        # startup event in __post_init__ -- the same mechanism the lunchbox and rind use.
        # Without that these would silently fall back to the simulator default and the
        # grasp behaviour would change for a purely visual switch.
        return sim_utils.UsdFileCfg(
            usd_path=PROP_USD[cls], scale=(1.0, 1.0, 1.0),
            rigid_props=_RIGID, collision_props=_COLL,
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.20 if cls in ("can", "bottle") else 0.10))
    if cls == "can":
        return sim_utils.CylinderCfg(
            radius=0.033, height=0.115, rigid_props=_RIGID, collision_props=_COLL,
            physics_material=_PHYS, mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=OBJ_RGB["can"]))
    if cls == "bottle":
        return sim_utils.CylinderCfg(
            radius=0.030, height=0.170, rigid_props=_RIGID, collision_props=_COLL,
            physics_material=_PHYS, mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=OBJ_RGB["bottle"]))
    if cls == "rind":
        # MISSION_RIND_BOX=1 falls back to the old uniform green cuboid for A/B.
        if os.environ.get("MISSION_RIND_BOX") == "1":
            return sim_utils.CuboidCfg(
                size=OBJ_DIMS["rind"], rigid_props=_RIGID, collision_props=_COLL,
                physics_material=_PHYS, mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=OBJ_RGB["rind"]))
        # UsdFileCfg takes no physics_material, so the surface comes from the per-rind
        # friction event in __post_init__, the same mechanism the lunchbox uses.
        return sim_utils.UsdFileCfg(
            usd_path=RIND_USD, scale=(1.0, 1.0, 1.0),
            rigid_props=_RIGID, collision_props=_COLL,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.10))
    if cls == "lunchbox":
        # The lunchbox was the ONLY object inheriting its surface and mass from its USD
        # instead of being given the same explicit values as the rest. It is also the one
        # object that has to be held by a shallow lid pinch, so a slippery default surface
        # or a heavy default mass shows up immediately: measured, the gripper closed on it
        # for 71 steps and lifted it 227 mm before it slid out. Same material and a
        # realistic empty-lunchbox mass as the other classes.
        # UsdFileCfg takes no physics_material, so the surface is set by the
        # per-lunchbox friction event in __post_init__ (same mechanism as the fingertips).
        return sim_utils.UsdFileCfg(
            usd_path=LUNCHBOX_USD, scale=(1.0, 1.0, 1.0),
            rigid_props=_RIGID, collision_props=_COLL,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15))
    color = OBJ_RGB["redbag"] if cls == "redbag" else OBJ_RGB["purplebag"]
    return sim_utils.CuboidCfg(
        size=OBJ_DIMS[cls], rigid_props=_RIGID, collision_props=_COLL,
        physics_material=_PHYS, mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))


# object sections: the walkable flat-ish ones (no objects on stairs/slope, matching
# course_sim). Order here fixes the traversal order the robot meets them in.
# Sections that may hold objects: the walkable flat-ish ones. Stairs and the grass
# SLOPE are excluded -- an object on a 9-degree ramp rolls, and one on a stair tread
# cannot be approached. Derived from the section list so the Hong Kong course (which
# has no "mixed" cell at all) does not silently end up with an empty or wrong set.
OBJ_SECTIONS = [s for s in ct.COURSE_SEQUENCE if s in OBJ_OK_SECTIONS]


# Copies of each class. Guide line 265: "각 목표 카테고리(4종)당 1개씩으로 구성된 목표
# 물체 세트가 3세트(총 12개)" -- three of each of the four TARGET categories. The
# distractor count is left unspecified by the guide ("유사한 물체가 추가로 배치"), so
# they get the same three. Legacy keeps four, which is what every 120/160 run used.
PER_TYPE = 4 if ct._VENUE == "legacy" else 3


def scatter_layout(seed: int = 0, per_type: int = None):
    """Deterministic object layout: [(name, cls, section, arc_x, lateral, z), ...].

    Distractors are placed too (they must never be grasped) — the guide requires the
    robot to reject them by class, not by absence.
    """
    per_type = PER_TYPE if per_type is None else per_type
    rng = np.random.default_rng(seed)
    classes = ["can", "rind", "lunchbox", "redbag", "bottle", "purplebag"]
    out, idx = [], 0
    for cls in classes:
        for k in range(per_type):
            sec = OBJ_SECTIONS[idx % len(OBJ_SECTIONS)]
            cx = SECTION_CENTERS[sec]
            x = float(cx + rng.uniform(-(SECTION_LEN - 1.6) / 2, (SECTION_LEN - 1.6) / 2))
            w = float(rng.uniform(-1.15, 1.15))
            # REVERTED, and worth recording why. Spawning each object at its own resting
            # height (dims[2]/2) was meant to stop the 27 cm paper bags toppling. It did
            # not — video showed them lying down anyway — and it broke something worse:
            # these heights are WORLD z, while the section surfaces sit at 0.008 (mud) to
            # 0.030 (pebble) plus relief. A 3 cm rind spawned at z = 0.020 has its base
            # 25 mm INSIDE the pebble bed, and the solver ejects it: objects were found at
            # z = -175 m and -936 m, which then showed up as phantom detections, a
            # poisoned aim bias and missions that stalled for their whole budget.
            # Dropping from clearly above the terrain is the robust choice.
            z = 0.30 if cls in ("redbag", "purplebag") else 0.20
            idx += 1
            out.append((f"trash_{idx}", cls, sec, x, w, z))
    return out


def bin_world_xy():
    """World (x, y) of each competition bin. Laterals come from terrain.BIN_LATERAL_OF
    so this cannot drift from what bins_section actually builds."""
    cx = SECTION_CENTERS["bins"]
    out = {}
    for name, lat in ct.BIN_LATERAL_OF.items():
        X, Y = curve_map(cx, lat)
        out[name] = (float(X), float(Y), float(lat))
    return out


@configclass
class CourseMissionEnvCfg(CourseEnvA2Cfg):
    """Course + competition trash + basket robot, grasp-policy-compatible physics."""

    layout_seed: int = 0
    per_type: int = PER_TYPE

    def __post_init__(self):
        super().__post_init__()
        r = self.scene.robot
        # basket-equipped robot (rear side-saddle basket; see eunha_rl/deposit_traj.json)
        if os.environ.get("MISSION_NO_BASKET") != "1":
            r.spawn.usd_path = BASKET_USD
        # grasp-env robot physics: stiff solver + tight contacts + dedicated finger gains
        r.spawn.articulation_props.solver_position_iteration_count = 16
        r.spawn.articulation_props.solver_velocity_iteration_count = 1
        r.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=0.002, rest_offset=0.0)
        r.actuators["arms"].joint_names_expr = "arm_joint[1-6]"
        # ARM SPEED CEILING. The robot description rates these joints at 10 rad/s
        # (a2pro_rl/assets/a2_piperx.urdf: <limit ... effort="100" velocity="10">) but the
        # asset config raises velocity_limit_sim to 100 (a2pro_rl/a2pkg/a2piper.py), so the
        # simulator lets the arm swing TEN TIMES faster than the hardware allows and never
        # objects. On the real robot that is exactly where a joint driver faults out and
        # takes the machine down mid-run — and the grasp policy does command bursts far
        # past the design contract (75-124 mm and 22-32 deg per 20 ms step). Setting this
        # to 10 makes the simulation honest about what the hardware can do; leaving it at
        # 100 preserves the behaviour the 120/160 runs were scored with.
        # DEFAULT 10 = the robot description's own rating. MEASURED: with the old value
        # (100, from a2pro_rl/a2pkg/a2piper.py) the grasp drives these joints to a peak of
        # 100.0 rad/s — the simulator ceiling, TEN TIMES the hardware rating — and nothing
        # complains, because only the simulator was permissive. On the real arm that is a
        # driver protection fault mid-run. Constraining it is not a cost: the constrained
        # run scored 120/160 in 54,619 steps with peaks at 10.4 rad/s while the
        # unconstrained control scored 80, losing exactly the lunchboxes that an
        # overshooting arm bats away. MISSION_ARM_VEL_LIMIT=100 restores the old behaviour.
        _avl = float(os.environ.get("MISSION_ARM_VEL_LIMIT", "10"))
        if _avl > 0.0:
            r.actuators["arms"].velocity_limit_sim = _avl
        # 30 N is a conservative figure for this gripper; a shallow lid pinch needs the
        # clamping force more than a wrap-around grasp does.
        r.actuators["gripper"] = ImplicitActuatorCfg(
            joint_names_expr="arm_joint[78]", effort_limit_sim=45.0, velocity_limit_sim=1.0,
            stiffness=1500.0, damping=50.0, friction=0.01, armature=0.01,
        )
        # the mission driver writes arm joint targets itself (scripted IK + policy IK)
        self.actions.joint_arm = None
        self.actions.joint_wheel = None

        # Rubberised fingertips. The converted PiPER-X URDF gives the fingers a default
        # low-friction material; on the 66 mm smooth can the gripper closed dead on the
        # object (TCP gap 1.7 cm) yet it never rose — it just slipped. Real gripper kits
        # have rubber pads, and grasp_env_cfg.py exposes the same knob as
        # GRASP_FINGER_FRICTION.
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg
        import isaaclab.envs.mdp as _mdp

        # The lunchbox comes from a USD, which cannot take physics_material inline, so it
        # would otherwise keep whatever surface its USD shipped with — the only object in
        # the scene not given an explicit one, and the only one that has to be held by a
        # shallow lid pinch. Measured before this: closed for 239 steps with the TCP 15 mm
        # from the object and it still would not come up.
        # The rind is now a USD too (two-sided shell), so it needs the same treatment or it
        # keeps whatever surface the asset shipped with. It is picked by a rim pinch, so a
        # slippery surface shows up the same way the lunchbox's did — give it the explicit
        # 0.8/0.7 the cuboid version had, not the lunchbox's grippier 1.3/1.1.
        for _n, _c, *_ in self.mission_layout:
            # Textured can/bottle/bags are USD too, so they lose the inline _PHYS the
            # primitives carried. Restore exactly the 0.8/0.7 those primitives had, so
            # MISSION_PROP_TEX changes appearance and NOTHING else.
            if prop_tex_on() and _c in PROP_USD:
                # The HANDLE bag is held by an 8 mm strap pinch: at 0.8/0.7 the strap
                # slides straight through the fingers (gw 12 mm dead on the bar, rise
                # -0.001 m over a 12 cm lift). Same cure as the lunchbox lid pinch —
                # the grippier 1.3/1.1 — applied only when the handle asset is in play.
                _grippy = (_c == "redbag"
                           and os.environ.get("MISSION_BAG_HANDLE") == "1")
                setattr(self.events, f"friction_{_n}", EventTerm(
                    func=_mdp.randomize_rigid_body_material, mode="startup",
                    params={"asset_cfg": SceneEntityCfg(_n),
                            "static_friction_range": (1.3, 1.3) if _grippy else (0.8, 0.8),
                            "dynamic_friction_range": (1.1, 1.1) if _grippy else (0.7, 0.7),
                            "restitution_range": (0.0, 0.0), "num_buckets": 2,
                            "make_consistent": True},
                ))
            if _c == "rind" and os.environ.get("MISSION_RIND_BOX") != "1":
                setattr(self.events, f"friction_{_n}", EventTerm(
                    func=_mdp.randomize_rigid_body_material, mode="startup",
                    params={"asset_cfg": SceneEntityCfg(_n),
                            "static_friction_range": (0.8, 0.8),
                            "dynamic_friction_range": (0.7, 0.7),
                            "restitution_range": (0.0, 0.0), "num_buckets": 2,
                            "make_consistent": True},
                ))
            # Paper bags: the surface they were given is the generic 0.8/0.7 from _PHYS,
            # which is LESS grip than the lunchbox gets even though the grasp is harder --
            # a squeeze on one 3 mm paper wall, with nothing wrapped around anything. Dry
            # kraft paper against rubber fingertips is closer to 1.4, and the whole hold is
            # friction: there is no geometry keeping the bag in the gripper.
            if _c in ("redbag", "purplebag") and prop_tex_on():
                _bf = float(os.environ.get("MISSION_BAG_FRICTION", "1.4"))
                setattr(self.events, f"friction_{_n}", EventTerm(
                    func=_mdp.randomize_rigid_body_material, mode="startup",
                    params={"asset_cfg": SceneEntityCfg(_n),
                            "static_friction_range": (_bf, _bf),
                            "dynamic_friction_range": (round(0.85 * _bf, 3),) * 2,
                            "restitution_range": (0.0, 0.0), "num_buckets": 2,
                            "make_consistent": True},
                ))
                continue
            if _c != "lunchbox":
                continue
            setattr(self.events, f"friction_{_n}", EventTerm(
                func=_mdp.randomize_rigid_body_material, mode="startup",
                params={"asset_cfg": SceneEntityCfg(_n),
                        # 0.9/0.8 was not enough: the lid pinch closed for up to 299
                        # steps and still slipped after 1-8 cm of lift, 11 times out of 17
                        # lunchbox attempts in one run. A plastic box against rubber
                        # fingertips (1.6/1.4) realistically sits nearer 1.2.
                        "static_friction_range": (1.3, 1.3),
                        "dynamic_friction_range": (1.1, 1.1),
                        "restitution_range": (0.0, 0.0), "num_buckets": 2,
                        "make_consistent": True},
            ))

        self.events.finger_friction = EventTerm(
            func=_mdp.randomize_rigid_body_material, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["arm_link7", "arm_link8"]),
                    # 1.2/1.1 still let a gripped paper bag slip out after a 73 mm lift;
                    # rubber pads on paper and plastic are grippier than that
                    "static_friction_range": (1.6, 1.6), "dynamic_friction_range": (1.4, 1.4),
                    "restitution_range": (0.0, 0.0), "num_buckets": 2, "make_consistent": True},
        )

        # finger contact sensors: unfiltered net force (the filtered matrix would need one
        # column per trash prim; the grasp helpers' fallback path uses net force + proximity)
        self.scene.contact_finger_l = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/arm_link7", update_period=0.0)
        self.scene.contact_finger_r = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/arm_link8", update_period=0.0)

        # continuous mission: only the (very long) timeout ends the episode
        for attr in list(vars(self.terminations)):
            if attr != "time_out":
                setattr(self.terminations, attr, None)
        self.episode_length_s = 100000.0

    def _course_setup(self):
        super()._course_setup()
        # drop course_sim's generic scatter, install the labelled competition layout
        for i in range(1, 40):
            if hasattr(self.scene, f"trash_{i}"):
                setattr(self.scene, f"trash_{i}", None)
        # MISSION_LAYOUT_SEED overrides the config field HERE, before anything is built.
        # Assigning env_cfg.layout_seed after parse_env_cfg looks like it works and does
        # nothing: this __post_init__ has already run by then, and the very next lines
        # turn the layout into scene spawn entries — so the world keeps the default-seed
        # placement no matter what the flag said. Every "--seed N" run before 2026-08-25
        # used the seed-0 layout for this reason.
        self.layout_seed = int(os.environ.get("MISSION_LAYOUT_SEED", self.layout_seed))
        self.mission_layout = scatter_layout(self.layout_seed, self.per_type)
        for name, cls, _sec, x, w, z in self.mission_layout:
            setattr(self.scene, name, _obj(name, _rel_course(x, w, z), _spawn_for(cls)))
