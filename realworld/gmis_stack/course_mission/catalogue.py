"""Competition object catalogue — pure data, no Isaac imports.

Kept separate from env_cfg.py so the detector can be scored offline (scripts/
tune_detector.py) against recorded frames without starting a simulator: importing
env_cfg pulls in isaaclab.sim, which refuses to load outside a running SimulationApp.

Classes and bins follow the 참가안내서 (2026-08-03) 과제 2:
  can, lunchbox, redbag -> BLUE (재활용) | rind -> GREEN (음식물)
  bottle, purplebag are distractors: not in the sub-task, so grasping them scores 0.
"""

import os

# COURSE_VENUE=legacy keeps the object sizes the four 120/160 runs were scored on.
# Read from the environment rather than imported from course_sim.terrain, because
# this module must stay importable without Isaac (see the docstring above).
_LEGACY = os.environ.get("COURSE_VENUE", "hk").lower() == "legacy"


BIN_OF = {"can": "blue", "rind": "green", "lunchbox": "blue", "redbag": "blue",
          "bottle": None, "purplebag": None}

# True OBB extents [m], handed to the grasp policy's obb_dims slot.
OBJ_DIMS = {
    "can": (0.066, 0.066, 0.115),
    "bottle": (0.060, 0.060, 0.170),
    # 2026-08-20 guide line 289: 모의 수박 껍질 — 윗면 폭 5cm, 아랫면 폭 6cm, 길이 20cm.
    # The 07-25 guide said "길이 약 7~10cm, 폭 약 5cm" for a NATURAL rind and this was
    # 90 mm to match; the venue prop is now a 200 mm moulded mock-up, more than twice
    # as long. The 5 vs 6 cm pair is the chord across the open face against the arc
    # across the curved outside, i.e. a shallow shell -- confirmed against the
    # organiser photos (course_sim/image/watermelon_top.jpg, _back.jpg), which show a
    # long lens with pointed ends, not a straight prism. So the shape
    # a2pro_rl/scripts/build_rind_usd.py already builds is right and only the size
    # changes: chord 50 mm, trough depth 14 mm (outer arc ~60 mm), shell 6 mm.
    # COURSE_VENUE=legacy keeps 90 mm so the four 120/160 runs stay reproducible.
    "rind": (0.090, 0.050, 0.036) if _LEGACY else (0.200, 0.050, 0.020),
    "lunchbox": (0.150, 0.110, 0.060),
    # Paper bags: the guide lists 13 x 9.5 x 27 cm FLAT-PACKED, but a rigid 95 mm box
    # cannot be picked up at all — the PiPER-X gripper opens 100 mm, leaving 5 mm of
    # clearance that any yaw error eats. Real paper crushes the moment the fingers close,
    # so the bag is modelled in its GRIPPED cross-section (55 mm). Height and the long
    # axis stay at spec, so the detector's size gate and the visual footprint are intact.
    "redbag": (0.130, 0.055, 0.270),
    "purplebag": (0.130, 0.055, 0.270),
}

# RGB the detector keys on (matches the visual materials; lunchbox comes from its USD)
OBJ_RGB = {
    "can": (0.95, 0.85, 0.10),
    "bottle": (0.75, 0.85, 0.95),
    "rind": (0.15, 0.55, 0.20),
    "lunchbox": (0.93, 0.90, 0.80),
    "redbag": (0.85, 0.10, 0.10),
    "purplebag": (0.55, 0.15, 0.70),
}

# grasping preference: flatter ground first (사용자 지정 순서 잔디 > 진흙 > 자갈 > 계단)
TERRAIN_RANK = {"grass": 0, "mud": 1, "pebble": 2, "mixed": 2, "slope": 3, "stairs": 4}


# Gripper reality: the PiPER-X opens 100 mm, so a target width above ~90 mm is a command
# it can never reach. The old code asked for min(dims[0], dims[1]), which for the lunchbox
# is 110 mm — the policy was being told to do something impossible, and lunchboxes scored
# 0/3 in the mission while cans and rinds went 4/4.
#
# GRASP_HINT[cls] = (target finger width [m], depth below the object's top to pinch [m])
#   can/bottle/rind  — pinch the body across its minor axis, a little below the top
#   lunchbox         — 150 x 110 x 60 lying flat has NO horizontal dimension the gripper
#                      can close on, so the only grasp is a shallow pinch on the lid,
#                      8 mm down, closing to less than the lid's local width (measured
#                      previously: this is the one thing that ever picked it up)
#   redbag/purplebag — a real paper bag stands on its base; grab the 55 mm rim near the top
def _bag_grip():
    """(finger width target, pinch depth) for a paper bag's wall. MISSION_BAG_GRIP="w,d"."""
    raw = os.environ.get("MISSION_BAG_GRIP", "")
    if raw:
        w, d = (float(v) for v in raw.split(","))
        return (w, d)
    return (0.003, 0.030)


GRIPPER_MAX_WIDTH = 0.090
GRASP_HINT = {
    "can":       (0.066, 0.015),
    "bottle":    (0.060, 0.015),
    "rind":      (0.030, 0.010),
    # MEASURED, two ways round:
    #   8 mm  + friction 0.9 / 30 N -> reaches beautifully (TCP 0.009-0.013 m), closes for
    #                                  up to 299 steps, and slips: 11 of 17 attempts.
    #   12 mm + friction 1.3 / 45 N -> never slips, but the target sits 4 mm lower than the
    #                                  arm can comfortably get to: TCP stalls 0.05-0.07 m
    #                                  away and the policy never closes at all (3 of 4).
    # So the depth was never the problem — holding force was. Back to the 8 mm bite that
    # reaches, with the friction and clamping force that hold.
    "lunchbox":  (0.055, 0.008),
    # Bags come to rest on their side (confirmed on video), so like the lunchbox they
    # present a flat slab whose horizontal dimensions (130 x 270 mm) the gripper cannot
    # close on at all. The only grasp is the same shallow lip pinch that finally worked
    # for the lunchbox: catch the top few mm, where the fingers meet the 55 mm thickness
    # instead of the 130 mm width. A 20 mm depth put the fingers mid-slab, which closed
    # for 141-281 steps and slid out every time.
    # PINCH ONE WALL, not the whole bag. Asking for the 55 mm cross-section meant closing
    # on a void: the bag was six sealed faces with nothing between them, and every red-bag
    # grasp failed across two full runs (6 of 8 attempts, then 6 of 6, zero conversions)
    # while cans went 2 for 2. It is a paper bag, so it is picked up the way a hand does
    # it -- hook into the mouth and take a single layer of the wall. a2pro_rl/scripts/
    # build_prop_usds.build_bag now gives that layer a real 3 mm thickness and leaves the
    # top open so the rim exists to be hooked.
    #
    # 0.003 -> grasp_hint() asks the fingers for 3 + 5 = 8 mm, well inside their travel.
    # 0.030 of depth puts the bite 3 cm down the wall: deep enough that the fingers are
    # past the rim's rounding and onto flat material, shallow enough to stay in the top
    # third where a bag is stiffest. MISSION_BAG_GRIP overrides both for a sweep.
    "redbag":    _bag_grip(),
    "purplebag": _bag_grip(),
}


# What the grasp policy should be TOLD the object is, when that differs from its true
# extents. The policy was trained (grasp_rl/graspenv/mdp/events.py: reset_pregrasp_rsi)
# on objects placed directly under the fingertips, so nothing in its experience is 27 cm
# tall. A standing paper bag is, and it showed: the fingers reached to within 0.035 m and
# then never closed. The fingers only ever meet the top of the bag, so that is the slab
# described to the policy — true footprint, grasped height.
# The minor axis is what the FINGERS MEET, and that is no longer the bag's 55 mm
# cross-section: with the walls modelled (build_prop_usds.build_bag) the grasp is a pinch
# on one 3 mm layer. Describing a 55 mm slab while asking the fingers to close to 8 mm is
# a contradiction the policy cannot satisfy. 10 mm rather than a literal 3 mm because the
# policy was trained on objects sat under its fingertips and nothing in that experience is
# 3 mm thin -- this is the smallest value that stays in distribution while telling the
# truth about the direction. MISSION_BAG_OBB="l,w,h" overrides it for a sweep; the probe
# (course_mission/scripts/run_probe_bag.sh) is what settles the number.
_BAG_OBB = tuple(float(v) for v in os.environ["MISSION_BAG_OBB"].split(",")) \
    if os.environ.get("MISSION_BAG_OBB") else (0.130, 0.010, 0.060)
GRASP_OBB = {
    "redbag": _BAG_OBB,
    "purplebag": _BAG_OBB,
}
# The lunchbox's true minor extent (110 mm) is outside the policy's training range
# (W 30-85 mm) and beyond the 90 mm the fingers can span; what the fingers actually
# meet is the lid taper (85 mm at the top). MISSION_LBOX_OBB="l,w,h" lets the probe
# measure whether describing the lid instead of the body helps the policy reach it.
if os.environ.get("MISSION_LBOX_OBB"):
    GRASP_OBB["lunchbox"] = tuple(float(v) for v in
                                  os.environ["MISSION_LBOX_OBB"].split(","))


# Height of the object's TOP above its RIGID-BODY ORIGIN [m], for classes whose origin is
# NOT the geometric centre. The lunchbox is the only object spawned from a USD file
# (env_cfg.py uses UsdFileCfg for it and centre-origin Cylinder/CuboidCfg for everything
# else), and a2pro_rl/scripts/build_lunchbox_usd.py puts its meshes at local z = 0.000 to
# 0.065 on a bare Xform with NO transform at all (verified: `strings lunchbox.usd` finds
# zero xformOp / translate / matrix4d / orient tokens). IsaacLab's root_pos_w is that
# actor frame, so for the lunchbox it is the BOTTOM FACE.
#
# _grasp_point_w's generic rule — "an object at rest has its centre one half-height up, so
# the top is z + half" — is therefore wrong by a full half-height here, and it aimed
# 35 mm low on EVERY terrain. That is not merely low: the lid's seam lip (local z 0.031 to
# 0.037) is 205 x 115 mm and the gripper opens 100 mm, so nothing can descend below local
# z = 0.044 at all. The commanded point sat at 0.022 — 22 mm inside solid, unreachable
# geometry. It explains all three logged signatures (never closed / closed then slipped /
# object batted away), and it explains why going DEEPER to 12 mm made reaching worse
# rather than better, which the friction story never could.
TOP_ABOVE_ORIGIN = {"lunchbox": 0.065}


# Below this, a feature is thin enough that the fingers must SQUEEZE it rather than close
# around it, and the fixed +5 mm clearance becomes the whole problem.
THIN_FEATURE = 0.012
SQUEEZE = float(os.environ.get("MISSION_SQUEEZE", "0.002"))


def grasp_hint(cls: str, dims):
    """(finger width target, pinch depth) for a class, never exceeding what the gripper
    can actually open to.

    The clearance is NOT a constant. It was +5 mm for everything, which is fine when the
    fingers close around a 66 mm can, but on the paper bag's 3 mm wall it asked them to
    stop 8 mm apart -- 1 mm clear of each face, never touching it. Measured: 24 bag trials
    across 8 parameter settings, zero grasps, and the TCP reaching within 2.8 cm on the
    good ones. The fingers were arriving and closing on air.

    A thin feature is gripped by closing PAST it: paper and a bagasse lid both crush, and
    the hold comes from the clamp, not from wrapping. So below THIN_FEATURE the target goes
    SQUEEZE millimetres under the material instead of over it.
    """
    w, d = GRASP_HINT.get(cls, (min(dims), 0.015))
    target = max(0.0, w - SQUEEZE) if w <= THIN_FEATURE else w + 0.005
    return min(target, GRIPPER_MAX_WIDTH), d


def grasp_obb(cls: str, dims):
    """The extents to hand the policy's obb_dims slot (see GRASP_OBB)."""
    return GRASP_OBB.get(cls, tuple(dims))


# Sections that may hold objects: the walkable flat-ish ones. Stairs and the grass SLOPE
# are excluded -- an object on a 9-degree ramp rolls, and one on a stair tread cannot be
# approached. The LIST (OBJ_SECTIONS) is course-dependent and lives in env_cfg, but the
# membership rule is not, and the hardware build needs it without importing the PhysX
# scene -- so the tuple lives here, where nothing imports a simulator.
OBJ_OK_SECTIONS = ("mud", "pebble", "grass", "mixed")
