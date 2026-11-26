"""The mission finite-state machine: zigzag search -> classify -> crouch -> RL grasp ->
basket -> carry -> sort into the colour-matched bin, with a crouch-stability gate that
downgrades a terrain to traverse-only when the squat is not solid.

Imported by the solution_v*.py scripts AFTER AppLauncher has started the simulator.

Terrain policy (사용자 지정): attempt objects on the flattest ground first,
grass > mud > pebble/mixed, never on stairs/slope. A terrain whose first crouch is
unstable is struck off for the rest of the run — the robot then only walks through it.

Scoring follows 참가안내서 (2026-08-03) 과제 2: 20 points for the correct bin, 10 for the
wrong bin, 5 for the designated end area, 0 for a distractor; at most 2 scored objects
per category (4 categories x 2 x 20 = 160).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# On the robot there is no Isaac Lab, and these two imports are the only reason this
# module needs one. mathx is a verified line-for-line stand-in (scripts/check_mathx.py);
# env_cfg IS the PhysX scene, but the four names taken from it are re-exports of
# catalogue.py plus one list derived from the course sequence.
try:
    import isaaclab.utils.math as math_utils
except ModuleNotFoundError:
    from course_mission import mathx as math_utils

from course_sim import terrain as ct
from course_mission.catalogue import GRIPPER_MAX_WIDTH, TOP_ABOVE_ORIGIN, grasp_hint
from course_mission.perception import detect_bins
try:
    from course_mission.env_cfg import BIN_OF, OBJ_DIMS, OBJ_SECTIONS, TERRAIN_RANK
except ModuleNotFoundError:
    from course_mission.catalogue import (
        BIN_OF, OBJ_DIMS, OBJ_OK_SECTIONS, TERRAIN_RANK,
    )
    OBJ_SECTIONS = [s for s in ct.COURSE_SEQUENCE if s in OBJ_OK_SECTIONS]
from course_mission.mission import (
    ARM_JOINTS, COMPUTE, COMPUTE_LIMITS, GRIP_OPEN, REAL_NOISE, TCP_OFFSET, ArmDriver,
    GraspObsBuilder, GraspPolicy, HeadCamMount, LegDriver, Perception, RW, arc_of,
    bin_targets, section_of,
)

SCORE = {"correct": 20, "wrong": 10, "area": 5}
MAX_SCORED_PER_CLASS = 2
# Lateral distance from the base centreline to the outside edge of a standing foot, from
# a2pro_rl/assets/a2_piperx.urdf: hip abduction 0.075113 + thigh 0.12779 = 0.2029 m.
STANCE_HALF_WIDTH = 0.203
# Lateral guard, foot-aware. MISSION_FOOT_GUARD=1 replaces the 1.8 m body-centre trigger
# with one derived from the geometry that actually traps the robot:
#     BOUNDARY_Y 1.9775  - the judged line, where the stanchions stand
#   - BASE_R     0.110   - the posts sit on black bases; their INNER edge is at 1.8675
#   - STANCE_HALF_WIDTH  - the outer foot is this far outboard of the body centre
#   - 0.02               - margin
#   = 1.645
# Why it matters, measured: run videos_venue/0824_1104_dep_detect2 wedged at lat +1.80 and
# logged EIGHT consecutive "off the side"/"cannot hold the centreline" strikes before
# abandoning the search and the finish line. At w=1.80 the outer foot is at 2.003 m --
# already 0.136 m INSIDE the stanchion base. The old guard fires only once the robot is
# standing on the barrier, which is exactly when it can no longer steer out of it. The
# flat-strip course had nothing to catch a foot, so this trap arrived with the belt
# stanchions on 2026-08-22 and the runs before it were simply lucky.
FOOT_GUARD = os.environ.get("MISSION_FOOT_GUARD") == "1"
LAT_GUARD = 1.645 if FOOT_GUARD else 1.8

# --- venue clock model ----------------------------------------------------------------
# Fitted over the 33 completed goto_bin legs in videos_persist/*/report_persist.json by
# counting the 250-step "approaching <c> bin:" prints (runner.py:888) and the 400-step
# "shuttle:" prints (runner.py:1093): steps = 85 x arc-metres + 620, R^2 = 0.97. The fixed
# term is real (accel, the final lateral strafe, settle) and matters here because the
# reserve has to cover it even when the robot is already next to the bins.
# DERIVED, not a literal. This was 17.44, which is 0.56 m short of the bins section
# centre on the seven-section course -- correct there and silently wrong anywhere
# else. The Hong Kong course puts the bins at arc +15, so a hard 17.44 would park
# the robot 2.4 m PAST them, off the end of the course.
BIN_PARK_ARC = ct.SECTION_CENTERS["bins"] - 0.56
TRAVEL_STEPS_PER_M = 85.0
TRIP_FIXED_STEPS = 620.0
UNLOAD_STEPS = 1100           # release_over_bin worst case (runner.py:1183-1221)
# approach + stop-and-look + settle + crouch + grasp incl. retries, per object; measured
# 3,480 median over run 0817_0153 (5 objects, travel and search subtracted out).
OBJECT_WORK_STEPS = 3500


# ======================================================================================
# terrain height (arc frame) — mirrors course_sim/terrain.py section builders
# ======================================================================================
def ground_z_arc(arc_x: float) -> float:
    sec = section_of(arc_x)
    local = arc_x - ct.SECTION_CENTERS[sec]          # [-3, 3]
    if sec == "mud":
        return 0.008
    if sec == "grass":
        return 0.005
    if sec == "pebble":
        return 0.013                                  # mean of the 0.004-0.022 gravel bed
    if sec == "mixed":
        return 0.020
    if sec == "bins":
        return 0.0
    if sec == "stairs":
        x = local + ct.SECTION_LEN / 2                # 0..6 local corner coords
        x0, step_h, step_d, n = 0.8, 0.15, 0.35, 4
        plat_x0 = x0 + n * step_d
        plat_len = ct.SECTION_LEN - 2 * 0.8 - 2 * n * step_d
        if x < x0:
            return 0.02
        if x < plat_x0:
            return step_h * min(int((x - x0) / step_d) + 1, n)
        if x < plat_x0 + plat_len:
            return step_h * n
        if x < plat_x0 + plat_len + n * step_d:
            return step_h * (n - min(int((x - plat_x0 - plat_len) / step_d), n - 1))
        return 0.02
    if sec == "slope":
        grade = np.tan(np.deg2rad(9.0))
        half = ct.SECTION_LEN / 2
        up = np.clip((local + half) - 0.8, 0.0, 2.2) * grade
        down = np.clip((half - local) - 0.8, 0.0, 2.2) * grade
        return float(min(up, down, grade * 2.2))
    return 0.0


def ground_z_world(X: float, Y: float) -> float:
    ax, _ = arc_of(X, Y)
    return ground_z_arc(ax)


# ======================================================================================
@dataclass
class Profile:
    """What separates the solution variants."""
    name: str = "balanced"
    max_terrain_rank: int = 2          # 0 grass | 1 +mud | 2 +pebble/mixed | 3 +slope
    probe_terrains: tuple = ("pebble", "mixed")   # allowed only after a stable probe
    # MEASURED capacity limit: the 0.28 x 0.20 x 0.12 m basket cannot carry four
    # competition objects over the 30 m walk to the bins — they crowd each other into the
    # rear third (outside the arm's in-basket reach; servo residual 0.01 m with two
    # objects vs 0.09-0.11 m with four) and some bounce out entirely. Two is the load
    # that stayed retrievable in every probe, and the guide only scores 2 per category.
    max_picks: int = 6          # total objects to handle across all trips
    # MEASURED basket capacity: two competition objects. With four aboard they crowd each
    # other into the rear third (outside the arm's in-basket reach: servo residual 0.01 m
    # with two vs 0.09-0.11 m with four) and some bounce out during the walk. When the
    # basket is full the robot SHUTTLES: deliver, walk back to where it left off, resume.
    max_carry: int = 2
    # Classes the arm can actually get back OUT of the basket. Measured 0826_0226: cans
    # servo GRIPPED on try 1 (residual 0.03 m, gap 0.02 m); the lunchbox MISSED 4/4
    # (residual 0.08-0.16 m) because its body is 110 mm across against a 100 mm gripper
    # opening and the only grip that works on it — the lid-slope shallow pinch — needs an
    # approach geometry that does not exist inside the basket. A rind lies flush with the
    # basket floor, same top-pinch problem. Anything NOT in this set is carried to its bin
    # in the gripper immediately (a solo trip), which is the mode that scored with it before.
    # rind ADDED 0827 (user: "바구니에 4개까지 넣어야지"): its original exclusion
    # ("lies flush with the floor, top-pinch can't get under it") predates the wrist
    # alternation, isolation-first unload, low release and the rescanned window — none
    # of which has ever been tried on a basketed rind. With rind aboard the basket
    # fills to its quota pair (can 2 + rind 2) and the lunchbox rides the hand: the
    # full 4+1 load. The lunchbox stays hand-only — its 110 mm body against the 100 mm
    # finger span is a geometric fact no retrieval trick changes (measured 0/8).
    basket_classes: tuple = ("can", "bottle", "rind")
    max_trips: int = 3
    grasp_steps: int = 320
    grasp_retries: int = 1
    stability_steps: int = 70
    max_roll: float = 0.28
    max_pitch: float = 0.32
    max_height_err: float = 0.09
    vx: float = 0.45          # slower search = more looks per metre of course
    # Objects scatter to +-1.15 m laterally; the old 1.05 amp left the outer 10 cm
    # band uncovered, and the seed-0 audit over 27 runs shows exactly that: the
    # lat -1.07 lunchbox missed 96%, the lat -1.14 rind 82%, while every |lat|<1.0
    # target was engaged. 1.2 covers the whole scatter with 0.4 m to the foot guard.
    zigzag_amp: float = 1.2
    zigzag_period: float = 5.5
    detect_every: int = 3
    approach_timeout: int = 900
    # Squatting shifts the base by a repeatable amount (measured over 6 probe attempts:
    # about +0.04 m forward and -0.21 m lateral), so the approach aims off by that much
    # up front instead of paying a stand-up/re-approach correction on every object.
    # Fixed feed-forward for the settle+crouch shift, applied on EVERY approach.
    # Recalibrated 0826 from 398 mission first-crouch errors: residual with the old
    # (0.04, -0.20) was median (+0.005, +0.052), so the y half was 5 cm overcorrected.
    # The shift is treated as a CONSTANT, not re-measured per grasp: paired analysis of
    # 232 measure-stand-recrouch retries showed the re-measurement chased noise (retry
    # improved |err| only 46% of the time and made x WORSE 81% of the time, median
    # -0.070 -> -0.183, because the stand-up transient lunges forward and the
    # forward-trained walk cannot back away from an object it is already on top of).
    aim_bias0: tuple = (0.05, -0.15)
    basket_retries: int = 3      # confirmed-grip retries before giving up on a payload
    # Classes to ignore entirely. The guide scores 4 categories x 2 objects x 20 points,
    # so leaving the paper bags out caps the run at 120 — which is the right trade when
    # bags are 0/6 and every attempt on one costs an approach, a squat and 320 policy
    # steps that a can or a lunchbox could have used.
    skip_classes: tuple = ()
    # Classes to attempt only when nothing better is on offer, as a {class: rank} map --
    # lower goes first, absent means rank 0. This exists because skip_classes is all or
    # nothing and both settings are wrong for the red paper bag: skipping it forfeits its
    # 40 points outright (it is 하위 과제 4, guide line 260), while treating it as a peer
    # let it eat the run. MEASURED over two Hong Kong runs: 6 of 8 and 4 of 6 targets were
    # red bags, every one of them failed to grasp, and the pick budget was gone before the
    # classes that DO convert had their turn.
    # Empty = today's behaviour, so this changes nothing until it is set.
    class_rank: dict = field(default_factory=dict)
    # How many times one object may be re-approached after a failed grasp. The old code
    # blacklisted an object the moment a grasp failed, so a lunchbox that needs two or
    # three tries was never picked up at all.
    object_attempts: int = 1
    # Extra picks allowed per class beyond the 2 that can score, so an object lost on the
    # way to the bins can be replaced instead of costing the category outright.
    spare_picks: int = 0
    # How many times to sweep the course. The search only ever walks FORWARD, so once it
    # reaches the last object-bearing section it stops — a run that had scored two objects
    # ended there with four still on the ground. Extra passes go back to the start and
    # look again; objects already given up on stay blacklisted.
    max_passes: int = 1
    # Carry the object in the GRIPPER to the bin instead of dropping it in the basket.
    # The basket costs three separate failure modes — the deposit swing, bouncing out
    # over the wall during a 30 m walk (measured repeatedly: an object deposited at
    # base-frame [-0.09, -0.27, 0.10] was on the ground by the time the robot arrived),
    # and the in-basket retrieval that only reaches 54% of the floor. Holding the object
    # removes all three at the cost of one object per trip.
    carry_in_gripper: bool = False
    # a shuttle round trip is ~60 m of walking, so the caps have to allow for several
    search_budget: int = 60000   # hard cap so a stuck robot cannot hang the run
    # THE VENUE CLOCK. 30 minutes at 0.02 s/step (sim.dt 0.005 x decimation 4) = 90000
    # steps, and it does not pause. This used to be dead code — declared here, assigned
    # in the solution file, read nowhere — so nothing stopped a slow run walking past the
    # whistle, and an object still in the gripper at that moment scores zero. `deadline`
    # is now enforced by steps_left()/must_head_home() below.
    total_budget: int = 90000


@dataclass
class Attempt:
    name: str
    cls: str
    section: str
    stable: bool = False
    grasped: bool = False
    deposited: bool = False
    binned: str | None = None
    note: str = ""


@dataclass
class MissionLog:
    attempts: list = field(default_factory=list)
    events: list = field(default_factory=list)
    # Step clock, installed by the runner. Every event is stamped "@<step>" so the
    # 30-minute budget can be audited per phase from report_venue.json alone (stdout
    # is discarded in detached runs, and the 500-step traverse lines were the only
    # timing information the reports carried).
    clock: object = None

    def ev(self, msg: str):
        if self.clock is not None:
            msg = f"@{int(self.clock())} {msg}"
        self.events.append(msg)
        print(f"[MISSION] {msg}", flush=True)


class MissionRunner:
    def __init__(self, env, uenv, cfg, profile: Profile, ckpts: dict, cams: dict,
                 writers: dict, out_dir: Path, device: str):
        self.env, self.uenv, self.cfg, self.p = env, uenv, cfg, profile
        self.device = device
        self.robot = uenv.scene["robot"]
        self.out_dir = out_dir
        self.cams, self.writers = cams, writers
        self.log = MissionLog()

        self.legs = LegDriver(self.robot, device, ckpts["walk"], ckpts["crouch"], ckpts["adapt"])
        self.arm = ArmDriver(self.robot, device)
        self.grasp = GraspPolicy(ckpts["grasp"], device)
        # Optional retry-diversity ensemble: a second checkpoint (a different training
        # seed) fails on DIFFERENT objects, so alternating policies across the retry
        # budget converts objects the primary repeatedly misses. Off unless set.
        self._grasp_pri = self.grasp
        self._grasp_alt = None
        _alt = os.environ.get("MISSION_GRASP_CKPT2")
        if _alt:
            _alt_p = _alt if os.path.isabs(_alt) else str(RW / _alt)
            self._grasp_alt = GraspPolicy(_alt_p, device)
        self.obs_builder = GraspObsBuilder(self.robot, self.arm, device)
        # the head camera neither follows base_link nor reports its own pose — the mount
        # drives it and is the pose the detector unprojects with (see HeadCamMount)
        from course_mission.launch import HEAD_POS, HEAD_ROT
        self.head_mount = HeadCamMount(cams["head"], self.robot, device,
                                       HEAD_POS, HEAD_ROT)
        self.percep = Perception(cams["head"], device, mount=self.head_mount)

        names = list(self.robot.joint_names)
        self.arm_ids = [names.index(j) for j in ARM_JOINTS]
        self.n_act = uenv.action_manager.total_action_dim
        self.act = torch.zeros(1, self.n_act, device=device)

        # pregrasp bank (defines where the policy expects the object to be)
        bank = np.load(ckpts["bank"], allow_pickle=True)
        ee_b = bank["ee_pos_b"]
        keep = ee_b[:, 0] >= 0.50
        self.bank_q = bank["arm_q"][keep]
        self.bank_ee = ee_b[keep]
        self.grasp_anchor = np.median(self.bank_ee, axis=0)   # base-frame sweet spot
        self.log.ev(f"pregrasp bank: {len(self.bank_q)} configs, anchor="
                    f"{np.round(self.grasp_anchor, 3).tolist()}")
        # MISSION_BANK_EE: the bank's gripper_base QUATERNIONS (base frame), dumped from
        # this sim per config (exp_runs/banks/pregrasp_bank_<depth>_ee.npz). The bank
        # stores positions only, and the actor is not sign-invariant in the quaternion
        # (flipping ee_q at a held start turned its approach score from +3.4 to -4.0), so
        # the sign the sim actually produces is what must be scored — analytic FK gets
        # the rotation to 3e-3 but the sign right only 44% of the time. Needed by
        # MISSION_BANK_PICK=policy; the bank pick falls back to "far" without it.
        self.bank_quat = None
        self._pick_used, self._pick_pred = [], None
        # extra hand clearance over the object demanded of later picks in the same squat
        # once a joint ramp was blocked (the hand met the object on the way down)
        self._pick_clear_bump = 0.0
        _ee = os.environ.get("MISSION_BANK_EE", "")
        if _ee:
            _d = np.load(_ee if os.path.isabs(_ee) else str(RW / _ee))
            if len(_d["arm_q"]) != len(bank["arm_q"]) or \
                    np.abs(_d["arm_q"] - bank["arm_q"]).max() > 1e-5:
                raise RuntimeError(f"MISSION_BANK_EE {_ee} was dumped from a different bank")
            self.bank_quat = torch.tensor(_d["ee_quat_b"][keep], dtype=torch.float32,
                                          device=device)
            self.bank_pos_sim = torch.tensor(_d["ee_pos_b"][keep], dtype=torch.float32,
                                             device=device)
            _perr = np.linalg.norm(_d["ee_pos_b"][keep] - self.bank_ee, axis=1)
            self.log.ev(f"pregrasp bank EE dump: {_ee} ({len(self.bank_quat)} configs, "
                        f"sim-vs-bank position max {_perr.max():.4f} m)")
        # MISSION_BANK_RAMP_MASK: which kept configs the joint ramp from the travel pose
        # actually REACHES in the squat with no object anywhere near (scratch ramp_sweep.py:
        # the mission's own 60-step ramp + settle, reachable = every joint within 0.10 rad).
        # The policy-pick probe had ramps ending 0.17-0.45 rad off their config with the
        # fingers pushed half shut while the hand was still 8 cm above the lunchbox — and
        # the no-object sweep reproduces that on the same configs, so it is the arm meeting
        # the robot's own crouched front legs or the ground, not the object. Such a config
        # is never a usable start, so it is dropped from the pick outright.
        self.bank_ramp_ok = None
        _rm = os.environ.get("MISSION_BANK_RAMP_MASK", "")
        if _rm:
            _m = np.load(_rm if os.path.isabs(_rm) else str(RW / _rm))
            if len(_m["ok"]) != len(self.bank_q):
                raise RuntimeError(f"MISSION_BANK_RAMP_MASK {_rm} covers {len(_m['ok'])} configs, "
                                   f"the kept bank has {len(self.bank_q)}")
            self.bank_ramp_ok = torch.tensor(np.asarray(_m["ok"], dtype=bool), device=device)
            self.log.ev(f"pregrasp bank ramp mask: {_rm} ({int(self.bank_ramp_ok.sum())} of "
                        f"{len(self.bank_q)} configs reachable)")

        self.travel_q = torch.tensor([0.0, 1.2, -1.2, 0.0, 0.6, 0.0], device=device)
        # BASKET MODE: start (and travel empty) with the ARM RAISED at the deposit
        # trajectory's staging pose instead of folded. Two reasons: the folded travel_q
        # tucks the forearm across the space the rear basket now occupies, and starting
        # raised means the very first grasp needs no unfold swing next to a loaded
        # basket. q_staging is the pose the deposit trajectory itself validated as clear
        # of both the body and the basket.
        if os.environ.get("MISSION_USE_BASKET") == "1":
            _dep = json.loads((RW / "eunha_rl" / "deposit_traj.json").read_text())
            self.travel_q = torch.tensor([float(x) for x in _dep["q_staging"]],
                                         device=device)
        # travel_q folds the arm tight against the body. That is right for walking empty,
        # but an object held in the gripper gets pressed against the robot and squeezed
        # out: the first carry attempt dropped the rind during stand_up() and "delivered"
        # nothing (payload already OUT at base-frame [1.15, -3.69, -0.49] before the walk
        # even started). While carrying, the arm holds the retrieve trajectory's mid
        # waypoint instead, which is clear of both the body and the basket.
        self.travel_q_empty = self.travel_q.clone()
        self.deposit = json.loads((RW / "eunha_rl" / "deposit_traj.json").read_text())
        rt = Path(__file__).resolve().parent / "retrieve_traj.json"
        self.retrieve = json.loads(rt.read_text()) if rt.exists() else None
        # CARRY POSE. retrieve q_mid holds the object at base-frame z = +0.548 m for the
        # whole 30 m haul. That height was chosen to clear the BASKET, and the basket is
        # gone (MISSION_NO_BASKET=1), so it is now pure cost: a mass 0.55 m above the base
        # raises the CoM and hangs a pendulum off a walk policy that never saw one.
        # scripts/solve_carry.py re-solves it without the basket for a pose that tucks the
        # load low and aft — TCP [-0.023, -0.217, -0.022], i.e. beside the torso at base
        # level where the basket used to sit, tool 23 deg off vertical so a shallow lid
        # pinch still holds, object bottom 0.31 m off the ground.
        ct_p = Path(__file__).resolve().parent / "carry_traj.json"
        self.carry = json.loads(ct_p.read_text()) if ct_p.exists() else None
        # DEFAULT REVERTED TO "mid". The low/aft pose is a REGRESSION, measured: three runs
        # with it scored 0/160 between them and lost EVERY object in transit — TCP gap was
        # a healthy 0.014-0.06 m at pickup, then "unload -> none" every time, with the
        # payload logged sliding back and DOWN out of the grip (trash_10 [-0.27,-0.12,-0.40]
        # DROPPED, landing 30 m short of the bins). The object was not knocked out by a leg,
        # it peeled: 23 deg off vertical is still too much tilt for the shallow pinches this
        # mission relies on. Carrying high is not free — it raises the CoM for the whole haul
        # — but a dropped object scores zero, and that dominates.
        # MISSION_CARRY=low re-enables the low pose for further work on it.
        self.carry_mode = os.environ.get("MISSION_CARRY", "mid")
        if self.carry_mode == "low" and self.carry and self.carry["checks"]["valid"]:
            self.carry_q = list(self.carry["q_carry"])
            self.log.ev(f"carry pose: LOW/aft, TCP {np.round(self.carry['tcp_carry_b'], 3).tolist()}")
        else:
            self.carry_q = list(self.retrieve["q_mid"]) if self.retrieve else None
            self.log.ev("carry pose: retrieve q_mid (high, over the torso)")
        # Per-class carry pose. The lunchbox is held by an 8 mm lid pinch; at q_mid the
        # tool is 77 deg off vertical, so the box hangs cantilevered and its own weight
        # peels the pinch open on the stair climb. carry_shelf.json's q_shelf_above puts
        # the TCP 0.33 m up beside the basket with the tool 13 deg off vertical: gravity
        # loads the pads straight down (the tapered lid wedges tighter, not looser) and a
        # box that does let go lands in the basket. scripts/probe_lbox_carry.py "hang":
        # held through the stairs at 0.24 m/s and released over the bin, every seed.
        # MISSION_CARRY_LUNCHBOX=mid restores the shared pose.
        self.carry_q_cls = {}
        sh_p = Path(__file__).resolve().parent / "carry_shelf.json"
        if os.environ.get("MISSION_CARRY_LUNCHBOX", "hang") == "hang" and sh_p.exists():
            _sh = json.loads(sh_p.read_text())
            self.carry_q_cls["lunchbox"] = [float(v) for v in _sh["q_shelf_above"]]
            self.log.ev("carry pose (lunchbox): HANG at q_shelf_above, TCP "
                        f"{np.round(_sh['tcp_above_b'], 3).tolist()}")

        # object bookkeeping
        self.objects = {n: uenv.scene[n] for n, *_ in cfg.mission_layout}
        self.obj_cls = {n: c for n, c, *_ in cfg.mission_layout}
        self.blacklist = []                 # world xy of attempted spots
        self.terrain_strikes = {}
        self.terrain_ok = {}                # probed terrains that proved stable
        self.carried = []                   # [(name, cls)] currently in the basket
        self._basket_slot = {}              # name -> release-slot twist (deposit_to_basket)
        self._basket_slot_p = {}            # name -> TCP base-frame xyz at release (FK)
        # SHADOW bin-memory estimator (MISSION_BINMEM=1): answers "after wandering
        # and turning through grasp cycles, is the robot's mapless guess of where the
        # bins are still right?" It uses ONLY deployment-honest inputs — body-frame
        # velocity + yaw rate with noise (leg odometry / IMU), and detect_bins fixes
        # taken ROBOT-RELATIVE — and is scored against truth without ever steering.
        self._bm_on = os.environ.get("MISSION_BINMEM") == "1"
        self._bm = {"x": 0.0, "y": 0.0, "yaw": None, "bin": None, "fixes": 0,
                    "gyro_bias": float(np.random.normal(0.0, 0.0017))}  # ~0.1 deg/s
        self._stale_hits = {}               # honest: fruitless re-acquires per object
        self.perma_blacklist = []           # positions the PERSIST re-sweep must NOT
                                            # resurrect (e.g. a bag lying handle-down:
                                            # its pose will not fix itself, and persist
                                            # kept feeding the mission back to it)
        self._bag_down_hits = {}
        self._recalling = False             # walking back to a REMEMBERED object: the
                                            # defer rule below must not re-skip it
        self._hand_item = None              # basket mode: object riding in the GRIPPER
                                            # (a non-basket_classes grasp) — must be
                                            # delivered before anything else is picked
        self.picked_per_cls = {}       # deposited into the basket
        self.binned_ok_per_cls = {}    # actually landed in the right bin
        self.step_i = 0
        self.log.clock = lambda: self.step_i
        # Optional per-step grasp trace sink (list). None = no tracing.
        self.grasp_trace = None
        self.grasp_cam_target = None
        try:
            _bn = self.robot.body_names
            self._foot_idx = [i for i, n in enumerate(_bn) if n.endswith("_foot")]
            self._foot_idx = self._foot_idx or None
        except Exception:
            self._foot_idx = None
        # task 1 (하이킹) + penalty state; see _score_task1
        self.cp_done: set[int] = set()
        self.task1 = 0
        self.oob_count = 0
        self.oob_feet_count = 0
        self._last_strike_lat = 0.0
        self._oob_feet_active = False
        self._oob_active = False
        self.finished = False
        self.finish_step = None
        self.frame_every = 2
        _mt_env = os.environ.get("MISSION_MAX_TRIPS")
        if _mt_env:
            # route-economy tuning: the stock cap of 3 was sized for slow single-item
            # trips; r10's basket batching finished in 17.3 of 30 min and the cap sent
            # it home with 12 min unspent
            profile.max_trips = int(_mt_env)
        _mp_env = os.environ.get("MISSION_MAX_PASSES")
        if _mp_env:
            profile.max_passes = int(_mp_env)
        _br_env = os.environ.get("MISSION_BASKET_RETRIES")
        if _br_env:
            # unload leak (r14-r17 mining): 8 of 10 lost deliveries were basket-
            # retrieval failures with 4-deep loads
            profile.basket_retries = int(_br_env)
        _mp_env2 = os.environ.get("MISSION_MAX_PICKS")
        if _mp_env2 is not None:
            # nav-only hardening runs: 0 disables manipulation so the honest
            # navigation stack can be iterated at ~10 min/run
            profile.max_picks = int(_mp_env2)
        _zz_env = os.environ.get("MISSION_ZIGZAG_AMP")
        if _zz_env:
            # honest nav: est lat err ~0.5 m rides on top of the swing — 1.05 put the
            # feet over the 1.645 foot-guard line and every wreck chain started with
            # exactly that first boundary contact
            profile.zigzag_amp = float(_zz_env)
        _oa_env = os.environ.get("MISSION_OBJ_ATTEMPTS")
        if _oa_env:
            # grasp-rate tuning: with the route economy the mission banks 8-13 spare
            # minutes; a second full cycle per object converts ~15-30 %/attempt into
            # meaningfully more picks
            profile.object_attempts = int(_oa_env)
        _vx_env = os.environ.get("MISSION_VX")
        if _vx_env:
            # deploy tuning: the 0824 pair tracks 0.7 solo; the stock 0.45 search
            # spends the 30-min budget walking
            profile.vx = float(_vx_env)
        self.aim_bias = np.array(profile.aim_bias0, float)  # base shift from settle+crouch
        self.rej = {k: 0 for k in ("raw", "distractor", "skipped", "quota", "blacklist",
                                   "terrain", "corridor", "range", "accepted", "unmatched")}
        self.obj_fails = {}          # name -> failed approach/grasp cycles so far
        self._grasp_out_of_bank = False   # set by run_grasp; see the retry loop in run()
        self._grasp_displaced = False     # ditto: the attempt shifted the object
        self.lat_strikes = 0         # times traverse had to steer off the side edge
        self.off_platform = False    # terminal: robot left the course and cannot return
        self._ns_on = os.environ.get("MISSION_NAVSHADOW") == "1"
        self._ns = {}
        # MISSION_TORQUE_AUDIT=1: per-joint effort/velocity book-keeping against the
        # actuator limits (see _torque_audit_step). Both robot actuator groups are
        # ImplicitActuatorCfg, so PhysX itself clamps the drive at effort_limit_sim and
        # "exceeding" the limit is impossible in the plant; what CAN be over the limit is
        # the PD demand (data.computed_torque, before the clip) — that is the saturation
        # this audit counts, plus the joint speed against joint_vel_limits.
        self._taud = None
        if os.environ.get("MISSION_TORQUE_AUDIT") == "1":
            self._torque_audit_init()
        from course_mission.navest import NavEst
        self.nav = NavEst(self)
        # MISSION_OBJ_HONEST=1: no simulator OBJECT truth on any control path (the
        # navigation half is MISSION_NAV_HONEST). What changes: the grasp policy's
        # target is the camera detection, frozen for the attempt; HELD, carry-loss and
        # let-go verdicts come from finger width + finger effort; the basket pick aims
        # at the slot the robot itself deposited into; landing/payload checks that read
        # the object pose become bookkeeping from the robot's own actions. The sim
        # object handles stay for logging ([TRUTH] lines) and for report()/scoring.
        # Fitted on 1,156 [VERDICT] lines (298 HELD / 858 MISS): gw_end >= 3 mm and
        # effort_end >= 5 recovers 298/298 holds with 33 false positives, all of them
        # with no lift — which is why the LIFT term does the separating in
        # _sensed_hold(). The width ceiling only rules out the physically impossible:
        # hs4b trash_9 was genuinely HELD at gw_end 99.4 mm (a wide object against a
        # ~100 mm span, effort 32), so a 95 mm ceiling would have thrown a real hold
        # away — it sits just under the mechanical limit instead.
        self.obj_honest = os.environ.get("MISSION_OBJ_HONEST", "0") == "1"
        self._hold_gw_lo = float(os.environ.get("MISSION_HOLD_GW_LO", "0.003"))
        self._hold_gw_hi = float(os.environ.get("MISSION_HOLD_GW_HI", "0.0995"))
        self._hold_effort = float(os.environ.get("MISSION_HOLD_EFFORT", "5.0"))
        self._hold_lift = float(os.environ.get("MISSION_HOLD_LIFT", "0.04"))
        self._hold_z_load = None     # TCP z when the fingers first loaded (sensed lift)
        self._grasp_target_S = None  # frozen detection-seeded grasp point (decision frame)
        # Map of everything seen so far: name -> (cls, world xy, step last seen). The
        # search used to be purely reactive — a can noticed while carrying a rind, or
        # passed during the 30 m walk back from the bins, was simply forgotten, and the
        # robot had to stumble across it again. One run swept the whole course three
        # times and only ever considered one of the four cans.
        self.seen = {}
        self._gap = {}               # name -> [ray/ground length - blob depth] (see scan)
        self._h_seen = {}            # name -> [measured heights]  (shape memory)
        self._span_seen = {}         # name -> [footprint spans]
        self._yaw_seen = {}          # name -> last blob long-axis yaw
        self._elong_seen = {}        # name -> last blob elongation
        self._edge_seen = {}         # name -> (near-edge midpoint, view direction)
        self.recall_tries = {}       # name -> how often we have walked back for it
        self._obj_steps = {}         # name -> sim steps spent on failed cycles
        # Objects that have been UNLOADED at the bins, whatever the result. hs1: a
        # lunchbox scored "area" against the blue bin's face was re-seen 45 times, re-
        # targeted twice, and the crouch beside the bin ended with the robot over the
        # deck edge with 40 points still in the basket. deliver_payload cleared
        # `carried` and nothing else remembered the object.
        self.done_names = set()
        # name -> xy of the ground spot blacklisted when it was picked (undone when a
        # payload check finds the object is not aboard after all)
        self._pick_spot = {}
        # Everything the camera has ever reported, pickable or not (S-frame when honest):
        # the crouch-site check and the approach-side choice read it. hs1 F5: the robot
        # walked over a bag + lunchbox + bottle cluster to reach a rind, crouched on
        # top of them and rolled 0.92 rad, ejecting the can it was carrying.
        self._obst = []              # dicts: xy, cls, step, n
        self._allow_pad = False      # scan(): let the bin-rescue see the drop zone
        # On-board compute model (ASUS NUC14MNK15 / Intel N150). The detector runs at the
        # rate that machine can actually sustain, and every leg command is delayed by the
        # sensor-read + bus + inference latency instead of being applied the same tick.
        lim = COMPUTE_LIMITS.get(COMPUTE, COMPUTE_LIMITS["ideal"])
        self.detect_every_eff = max(1, int(round(
            50.0 / float(os.environ.get("MISSION_DETECT_HZ", lim["detect_hz"])))))
        # stop_and_look/reacquire scan MUCH faster than the modelled detector (k % 4 =
        # 12.5 Hz) because in sim a scan costs zero time. When an explicit detector rate
        # is being simulated, honour it in the acquisition phase too — otherwise it
        # quietly enjoys 6x the detections the N150 will produce. Default (env var unset)
        # keeps the historical cadence so scored runs stay comparable.
        self.look_every = (max(4, self.detect_every_eff)
                           if os.environ.get("MISSION_DETECT_HZ") else 4)
        self.action_delay = int(os.environ.get("MISSION_ACTION_DELAY",
                                              lim["action_delay_steps"]))
        self._act_queue = []
        self.log.ev(f"deployment model: compute={COMPUTE} "
                    f"(detector {lim['detect_hz']:.0f} Hz -> every {self.detect_every_eff} "
                    f"steps, action latency {self.action_delay * 20} ms), "
                    f"sensor noise={'ON' if REAL_NOISE else 'OFF'}")
        self.oracle = os.environ.get("MISSION_ORACLE_DETECT") == "1"
        if self.oracle:
            self.log.ev("ORACLE detection enabled (ablation: camera bypassed)")

    # ---------------------------------------------------------------- plumbing
    def _write_arm(self, q6=None, open_=None):
        if q6 is not None:
            self.arm.set_joints(q6)
        if open_ is not None:
            self.arm.set_gripper(open_)
        self.arm.write_gripper()

    def _capture(self):
        if self.step_i % self.frame_every:
            return
        rp = self.robot.data.root_pos_w[0]
        # probes run with the head camera only (no video), so this must not require one
        # Grasp cam: low, off to the side of the OBJECT, looking back at it. Aimed every
        # capture so the gripper-object gap stays in frame; the follow cam cannot show this
        # because the basket on the robot's back is between it and the hand.
        gc = self.cams.get("grasp")
        tgt = getattr(self, "grasp_cam_target", None)
        if gc is not None and tgt is not None:
            op = tgt.data.root_pos_w[0]
            to_rob = rp[:2] - op[:2]
            n = torch.linalg.norm(to_rob).clamp(min=1e-6)
            side = torch.stack([-to_rob[1] / n, to_rob[0] / n])
            eye = torch.stack([op[0] + side[0] * 1.05 + to_rob[0] / n * 0.35,
                               op[1] + side[1] * 1.05 + to_rob[1] / n * 0.35,
                               op[2] + 0.42])
            gc.set_world_poses_from_view(
                eye.unsqueeze(0),
                torch.stack([op[0], op[1], op[2] + 0.06]).unsqueeze(0))
        fc = self.cams.get("follow")
        if fc is not None:
            fc.set_world_poses_from_view(
                (rp + torch.tensor([-2.6, -2.6, 1.8], device=self.device)).unsqueeze(0),
                (rp + torch.tensor([0.0, 0.0, 0.25], device=self.device)).unsqueeze(0))
        for key, w in self.writers.items():
            # A writer may be a SINK attached to an existing camera rather than a camera of
            # its own (probe_manip --trace_dir pairs frames with per-step grasp values).
            # Resolve through cam_of when the key is not itself a camera.
            cam = self.cams.get(key) or self.cams.get(getattr(w, "cam_key", ""))
            if cam is None:
                continue
            w.append_data(cam.data.output["rgb"][0, ..., :3].cpu().numpy())

    def _binmem_step(self):
        bm = self._bm
        d = self.robot.data
        X, Y, _, yaw_true = self.base_pose_true()
        if bm["yaw"] is None:                       # anchor odom at the start pose
            bm["x"], bm["y"], bm["yaw"] = X, Y, yaw_true
            self.log.ev("[BINMEM] shadow estimator ON (odom anchored)")
            return
        # the CONTROL dt, read from the env (a hardcoded 0.02 put a 22 deg yaw jump
        # into one fast turn — a few % of dt mismatch scales every revolution)
        if "dt" not in bm:
            _ue = self.env.unwrapped
            bm["dt"] = float(getattr(_ue, "step_dt", None)
                             or _ue.physics_dt * getattr(_ue.cfg, "decimation", 4))
            self.log.ev(f"[BINMEM] control dt = {bm['dt']:.4f} s")
        dt = bm["dt"]
        v = d.root_lin_vel_b[0].cpu().numpy()
        # HEADING: a real AHRS integrates the gyro onboard at ~1 kHz, so its
        # short-horizon yaw DELTA is essentially exact and the long-horizon error is
        # bias-shaped. Integrating world wz at this loop's 50 Hz aliased the grasp
        # reaction wobble into a 22 deg jump in one turn — an artifact of the shadow,
        # not of any deployable sensor. Model the AHRS honestly: true per-step yaw
        # delta plus bias plus noise.
        _yt = self.base_pose_true()[3]
        _dyaw = (_yt - bm.get("prev_yaw_true", _yt) + np.pi) % (2 * np.pi) - np.pi
        bm["prev_yaw_true"] = _yt
        # leg-odometry realism: 3 % velocity scale noise, gyro bias + noise
        vx = float(v[0]) * (1.0 + np.random.normal(0, 0.03))
        vy = float(v[1]) * (1.0 + np.random.normal(0, 0.03))
        # outlier clamp, as any deployed odometry has: b40 integrated one physics
        # velocity spike into a 388 m position jump in <1000 steps. The robot never
        # exceeds ~1 m/s; hold the previous sample instead of eating a spike.
        if abs(vx) > 2.5 or abs(vy) > 2.5:
            vx, vy = bm.get("v_prev", (0.0, 0.0))
        bm["v_prev"] = (vx, vy)
        bm["yaw"] += _dyaw + (bm["gyro_bias"] + np.random.normal(0, 0.003)) * dt
        c, s_ = np.cos(bm["yaw"]), np.sin(bm["yaw"])
        bm["x"] += (c * vx - s_ * vy) * dt
        bm["y"] += (s_ * vx + c * vy) * dt
        # 100-step cadence (was 250): the head cam's search pitch keeps the bins in
        # frame only inside ~1-4 m, a band the approach crosses in a few seconds —
        # 250 steps sampled it 1-2 times per bin visit
        if self.step_i % 100 == 0:
            try:
                cd = self.percep.cam.data
                if self.percep.mount is not None:
                    cam_p, cam_R = self.percep.mount.pos_w, self.percep.mount.R_ros
                else:
                    cam_p = cd.pos_w[0].cpu().numpy()
                    cam_R = cd.quat_w_ros[0].cpu().numpy()
                gz = ground_z_arc(arc_of(X, Y)[0])
                hit = detect_bins(cd.output["rgb"][0, ..., :3].cpu().numpy(),
                                  cd.intrinsic_matrices[0].cpu().numpy(),
                                  cam_p, cam_R, gz)
            except Exception as e:
                hit = None
                if not bm.get("err_logged"):
                    bm["err_logged"] = True
                    self.log.ev(f"[BINMEM] detect error (once): {type(e).__name__} {e}")
            if hit is not None:
                centre_w = np.asarray(hit["centre_w"], float)[:2]
                # take the measurement ROBOT-RELATIVE (what a real camera gives),
                # then place it in the ODOM frame — never store the true position
                rel = centre_w - np.array([X, Y])
                rb = np.array([np.cos(-yaw_true) * rel[0] - np.sin(-yaw_true) * rel[1],
                               np.sin(-yaw_true) * rel[0] + np.cos(-yaw_true) * rel[1]])
                c, s_ = np.cos(bm["yaw"]), np.sin(bm["yaw"])
                meas = (bm["x"] + c * rb[0] - s_ * rb[1],
                        bm["y"] + s_ * rb[0] + c * rb[1])
                corr = (float(np.hypot(bm["bin"][0] - meas[0], bm["bin"][1] - meas[1]))
                        if bm["bin"] is not None else float("nan"))
                bm["bin"] = meas
                bm["fixes"] += 1
                self.log.ev(f"[BINMEM] fix #{bm['fixes']} at step {self.step_i}: "
                            f"range {float(np.hypot(*rb)):.1f} m, belief correction "
                            f"{corr:.2f} m, n_seen {hit['n_seen']}")
        if self.step_i % 1000 == 0:
            _oderr = float(np.hypot(bm["x"] - X, bm["y"] - Y))
            _yerr = float(np.degrees((bm["yaw"] - yaw_true + np.pi) % (2 * np.pi)
                                     - np.pi))
            self.log.ev(f"[BINMEM] step {self.step_i}: odom drift {_oderr:.2f} m, "
                        f"yaw drift {_yerr:+.1f} deg")
        if bm["bin"] is not None and self.step_i % 1000 == 0:
            bx, blat = bin_targets()["blue"]
            BX, BY = ct.curve_map(bx, 0.0)
            err = float(np.hypot(bm["bin"][0] - float(BX), bm["bin"][1] - float(BY)))
            brg_est = np.degrees(np.arctan2(bm["bin"][1] - Y, bm["bin"][0] - X))
            brg_true = np.degrees(np.arctan2(float(BY) - Y, float(BX) - X))
            dbrg = (brg_est - brg_true + 180) % 360 - 180
            # nav err = belief vs truth IN THE ROBOT FRAME — the steering error a
            # return trip would actually experience (odom-frame err conflates anchor
            # drift the robot never sees; right after a fix nav err ~ measurement err)
            _c, _s = np.cos(-bm["yaw"]), np.sin(-bm["yaw"])
            _dx, _dy = bm["bin"][0] - bm["x"], bm["bin"][1] - bm["y"]
            _re = (_c * _dx - _s * _dy, _s * _dx + _c * _dy)
            _ct, _st = np.cos(-yaw_true), np.sin(-yaw_true)
            _tx, _ty = float(BX) - X, float(BY) - Y
            _rt = (_ct * _tx - _st * _ty, _st * _tx + _ct * _ty)
            _nav = float(np.hypot(_re[0] - _rt[0], _re[1] - _rt[1]))
            self.log.ev(f"[BINMEM] step {self.step_i}: bin-est err {err:.2f} m, "
                        f"nav err {_nav:.2f} m, bearing err {dbrg:+.1f} deg, "
                        f"fixes {bm['fixes']}")

    def _navshadow_step(self):
        ns = self._ns
        d = self.robot.data
        X, Y, _, yaw_true = self.base_pose_true()
        ax, w_true = arc_of(X, Y)
        th = float(ct.curve_theta(ax))
        psi_true = (th - yaw_true + np.pi) % (2 * np.pi) - np.pi
        if ns.get("fuser") is None:
            from course_mission.navfuse import BeltOdomFuser
            ns["fuser"] = BeltOdomFuser(lat0=0.0, psi0=0.0)
            self.log.ev("[NAVSHADOW] belt+odom fuser ON (shadow, no steering)")
            _ue = self.env.unwrapped
            ns["dt"] = float(getattr(_ue, "step_dt", None)
                             or _ue.physics_dt * getattr(_ue.cfg, "decimation", 4))
        v = d.root_lin_vel_b[0].cpu().numpy()
        wz = float(d.root_ang_vel_b[0, 2].cpu())
        vx = float(v[0]) * (1.0 + np.random.normal(0, 0.03))
        vy = float(v[1]) * (1.0 + np.random.normal(0, 0.03))
        if abs(vx) > 2.5 or abs(vy) > 2.5:
            vx, vy = ns.get("v_prev", (0.0, 0.0))
        ns["v_prev"] = (vx, vy)
        ns["fuser"].predict(vx, vy, wz + np.random.normal(0, 0.01), ns["dt"])
        if self.step_i % 100 == 0:
            try:
                from course_mission.perception import detect_belt_hybrid
                cd = self.percep.cam.data
                if self.percep.mount is not None:
                    cam_p, cam_R = self.percep.mount.pos_w, self.percep.mount.R_ros
                else:
                    cam_p = cd.pos_w[0].cpu().numpy()
                    cam_R = cd.quat_w_ros[0].cpu().numpy()
                hit = detect_belt_hybrid(
                    cd.output["rgb"][0, ..., :3].cpu().numpy(),
                    cd.intrinsic_matrices[0].cpu().numpy(),
                    cam_p, cam_R, ground_z_arc(ax))
                if hit is not None:
                    ns["fuser"].update(hit.get("lat"), hit.get("heading"))
                    ns["fixes"] = ns.get("fixes", 0) + 1
            except Exception as e:
                if not ns.get("err_logged"):
                    ns["err_logged"] = True
                    self.log.ev(f"[NAVSHADOW] detect error (once): "
                                f"{type(e).__name__} {e}")
        if self.step_i % 1000 == 0:
            _ew = ns["fuser"].lat - w_true
            _ep = (ns["fuser"].heading - psi_true + np.pi) % (2 * np.pi) - np.pi
            self.log.ev(f"[NAVSHADOW] step {self.step_i}: lat err {_ew:+.2f} m, "
                        f"psi err {np.degrees(_ep):+.1f} deg, "
                        f"fixes {ns.get('fixes', 0)}")

    # ---- effort / velocity audit (MISSION_TORQUE_AUDIT=1) --------------------------
    def _torque_audit_init(self):
        d = self.robot.data
        n = len(self.robot.joint_names)
        lim_t = d.joint_effort_limits[0].clone()
        lim_v = d.joint_vel_limits[0].clone()
        self._taud = dict(
            lim_t=lim_t, lim_v=lim_v,
            max_c=torch.zeros(n, device=self.device),   # peak |computed| (PD demand)
            max_a=torch.zeros(n, device=self.device),   # peak |applied|  (after clip)
            max_v=torch.zeros(n, device=self.device),   # peak |joint_vel|
            sat_t=torch.zeros(n, device=self.device),   # steps with |computed| >= 0.95 lim
            sat_v=torch.zeros(n, device=self.device),   # steps with |vel| >= 0.95 lim
            at_c=torch.zeros(n, dtype=torch.long, device=self.device),  # step of max_c
            at_v=torch.zeros(n, dtype=torch.long, device=self.device),  # step of max_v
            over_steps=torch.zeros(n, device=self.device),  # steps with |computed| > lim
            over_v=torch.zeros(n, device=self.device),      # steps with |vel| > lim
            events=[],      # first over-limit bursts: (step, legs mode, joint, demand)
            vevents=[],     # same for the VELOCITY rating
            burst_until=-1, vburst_until=-1,
            n=0, next_print=5000,
        )
        self.log.ev("[TORQUE AUDIT] on: limits " + ", ".join(
            f"{nm}={float(t):.0f}Nm/{float(v):.1f}rad/s"
            for nm, t, v in zip(self.robot.joint_names, lim_t, lim_v)))
        # WHAT THE SOLVER ACTUALLY ENFORCES. `velocity_limit_sim` is written through to
        # PhysX as the joint's maxJointVelocity (Articulation.write_joint_velocity_limit
        # _to_sim -> set_dof_max_velocities), so the achieved speed should never pass it
        # — and the audit has been recording calf peaks of 20 rad/s against a 14 rad/s
        # limit. Print what PhysX holds so the difference between "we command too fast"
        # and "a contact impulse back-drove the joint" is settled from data, not assumed.
        try:
            _phys = self.robot.root_physx_view.get_dof_max_velocities()[0]
            _bad = [f"{nm} cfg {float(v):.1f} vs physx {float(p):.1f}"
                    for nm, v, p in zip(self.robot.joint_names, lim_v, _phys)
                    if abs(float(p) - float(v)) > 0.05]
            self.log.ev("[TORQUE AUDIT] PhysX max joint velocity "
                        + ("matches the actuator config on every joint"
                           if not _bad else "DIFFERS: " + ", ".join(_bad)))
        except Exception as e:
            self.log.ev(f"[TORQUE AUDIT] could not read the PhysX velocity limits: "
                        f"{type(e).__name__} {e}")

    def _torque_audit_step(self):
        a = self._taud
        try:
            d = self.robot.data
            c = d.computed_torque[0].abs()
            ap = d.applied_torque[0].abs()
            v = d.joint_vel[0].abs()
            a["at_c"] = torch.where(c > a["max_c"], self.step_i, a["at_c"])
            a["at_v"] = torch.where(v > a["max_v"], self.step_i, a["at_v"])
            a["max_c"] = torch.maximum(a["max_c"], c)
            a["max_a"] = torch.maximum(a["max_a"], ap)
            a["max_v"] = torch.maximum(a["max_v"], v)
            a["sat_t"] += (c >= 0.95 * a["lim_t"]).float()
            a["sat_v"] += (v >= 0.95 * a["lim_v"]).float()
            # OVER the rating, not merely near it. The rating is the hardware's, so a
            # step above it is a step the real machine could not have taken — recorded
            # per joint, with the leg mode of the worst one, so the report can say
            # plainly whether the run stayed inside the envelope.
            _vo = v > a["lim_v"]
            a["over_v"] += _vo.float()
            if bool(_vo.any()) and self.step_i > a["vburst_until"] and len(a["vevents"]) < 40:
                a["vburst_until"] = self.step_i + 100
                _jv = int(torch.argmax(v / a["lim_v"]))
                a["vevents"].append((self.step_i, self.legs.mode,
                                     self.robot.joint_names[_jv], round(float(v[_jv]), 1)))
            over = c > a["lim_t"]
            a["over_steps"] += over.float()
            a["n"] += 1
            # keep the first 40 over-limit bursts (one entry per burst, 100-step
            # debounce) with the leg mode so saturation can be matched to a phase
            if bool(over.any()) and self.step_i > a["burst_until"] and len(a["events"]) < 40:
                a["burst_until"] = self.step_i + 100
                j = int(torch.argmax(c / a["lim_t"]))
                a["events"].append((self.step_i, self.legs.mode, self.robot.joint_names[j],
                                    round(float(c[j]), 1)))
            if self.step_i >= a["next_print"]:
                a["next_print"] += 5000
                self.log.ev("[TORQUE AUDIT] " + self._torque_audit_summary())
        except Exception as e:
            if not a.get("err"):
                a["err"] = True
                self.log.ev(f"[TORQUE AUDIT] disabled after error: {type(e).__name__} {e}")
                self._taud = None

    def _torque_audit_rows(self):
        a = self._taud
        n = max(1, a["n"])
        rows = []
        for i, nm in enumerate(self.robot.joint_names):
            rows.append(dict(
                joint=nm, effort_limit=float(a["lim_t"][i]), vel_limit=float(a["lim_v"][i]),
                peak_computed=float(a["max_c"][i]), peak_applied=float(a["max_a"][i]),
                peak_vel=float(a["max_v"][i]),
                peak_computed_step=int(a["at_c"][i]), peak_vel_step=int(a["at_v"][i]),
                over_limit=bool(a["max_c"][i] > a["lim_t"][i]),
                over_limit_pct=100.0 * float(a["over_steps"][i]) / n,
                sat_effort_pct=100.0 * float(a["sat_t"][i]) / n,
                sat_vel_pct=100.0 * float(a["sat_v"][i]) / n,
                over_vel=bool(a["max_v"][i] > a["lim_v"][i]),
                over_vel_pct=100.0 * float(a["over_v"][i]) / n))
        return rows

    def _torque_audit_summary(self):
        rows = self._torque_audit_rows()
        worst = sorted(rows, key=lambda r: -r["peak_computed"] / max(1e-6, r["effort_limit"]))[:4]
        over = [r["joint"] for r in rows if r["over_limit"]]
        _ov = [f"{r['joint']} {r['peak_vel']:.1f}/{r['vel_limit']:.1f} "
               f"({r['over_vel_pct']:.3f}%)" for r in rows if r["over_vel"]]
        return (("VELOCITY OVER RATING: " + "; ".join(_ov) + " | " if _ov
                 else "velocity within rating on every joint | ")
                + f"step {self.step_i}: PD demand over limit on {len(over)} joints "
                f"{over}; worst " + "; ".join(
                    f"{r['joint']} {r['peak_computed']:.0f}/{r['effort_limit']:.0f}Nm "
                    f"@{r['peak_computed_step']} (applied {r['peak_applied']:.0f}, "
                    f"over {r['over_limit_pct']:.2f}%, vel {r['peak_vel']:.1f}/"
                    f"{r['vel_limit']:.0f} @{r['peak_vel_step']})" for r in worst)
                + (f"; bursts {self._taud['events'][-6:]}" if self._taud["events"] else ""))

    def _torque_audit_write(self):
        if self._taud is None:
            return
        try:
            self.log.ev("[TORQUE AUDIT] final " + self._torque_audit_summary())
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "torque_audit.json").write_text(json.dumps(dict(
                steps_sampled=self._taud["n"],
                note=("sampled once per env step (50 Hz, last physics substep); "
                      "computed = PD demand before the actuator clip, applied = after; "
                      "ImplicitActuator -> PhysX clamps the drive at effort_limit_sim"),
                over_limit_bursts=[dict(step=s, legs_mode=m, joint=j, demand=d)
                                   for s, m, j, d in self._taud["events"]],
                joints=self._torque_audit_rows()), indent=2))
        except Exception as e:
            print(f"[MISSION] torque audit write failed: {e}", flush=True)

    def _step(self, leg_action):
        self.head_mount.sync()
        try:
            self.nav.step()
        except Exception as e:
            if not getattr(self, "_nav_err_logged", False):
                self._nav_err_logged = True
                self.log.ev(f"[NAVEST] step error (once): {type(e).__name__} {e}")
        if getattr(self, "_ns_on", False):
            try:
                self._navshadow_step()
            except Exception:
                pass
        if getattr(self, "_hand_item", None) is not None and self.step_i % 200 == 0:
            try:
                _ho = self._sim_obj(self._hand_item).data.root_pos_w[0]
                _gap = float((_ho - self.tcp_w()[0]).norm())
                # a held item sits 0.04-0.07 m from the TCP (the lunchbox lid pinch is
                # the widest). svB's lunchbox slipped on the stairs to a gap of 0.40 —
                # under the old 0.45 — rode the rest of the way ON THE ROBOT'S BACK,
                # and the bin stop "unloaded" an empty hand for 1,100 steps.
                # THE FINGERS ARE THE AUTHORITY, in every mode. The truth gap test is
                # blind to the one way the hang carry actually fails: the lunchbox
                # slips out and lands in the BASKET, which the carry pose hangs
                # directly over — hs11a @25000-@26884 read "gap 0.09 m" all the way to
                # the bin, swung an empty hand over the opening, and scored the box
                # `none` while it sat in the basket the whole time. Width + effort
                # cannot be fooled that way, and they are what the robot has.
                # Two samples 200 steps apart so a stair jolt cannot write cargo off.
                _gw = float(self.arm.gripper_width()[0, 0])
                _ge = self.arm.grip_effort()
                _empty = _gw < self._hold_gw_lo or (_ge is not None
                                                    and _ge < 0.4 * self._hold_effort)
                self._carry_empty_n = (getattr(self, "_carry_empty_n", 0) + 1) if _empty else 0
                _sensed_lost = self._carry_empty_n >= 2
                if self.obj_honest:
                    _lost = _sensed_lost
                    _why = (f"width {_gw*1000:.1f} mm effort "
                            f"{-1.0 if _ge is None else _ge:.1f} (truth gap {_gap:.2f} m)")
                else:
                    _lost = _sensed_lost or _gap > 0.20
                    _why = (f"gap {_gap:.2f} m, width {_gw*1000:.1f} mm effort "
                            f"{-1.0 if _ge is None else _ge:.1f}")
                if _lost and not getattr(self, "_carry_lost_logged", False):
                    self._carry_lost_logged = True
                    self._carry_lost = self._hand_item
                    self.log.ev(f"  CARRY LOST {self._hand_item}: {_why} at "
                                f"step {self.step_i} (obj z "
                                f"{float(_ho[2]):.2f})")
                    # Write the loss off IMMEDIATELY. The old flow only aborted when
                    # the lost item was the SOLE cargo, so with basket items aboard
                    # (s5600d) the watch saw gap 3.83 m and the robot still walked
                    # 20 m and "unloaded" an empty hand into the blue bin. Dropping
                    # it from the books here keeps the trip for the remaining cargo,
                    # skips the air-unload, and leaves the object recallable later.
                    _lostn = self._hand_item
                    # durable record: _carry_lost is cleared by _goto_bin_body as soon
                    # as the name leaves `carried` (which happens on the next line), and
                    # unload_one needs to know at the BIN that this one emptied out of
                    # the hand — that is what makes the basket worth a look.
                    self._lost_in_hand = getattr(self, "_lost_in_hand", set()) | {_lostn}
                    self.carried = [(n, c) for n, c in self.carried if n != _lostn]
                    self._hand_item = None
                    if getattr(self, "travel_q_empty", None) is not None:
                        self.travel_q = self.travel_q_empty.clone()
                elif self.step_i % 1000 == 0:
                    self.log.ev(f"  carry-watch {self._hand_item}: gap {_gap:.2f} m")
            except Exception:
                pass
        if self.action_delay > 0:
            self._act_queue.append(leg_action.clone())
            if len(self._act_queue) > self.action_delay:
                leg_action = self._act_queue.pop(0)
            else:
                leg_action = torch.zeros_like(leg_action)
        self.act[:, :12] = leg_action
        self.env.step(self.act)
        self.step_i += 1
        self._score_task1()
        if self._bm_on:
            self._binmem_step()
        if self._taud is not None:
            self._torque_audit_step()
        # Trace EVERY step once a target is attached, not just the grasp loop. The bag is
        # knocked over somewhere between the approach starting and the grasp policy's
        # fourth step, and a grasp-only trace cannot say which -- on mud it is already flat
        # (90.2 deg) before the policy runs at all.
        if self.grasp_trace is not None and self.grasp_cam_target is not None:
            self.grasp_trace.append(self._trace_row(
                getattr(self, "_trace_name", "?"), self.grasp_cam_target,
                float((self.grasp_cam_target.data.root_pos_w[0]
                       - self.tcp_w()[0]).norm()),
                getattr(self, "_last_a7", None),
                getattr(self, "_trace_rest_z", 0.0)))
        self._capture()

    # ------------------------------------------------------------ task 1 + penalties
    def _score_task1(self):
        """Hiking checkpoints and out-of-bounds penalties, evaluated once per sim step.

        Both are things the venue judges continuously and the code did not track at all --
        report() emitted a task-2 total with no task-1 block and no penalty term, so the
        number it printed was a gross score, not the one the ranking uses.

        Checkpoints are one-shot (guide 239: 각 하위 구간은 1회만 채점됩니다), which matters
        because the robot shuttles up and down the course many times per run and a
        high-water-mark on arc would re-award every pass.

        Out-of-bounds is judged on where the SUPPORTING SURFACE touches the ground, not on
        body overhang (guide 133), so this tests the base's ground-projected lateral offset
        against the strip line and not the arm or the basket. Each fresh excursion is -10
        with no cap (guide 392); it is counted per CROSSING, with a hysteresis band so a
        robot sitting on the line does not bleed a point every step.
        """
        if not ct.CHECKPOINTS and ct.FINISH_ARC is None:
            return
        if self.step_i % 5:                      # 10 Hz is ample for a 0.02 s step
            return
        x, y, z, _ = self.base_pose_true()
        ax, lat = arc_of(x, y)
        # A checkpoint or the finish is only earned ON THE COURSE. Scoring on arc alone
        # awarded svU (seed 0) the finish line at lat +29 while it was walking on the
        # ground BELOW the platform after falling off the deck at a bag; the venue judge
        # would have scored nothing past the fall. Same truth the fall check in
        # _reenter_course uses (this is self-scoring, not navigation).
        on_course = abs(lat) < ct.BOUNDARY_Y and z > ground_z_arc(ax) - 0.25
        # THE RUN IS OVER WHEN THE ROBOT IS BELOW THE DECK. Only _reenter_course
        # noticed a fall, and only when a routine happened to call it: svW (seed 0)
        # went over the side at lat -1.98 chasing a mud can, was noticed 1,900 steps
        # later at lat -19, and kept "steering back to the centreline" from lat -30 —
        # 59k steps of budget to burn for nothing. A venue marshal stops the run at
        # the fall; so does this (traverse/shuttle/finish all honour off_platform).
        # Honest navigation gets the attitude test instead of the surveyed deck
        # height, so no prior enters the honest path through this flag.
        if not self.off_platform:
            if self.nav.honest:
                _below = float(self.robot.data.projected_gravity_b[0, 2].cpu()) > -0.55
            else:
                _below = z < ground_z_arc(ax) - 0.25
            if _below:
                self._below_deck = getattr(self, "_below_deck", 0) + 1
                if self._below_deck >= 20:                  # 2 s, not a stair bounce
                    self.off_platform = True
                    self.log.ev(f"FALLEN OFF THE COURSE at arc {ax:.1f} lat {lat:+.2f} "
                                f"(base z {z:.2f} vs deck "
                                f"{ground_z_arc(ax):.2f}) — the run is over")
            else:
                self._below_deck = 0
        for i, (cp_arc, pts, label) in enumerate(ct.CHECKPOINTS):
            if i not in self.cp_done and ax >= cp_arc and on_course:
                self.cp_done.add(i)
                self.task1 += pts
                self.log.ev(f"checkpoint {label}: +{pts} (task1 {self.task1}/{ct.TASK1_MAX})")
        # THE VENUE JUDGES FEET, NOT THE BODY. Guide 131: out of bounds is where the
        # SUPPORTING SURFACE touches the ground, and body/arm/gripper overhang explicitly
        # does not count. This test uses the base centre, and the A2's feet stand about
        # 0.203 m outboard of it (URDF: hip abduction 0.0751 + thigh 0.1278), so the run
        # can report a clean sheet while an outside foot is already over the line -- at
        # -10 points per crossing with no cap. Reported alongside rather than instead of
        # the body number so the 120-run baseline stays comparable.
        out_feet = abs(lat) + STANCE_HALF_WIDTH >= ct.BOUNDARY_Y
        if out_feet and not self._oob_feet_active:
            self._oob_feet_active = True
            self.oob_feet_count += 1
            self.log.ev(f"OUT OF BOUNDS (FEET) #{self.oob_feet_count} at arc {ax:.1f} "
                        f"lat {lat:+.2f} (foot edge {abs(lat) + STANCE_HALF_WIDTH:.2f} "
                        f"vs line {ct.BOUNDARY_Y:.2f})")
        elif self._oob_feet_active and abs(lat) + STANCE_HALF_WIDTH < ct.BOUNDARY_Y - 0.10:
            self._oob_feet_active = False
        out = abs(lat) >= ct.BOUNDARY_Y
        if out and not self._oob_active:
            self._oob_active = True
            self.oob_count += 1
            self.log.ev(f"OUT OF BOUNDS #{self.oob_count} at arc {ax:.1f} lat {lat:+.2f} "
                        f"(line +-{ct.BOUNDARY_Y:.2f}) -10")
        elif self._oob_active and abs(lat) < ct.BOUNDARY_Y - 0.10:
            self._oob_active = False
        if ct.FINISH_ARC is not None and not self.finished and ax >= ct.FINISH_ARC \
                and on_course:
            self.finished = True
            self.finish_step = self.step_i
            self.log.ev(f"FINISH LINE crossed at step {self.step_i} "
                        f"({self.step_i * 0.02 / 60:.1f} min)")

    # ------------------------------------------------------------------ pose access
    # TWO KINDS OF POSE. `*_true` is the simulator's world truth: the judge, the O-object
    # bookkeeping (memory checks, landing classification) and the sim-only basket
    # checks use it. The bare names are what the ROBOT decides with: identical to the
    # truth under the oracle, but the S-frame estimate (est arc, est lat, root z,
    # heading-vs-tangent) under MISSION_NAV_HONEST=1 — see navest.py. Every position
    # the robot stores while honest (detections, memory, bins, blacklists) is in that
    # same frame, so `arc_of`/`section_of`/`curve_map` must NOT be applied to it; the
    # `_arc_lat`/`_section`/`_tan_yaw`/`_curve_map` helpers below do the right thing in
    # both modes.
    def base_pose_true(self):
        d = self.robot.data
        p = d.root_pos_w[0]
        yaw = math_utils.euler_xyz_from_quat(d.root_quat_w)[2][0]
        return float(p[0]), float(p[1]), float(p[2]), float(yaw)

    def to_base_frame_true(self, pos_w: np.ndarray) -> np.ndarray:
        d = self.robot.data
        v = torch.tensor(pos_w, dtype=torch.float32, device=self.device).reshape(1, 3) - d.root_pos_w
        return math_utils.quat_apply_inverse(d.root_quat_w, v)[0].cpu().numpy()

    def base_pose(self):
        """Robot-side base pose: (x, y, z, yaw) truth under the oracle; (s, w, z, yaw_S)
        when honest."""
        if self.nav.honest:
            return self.nav.pose()
        return self.base_pose_true()

    def to_base_frame(self, pos: np.ndarray) -> np.ndarray:
        """Robot-side frame change: a stored position (world / S) -> body frame."""
        if self.nav.honest:
            return self.nav.to_base(np.asarray(pos, np.float64))
        return self.to_base_frame_true(pos)

    def of_base_frame(self, pos_b: np.ndarray) -> np.ndarray:
        """Inverse of `to_base_frame`: body frame -> the decision frame (S / world)."""
        if self.nav.honest:
            return self.nav.of_base(np.asarray(pos_b, np.float64))
        d = self.robot.data
        v = torch.tensor(np.asarray(pos_b, np.float64), dtype=torch.float32,
                         device=self.device).reshape(1, 3)
        return (math_utils.quat_apply(d.root_quat_w, v) + d.root_pos_w)[0].cpu().numpy()

    # OBJECT-HONEST grasp geometry. The detector reports where the blob's bottom
    # edge meets the ground — the near CONTACT line of the footprint, not the
    # centre. The centre sits half the object's depth further along the line of
    # sight; MISSION_DET_CENTRE_<CLS> overrides the catalogue half-depth once the
    # [TRUTH] calibration lines say otherwise.
    def _det_centre_offset(self, cls: str, det: dict) -> float:
        ov = os.environ.get(f"MISSION_DET_CENTRE_{cls.upper()}")
        if ov:
            return float(ov)
        dims = OBJ_DIMS.get(cls, (0.10, 0.10, 0.10))
        # MEASURED, not derived. The geometric argument says "add half the footprint's
        # depth to the contact line", but the detector's point is not the contact line:
        # it is the ray through the mean of the blob's bottom FIFTH of rows, which sits
        # above the true bottom edge and therefore already lands inside the footprint —
        # the taller the object, the further in. Calibrated against the objects' own
        # poses on the object-honest probe (oracle navigation, so the only error left is
        # the detector's), first attempt per object, from the [TRUTH] lines of
        # exp_runs/logs/probe_ohB.log:
        #   lunchbox  0.055 applied -> +0.025 / +0.011 too far  => 0.037
        #   can       0.033 applied -> +0.030 / +0.017 too far  => 0.010
        #   rind      0.025 applied -> -0.004 / -0.022 short    => 0.035
        # The can is worst because it is the tallest: the bottom fifth of its blob is a
        # full 2 cm up the cylinder. Re-fit (re-run the probe and read the [TRUTH]
        # lines) if the camera mount, the blob percentile or the crouch height changes.
        # No bag entry on purpose. The bags DO land ~0.21 m past the anchor on every
        # probe trial, but that is not a detector bias to cancel here: 0.12 m of it is
        # the deliberate cage-lift standoff in approach() and the rest is the ordinary
        # approach residual. Fixing it there is the honest place; a constant here would
        # only bend the obj-honest path, which the bags do not use.
        return {"lunchbox": 0.037, "can": 0.010, "rind": 0.035}.get(
            cls, 0.5 * min(dims[0], dims[1]))

    def _det_centre_S(self, cls: str, det: dict) -> np.ndarray:
        """The object's CENTRE in the decision frame, from one detection.

        Preferred form: the detector's near-edge midpoint (`edge_w`, the silhouette's
        bottom profile dropped on the ground — unbiased in both axes) plus the class's
        support half-width along the line of sight, which for a box of half-extents
        a (long) and b (short) seen at angle t off its long axis is a|cos t| + b|sin t|.
        Checked offline against a synthetic 150x110 box at 1.6 m: 4 mm, against 12 mm
        for the bottom-fifth point the old form starts from (25-30 mm on real blobs).
        Falls back to `pos_w` + the calibrated per-class constant when the detection
        carries no edge (a remembered pick, or a degenerate silhouette).
        """
        e, v = det.get("edge_w"), det.get("view_w")
        if e is not None and v is not None:
            e = np.asarray(e, float)
            v = np.asarray(v, float)
            # The FOOTPRINT is whichever two extents are not the vertical one, and the
            # vertical one is the measured height: a can on its side stands 66 mm tall
            # with a 115x66 footprint, standing it is 115 tall on 66x66. Taking dims[0]
            # and dims[1] assumes the nominal attitude and is 2.5 cm wrong for exactly
            # the case the height measurement exists to catch.
            dims = list(OBJ_DIMS.get(cls, (0.10, 0.10, 0.10)))
            _h = self._det_top_height(cls, det)
            _iv = int(np.argmin([abs(d - _h) for d in dims]))
            foot = [d for i, d in enumerate(dims) if i != _iv]
            a, b = max(foot) / 2, min(foot) / 2
            yaw = float(det.get("yaw_ref", det.get("yaw", np.nan)) or np.nan)
            el = float(det.get("elong", 1.0) or 1.0)
            if np.isfinite(yaw) and el > 1.3:
                t = yaw - float(np.arctan2(v[1], v[0]))
                half = a * abs(np.cos(t)) + b * abs(np.sin(t))
            else:
                half = 0.5 * (a + b)
            # CALIBRATED FORWARD PULL-BACK (0909). Mined from 136 grasp-start
            # [est-truth] base-frame lines across fv36-43: the centre this model
            # produces sits a systematically POSITIVE x (forward) off the article —
            # can +66 mm median (IQR +23..+108), rind +39 (+7..+85), y already
            # centred (+14/-4). Whatever mix of near-edge bias and half-depth
            # overshoot causes it, it is a stable per-class constant at grasp range,
            # and subtracting it along the line of sight recentres the whole error
            # distribution onto the object — the single largest known gap between
            # the eval's 12 mm world (17% policy) and the mission's (~1%/attempt).
            _fb = {"can": 0.028, "rind": 0.064}
            _fbov = os.environ.get(f"MISSION_DET_FWD_BIAS_{cls.upper()}")
            _pull = float(_fbov) if _fbov else _fb.get(cls, 0.0)
            half = half - _pull
            return np.array([e[0] + half * v[0], e[1] + half * v[1], e[2]], float)
        b0 = self.to_base_frame(np.asarray(det["pos_w"], float))
        cen = self._det_centre_offset(cls, det)
        _fb = {"can": 0.028, "rind": 0.064}
        _fbov = os.environ.get(f"MISSION_DET_FWD_BIAS_{cls.upper()}")
        cen = cen - (float(_fbov) if _fbov else _fb.get(cls, 0.0))
        u = b0[:2] / max(float(np.linalg.norm(b0[:2])), 1e-6)
        return self.of_base_frame(np.array([b0[0] + cen * u[0],
                                            b0[1] + cen * u[1], b0[2]], float))

    def _det_top_height(self, cls: str, det: dict) -> float:
        """Resting top-of-object height above the ground it stands on.

        MEASURED first (perception's ray-through-the-blob-top height), catalogue only
        as the fallback. Which way a can came to rest changes its top by 5 cm and the
        catalogue cannot know it; the measurement is snapped to the nearer of the
        class's two possible attitudes so a noisy pixel row cannot invent a height.
        """
        ov = os.environ.get(f"MISSION_DET_TOP_{cls.upper()}")
        if ov:
            return float(ov)
        dims = OBJ_DIMS.get(cls, (0.10, 0.10, 0.10))
        if cls == "lunchbox":
            cands = [TOP_ABOVE_ORIGIN.get("lunchbox", 0.065)]
        elif cls == "can":
            cands = [dims[2], dims[0]]            # standing 0.115 / on its side 0.066
        else:
            cands = [dims[2], min(dims[0], dims[1])]
        h = float(det.get("h_est", np.nan) or np.nan)
        if np.isfinite(h) and len(cands) > 1:
            return min(cands, key=lambda c: abs(c - h))
        if cls == "can" and not np.isfinite(h):
            # no height measurement: the footprint's own size still separates the two
            span = float(det.get("span", 0.0) or 0.0)
            return dims[2] if span < 0.09 else dims[0]
        return cands[0]

    def _sensed_load(self):
        """(loaded, width, effort): the fingers hold something — width inside the
        band a real object leaves between them, with the gripper working against it."""
        gw = float(self.arm.gripper_width()[0, 0])
        ge = self.arm.grip_effort()
        loaded = (self._hold_gw_lo <= gw <= self._hold_gw_hi
                  and (ge is None or ge >= self._hold_effort))
        return loaded, gw, ge

    def _sensed_hold(self, closed_cmd: bool) -> bool:
        """HELD without the simulator: fingers commanded shut AND loaded AND the TCP
        has risen `_hold_lift` since they first loaded (a pinch on the ground, or on
        the lid of something still resting, never lifts)."""
        if not closed_cmd:
            self._hold_z_load = None
            return False
        loaded, _, _ = self._sensed_load()
        if not loaded:
            self._hold_z_load = None
            return False
        z = float(self.arm.frame_pose()[0][0, 2])
        if self._hold_z_load is None:
            self._hold_z_load = z
            return False
        return z - self._hold_z_load >= self._hold_lift

    def _arc_lat(self):
        """(arc, lat) of the robot for decisions: est when honest."""
        return self.nav.arc_lat()

    def _arc_of(self, p):
        """(arc, lat) of a stored decision-frame position: identity when honest (the
        S-frame IS (arc, lat)), the course map under the oracle."""
        p = np.asarray(p, np.float64)
        if self.nav.honest:
            return float(p[0]), float(p[1])
        return arc_of(float(p[0]), float(p[1]))

    def _on_stairs_at(self, ax: float) -> bool:
        """Is arc `ax` on the stairs? Surveyed under the oracle; honest mode has no
        map, so only the robot's OWN place is known (IMU pitch) — for a remote arc it
        answers 'unknown' as False."""
        if self.nav.honest:
            return False
        return section_of(ax) == "stairs"

    def _S_of_true(self, p_w):
        """A TRUE world point expressed in the robot's decision frame (S when honest;
        identity under the oracle). Sim bookkeeping for the O-object memory and the
        blacklists, which the runner builds from spawned poses."""
        p_w = np.asarray(p_w, np.float64)
        if self.nav.honest:
            return self.nav.S_of_true(p_w)
        return p_w

    def _tan_yaw(self, ax: float) -> float:
        """Course tangent heading at arc `ax` in the decision frame: the S-frame's x
        axis IS the tangent, so 0 when honest."""
        if self.nav.honest:
            return 0.0
        return float(ct.curve_theta(ax))

    def _curve_map(self, ax: float, w: float):
        """(arc, lat) -> decision-frame xy: identity when honest."""
        if self.nav.honest:
            return float(ax), float(w)
        X, Y = ct.curve_map(ax, w)
        return float(X), float(Y)

    def _section(self, ax: float) -> str:
        """Section label for the robot's own arc: the surveyed table under the oracle,
        a plain 6 m pseudo-segment ('seg0'..) when honest (the rules give the section
        LENGTH, not what each one is). Terrain identity while honest comes from the
        IMU (nav.on_stairs / nav.rough), never from this label."""
        if self.nav.honest:
            k = int(np.floor((float(ax) - self.nav.s_start()) / ct.SECTION_LEN))
            return f"seg{max(k, 0)}"
        return section_of(ax)

    def _ground_z(self) -> float:
        """Local ground height under the robot in the decision frame."""
        return self.nav.ground_z()

    def _stance_height(self) -> float:
        """Base height above the ground the robot is standing on. Oracle: the
        surveyed height map; honest: leg kinematics (base minus the lowest foot —
        what the joint encoders give), no map."""
        X, Y, Z, _ = self.base_pose_true()
        if not self.nav.honest or self._foot_idx is None:
            return float(Z - ground_z_world(X, Y))
        fz = self.robot.data.body_pos_w[0, self._foot_idx, 2].cpu().numpy()
        return float(Z - float(np.min(fz)))

    def _bins_arc(self) -> float:
        """Arc of the bin row for planning: surveyed (oracle) / sighted (honest); the
        honest fallback before any sighting is the rules' nominal course length."""
        b = self.nav.bins_arc()
        if b is not None:
            return float(b)
        from course_mission.navest import COURSE_LEN_NOMINAL
        return self.nav.s_start() + COURSE_LEN_NOMINAL - 3.0

    def _bin_park_arc(self) -> float:
        """BIN_PARK_ARC in the decision frame (the surveyed constant under the oracle)."""
        if self.nav.honest:
            return self._bins_arc() - 0.56
        return float(BIN_PARK_ARC)

    def _course_end(self) -> float:
        """Arc of the far edge of the course for planning bounds: surveyed under the
        oracle; the rules' nominal length past the start line when honest."""
        if self.nav.honest:
            from course_mission.navest import COURSE_LEN_NOMINAL
            # the true arc length of a curved course exceeds its nominal length (the
            # S-course: 39.6 m for 36), so once the bins are sighted they set the end:
            # the deck runs 3 m past the bin row
            b = self.nav.bins_arc()
            end = self.nav.s_start() + COURSE_LEN_NOMINAL
            return max(end, float(b) + 3.0) if b is not None else end
        return float(ct.COURSE_X0 + ct.COURSE_LEN)

    # ---------------------------------------------------------------- locomotion
    def _reenter_course(self, max_steps: int = 700):
        """Walk back inside the belt after a leaving-course abandon.

        Without this every abandon handed the NEXT routine a start point further out
        (r4: lat -2.8 -> -6.8 -> -10.8 -> -14.8 across four abandons) and one r3
        excursion reached lat +63 before the step budget ended it."""
        for _ in range(max_steps):
            _, w = self.nav.arc_lat()
            # fall check: oracle mode compares base z to the surveyed ground; honest
            # mode uses what a real robot has — its attitude (a body tipped past
            # ~55 deg is not walking anywhere)
            if self.nav.honest:
                _fallen = float(self.robot.data.projected_gravity_b[0, 2].cpu()) > -0.55
            else:
                _fallen = float(self.robot.data.root_pos_w[0, 2].cpu()) < \
                    ground_z_arc(arc_of(*self.base_pose_true()[:2])[0]) - 0.25
            if _fallen:
                # TERMINAL, like the lat-strike cap in traverse(): both callers ignore
                # this verdict and the main loop went straight back to searching —
                # svU (seed 0, 120/160 banked) fell off the deck orbiting a bag at
                # step 53.7k and then "recentred" to lat +22 and kept walking in the
                # void, on course to burn the remaining 36k steps (2 h of GPU time)
                # for a score that could no longer change.
                self.off_platform = True
                self.log.ev("  fallen below the platform — re-entry impossible; "
                            "ending the search and the run")
                return False
            if abs(w) < 1.2:
                self.log.ev(f"  re-entered the course at lat {w:+.2f}")
                return True
            cmd, _, _ = self.walk_cmd(0.0, vx=0.35)
            self._write_arm(self.travel_q, open_=False)
            self._step(self.legs.step(cmd))
        self.log.ev("  could not re-enter the course within the budget")
        return False

    def face_forward(self, steps: int = 400, dir_sign: float = 1.0) -> bool:
        steps = int(steps * float(os.environ.get("MISSION_TURN_SCALE", "1")))
        """Turn IN PLACE until the robot points along the course again.

        goto_arc walks backwards by facing the reversed course tangent, so the robot
        arrives at the resume point pointing the wrong way. walk_cmd does correct heading
        while driving, but at a 0.7 rad/s yaw cap and 0.6 m/s forward it travels further
        backwards than it turns: one run drifted arc -15.3 -> -17.1 over 14000 steps,
        printing "stalled near arc ...; pushing through" the whole way. Turning with no
        forward command converges in a couple of seconds.
        """
        _tol = 0.35 if self.nav.honest else 0.25   # honest psi carries a residual bias
        for k in range(steps):
            err = self.nav.psi(dir_sign)
            if abs(err) < _tol:
                self.log.ev(f"  facing forward again after {k} steps "
                            f"(heading error {np.degrees(err):+.0f} deg)")
                return True
            cmd = torch.zeros(1, 3, device=self.device)
            cmd[0, 2] = float(np.clip(1.5 * err, -0.7, 0.7))
            self._write_arm(self.travel_q, open_=False)
            self._step(self.legs.step(cmd))
        self.log.ev("  could not face forward within the turn budget")
        return False

    def _nose_inboard(self) -> bool:
        """True when the nose points toward the centreline, i.e. a reverse gait walks
        the base OUTBOARD (toward the platform edge)."""
        ax, w = self.nav.arc_lat()
        _in = self._tan_yaw(ax) + (np.pi / 2 if w < 0 else -np.pi / 2)
        _, _, _, _yaw = self.base_pose()
        _e = (_in - _yaw + np.pi) % (2 * np.pi) - np.pi
        return abs(_e) < np.pi / 2

    def _walk_inboard(self, lim: float, max_steps: int = 600) -> None:
        """Bring the base to |lat| < lim WITHOUT turning in place.

        A spin sweeps the feet ~0.3 m around the base, so at |lat| 1.7 they cross the
        1.98 platform line: svK slid off turning at 1.99, svM at 1.78 while _pan_look
        turned for a re-acquire after a look-back had already stopped at 1.69. Walk
        straight instead — forward when the nose points inboard, backward when it
        points outboard — with only a gentle steer, until the base is inside lim;
        then a turn is safe. MISSION_EDGE_TURN (1.45) is the |lat| the callers use."""
        ax, w = self.nav.arc_lat()
        if abs(w) < lim:
            return
        t0, w0 = self.step_i, w
        _w_ref, _k_ref, _vy = w, 0, 0.0
        for k in range(max_steps):
            ax, w = self.nav.arc_lat()
            if abs(w) < lim:
                break
            _in = self._tan_yaw(ax) + (np.pi / 2 if w < 0 else -np.pi / 2)
            _, _, _, _yaw = self.base_pose()
            _e = (_in - _yaw + np.pi) % (2 * np.pi) - np.pi      # nose vs inboard
            _fwd = abs(_e) < np.pi / 2
            _es = _e if _fwd else (_e - np.pi + np.pi) % (2 * np.pi) - np.pi
            if k - _k_ref >= 150:
                if abs(abs(w) - abs(_w_ref)) < 0.03:
                    # STALLED. swF: reversing off the mud edge with the nose outboard,
                    # the basket was snagged on a boundary post and two 600-step
                    # calls moved the base lat +1.82 -> +1.83 -> +1.82 before the
                    # robot slid off. Side-step along the belt to slide off whatever
                    # holds the base, alternating sides each stalled window.
                    _vy = 0.20 if _vy <= 0 else -0.20
                    self.log.ev(f"  walk-inboard stalled at lat {w:+.2f} for 150 steps — "
                                f"side-stepping {'left' if _vy > 0 else 'right'} to clear "
                                f"whatever holds the base")
                else:
                    _vy = 0.0
                _w_ref, _k_ref = w, k
            c = torch.zeros(1, 3, device=self.device)
            # Drive only once the body actually points (in)board. With the nose 70 deg
            # off, 0.30 m/s puts 0.28 of it ALONG the edge and 0.10 inboard, and the
            # old 0.3 rad/s clip needed 4.1 s to swing round — longer than the margin
            # the falls had left. Turn first at the policy's own rate, then walk.
            _face = float(os.environ.get("MISSION_INBOARD_FACE", "1.0"))
            c[0, 0] = (0.30 if _fwd else -0.30) if abs(_es) < _face else 0.0
            c[0, 1] = _vy
            c[0, 2] = float(np.clip(0.8 * _es, -0.7, 0.7))
            self._write_arm(self.travel_q, open_=False)
            self._step(self.legs.step(c))
        self.log.ev(f"  walked inboard without turning: lat {w0:+.2f} -> {w:+.2f} "
                    f"({self.step_i - t0} steps)")

    def _recentre_after_pick(self, lim: float | None = None, goal: float = 0.6) -> None:
        """Turn in place toward the centreline and walk in BEFORE the search resumes
        from a pick cycle that ended near the edge.

        svK: the grass lunchbox (lat -1.07) was given up with the robot standing at
        about lat -1.2, traverse walked its first 100 steps at 0.55 on the stale
        heading to lat -1.65, the edge reflex then turned in place with the feet
        already AT the platform edge (foot 1.99 vs line 1.98) and the robot slid
        off (lat -2.9 -> -9.3, can + rind aboard, run lost). Turning here, at
        |lat| ~1.2, keeps the stance inside the platform, and the walk-in uses the
        normal steering with a low speed so the arc stays shallow. If the cycle
        ended beyond MISSION_EDGE_TURN the robot first walks straight in
        (_walk_inboard) — the turn itself is what tips it off the platform."""
        _lim = float(os.environ.get("MISSION_RECENTRE_LAT", "0.9")) if lim is None else lim
        if _lim <= 0:
            return
        ax, w = self.nav.arc_lat()
        if abs(w) < _lim:
            return
        t0 = self.step_i
        self._walk_inboard(float(os.environ.get("MISSION_EDGE_TURN", "1.45")))
        ax, _ = self.nav.arc_lat()
        _th = self._tan_yaw(ax)
        _want = _th + (0.7 if w < 0 else -0.7)     # ~40 deg toward the centreline
        for _ in range(300):
            _, _, _, _yaw = self.base_pose()
            _e = (_want - _yaw + np.pi) % (2 * np.pi) - np.pi
            if abs(_e) < 0.3:
                break
            c = torch.zeros(1, 3, device=self.device)
            c[0, 2] = float(np.clip(1.5 * _e, -0.6, 0.6))
            self._write_arm(self.travel_q, open_=False)
            self._step(self.legs.step(c))
        for _ in range(500):
            _, w2 = self.nav.arc_lat()
            if abs(w2) < goal:
                break
            c, _, _ = self.walk_cmd(0.0, vx=0.35)
            self._write_arm(self.travel_q, open_=False)
            self._step(self.legs.step(c))
        ax2, w2 = self.nav.arc_lat()
        self.log.ev(f"recentred after the pick: lat {w:+.2f} -> {w2:+.2f} at arc "
                    f"{ax2:.1f} ({self.step_i - t0} steps)")

    def walk_cmd(self, lat_target: float, dir_sign: float = 1.0, vx: float | None = None):
        # NAV: (arc, lat, heading-err) come from NavEst — the oracle passthrough by
        # default, the belt+odom estimator under MISSION_NAV_HONEST=1 (실전형: no
        # course map, no surveyed tangent).
        ax, w = self.nav.arc_lat()
        lat_err = w - lat_target
        _lg = float(os.environ.get("MISSION_LAT_GAIN", "1"))
        psi = self.nav.psi(dir_sign)          # + means "turn left to align"
        # fix famine -> caution: with no recent belt fix the fused lat/psi are
        # dead-reckoned; a confident pull on a stale estimate is how the first
        # honest run left the course in 80 s. Halve the pull and slow down.
        _sf = getattr(self.nav, "since_fix", 0) if self.nav.honest else 0
        _starve = self.nav.honest and _sf > 400
        if self.nav.honest and _sf > 1200 and os.environ.get("NAV_MINIMAL") != "1":
            # BUILT-IN STARVATION SCAN. Every honest wreck shared one shape: the
            # heading went bad, the belt left the frame, updates stopped, and the
            # fuser's own re-seed could never fire because re-seeding needs a
            # measurement. When the walk itself has been blind for 24 s, stop
            # walking and turn in place until the belt is back — wherever we are.
            cmd = torch.zeros(1, 3, device=self.device)
            cmd[0, 0] = 0.05
            cmd[0, 2] = 0.40 if (self.step_i // 900) % 2 == 0 else -0.40
            ax, w = self.nav.arc_lat()
            return cmd, ax, w
        if _starve:
            lat_err = lat_err * 0.5
        err = (psi - np.clip(0.8 * _lg * lat_err * dir_sign, -0.7, 0.7)
               + np.pi) % (2 * np.pi) - np.pi
        cmd = torch.zeros(1, 3, device=self.device)
        _v = self.p.vx if vx is None else vx
        # SLOW WHILE HAND-CARRYING. A solidly-verified lunchbox grip (gw 73 mm,
        # effort 35, TCP gap 0.056 at pickup) still shook out within 2 m of trot on
        # the way to the bins (basket10: found at arc 6.5, 8.8 m short of the bin).
        # Hand-carry classes are exactly the fragile-grip ones, so cap the walk while
        # one is aboard; ~9 m at 0.45 costs ~8 s against a 20-point object.
        if getattr(self, "_hand_item", None) is not None and _v > 0 \
                and not getattr(self, "_bin_final", False):
            # _bin_final: goto_bin's approach needs 0.35-0.65 to actually arrive — the
            # walk policy marches in place below ~0.3 m/s, and a 0.24 carry cap left
            # the robot 0.3 m short of the blue bin until the budget died (s5600m,
            # lunchbox demoted to the apron for 5 instead of 20).
            _v = min(_v, float(os.environ.get("MISSION_CARRY_VX", "0.45")))
        # SLOW THE LOADED BASKET ON THE STAIRS. Measured (s5600b): a deposited rind
        # rode ~20 m of flat course fine and bounced OUT on the stair crossing (found
        # at arc 7.7, section stairs — "lost in transit"). The basket is open-top;
        # stair impacts are the one place the payload gets launched.
        # JOINT SPEED IS A HARDWARE LIMIT, and the stairs are where it is broken. The
        # drive itself never exceeds the rating — PhysX holds maxJointVelocity, and the
        # thigh joints sit exactly at their 22.0 rad/s ceiling — but a hard footfall
        # BACK-DRIVES a calf past its 14.67 rad/s rating: measured over the 105-point
        # run, FL_calf peaked at 20.1 (+37%) and the calves spent 0.08-1.08% of the run
        # within 5% of the ceiling. Impact speed is what sets that peak, so cap the
        # crossing whether or not anything is aboard. MISSION_STAIR_VX=0 restores the
        # old behaviour (0.55 m/s empty).
        if _v > 0 and self.nav.on_stairs():
            _sv = float(os.environ.get("MISSION_STAIR_VX", "0.35"))
            if _sv > 0.0:
                _v = min(_v, _sv)
        if getattr(self, "carried", None) and _v > 0 and self.nav.on_stairs():
            _v = min(_v, float(os.environ.get("MISSION_STAIR_CARRY_VX", "0.28")))
            # the hand-held lunchbox (6 mm lid pinch, 0.55 m up on a straight arm) is
            # the payload the stair bounce shakes out (svB, @25000-26000). Separate cap
            # so it can go slower than the basket without stalling the empty-hand climb.
            if getattr(self, "_hand_item", None) is not None:
                _v = min(_v, float(os.environ.get("MISSION_STAIR_HAND_VX", "0.28")))
        if _starve and _v > 0:
            _v = min(_v, 0.35)
        # THE DECK EDGE IS THE MOST EXPENSIVE THING ON THE COURSE. A fall ends the run
        # and forfeits everything aboard: hs5d went over at step 11,165 with three
        # objects in the basket (60 points), hs4d and hs4e the same way later on. Every
        # one of those falls has the same shape — the robot is already outside the rope
        # (|lat| 1.7-1.9, the estimate ACCURATE) and keeps marching forward at 0.5 m/s
        # while it slowly steers in, and a foot finds the edge first. Forward speed is
        # what converts "outside" into "over": going inboard is the only motion that
        # helps, so past MISSION_EDGE_SAFE cap the march and widen the lateral command
        # while the heading term steers in. MISSION_EDGE_HOLD releases the cap after a
        # while so a lying estimate cannot pin the mission against an edge that is not
        # there.
        # ...and DON'T DRIVE BROADSIDE. cmd[0,0] is body-forward, so while the nose is
        # still 60-130 deg off where it wants to go, "walking forward" is walking
        # sideways: hs5d entered the bins search pointing 132 deg off and drove 0.5 m/s
        # for the 3.2 s the yaw loop needed, sweeping the base 0.87 m outboard and over
        # the edge. Scaling the drive by how well the body already points at its own
        # heading target costs nothing when aligned (err ~ 0, factor 1) — a deliberate
        # 40 deg inboard crab is ALIGNED, because err is measured against that target,
        # not against the tangent. The floor keeps the gait alive while it turns.
        _align = float(np.cos(err))
        _v *= max(float(os.environ.get("MISSION_PSI_VX_FLOOR", "0.15")), _align) \
            if _v > 0 else 1.0
        _lat_clamp = 0.3
        _over = abs(w) - float(os.environ.get("MISSION_EDGE_SAFE",
                                              os.environ.get("MISSION_EDGE_LAT", "1.45")))
        if _over > 0.0:
            self._edge_hold = getattr(self, "_edge_hold", 0) + 1
            _lat_clamp = 0.45
            # NOT a full stop: this walking policy marches in place below ~0.3 m/s, so
            # commanding 0 would pin the robot AT the edge instead of bringing it in.
            # The heading term already aims up to 40 deg inboard; halving the forward
            # speed doubles the time that has to work in, which is what the falls were
            # short of (hs5d had 0.34 m of margin and 250 steps).
            _cap = 0.30 if _over < 0.25 else 0.25
            if self._edge_hold > int(os.environ.get("MISSION_EDGE_HOLD", "600")):
                _cap = 0.35                  # the estimate may be lying — press on
            _v = min(_v, _cap) if _v > 0 else max(_v, -_cap)
            if self._edge_hold % 100 == 1:
                self.log.ev(f"  edge guard: lat {w:+.2f} is {_over:.2f} m past the safe "
                            f"line — forward capped at {_cap:.2f} m/s while it comes in")
        else:
            self._edge_hold = 0
        cmd[0, 0] = _v
        cmd[0, 1] = float(np.clip(-0.5 * _lg * lat_err * dir_sign, -_lat_clamp, _lat_clamp))
        cmd[0, 2] = float(np.clip(1.2 * err, -0.7, 0.7))
        return cmd, ax, w

    def crouch_cmd(self, c: float):
        cmd = torch.zeros(1, 3, device=self.device)
        cmd[0, 0] = c
        return cmd

    # ---------------------------------------------------------------- perception
    def scan(self):
        if self.oracle:
            dets = self._oracle_detect()
        else:
            try:
                if self.nav.honest:
                    # ground = own stance height below the base; the detector's
                    # extrinsics come from NavEst (mount.use_est), so every pos_w
                    # below is an S-frame (est) point
                    dets = self.percep.detect(self._ground_z())
                else:
                    X, Y, _, _ = self.base_pose_true()
                    dets = self.percep.detect(ground_z_world(X, Y))
            except Exception as e:                   # a bad frame must not kill the run
                self.log.ev(f"scan error: {e}")
                return []
        out = []
        self.rej["raw"] += len(dets)
        # the drop zone is unpickable ground once the bins are in sight: whatever lies
        # there was put there (by us or a rival) and the pad is where hs1 died
        _pad_arc = None
        if self.nav.honest and not self._allow_pad and self.nav.bins_arc() is not None:
            _pad_arc = float(self.nav.bins_arc()) - 1.5
        for d in dets:
            self._remember_obstacle(d)
            if BIN_OF.get(d["cls"]) is None:
                self.rej["distractor"] += 1
                continue                              # distractor: never grasp
            if d["cls"] in self.p.skip_classes:
                self.rej["skipped"] += 1
                continue                              # deliberately out of scope
            if _pad_arc is not None and float(d["pos_w"][0]) > _pad_arc:
                self.rej["pad"] = self.rej.get("pad", 0) + 1
                continue
            # Guide: only 2 per category score. But the quota used to count objects put
            # in the BASKET, so a can that bounced out on the walk to the bins still
            # burned one of its two slots and the robot would not pick a replacement.
            # Stop when two of the class have actually been binned correctly, and
            # otherwise allow a couple of spare attempts.
            if self.binned_ok_per_cls.get(d["cls"], 0) >= MAX_SCORED_PER_CLASS or \
                    self.picked_per_cls.get(d["cls"], 0) >= MAX_SCORED_PER_CLASS + self.p.spare_picks:
                self.rej["quota"] += 1
                continue
            p = d["pos_w"]
            if any(np.hypot(p[0] - b[0], p[1] - b[1]) < 0.30
                   for b in self.blacklist + self.perma_blacklist):
                self.rej["blacklist"] += 1
                continue
            ax, w = self._arc_of(p)
            sec = self._section(ax)
            if not self.terrain_allowed(sec):
                self.rej["terrain"] += 1
                continue
            if self.nav.honest and ax < self.nav.s_start() + 2.0:
                self.rej["corridor"] += 1     # start-cliff margin under est error
                continue
            if abs(w) > float(os.environ.get("MISSION_CORRIDOR", "1.35")):
                # red boundary strip / off course. Tightenable (deploy 1.10): orbiting
                # a lat-1.3 target swings the base past the 1.98 line — most 0824-run
                # wrecks started exactly there
                self.rej["corridor"] += 1
                continue
            d.update(arc=ax, lat=w, section=sec)
            nm, _dd = self.match_object(d)
            if nm is not None:
                # MULTI-VIEW CONFIRMATION replaces the simulator's false-positive
                # filter. The old match_object discarded any blob with no true article
                # within 0.60 m — 30 % of detections in the 120-point run — and that is
                # not available at a venue. The honest substitute is the robot's own
                # evidence: a track seen in only one frame is as likely to be a texture
                # artefact as an article, and dh0's first sweep shows the cost of
                # believing them (three lunchbox tracks, two of them 1.10 and 1.62 m
                # from any real lunchbox). Remembering costs nothing, so a young track
                # is still tracked and still accumulates views — it just cannot be
                # engaged, which is where the ~4,000-step approach cycle is spent.
                _tv = len(getattr(self, "_tracks", {}).get(nm, {}).get("views", []) or [1])
                d["views"] = _tv
                d["name"] = nm
                if self.det_honest() and _tv < int(
                        os.environ.get("MISSION_TRACK_MIN_VIEWS", "2")):
                    self.seen.pop(nm, None)
                    self.rej["young"] = self.rej.get("young", 0) + 1
                    continue
                # Written off by NAME, not by position. A failed grasp bats the object
                # out of the 0.30 m positional blacklist, so it came back as a fresh
                # candidate over and over: one run logged "giving up on trash_12" after
                # 4, then 5, then 6 cycles, burning a full approach each time.
                if self.obj_fails.get(nm, 0) >= self._attempts_for(self.obj_cls.get(nm)):
                    self.rej["blacklist"] += 1
                    continue
                if nm in self.done_names:            # already delivered (see __init__)
                    self.rej["done"] = self.rej.get("done", 0) + 1
                    continue
                self.seen[nm] = (d["cls"], np.asarray(d["pos_w"], float)[:2], self.step_i)
                # ray/ground length minus the blob's median depth: ~0 for an object
                # on the ground, +0.25 for one propped up (the perception assumes
                # ground contact). Logged per object at each survey station so the
                # truth-based perch gate can be replaced by this deployable signal.
                if "ray_t" in d:
                    self._gap.setdefault(nm, []).append(float(d["ray_t"] - d["dist"]))
                # shape memory, so a remembered pick still knows how tall the object
                # looked and which way its long axis ran (see _pick_from_memory)
                if np.isfinite(d.get("h_est", np.nan) or np.nan):
                    self._h_seen.setdefault(nm, []).append(float(d["h_est"]))
                if float(d.get("span", 0.0) or 0.0) > 0.0:
                    self._span_seen.setdefault(nm, []).append(float(d["span"]))
                if np.isfinite(d.get("yaw", np.nan) or np.nan):
                    self._yaw_seen[nm] = float(d["yaw"])
                    self._elong_seen[nm] = float(d.get("elong", 1.0) or 1.0)
                if d.get("edge_w") is not None:
                    self._edge_seen[nm] = (np.asarray(d["edge_w"], float),
                                           np.asarray(d["view_w"], float),
                                           np.asarray(d["pos_w"], float))
                if self._bag_deferred(d["cls"]):
                    # remembered (recall can fetch it once the core is done), not served
                    self.rej["bag_wait"] = self.rej.get("bag_wait", 0) + 1
                    continue
                out.append(d)
            else:
                self.rej["unmatched"] += 1
        self.rej["accepted"] += len(out)
        return out

    # ------------------------------------------------------------ obstacle memory
    OBST_MERGE = 0.40        # m: same class within this = the same thing
    OBST_MAX_AGE = 9000      # steps: older sightings are stale (drift, or it was picked)
    OBST_MAX_RANGE = 3.0     # m: farther sightings scatter 0.3-0.5 m (hs2: one can became
                             # "5 remembered objects" from the survey stations)
    OBST_MIN_N = 2           # sightings before it counts (7 % of hs1's pickable-class
                             # detections matched no object at all — single-frame ghosts)

    def _remember_obstacle(self, d):
        """Every plausible detection, whatever its class, in the decision frame."""
        try:
            p = np.asarray(d["pos_w"], float)[:2]
        except Exception:
            return
        if not np.all(np.isfinite(p)):
            return
        if float(d.get("dist", 0.0)) > self.OBST_MAX_RANGE:
            return
        if abs(self._arc_of(p)[1]) > self._edge_lat() + 0.3:
            return                                   # beyond the rope: not a crouch site
        cls = d.get("cls")
        for o in self._obst:
            if o["cls"] == cls and float(np.hypot(*(o["xy"] - p))) < self.OBST_MERGE:
                # running mean, later views weigh more (the robot is closer by then)
                o["xy"] = 0.7 * p + 0.3 * o["xy"] if o["n"] >= 2 else 0.5 * (o["xy"] + p)
                o["step"] = self.step_i
                o["n"] += 1
                return
        self._obst.append({"xy": p.copy(), "cls": cls, "step": self.step_i, "n": 1})
        if len(self._obst) > 400:
            del self._obst[:100]

    def obstacles_near(self, xy, radius: float, exclude_xy=None, exclude_r: float = 0.45):
        """Remembered detections within `radius` of xy (decision frame), minus the
        target itself (anything within exclude_r of exclude_xy). Fresh, repeatedly
        seen ones only."""
        xy = np.asarray(xy, float)[:2]
        out = []
        for o in self._obst:
            if self.step_i - o["step"] > self.OBST_MAX_AGE or o["n"] < self.OBST_MIN_N:
                continue
            if exclude_xy is not None and \
                    float(np.hypot(*(o["xy"] - np.asarray(exclude_xy, float)[:2]))) < exclude_r:
                continue
            dd = float(np.hypot(*(o["xy"] - xy)))
            if dd < radius:
                out.append((dd, o))
        out.sort(key=lambda t: t[0])
        return [o for _, o in out]

    def _forget_obstacle(self, xy, cls=None, r: float = 0.35):
        """The thing at xy is gone (picked up): drop its memory so the next target
        beside it is not refused a crouch site over a ghost."""
        xy = np.asarray(xy, float)[:2]
        self._obst = [o for o in self._obst
                      if (cls is not None and o["cls"] != cls)
                      or float(np.hypot(*(o["xy"] - xy))) > r]

    # the crouch footprint in the base frame: feet + belly with a small margin (the
    # A2 stands ~0.6 x 0.3; hs2's first cut also covered a metre of walk-in behind
    # the body and detoured on nearly every target)
    FOOT_FRONT, FOOT_BACK, FOOT_HALF = 0.40, 0.50, 0.25

    def _site_clearance(self, target_w, aim, theta, obst) -> float:
        """How far the nearest remembered obstacle sits outside the crouch footprint
        when the robot approaches `target_w` along heading `theta` and stops with the
        object at base-frame `aim`. 0 = something is under the body."""
        u = np.array([np.cos(theta), np.sin(theta)])
        n = np.array([-u[1], u[0]])
        c = np.asarray(target_w, float)[:2] - aim[0] * u - aim[1] * n
        best = 9.0
        for o in obst:
            r = o["xy"] - c
            bx, by = float(r @ u), float(r @ n)
            dx = max(bx - self.FOOT_FRONT, -self.FOOT_BACK - bx, 0.0)
            dy = max(abs(by) - self.FOOT_HALF, 0.0)
            best = min(best, float(np.hypot(dx, dy)))
        return best

    def _choose_heading(self, target_w, aim, theta0):
        """Pick the approach heading (decision frame) with the clearest crouch site
        among 12 candidates; the natural one (theta0, straight at the object) wins
        every tie. Returns (theta, clearance, natural_clearance, n_obstacles)."""
        obst = self.obstacles_near(target_w, 1.6, exclude_xy=target_w)
        if not obst:
            return theta0, 9.0, 9.0, 0
        t = np.asarray(target_w, float)[:2]
        lim = self._edge_lat() - 0.2
        cands = [theta0 + k * np.pi / 6 for k in range(12)]
        scored = []
        for th in cands:
            u = np.array([np.cos(th), np.sin(th)])
            c = t - aim[0] * u                      # stance point
            w0 = t - 1.3 * u                        # walk-in waypoint
            cs, cw = self._arc_of(c)
            ws, ww = self._arc_of(w0)
            _s0 = self.nav.s_start() if self.nav.honest else ct.COURSE_X0
            if abs(cw) > lim or abs(ww) > lim or cs < _s0 + 1.0:
                continue
            clr = min(self._site_clearance(target_w, aim, th, obst), 0.5)
            turn = abs((th - theta0 + np.pi) % (2 * np.pi) - np.pi)
            pen = 0.05 if ws < _s0 + 2.8 else 0.0   # goto_arc would clip that waypoint
            scored.append((clr - pen - 0.02 * turn, th, clr))
        nat = min(self._site_clearance(target_w, aim, theta0, obst), 0.5)
        if not scored:
            return theta0, nat, nat, len(obst)
        _, th, clr = max(scored, key=lambda s: s[0])
        return th, clr, nat, len(obst)

    def _footprint_blocked(self, target_w, fresh: int = 3000):
        """A remembered obstacle (seen within `fresh` steps) under the body right now,
        excluding the target: (cls, bx, by) or None."""
        for o in self.obstacles_near(target_w, 1.6, exclude_xy=target_w):
            if self.step_i - o["step"] > fresh:
                continue
            b = self.to_base_frame(np.array([o["xy"][0], o["xy"][1], 0.0]))
            if -self.FOOT_BACK < b[0] < self.FOOT_FRONT and abs(b[1]) < self.FOOT_HALF:
                return o["cls"], float(b[0]), float(b[1])
        return None

    def _obst_desc(self, target_w, obst) -> str:
        """'cls@(dx,dy)xn' list relative to the target, for the log."""
        t = np.asarray(target_w, float)[:2]
        return " ".join(f"{o['cls']}@({o['xy'][0]-t[0]:+.2f},{o['xy'][1]-t[1]:+.2f})x{o['n']}"
                        for o in obst[:6])

    def _oracle_detect(self):
        """MISSION_ORACLE_DETECT=1 ablation: ground-truth positions inside the camera's
        forward cone, with a 1.5 cm localisation error stand-in. Used to separate
        perception failures from manipulation failures — NOT the default path.
        """
        d = self.robot.data
        out = []
        for name, obj in self.objects.items():
            p = obj.data.root_pos_w[0].cpu().numpy()
            b = self.to_base_frame_true(p)
            dist = float(np.linalg.norm(b[:2]))
            if b[0] < 0.25 or dist > 3.2 or abs(np.arctan2(b[1], b[0])) > 0.55:
                continue
            # An object that bounced out of the basket and fell off the course is still
            # "in the cone" by its xy alone: one sat at z = -717 m and the robot spent
            # thousands of steps approaching it. A camera could never see it, so the
            # ablation must not report it either.
            if abs(float(p[2]) - ground_z_world(float(p[0]), float(p[1]))) > 0.5:
                continue
            noisy = p + np.random.normal(0, 0.015, 3)
            out.append({"cls": self.obj_cls[name], "pos_w": noisy, "n_pix": 999,
                        "dist": dist, "span": 0.1})
        return out

    def _cls_credit(self, cls: str) -> int:
        """Objects of a class that are binned correctly OR still aboard — the count
        the soft per-class quota should compare against. picked_per_cls also counts
        objects that were LOST after pickup (svC: can trash_2 "unload -> none"), so
        with picked_per_cls the memory pick refused the pebble can trash_3 twice
        while the recall path — which only looks at binned_ok — kept walking the
        robot back for it (two 3.5 m shuttles, then a 20 m sweep, for nothing)."""
        return self.binned_ok_per_cls.get(cls, 0) + sum(1 for _, c in self.carried if c == cls)

    def needed_classes(self) -> set:
        """Classes that can still earn points."""
        return {c for c, b in BIN_OF.items()
                if b and c not in self.p.skip_classes
                and self.binned_ok_per_cls.get(c, 0) < MAX_SCORED_PER_CLASS}

    BAG_CLASSES = ("redbag", "purplebag")

    def _bag_deferred(self, cls) -> bool:
        """Bags wait until every CORE class (can/rind/lunchbox) has its scoring pair
        binned. MISSION_BAG_AFTER_CORE=0 restores rank-only ordering.

        The class_rank deprioritisation only orders candidates that are visible at the
        same moment; a bag that is the only thing in view still gets engaged. Measured
        over the 100-point runs (s5600w/z): 8-17k steps per run on bags, ZERO
        conversions across the whole record, one run driven off the deck orbiting one.
        120 needs two lunchboxes, and those are what that time should buy. The bag is
        still remembered and comes back as recall once the core quota is complete."""
        if cls not in self.BAG_CLASSES:
            return False
        if os.environ.get("MISSION_BAG_AFTER_CORE", "1") != "1":
            return False
        return bool(self.needed_classes() - set(self.BAG_CLASSES))

    def _perched(self, name: str, what: str) -> bool:
        """PROPPED OBJECT GATE (MISSION_PERCH_SKIP, default 1). Seed 0's mud can stands
        upright on the foot of the red bag 25 cm away (ctlA video, 190-232 s): every
        grasp start put its centre 8.5 cm above the ground — 5-6 cm higher than a can
        lying free (base-frame z -0.14..-0.16 vs -0.20..-0.22) — the ray/ground
        perception lands it 0.25-0.29 m too far (it assumes ground contact), and with
        the bag standing on the approach line the policy closed on the BAG and lifted
        it instead, "never reached it" in 4 runs (ctlA 0/6, svA 1/7, svB 1/4): 5,000-
        6,000 steps per run while a clean copy of the same class sat one section on.
        An object propped above its resting height is a fact about the SCENE the
        robot cannot fix, so say so once and let the planner take the other copy.
        Threshold 4.5 cm above the lying height keeps an upright can on the ground
        (+2.5 cm) in play. Truth-based for now (like the VERDICT lines); the deployable
        signal is the gap between the blob's depth range and its ray/ground length."""
        # OFF THE DECK ENTIRELY — the opposite of propped, and far more expensive.
        # hs21a chased trash_3 at base-frame z -0.616 (0.6 m BELOW the robot's own
        # feet, i.e. over the edge and gone): five grasp attempts that never got within
        # 1.0 m, the approach walking it to lat 1.78, then 1.99, then off the platform
        # at lat 3.46 — the whole run lost 66,000 steps in, for an object that could not
        # have scored from down there.
        #
        # THE FIRST VERSION OF THIS GATE COST MORE THAN IT SAVED. It read the object's
        # TRUE base-frame z (a simulator read on the target-selection path, gated by
        # neither honesty flag) and tested it against -0.45 m. That is a body-frame z
        # measured at 1.7-2.3 m range, so base pitch dominates the lever arm: 2-3 deg
        # nose-down adds -0.07..-0.10 m at 2 m and trips the bar on articles lying flat
        # on the pebble bed. Measured firings, all on objects demonstrably ON the deck:
        # hs23a @53726 wrote off trash_9 (lunchbox, seen 56 times, perch gap +0.06 m)
        # and trash_6 (rind) in the same instant — neither ever attempted, run finished
        # 60/160; hs23e @24789/@53602 lost trash_8 the same way (65/160). ~40 points in
        # one run, ~20 in another.
        #
        # The honest signal is the one scan() already stores: `_gap` = the ray/ground
        # length minus the blob's median depth, ~0 for an article on the ground and
        # growing with however far the real surface is below where the ray-to-ground
        # model put it. An article over the edge is the extreme of that, so it is the
        # same test as the perch gate below with a much larger bar and the same
        # multi-view demand. The lateral corridor (scan(): |w| > MISSION_CORRIDOR) is
        # the other half — an article over the SIDE is outside the corridor in the
        # estimate too, which is what actually protects against the hs21a case.
        _g = self._gap.get(name, [])
        _gapdrop = float(os.environ.get("MISSION_OFFDECK_GAP", "0.60"))
        if len(_g) >= 4 and float(np.median(_g)) > _gapdrop:
            self.log.ev(f"{what} {name}: depth gap {float(np.median(_g)):+.2f} m over "
                        f"{len(_g)} views — the ground under it is far below the deck "
                        f"the robot is standing on; writing it off")
            # ...and blacklist in the DECISION frame. The old append used world truth
            # while scan() and _pick_from_memory compare S-frame detections, so at these
            # runs' 0.7-2.1 m estimate error the 0.30 m match never hit and the
            # "permanent" write-off silently did nothing (hs24h picked trash_8 after
            # four of them). Frame and test have to be fixed together — fixing only the
            # frame would have made the loss above real instead of accidental.
            _s = self.seen.get(name)
            if _s is not None:
                self.perma_blacklist.append(np.asarray(_s[1], float)[:2])
            return True
        if os.environ.get("MISSION_PERCH_SKIP", "1") != "1":
            return False
        cls = self.obj_cls.get(name)
        if cls not in ("can", "rind", "lunchbox"):
            return False
        if self.nav.honest or self.obj_honest:
            # the deployable signal: ray/ground length minus blob depth (~0 on the
            # ground, ~+0.25 propped up), median over the sweeps that saw it
            # (hs1 calibration: a lunchbox first seen from 3 m read +0.14 over two
            # views, then +0.05 over 16 — so demand a few views and a clear margin
            # below the propped-up value)
            g = self._gap.get(name, [])
            if len(g) < 4 or float(np.median(g)) <= 0.16:
                return False
            self.log.ev(f"{what} {name} ({cls}): depth gap {float(np.median(g)):+.2f} m "
                        f"over {len(g)} views — propped up, not on the ground")
            return True
        op = self._sim_obj(name).data.root_pos_w[0].cpu().numpy()
        rest = 0.5 * float(min(OBJ_DIMS[cls]))
        above = float(op[2]) - ground_z_world(float(op[0]), float(op[1])) - rest
        if above <= 0.045:
            return False
        nb = [f"{n} ({self.obj_cls[n]}) {float(np.linalg.norm(o.data.root_pos_w[0, :2].cpu().numpy() - op[:2])):.2f} m"
              for n, o in self.objects.items() if n != name and float(
                  np.linalg.norm(o.data.root_pos_w[0, :2].cpu().numpy() - op[:2])) < 0.35]
        self.log.ev(f"{what} {name} ({cls}): it rests {above * 100:.0f} cm higher than a "
                    f"{cls} lying on the ground (propped up; nearby: "
                    f"{', '.join(nb) if nb else 'nothing within 0.35 m'}) — writing it off")
        return True

    def _attempts_for(self, cls) -> int:
        # Hand classes (rind/lunchbox) are the 80->100->120 deciders and their pinch is
        # pose-dependent — one extra full cycle beats writing the object off (the second
        # lunchbox has NEVER converted; its recalls burn out at the 2-cycle cap).
        if cls in ("redbag", "purplebag"):
            # One cycle only: with the per-crouch budget already time-boxed to 2, the
            # bags still burned 4 attempts per run (s5600p/q) while cans and lunchboxes
            # went unengaged. One crouch is the lottery ticket; the reclaimed minutes
            # buy a whole extra object elsewhere.
            return 1
        return self.p.object_attempts + (0 if cls in self.p.basket_classes else 1)

    def recall_target(self):
        """Nearest remembered object of a class we still need, or None.

        Turns the search from "walk forward and hope" into "go to the one I already
        saw". Blacklisted objects and disallowed terrain are excluded, and so is
        anything the object bookkeeping has already given up on.
        """
        want = self.needed_classes()
        if not want:
            return None
        # Bounded, or it loops forever: a remembered object that the detector cannot pick
        # up again (one sat at lateral -0.84, outside the forward cone) never registers a
        # failure, so the robot walked back to the same lunchbox three times in a row and
        # would have kept going until the budget ran out.
        MAX_RECALLS = 2
        X, Y, _, _ = self.base_pose()
        _park = self._bin_park_arc()
        best, bd = None, 1e9
        for name, (cls, xy, _seen_at) in self.seen.items():
            if cls not in want or self._bag_deferred(cls):
                continue
            if self.obj_fails.get(name, 0) >= self._attempts_for(cls):
                continue
            # The hand classes (rind/lunchbox) are what separate 60 from 80 from 100 —
            # cans fill their quota nearly every run — so they get one extra recall.
            _cap = MAX_RECALLS + (0 if cls in self.p.basket_classes else 1)
            if self.recall_tries.get(name, 0) >= _cap:
                continue
            if any(np.hypot(xy[0] - b[0], xy[1] - b[1]) < 0.30 for b in self.blacklist):
                continue
            ax, w = self._arc_of(xy)
            if not self.terrain_allowed(self._section(ax)) or abs(w) > 1.35:
                continue
            # Never commit to something the clock cannot finish — an object still in the
            # gripper at the whistle scores zero, and recall_target was the one path that
            # bypassed afford_object entirely.
            if not self.afford_object(ax):
                continue
            # Rank by cost TO THE BIN, not by raw distance from the robot. With
            # max_carry=1 the return leg dominates, so a mixed copy at arc 12.75 (399
            # steps of haul) beats a pebble copy at -12.88 (2,577) even when the robot is
            # standing closer to the pebble one.
            d = float(np.hypot(xy[0] - X, xy[1] - Y))
            cost = TRAVEL_STEPS_PER_M * (d + max(0.0, _park - ax))
            # Hand classes first: a remembered rind/lunchbox is the 80/100 decider while
            # a remembered can is at best quota insurance, so a can only wins the recall
            # when no hand-class candidate is on the books (half a course of travel cost
            # as the tiebreak-proof offset).
            if cls in self.p.basket_classes:
                cost += TRAVEL_STEPS_PER_M * ct.COURSE_LEN * 0.5
            # A deprioritised class is pushed behind every other candidate rather than
            # excluded: one whole course length of travel cost per rank, so it only wins
            # when nothing else is left.
            cost += self.p.class_rank.get(cls, 0) * TRAVEL_STEPS_PER_M * ct.COURSE_LEN
            if cost < bd:
                best, bd = (name, cls, xy, ax, w), cost
        return best

    # ---------------------------------------------------------------- venue clock
    def steps_left(self) -> int:
        return self.p.total_budget - self.step_i

    def home_reserve(self) -> int:
        """Steps needed to get what is in the gripper into a bin from where we stand.

        An object still held when the clock stops scores NOTHING, so this is the one
        reserve the mission must never spend. Empty-handed it is zero — there is nothing
        to protect and the robot should keep searching to the last step.
        """
        if not self.carried:
            return 0
        ax, _ = self.nav.arc_lat()
        d = max(0.0, self._bin_park_arc() - float(ax))
        return int(TRAVEL_STEPS_PER_M * d + TRIP_FIXED_STEPS + UNLOAD_STEPS)

    def must_head_home(self) -> bool:
        return self.steps_left() <= self.home_reserve()

    def afford_object(self, arc_x: float) -> bool:
        """Enough clock left to service an object at this arc AND deliver it?

        Committing to an object the mission cannot finish is worse than ignoring it: the
        approach and squat are spent either way, and the object ends up in the gripper at
        the whistle, where it is worth zero.
        """
        d = max(0.0, self._bin_park_arc() - float(arc_x))
        need = OBJECT_WORK_STEPS + TRAVEL_STEPS_PER_M * d + TRIP_FIXED_STEPS + UNLOAD_STEPS
        return self.steps_left() > need

    def pass_start_arc(self) -> float:
        """Where a fresh sweep should begin.

        Starting every pass at the very back of the course (COURSE_X0 + 2.5) is both
        dangerous — that is the edge the robot falls off — and pointless when the first
        sections are excluded from picking anyway. Begin a short lead-in ahead of the
        earliest section we are actually allowed to grasp from.
        """
        if self.nav.honest:
            # no section map: the sweep begins the same 2.8 m lead-in past the start
            # line (the only anchor the robot has)
            return float(self.nav.s_start() + 2.8)
        allowed = [ct.SECTION_CENTERS[s] - ct.SECTION_LEN / 2
                   for s in ct.COURSE_SEQUENCE
                   if s in TERRAIN_RANK and self.terrain_allowed(s)]
        # Floor moved 4.0 -> 2.8 m from the back edge: the 27-run seed-0 audit shows
        # the -14 floor put the sweep start 13 cm AHEAD of a rind (missed 82%) and
        # left the arc<-14 strip structurally unreachable (its redbag: 100% missed).
        # 2.8 m of margin still clears the measured ~2 m turn-in-place drift.
        start = (min(allowed) - 3.0) if allowed else ct.COURSE_X0 + 2.8
        return float(max(start, ct.COURSE_X0 + 2.8))

    def _terrain_rank(self, sec: str) -> int:
        """Difficulty rank of a section label. Honest pseudo-segments ('seg0'..) carry
        no surveyed identity — every one ranks 0 (allowed); the IMU-strike bookkeeping
        (terrain_strikes / terrain_ok, keyed by the same label) is what excludes a
        segment the robot has actually failed on."""
        if self.nav.honest and str(sec).startswith("seg"):
            return 0
        return TERRAIN_RANK.get(sec, 9)

    def terrain_allowed(self, sec: str) -> bool:
        rank = self._terrain_rank(sec)
        if rank > self.p.max_terrain_rank:
            return False
        if self.terrain_strikes.get(sec, 0) >= 1 and not self.terrain_ok.get(sec):
            return False
        return True

    def reacquire(self, det, name, radius: float = 2.5):
        """Find the object again before a retry cycle.

        A failed grasp does not leave the object where it was — the fingers bat it aside.
        Re-running the approach on the stale detection made every retry WORSE, not better
        (measured: closest TCP 0.066 m on cycle 1, 0.171 m on cycle 2, then given up on).
        Stand up, look again, and take the nearest same-class detection.

        RADIUS 2.5, not 1.5. The measured displacement of a slipped object is 0.50, 0.77,
        0.89, 0.93, 1.41, 1.59 and 1.86 m — the common case lands just OUTSIDE 1.5, so
        re-acquire failed, returned the stale detection, and cycle 2 walked to where the
        object used to be (logged: "cam err 1.855 m", then "1.59 m outside the pregrasp
        bank"). With object_attempts=2 that made the second cycle worthless, so the robot
        effectively got one try per object. Widening cannot grab the wrong copy: the
        layout puts same-class copies at least 3.43 m apart (can 3.43, lunchbox 3.53,
        rind 6.18), so 2.5 m still has 1.4x of margin.
        """
        self.stand_up()
        self.settle(30)
        # the crouch spot is inside the camera's blind cone (see bring_into_view);
        # looking from there found nothing and "kept the old estimate" every time
        self.bring_into_view(det["pos_w"])
        # never scan standing at the platform line (swB: the feet crossed it while
        # the robot stood scanning on mud)
        self._walk_inboard(self._edge_lat())
        best, bd = None, 1e9
        for _ in range(4):
            for d2 in self.scan():
                if d2["cls"] != det["cls"]:
                    continue
                dd = float(np.linalg.norm(d2["pos_w"][:2] - det["pos_w"][:2]))
                if dd < bd:
                    best, bd = d2, dd
            if best is not None and bd < radius:
                break
            for _ in range(max(15, self.look_every)):   # a beat for a fresh frame
                self._write_arm(self.travel_q, open_=False)
                self._step(self.legs.step(torch.zeros(1, 3, device=self.device)))
        if best is None or bd >= radius:
            # The straight look covers +-33 deg x 1.25 m = +-0.8 m either side of the
            # old spot, and a batted object lands 0.5-1.9 m away (docstring above) —
            # mostly outside that. svA's mud can was dropped 0.8 m off the swing and
            # then chased at its OLD spot for 5,000 steps. Pan before giving up.
            best, bd = self._pan_look(det["cls"], det["pos_w"], radius)
        if best is None or bd >= radius:
            # an object batted TOWARDS the robot sits inside the 0.9 m blind disc of
            # every heading the pan just tried; back off and look once more — but not
            # off the belt (svF: hooked on the boundary rope at lat -1.8, run over)
            _w_prev = self.nav.arc_lat()[1]
            for _ in range(110):
                _, _w = self.nav.arc_lat()
                if abs(_w) > self._edge_lat() and abs(_w) >= abs(_w_prev):
                    self.log.ev(f"  back-off stopped at lat {_w:+.2f}: course edge")
                    self._walk_inboard(self._edge_lat() - 0.2)   # see bring_into_view
                    break
                if abs(_w) > self._edge_lat() - 0.35 and self._nose_inboard():
                    self.log.ev(f"  back-off skipped at lat {_w:+.2f}: it would walk "
                                f"outboard toward the edge")
                    break
                _w_prev = _w
                c = torch.zeros(1, 3, device=self.device)
                c[0, 0] = -0.30
                self._write_arm(self.travel_q, open_=False)
                self._step(self.legs.step(c))
            best, bd = self._pan_look(det["cls"], det["pos_w"], radius)
        if best is None or bd >= radius:
            # A TRACK THE ROBOT CANNOT FIND TWICE IS NOT THERE. With the simulator's
            # identity table gone, a detection that was a texture artefact becomes a
            # track like any other, and "keep the old estimate" then sends the robot
            # back to an empty patch of deck for ever. vh1 died of exactly that: after
            # its grasp on trk005 came up empty it logged "pan-look for the can:
            # nothing within reach", wandered from lat +1.79 across to -1.79, took two
            # -10s and went off the far side at -2.70, 9,860 steps in.
            # This is the honest write-off: the robot walked to the spot, looked
            # straight ahead, panned +-0.7 rad, backed off and panned again — twice —
            # and saw nothing of that class within 2.5 m. That is its own evidence, not
            # the simulator's. A real article batted out of the way still gets the
            # first miss for free.
            self._lost_looks = getattr(self, "_lost_looks", {})
            self._lost_looks[name] = self._lost_looks.get(name, 0) + 1
            _n = self._lost_looks[name]
            if self.det_honest() and _n >= int(
                    os.environ.get("MISSION_TRACK_LOST_LOOKS", "2")):
                self.log.ev(f"  re-acquire {name}: nothing of that class within "
                            f"{radius:.1f} m on {_n} separate looks — the track is a "
                            f"ghost; writing it off")
                self.obj_fails[name] = self._attempts_for(self.obj_cls.get(name))
                _s = self.seen.pop(name, None)
                if _s is not None:
                    self.perma_blacklist.append(np.asarray(_s[1], float)[:2])
                self._tracks.pop(name, None)
                return {**det, "stale": True, "ghost": True}
            self.log.ev(f"  re-acquire {name}: no fresh view within {radius:.1f} m "
                        f"— keeping the old estimate (look {_n})")
            return {**det, "stale": True}
        self.log.ev(f"  re-acquire {name}: moved {bd:.2f} m to "
                    f"{np.round(best['pos_w'], 2).tolist()}")
        return {**det, **best, "stale": False}

    def _pan_look(self, cls: str, ref_w, radius: float, amp: float = 0.7):
        """Turn +-amp rad in place, scanning; nearest same-class detection to ref_w."""
        best, bd = None, 1e9
        # never spin with the feet at the platform line (svM: pan at lat -1.69,
        # feet over 1.98 fifty steps later, off the deck)
        self._walk_inboard(float(os.environ.get("MISSION_EDGE_TURN", "1.45")))
        _, _, _, yaw0 = self.base_pose()
        ref = np.asarray(ref_w, float)[:2]
        for tgt in (yaw0 + amp, yaw0 - amp, yaw0):
            for _ in range(160):
                _, _, _, yaw = self.base_pose()
                e = (tgt - yaw + np.pi) % (2 * np.pi) - np.pi
                if abs(e) < 0.08:
                    break
                c = torch.zeros(1, 3, device=self.device)
                c[0, 2] = float(np.clip(1.5 * e, -0.6, 0.6))
                self._write_arm(self.travel_q, open_=False)
                self._step(self.legs.step(c))
                if self.step_i % self.look_every == 0:
                    for d2 in self.scan():
                        if d2["cls"] != cls:
                            continue
                        dd = float(np.linalg.norm(d2["pos_w"][:2] - ref))
                        if dd < bd:
                            best, bd = d2, dd
            for _ in range(self.look_every + 4):        # one still frame per heading
                self._write_arm(self.travel_q, open_=False)
                self._step(self.legs.step(torch.zeros(1, 3, device=self.device)))
                if self.step_i % self.look_every == 0:
                    for d2 in self.scan():
                        if d2["cls"] != cls:
                            continue
                        dd = float(np.linalg.norm(d2["pos_w"][:2] - ref))
                        if dd < bd:
                            best, bd = d2, dd
            if best is not None and bd < radius:
                break
        self.log.ev(f"  pan-look for the {cls}: "
                    + (f"seen {bd:.2f} m from the old spot" if best is not None and bd < radius
                       else "nothing within reach"))
        return best, bd

    def det_honest(self) -> bool:
        """Identity comes from the robot's OWN detections, not the simulator's table.

        Defaults to MISSION_OBJ_HONEST: object honesty without detection honesty is
        half a measurement, because the simulator was still naming the blobs.
        """
        return os.environ.get("MISSION_DET_HONEST",
                              "1" if self.obj_honest else "0") == "1"

    def _sim_obj(self, name):
        """The simulator article a name refers to — LOGGING and SCORING only.

        With honest identity the robot's names are its own track ids, which the
        simulator knows nothing about. Every DECISION must run off honest signals; this
        exists so the [TRUTH] telemetry and the scorer can still say which physical
        article was involved. The binding is nearest-truth of the track's class, pinned
        the first time it is asked for so a track cannot silently change article
        mid-carry (which would make the score meaningless in either direction).
        """
        o = self.objects.get(name)
        if o is not None:
            return o
        b = getattr(self, "_bound", {}).get(name)
        if b is not None:
            return self.objects[b]
        t = getattr(self, "_tracks", {}).get(name)
        if t is None:
            raise KeyError(name)
        pw = self.nav.true_of_S(np.asarray(t["p"], float)) if self.nav.honest \
            else np.asarray(t["p"], float)
        best, bd = None, 1e9
        for n_, o_ in self.objects.items():
            if self.obj_cls.get(n_) != t["cls"]:
                continue
            d_ = float(np.linalg.norm(
                o_.data.root_pos_w[0].cpu().numpy()[:2] - pw[:2]))
            if d_ < bd:
                best, bd = n_, d_
        if best is None:                      # no article of that class exists at all
            best = next(iter(self.objects))   # a phantom track: bind anything so the
            bd = 9.9                          # telemetry has a handle; it scores 'none'
        self._bound = getattr(self, "_bound", {})
        self._bound[name] = best
        self.log.ev(f"  [TRUTH] track {name} ({t['cls']}) bound to {best} "
                    f"at {bd:.2f} m — telemetry/scoring only")
        return self.objects[best]

    def _match_track(self, det):
        """HONEST DATA ASSOCIATION: cluster detections into the robot's own tracks.

        The old association asked the SIMULATOR which article a blob was — by true world
        position and true class — and scan() then DISCARDED any blob that failed to
        match. The 0906 audit measured the cost in the 120-point run: at step 68,500,
        144 of the 476 detections that reached the gate (30.3 %) were killed by it, and
        because the test included true-class equality no misclassification could ever be
        engaged, while scoring is by class. That made the simulator both the perception
        stack's false-positive filter and its identity oracle. Neither exists at a venue.

        Here every detection lands somewhere: on the nearest same-class track inside the
        gate, or on a new track of its own. Gate 0.45 m — measured over 171 associations
        in this round's runs the detection-to-memory distance is median 70 mm, p90
        210 mm, max 560 mm, and the stop-and-look refinement moves the estimate by a
        median 50 mm / p90 221 mm; 0.45 covers the p90 of both while staying well under
        the spacing the layout gives two articles of the same class.
        """
        d = np.asarray(det["pos_w"], float)
        cls = det["cls"]
        rng = float(det.get("dist", 2.0) or 2.0)
        self._tracks = getattr(self, "_tracks", {})

        # RANGE IS THE WHOLE STORY. A ray/ground intersection is a lever arm: the same
        # angular error puts the point 20 mm out at 0.6 m and a metre out at 3 m, and
        # the mission's first sighting of an article is always the far one. Measured on
        # the first honest batch (vp*), a flat median-of-views track sits a median
        # 0.96 m from the article it belongs to and only 39 % land within 0.30 m —
        # which is why the grasp "arrived on its target and closed on nothing" every
        # time even though the servo residual was 7-11 mm. The old pipeline never saw
        # this because the simulator's 0.60 m truth gate refused any detection that far
        # out and the grasp then read the true pose anyway.
        # So: gate on the range of the evidence, and let the CLOSEST views set the
        # position. A track built from 3 m is allowed to be a metre wrong and still be
        # recognised when the robot finally stands next to the article.
        _g0 = float(os.environ.get("MISSION_TRACK_GATE", "0.45"))
        _gk = float(os.environ.get("MISSION_TRACK_GATE_PER_M", "0.30"))
        best, bd, bgate = None, 1e9, _g0
        for tid, t in self._tracks.items():
            if t["cls"] != cls:
                continue
            dd = float(np.linalg.norm(np.asarray(t["p"], float)[:2] - d[:2]))
            # the gate is as wide as the WORSE of the two observations deserves
            _g = _g0 + _gk * max(rng, float(t.get("best_r", rng)))
            if dd - _g < bd - bgate:
                best, bd, bgate = tid, dd, _g
        if best is not None and bd < bgate:
            t = self._tracks[best]
            t["views"].append((d.copy(), rng))
            # keep the ten nearest-range views and let the three closest decide: a
            # 0.6 m view is worth more than twenty taken from 3 m, and a plain median
            # over all of them is dominated by the far ones the sweep collects first
            t["views"].sort(key=lambda v: v[1])
            del t["views"][10:]
            _use = [v[0] for v in t["views"][:3]]
            t["p"] = np.median(np.stack(_use), axis=0)
            t["best_r"] = float(t["views"][0][1])
            t["last"] = self.step_i
            return best, bd
        tid = f"trk{len(self._tracks):03d}"
        self._tracks[tid] = {"cls": cls, "p": d.copy(), "views": [(d.copy(), rng)],
                             "best_r": rng, "last": self.step_i}
        self.obj_cls[tid] = cls
        return tid, 0.0

    def match_object(self, det):
        """Associate a detection with the sim object (needed to drive the grasp obs).

        Sim bookkeeping: the detection is in the decision frame (S when honest), the
        objects are world truth — compare in world through the current estimate."""
        if self.det_honest():
            return self._match_track(det)
        best, bd = None, 1e9
        dw = np.asarray(det["pos_w"], np.float64)
        if self.nav.honest:
            dw = self.nav.true_of_S(dw)
        for name, obj in self.objects.items():
            if self.obj_cls[name] != det["cls"]:
                continue
            p = obj.data.root_pos_w[0].cpu().numpy()
            dd = float(np.linalg.norm(p[:2] - dw[:2]))
            if dd < bd:
                best, bd = name, dd
        return (best, bd) if bd < 0.60 else (None, bd)

    # ---------------------------------------------------------------- survey
    # MISSION_SURVEY=1: look FIRST, then plan. The reactive sweep only engages what
    # happens to sit in the forward cone (+-33 deg, 2.2 m) at the moment a frame is
    # taken, so it walked straight past objects 1 m to the side — the seed-0 audit has
    # the grass rind un-engaged in 9 of 9 runs and the grass lunchbox in 8 of 9. In
    # survey mode the robot walks the centreline and, at fixed stations, stops and pans
    # the head +-MISSION_SURVEY_YAW rad; every view lands in self.seen. It then services
    # the remembered objects inside the window ahead in the normal priority order,
    # nearest first, before walking on — so a cluster is cleared in one visit instead of
    # being re-found on the next pass (or never).
    def _survey_stations(self):
        if getattr(self, "_survey_st", None) is None:
            offs = [float(v) for v in os.environ.get(
                "MISSION_SURVEY_OFFS", "-2.6,-1.1,0.4,1.9").split(",")]
            st = []
            if self.nav.honest:
                # NO SECTION MAP: the same offsets around the centre of every 6 m
                # pseudo-segment from the start line up to the nominal bin section.
                # Segments that turn out to be a slope or the stairs are skipped
                # at sweep time from the IMU (traverse), not from a table.
                from course_mission.navest import COURSE_LEN_NOMINAL
                n_seg = int(round(COURSE_LEN_NOMINAL / ct.SECTION_LEN)) - 1   # minus bins
                for k in range(n_seg):
                    c = self.nav.s_start() + (k + 0.5) * ct.SECTION_LEN
                    st += [c + o for o in offs]
            else:
                for sec in OBJ_SECTIONS:
                    if self.terrain_allowed(sec):
                        st += [ct.SECTION_CENTERS[sec] + o for o in offs]
            self._survey_st = sorted(st)
            self._survey_done = set()
        return self._survey_st

    def _obj_section(self, sec: str) -> bool:
        """Can this section label hold objects? Surveyed list under the oracle; every
        pseudo-segment when honest (nothing is known until it is looked at)."""
        if self.nav.honest:
            return True
        return sec in OBJ_SECTIONS

    def _section_start(self, ax: float) -> float:
        """Arc where the section containing `ax` begins (decision frame)."""
        if self.nav.honest:
            k = max(0, int(np.floor((float(ax) - self.nav.s_start()) / ct.SECTION_LEN)))
            return self.nav.s_start() + k * ct.SECTION_LEN
        return float(ct.SECTION_CENTERS[section_of(ax)] - ct.SECTION_LEN / 2)

    def _survey_sweep(self, ax: float):
        """Stop and pan the camera across the course ahead, remembering everything."""
        amp = float(os.environ.get("MISSION_SURVEY_YAW", "0.6"))
        # NEVER SWEEP AT THE EDGE. A 360 deg survey is 700-1,300 steps of spinning and
        # standing, and a spin sweeps the feet ~0.3 m around the base while the trot's
        # own footfalls walk the base outboard with nothing commanding otherwise —
        # hs4d fell out of the middle of one. Every other spin in this file is fronted
        # by the same call; this was the one that never got it.
        self._walk_inboard(float(os.environ.get("MISSION_SURVEY_TURN_LAT", "1.10")),
                           max_steps=300)
        th = self._tan_yaw(ax)
        before = set(self.seen)
        zero = torch.zeros(1, 3, device=self.device)
        t0 = self.step_i
        # MISSION_SURVEY_FULL=1: a whole turn in 6 stops instead of the +-amp forward
        # pan. The forward pan only covers 0.9-2.2 m AHEAD (the head cam's ground
        # band), so a station that lands next to an object never sees it, and the
        # sweep after a pick is taken from wherever the pick left the robot: svB/svC
        # swept from -14.7 with the grass rind 0.17 m ahead (blind) and again from
        # -13.2 with the lunchbox 0.23 m behind (blind) — "0 new" both times, and both
        # objects went unengaged. Looking back covers the previous station's blind
        # disc; with stations 1.5 m apart every object gets a <=1.6 m sightline.
        # ~700 steps a station instead of ~450.
        if os.environ.get("MISSION_SURVEY_FULL", "0") == "1":
            heads = [th + k * (2.0 * np.pi / 6.0) for k in range(1, 7)]
            hold = self.look_every + 4
        else:
            heads = [th + amp, th - amp, th]
            hold = 2 * self.look_every + 6
        for tgt in heads:
            for _ in range(200):
                _, _, _, yaw = self.base_pose()
                e = (tgt - yaw + np.pi) % (2 * np.pi) - np.pi
                if abs(e) < 0.08:
                    break
                c = torch.zeros(1, 3, device=self.device)
                c[0, 2] = float(np.clip(1.5 * e, -0.7, 0.7))   # face_forward's rate
                self._write_arm(self.travel_q, open_=False)
                self._step(self.legs.step(c))
                if self.step_i % self.look_every == 0:
                    self.scan()
            # a couple of STILL frames at the end of each pan: walking-frame estimates
            # are 0.2-0.3 m off (see stop_and_look), still ones are what recall needs
            for _ in range(hold):
                self._write_arm(self.travel_q, open_=False)
                self._step(self.legs.step(zero))
                if self.step_i % self.look_every == 0:
                    self.scan()
        new = [f"{self.seen[n][0]}@{self._arc_of(self.seen[n][1])[0]:.1f}"
               for n in self.seen if n not in before]
        _, _w = self.nav.arc_lat()
        self.log.ev(f"survey at arc {ax:.1f} lat {_w:+.2f} ({self.step_i - t0} steps): "
                    f"{len(new)} new {new}, {len(self.seen)} remembered")
        # what the sweep could see, with the ground-contact gap (deployable perch signal)
        # and the truth for calibration (log only)
        rows = []
        for n, (cls, xy, _t) in self.seen.items():
            oax, ow = self._arc_of(xy)
            g = self._gap.get(n, [])
            op = self._sim_obj(n).data.root_pos_w[0].cpu().numpy()
            tax, tw = arc_of(float(op[0]), float(op[1]))
            rows.append(f"{n}:{cls} arc {oax:.1f}/{ow:+.2f} (true {tax:.1f}/{tw:+.2f}) "
                        f"gap {np.median(g):+.2f}x{len(g)}" if g else
                        f"{n}:{cls} arc {oax:.1f}/{ow:+.2f} (true {tax:.1f}/{tw:+.2f})")
        if rows:
            self.log.ev("  seen: " + " | ".join(rows))

    def _section_surveyed(self, ax: float) -> bool:
        """SECTION-FIRST SURVEY (MISSION_SURVEY_SECTION_FIRST, default 1): no pick —
        from memory or reactive — until every station of the section the robot is
        in has been swept. The interleaved version (sweep, pick, sweep...) undid the
        survey's own purpose: svD swept -17.0, went straight for the can at -14.4,
        and that walk marked the -15.7 station "passed" — the ONLY vantage on the
        grass rind (1.3 m ahead of it; 2.5 m from -17.0 is past FAR_CUT, and from
        -14.3 it sits in the 0.9 m blind disc with the bottle and the purple bag).
        The rind went unengaged in svB/svC/svD alike, and svA only found it on a
        second 27 m walk out from the bins (7,000 steps). Sweeping the whole section
        first costs one walk back of at most ~3 m (~300 steps) and every remembered
        object is then served nearest-first with all of them known."""
        if os.environ.get("MISSION_SURVEY") != "1" \
                or os.environ.get("MISSION_SURVEY_SECTION_FIRST", "1") != "1":
            return True
        sec = self._section(ax)
        if not self._obj_section(sec):
            return True
        done = getattr(self, "_survey_done", set())
        return all(s in done or self._section(s) != sec for s in self._survey_stations())

    def _basket_fill_ahead(self, ax: float, want: set) -> bool:
        """True while the basket can still expect to be filled FURTHER ALONG the course.

        A hand-carry pick (lunchbox) sends the robot straight to the bins, so taking
        it with a near-empty basket turns the trip into a 2-object delivery: svA took
        the grass lunchbox as its 2nd object and walked 27 m out and 27 m back for a
        can and a box (~6,000 steps) while the mud/pebble cans and rinds were still
        unvisited. Every course section carries one copy of each class (the layout
        rule the guide describes and scatter_layout implements), so an unvisited
        survey station in a section ahead is an expected basket item. Only used by
        the survey planner; MISSION_LBOX_DEFER=0 turns it off."""
        _mode = os.environ.get("MISSION_LBOX_DEFER", "1")
        if _mode not in ("1", "2"):
            return False
        if not (want & set(self.p.basket_classes)):
            return False
        st = self._survey_stations()
        done = getattr(self, "_survey_done", set())
        if _mode == "1" and os.environ.get("MISSION_SURVEY_SECTION_FIRST", "1") == "1":
            # only THIS section's stations count: with the section swept first the
            # basket items here are all known, and the lunchbox is the last pick
            # before the trip. Deferring it across sections sent svB/svC/svD to the
            # mud lunchbox (0/7, 1/6, 0/3 attempts) while the grass one — which svA
            # took in one attempt — was 20 m behind.
            sec = self._section(ax)
            return any(s > ax + 0.5 and s not in done and self._section(s) == sec for s in st)
        return any(s > ax + 0.5 and s not in done for s in st)

    def _unswept_ahead(self, ax: float):
        """Arc of the first survey station still to be swept beyond `ax`, in a LATER
        section than the one `ax` is in — or None."""
        if os.environ.get("MISSION_SURVEY") != "1":
            return None
        done = getattr(self, "_survey_done", set())
        st = [s for s in self._survey_stations()
              if s > ax + 0.5 and s not in done and self._section(s) != self._section(ax)]
        return min(st) if st else None

    def _survey_resume_arc(self, resume_arc: float):
        """Where a delivery trip should resume the search in survey mode.

        The stock resume point is 1.5 m past the last pick, which in section-first
        mode is the far end of a section that has already been swept clean: svG
        picked its 3rd grass object at -13.0, delivered, and shuttled 26 m back to
        -11.4 only to walk the empty slope up to the first mud station at -5.0 —
        ~600 steps of walking with nothing to see. Every remembered object that can
        still be served, and every station not yet swept, lies AHEAD of the swept
        sections, so resume 1 m short of whichever of those comes first. Only ever
        moves the resume point FORWARD; the pull-back for a stepped-over neighbour
        stays with recall_target.
        """
        if os.environ.get("MISSION_SURVEY") != "1":
            return resume_arc
        done = getattr(self, "_survey_done", set())
        pend = [s for s in self._survey_stations() if s not in done]
        want = self.needed_classes()
        corridor = float(os.environ.get("MISSION_CORRIDOR", "1.35"))
        objs = []
        for name, (cls, xy, _t) in self.seen.items():
            if cls not in want or self._bag_deferred(cls) \
                    or self._cls_credit(cls) >= MAX_SCORED_PER_CLASS \
                    or self.obj_fails.get(name, 0) >= self._attempts_for(cls):
                continue
            if any(np.hypot(xy[0] - b[0], xy[1] - b[1]) < 0.30
                   for b in self.blacklist + self.perma_blacklist):
                continue
            oax, ow = self._arc_of(xy)
            if abs(ow) <= corridor and self.terrain_allowed(self._section(oax)):
                objs.append(oax)
        cands = ([min(pend) - 1.0] if pend else []) + ([min(objs) - 1.5] if objs else [])
        if not cands:
            return resume_arc
        new = min(cands)
        if new > resume_arc + 0.5:
            self.log.ev(f"resume advanced from arc {resume_arc:.1f} to {new:.1f}: "
                        f"everything behind it is swept "
                        f"({len(pend)} station(s) and {len(objs)} remembered object(s) ahead)")
            return new
        return resume_arc

    def _pick_from_memory(self, ax: float):
        """Best remembered object inside the window ahead, as a detection, or None.

        Same gates as the reactive path (quota, bag deferral, blacklist, terrain,
        corridor, clock) and the same ordering. Position-checked against the sim
        object the memory was associated with: a knocked object that moved > 0.6 m
        from where it was seen is forgotten, not chased as a phantom forever.
        """
        back = float(os.environ.get("MISSION_SURVEY_BACK", "0.8"))
        ahead = float(os.environ.get("MISSION_SURVEY_AHEAD", "3.5"))
        corridor = float(os.environ.get("MISSION_CORRIDOR", "1.35"))
        if os.environ.get("MISSION_SURVEY_SECTION_FIRST", "1") == "1" \
                and self._obj_section(self._section(ax)):
            # the section was swept before anything was picked, so the window
            # reaches back to its start: the mud can at -5.0 is 3 m behind the last
            # mud station and would otherwise never be served
            back = max(back, ax - self._section_start(ax))
        X, Y, _, _ = self.base_pose()
        want = self.needed_classes()
        cands = []
        for name, (cls, xy, _t) in list(self.seen.items()):
            if cls not in want or self._bag_deferred(cls):
                continue
            if self.obj_fails.get(name, 0) >= self._attempts_for(cls):
                continue
            if any(np.hypot(xy[0] - b[0], xy[1] - b[1]) < 0.30
                   for b in self.blacklist + self.perma_blacklist):
                continue
            oax, ow = self._arc_of(xy)
            if not (ax - back <= oax <= ax + ahead) or abs(ow) > corridor:
                continue
            sec = self._section(oax)
            if not self.terrain_allowed(sec):
                continue
            # sim bookkeeping (truth): a knocked object is forgotten, not chased.
            # Object-honest: only the re-acquire can say the memory is stale (see the
            # _stale_hits path in run()), so the entry stays and the walk decides.
            here = self._S_of_true(self._sim_obj(name).data.root_pos_w[0].cpu().numpy())
            if not self.obj_honest \
                    and float(np.hypot(here[0] - xy[0], here[1] - xy[1])) > 0.60:
                self.log.ev(f"  memory: {name} ({cls}) is no longer where it was seen — forgetting it")
                self.seen.pop(name, None)
                continue
            if self._cls_credit(cls) >= MAX_SCORED_PER_CLASS and any(
                    v[0] != cls and v[0] in want for v in self.seen.values()):
                continue
            if not self.afford_object(oax):
                continue
            dist = float(np.hypot(xy[0] - X, xy[1] - Y))
            key = (self.p.class_rank.get(cls, 0) + self.picked_per_cls.get(cls, 0),
                   self._terrain_rank(sec), dist)
            cands.append((key, name, cls, xy, oax, ow, sec, dist))
        # hand class (lunchbox) blocks the gripper: take it once the basket is
        # nearly full or nothing basket-class is remembered ahead — the trip goes
        # out basket + hand either way, and the alternative (defer, then walk 20 m
        # back for it at the end of the sweep) is what the old sweeps did. Only a
        # SERVABLE basket item defers it (one that passed every gate above): a
        # remembered can outside the corridor or past its attempts used to hold the
        # lunchbox back with nothing to pick instead, and the robot walked on.
        if (not self.p.carry_in_gripper and self._hand_item is None
                and not self._recalling and len(self.carried) < self.p.max_carry - 1
                and (any(c[2] in self.p.basket_classes for c in cands)
                     or self._basket_fill_ahead(ax, want))):
            cands = [c for c in cands if c[2] in self.p.basket_classes]
        # Hand-carry class as the ONLY class still worth points, with a section still
        # unswept ahead: take that section's copy instead. The pick ends the trip
        # either way (a hand item sends the robot to the bins), so the only question
        # is WHICH copy — and every section carries one. The one nearest the bins is
        # the cheapest to carry (hand-carry is the slowest gait, and the stairs have
        # knocked a box out of the gripper — svB), and the mud copy specifically has
        # eaten runs: trash_8 went 1 for 14 across the 0903 record at 5-9k steps a
        # visit (its crouches landed at base x 0.45-0.72 — mud approach precision),
        # while the pebble copy 6 m on took 1-2 attempts. Not applied while basket
        # classes are still wanted: the grass lunchbox rides out with the grass can
        # and rind, and deferring it would cost a 27 m walk back later.
        # MISSION_LBOX_NEAR_BINS=0 turns it off; the deferred copy stays remembered
        # for the end-of-pass recall if the nearer one fails.
        if (cands and not self.p.carry_in_gripper and self._hand_item is None
                and not self._recalling
                and os.environ.get("MISSION_LBOX_NEAR_BINS", "1") == "1"
                # "still wanted" counts what is aboard: the mud can and rind ride in
                # the basket unbinned while this decision is made
                and not any(self._cls_credit(c) < MAX_SCORED_PER_CLASS
                            for c in want & set(self.p.basket_classes))
                and all(c[2] not in self.p.basket_classes for c in cands)):
            _nx = self._unswept_ahead(ax)
            if _nx is not None:
                self.log.ev(f"hand-carry pick deferred: {self._section(_nx)} ahead is "
                            f"unswept and its copy is nearer the bins "
                            f"({[c[1] for c in cands]} stay remembered)")
                return None
        if not cands:
            return None
        cands.sort(key=lambda c: c[0])
        _, name, cls, xy, oax, ow, sec, dist = cands[0]
        # honest: no height map — the local ground under the robot stands in
        gz = self._ground_z() if self.nav.honest else ground_z_world(float(xy[0]), float(xy[1]))
        self.log.ev(f"memory pick: {name} ({cls}) at arc {oax:.1f} lat {ow:+.2f}, "
                    f"{dist:.1f} m away ({len(cands)} candidate(s) in the window)")
        # Carry the SHAPE measurements forward with the position. A remembered pick
        # used to arrive with span 0 and no height, and the grasp then had to assume an
        # attitude: every remembered can was treated as standing (top 0.115 m), which
        # aims the fingers 5 cm over one lying on its side. The medians over every view
        # that saw it are the honest summary of what the camera did measure.
        _hs = [v for v in self._h_seen.get(name, []) if np.isfinite(v)]
        _sp = self._span_seen.get(name, [])
        # The near edge is remembered together with the sighting it came from, so it
        # travels with the remembered position (they are usually the same sighting;
        # they differ once a later view has moved the memory).
        _ew, _vw, _pw = self._edge_seen.get(name, (None, None, None))
        if _ew is not None:
            _ew = np.array([_ew[0] + xy[0] - _pw[0], _ew[1] + xy[1] - _pw[1], gz], float)
        return {"cls": cls, "pos_w": np.array([xy[0], xy[1], gz], float),
                "edge_w": _ew, "view_w": _vw,
                "n_pix": 0, "dist": dist,
                "span": float(np.median(_sp)) if _sp else 0.0,
                "h_est": float(np.median(_hs)) if _hs else np.nan,
                "yaw": float(self._yaw_seen.get(name, np.nan)),
                "elong": float(self._elong_seen.get(name, 1.0)),
                "arc": oax, "lat": ow, "section": sec, "memory": True}

    # ---------------------------------------------------------------- phases
    def traverse(self, x_stop: float, pick_enabled=True):
        """Zigzag forward, scanning. Returns a detection to service, or None at x_stop."""
        stall_x, stall_at = None, self.step_i
        ax_hi = -1e9          # furthest arc reached in this traverse; see the retreat guard
        if self.off_platform:
            return None
        survey = pick_enabled and os.environ.get("MISSION_SURVEY") == "1"
        if survey:
            # clear the window ahead from memory before walking anywhere
            _ax0, _ = self.nav.arc_lat()
            if self.step_i <= self.p.search_budget and not self.must_head_home() \
                    and self._section_surveyed(_ax0):
                _d = self._pick_from_memory(_ax0)
                if _d is not None:
                    return _d
        while True:
            if self.step_i > self.p.search_budget:
                self.log.ev(f"search budget spent at step {self.step_i} — heading for the bins")
                return None
            if self.must_head_home():
                self.log.ev(f"venue clock: {self.steps_left()} steps left and "
                            f"{self.home_reserve()} needed to bin what I am holding "
                            f"— heading for the bins")
                return None
            # BANK A STALLED LOAD. Objects in the basket are worth their 20 points
            # only once they are in a bin; until then they are cargo at risk and the
            # clock is running. hs16a banked its 2nd basket object at step 34,398 and
            # then spent 15,000 steps (a sixth of the whole venue budget) failing to
            # find a 3rd — trash_8 gave up after 3 cycles, trash_6 after 4 attempts —
            # with 40 unscored points riding on its back the entire time. Once the
            # hunt has produced nothing for MISSION_TRIP_STALL steps, take what is
            # already aboard to the bins; the remembered targets are still there
            # afterwards (the caller recalls them straight after deliver_payload).
            if pick_enabled and len(self.carried) != getattr(self, "_carry_n_seen", -1):
                self._carry_n_seen = len(self.carried)
                self._last_bank_step = self.step_i
            _stall = int(os.environ.get("MISSION_TRIP_STALL", "7000"))
            _stall_min = int(os.environ.get("MISSION_TRIP_STALL_MIN", "2"))
            if self.steps_left() < 30000:
                _stall_min = 1          # late on, one banked object justifies the trip
            if (pick_enabled and _stall > 0 and self._hand_item is None
                    and len(self.carried) >= _stall_min
                    and self.step_i - getattr(self, "_last_bank_step", self.step_i)
                    > _stall):
                self.log.ev(f"nothing picked in {_stall} steps with "
                            f"{len(self.carried)} aboard — banking the load before "
                            f"hunting further")
                self._last_bank_step = self.step_i
                return None
            ax, w = self.nav.arc_lat()
            if self.nav.honest and pick_enabled:
                x_stop = min(x_stop, self._bins_arc() - 1.5)   # bins sighted en route
                if self.nav.on_stairs():
                    # objects are never placed on the stairs (guide) and the bins follow
                    # them: the sweep is over, whatever the estimate says
                    self.log.ev(f"stairs under the feet at arc {ax:.1f} — the sweep ends here")
                    return None
            if ax >= x_stop:
                return None
            # LATERAL GUARD — the sibling of the retreat guard below, on the other axis.
            # traverse used to discard `w` entirely and test only the longitudinal edge,
            # so a robot that wandered off the SIDE of the platform zigzagged out there
            # indefinitely. MEASURED in 4 of 8 runs, all in the arc 4..14.6 band: 1,667 to
            # 3,000 steps per metre against 91-139 elsewhere, with the detector's raw
            # counter frozen at delta 0 for 4,500-11,500 contiguous steps — the camera was
            # pointing at nothing. That band holds the three cheapest objects on the
            # course (trash_4/12/8, round trips of ~1,400 steps). 1.8 m sits between the
            # corridor reject (1.35, in scan) and goto_bin's abandon (2.5), and the
            # realised zigzag peaks near 1.05 at zigzag_amp 1.1, so there is 0.75 m of
            # margin against a false trigger. Recovery is the same straight-ahead push the
            # stall handler already uses, aimed at the centreline.
            if abs(w) > LAT_GUARD:
                self.lat_strikes += 1
                self.log.ev(f"off the side at arc {ax:.1f} lat {w:+.2f} — steering back "
                            f"to the centreline (strike {self.lat_strikes})")
                _lat_before = w
                # WEDGED IS NOT OFF-COURSE. Measured (run 0824_2208_deploy_final2):
                # five consecutive strikes at lat -1.84 with ZERO lateral movement — a
                # foot caught on a stanchion base plate. The forward push below is the
                # right recovery for having DRIFTED wide, but a wedged foot was walked
                # into forward, so pushing forward grinds it deeper (that run slid to
                # -2.08, took two OOB penalties, and never crossed the finish). If the
                # previous strike's recovery moved the robot laterally by less than 5 cm,
                # back straight out first — the reverse of the motion that caused it —
                # and only then push for the centreline.
                if (self.lat_strikes >= 2
                        and abs(abs(self._last_strike_lat) - abs(w)) < 0.05):
                    self.log.ev("  no lateral progress since the last strike — wedged; "
                                "backing out before pushing")
                    # PURE body-frame reverse: walk_cmd's heading control while
                    # reversing grinds the caught foot on the stanchion base
                    # (r36h: four back-outs, zero true movement)
                    _cmd = torch.zeros(1, 3, device=self.device)
                    _cmd[0, 0] = -0.35
                    for _ in range(300):
                        self._write_arm(self.travel_q, open_=False)
                        self._step(self.legs.step(_cmd))
                    _cmd[0, 0] = 0.02
                    _cmd[0, 2] = -0.5 * float(np.sign(w))
                    for _ in range(150):
                        self._write_arm(self.travel_q, open_=False)
                        self._step(self.legs.step(_cmd))
                    if self.nav.honest and getattr(self.nav, "since_fix", 0) > 1500:
                        self.nav.reacquire(self)
                self._last_strike_lat = w
                # threshold LOWERED 1.75 -> 1.60 (below the 1.645 guard, so every
                # strike takes the reflex): basket31 struck at +1.65, took the OLD
                # advancing recovery, and its outward arc was past the platform edge
                # before the next measurement (+6.2) — the campaign's third fall, and
                # the first WITH the reflex installed but gated too high.
                if abs(w) > 1.60:
                    # EDGE REFLEX. The normal recovery advances at 0.35 while steering,
                    # which sweeps an outward ARC as the heading comes around — at lat
                    # 1.95 that arc IS the platform edge, and once over it there is no
                    # climbing back (basket15: strikes 1.65 -> 6.5 -> 23, run lost the
                    # finish and 3 OOB penalties; basket2 died the same way). Deep
                    # excursions turn IN PLACE until facing the centreline, then walk
                    # straight in, and only resume the course after that.
                    # ...but the spin itself is fatal once the FEET are at the line
                    # (svK: reflex at lat -1.65 with the foot edge at 1.99 vs 1.98,
                    # slid off during the turn). Walk straight in first, no spin,
                    # until the base is inside MISSION_EDGE_TURN; then turn.
                    self._walk_inboard(float(os.environ.get("MISSION_EDGE_TURN", "1.45")))
                    ax, w = self.nav.arc_lat()
                    _th = self._tan_yaw(ax)
                    _want = _th + (np.pi / 2 if w < 0 else -np.pi / 2)
                    for _ in range(300):
                        _, _, _, _yaw = self.base_pose()
                        _e = (_want - _yaw + np.pi) % (2 * np.pi) - np.pi
                        if abs(_e) < 0.4:
                            break
                        c = torch.zeros(1, 3, device=self.device)
                        c[0, 2] = float(np.clip(1.5 * _e, -0.7, 0.7))
                        self._write_arm(self.travel_q, open_=False)
                        self._step(self.legs.step(c))
                    # Walk in until actually INSIDE, not for a fixed 280 steps (~2.2 m):
                    # s5600r drifted to lat +20.7 because each reflex recovered less
                    # ground than the outward arc between strikes lost.
                    for _ in range(900):
                        _, _, _, _yaw = self.base_pose()
                        _, _w2 = self.nav.arc_lat()
                        if abs(_w2) < 1.2:
                            break
                        _e = (_want - _yaw + np.pi) % (2 * np.pi) - np.pi
                        c = torch.zeros(1, 3, device=self.device)
                        c[0, 0] = 0.4
                        c[0, 2] = float(np.clip(1.2 * _e, -0.5, 0.5))
                        self._write_arm(self.travel_q, open_=False)
                        self._step(self.legs.step(c))
                    self.face_forward()
                else:
                    for _ in range(400):
                        c, _, _ = self.walk_cmd(0.0, vx=0.35)
                        self._write_arm(self.travel_q, open_=False)
                        self._step(self.legs.step(c))
                if self.lat_strikes >= 6:
                    # STICKY. lat_strikes is an instance counter, so simply returning None
                    # let the main loop walk straight back into traverse and spend another
                    # strike plus 400 steps of recovery every time — observed climbing to
                    # 7+ after the cap. Once the robot is off the platform it is not
                    # coming back; stop searching, and let the caller deliver what it is
                    # already holding rather than grinding to the budget.
                    self.off_platform = True
                    self.log.ev("cannot hold the centreline — off the platform, "
                                "ending the search and delivering what I have")
                    return None
                continue
            # RETREAT GUARD. The stall watchdog below only catches standing still — it
            # tests |ax - stall_x| < 0.5, so a robot walking BACKWARD at 1.2 m per 500
            # steps sails past it forever. That is exactly what happened: after a failed
            # approach left the body pointing down-course, "forward" crawled -16.5 ->
            # -20.9 with no warning at all, and off the back edge. The search is
            # forward-only by definition, so any sustained loss of ground is a heading
            # fault, and the fix is to re-aim, not to keep walking.
            if ax_hi > -1e8 and ax < ax_hi - 1.2:
                self.log.ev(f"retreating: arc {ax:.1f} is {ax_hi - ax:.1f} m behind the "
                            f"furthest point reached — re-aiming forward")
                self.face_forward()
                ax_hi = ax          # re-baseline so one bad turn is not re-reported
                continue
            ax_hi = max(ax_hi, ax)
            if self.step_i - stall_at >= 600:        # <0.5 m in 12 s -> nudge past it
                if stall_x is not None and abs(ax - stall_x) < 0.5:
                    off = ax < self.nav.s_start() + 0.3
                    if off:
                        self.off_course_strikes = getattr(self, "off_course_strikes", 0) + 1
                        if self.off_course_strikes >= 4:
                            # measured: once the robot is on the ground behind the raised
                            # platform it never climbs back on, and the recovery just
                            # burns the search budget four thousand steps at a time
                            self.log.ev(f"off the course at arc {ax:.1f} and not recovering "
                                        f"— ending the search")
                            return None
                    self.log.ev(f"stalled near arc {ax:.1f}; "
                                f"{'off the course — driving back on' if off else 'pushing through'}")
                    self._stall_reps = (getattr(self, "_stall_reps", 0) + 1
                                        if abs(ax - getattr(self, "_stall_prev", 1e9)) < 0.4
                                        else 1)
                    self._stall_prev = ax
                    if self._stall_reps >= 3:
                        # h65 pushed at the same arc for 30k steps: repeated stalls at
                        # one spot mean an OBSTRUCTION, not a heading error — take the
                        # wedge escape (body-frame reverse, turn to centre) and, in
                        # honest mode, re-find the belt before pushing again
                        self.log.ev("  three stalls at the same arc — wedge escape")
                        _cmdw = torch.zeros(1, 3, device=self.device)
                        _cmdw[0, 0] = -0.35
                        for _ in range(300):
                            self._write_arm(self.travel_q, open_=False)
                            self._step(self.legs.step(_cmdw))
                        _cmdw[0, 0] = 0.02
                        _cmdw[0, 2] = -0.5 * float(np.sign(w)) if abs(w) > 0.05 else 0.4
                        for _ in range(150):
                            self._write_arm(self.travel_q, open_=False)
                            self._step(self.legs.step(_cmdw))
                        if self.nav.honest:
                            self.nav.reacquire(self)
                        self._stall_reps = 0
                    # Pushing harder does not help when the robot is simply pointed the
                    # wrong way: one run stalled 13 times in a row while sliding arc -10.3
                    # -> -12.7, i.e. backwards, at vx 0.9. Re-aim first, in place.
                    self.face_forward(steps=250)
                    for _ in range(300 if off else 150):
                        # off the end there is nothing to zigzag along: aim straight at
                        # the centreline and push
                        c, _, _ = self.walk_cmd(0.0, vx=0.9)
                        self._write_arm(self.travel_q, open_=False)
                        self._step(self.legs.step(c))
                stall_x, stall_at = ax, self.step_i
            if survey:
                # stations behind us (arrived by shuttle/recall) are skipped, never
                # walked back to; the first one reached ahead gets a pan-and-look
                _st = [s for s in self._survey_stations()
                       if s not in self._survey_done and s <= ax + 0.15]
                _due = [s for s in _st if s >= ax - 0.6]
                for s in _st:
                    self._survey_done.add(s)
                if _due and self.nav.honest and self.nav.inclined():
                    # no section map, but the IMU says slope/stairs — ground the rules
                    # put no objects on, and no place for a 360 deg spin
                    self.log.ev(f"  survey station at arc {ax:.1f} skipped: the body "
                                f"is inclined (slope/stairs) — walking on")
                    _due = []
                if _due:
                    self._survey_sweep(ax)
                    self.face_forward(steps=150)
                    stall_x, stall_at = ax, self.step_i
                    if self._section_surveyed(ax):
                        _d = self._pick_from_memory(ax)
                        if _d is not None:
                            return _d
                    else:
                        self.log.ev(f"  {self._section(ax)} not fully surveyed yet — "
                                    f"walking on to the next station")
                    continue
                lat = float(os.environ.get("MISSION_SURVEY_LAT", "0.0"))
                if lat:
                    lat *= np.sin(2 * np.pi * ax / self.p.zigzag_period)
            else:
                lat = self.p.zigzag_amp * np.sin(2 * np.pi * ax / self.p.zigzag_period)
            cmd, _, _ = self.walk_cmd(lat)
            self._write_arm(self.travel_q, open_=False)
            self._step(self.legs.step(cmd))
            if pick_enabled and self.step_i % self.detect_every_eff == 0:
                _dets = self.scan()          # remembered either way (scan writes seen)
                if survey:
                    # section-first: look now, pick after the sweep. Gated on the
                    # OBJECT's section as well as the robot's: svG, walking the slope
                    # (no stations), engaged the mud can from 2.3 m out before a single
                    # mud station was swept — the pick then decides where the sweep
                    # starts from, and any station it ends up behind is marked passed
                    _dets = [d for d in _dets
                             if self._section_surveyed(ax)
                             and self._section_surveyed(float(d["arc"]))]
                # flattest terrain first, then nearest (사용자 지정 우선순위)
                for d in sorted(_dets,
                                key=lambda d: (self.p.class_rank.get(d["cls"], 0)
                                               + self.picked_per_cls.get(d["cls"], 0),
                                               self._terrain_rank(d["section"]),
                                               d["dist"])):
                    # Basket mode: a hand-carry class (lunchbox/rind) occupies the
                    # GRIPPER, which blocks all further grasping — so while the basket
                    # still has room, remember it (scan() already wrote self.seen) and
                    # keep collecting basket classes; the end-of-sweep recall fetches it
                    # LAST and the trip goes out basket-full plus one in hand (user:
                    # "4 in the basket + 1 in the gripper = 5 per trip"). Never defer
                    # while walking back FOR such an object, or it would be re-skipped
                    # forever.
                    if (not self.p.carry_in_gripper
                            and d["cls"] not in self.p.basket_classes
                            and self._hand_item is None
                            and len(self.carried) < self.p.max_carry
                            and not self._recalling):
                        self.rej["defer"] = self.rej.get("defer", 0) + 1
                        continue
                    # Only 2 per class SCORE. A third can is retrieval insurance at
                    # best, and its ~2500-step grasp cycle is better spent on the
                    # classes that decide 60-vs-80 — so once a class has its scoring
                    # pair picked, take a spare ONLY when nothing else scoreable is
                    # even remembered.
                    if (self._cls_credit(d["cls"]) >= MAX_SCORED_PER_CLASS
                            and any(v[0] != d["cls"] and v[0] in self.needed_classes()
                                    for v in self.seen.values())):
                        self.rej["quota_soft"] = self.rej.get("quota_soft", 0) + 1
                        continue
                    if 0.38 < d["dist"] < Perception.FAR_CUT and d["arc"] > ax - 0.8:
                        # Do not start what the clock cannot finish (see afford_object).
                        if not self.afford_object(d["arc"]):
                            self.rej["no_time"] = self.rej.get("no_time", 0) + 1
                            continue
                        return d
                    self.rej["range"] += 1
            if self.step_i % 500 == 0:
                self.log.ev(f"step {self.step_i}: arc={ax:.1f} sec={self._section(ax)} "
                            f"picks={len(self.carried)} det={self.rej}")

    def approach(self, det, aim_bias=None, reanchor=True):
        """Walk until the object sits at the pregrasp bank's sweet spot in the base frame.

        Servos on the PERCEIVED position, re-anchored by fresh detections (task B's
        `_reanchor_target` pattern) — the ground-truth pose is never used to steer.
        """
        # reanchor=False walks onto a GEOMETRIC point (e.g. the mouth end of a lying
        # bag) rather than a detection. The camera machinery must stay out of it: both
        # stop_and_look and the rolling-median block below snap the target back to the
        # OBJECT'S CENTRE, which is exactly the point being avoided -- measured, the
        # mouth re-approach ended up 0.33 m short of the mouth because a fresh detection
        # of the bag's centre overwrote it mid-walk.
        if reanchor:
            target_w = self.stop_and_look(det, target_w=np.asarray(det["pos_w"], float))
        else:
            target_w = np.asarray(det["pos_w"], float)
        # honest nav: the approach target doubles as a drift landmark — every fresh
        # view below becomes a pose fix through the belt-blind phase
        self.nav.object_anchor(self.to_base_frame(target_w)[:2])
        aim = self.grasp_anchor[:2] - (np.zeros(2) if aim_bias is None else aim_bias)
        # global standoff knob (0824 stack): 202 mined attempts show the approach
        # stops too CLOSE — bx<0.45 is 45 % of all grasp starts and converts at 10 %
        # (81 % "never reached it"); bx 0.45-0.70 converts at ~26 %. Default 0 keeps
        # the incumbent behaviour.
        _gax = float(os.environ.get("MISSION_AIM_X", "0"))
        aim[0] += _gax
        # per-class fore standoff on top of the global one — the lunchbox lid pinch
        # bites the NEAR edge, so the BOX must sit farther out than a can does
        # (fv2/lbox rounds: pinch points at base x 0.41-0.75, the near ones jam the
        # elbow, the far ones fall off the arm envelope; a class knob lets a short
        # round sweep the window without touching every other class)
        aim[0] += float(os.environ.get(
            f"MISSION_AIM_X_{str(det.get('cls', '')).upper()}", "0"))
        # lateral sweet spot (MISSION_AIM_Y, m, + = object further to the robot's
        # LEFT). 552 grasp starts on the s5600 record: with the object at base y
        # +0.05..+0.15 the policy held the lunchbox ~35 % and the can ~40 % of the
        # time; at y <= -0.05 the lunchbox converted 3-9 % and the can 14 %. The
        # unbiased first crouch lands at y -0.04, on the wrong side of that.
        aim[1] += self.aim_y_off(det.get("cls"))
        if det.get("cls") == "redbag":
            # cage-lift standoff: all three HELDs to date sat at post-crouch x
            # +0.10..+0.16 (bar further than the anchor), every miss at |x|<=0.09 —
            # including the descent stalls, which only ever happen at x-close. Aim
            # the bar 12 cm beyond the anchor. Lives HERE so the probe path gets it
            # too (a run()-loop version missed the probes and had the sign wrong).
            # ...but the THREAD wants the opposite. The standoff exists so the cage can
            # descend onto a bar that is further out than the anchor; threading has to
            # put the fingers THROUGH an 82 x 60 mm window, and the probes measured
            # what the extra 12 cm costs there — post-crouch alignment +0.208, +0.234,
            # +0.214, +0.208 m across four trials, which puts the handle outside the
            # arm's envelope and returns "no reachable entry pose" every time. So the
            # standoff follows the manoeuvre: keep it for the pinch, drop it for the
            # thread. MISSION_BAG_AIM_X overrides either way.
            # ...and it has to go NEGATIVE for the thread, because the approach's own
            # residual is already a standoff. The loop exits at err < 0.15 and the trot
            # settles around there, so the bag lands +0.15..+0.21 past the anchor
            # whatever the aim says: measured +0.208/+0.234/+0.214/+0.208 with the
            # +0.12 in, and still +0.183 with it taken out. Aim SHORT by the settling
            # distance instead. No max(0, ...) clamp — that is what made the knob
            # one-sided in the first place.
            _bax = os.environ.get("MISSION_BAG_AIM_X")
            _bax = (float(_bax) if _bax else
                    (-0.15 if os.environ.get("MISSION_BAG_THREAD_FIRST", "1") == "1"
                     else 0.12))
            aim[0] += _bax - _gax    # +0.12 fitted for the cage pinch, -0.15 for the thread
        # CROUCH-SITE CHOICE from obstacle memory (hs1 F5: the robot walked straight
        # over a bag + lunchbox + bottle cluster to a rind, squatted on top of them,
        # rolled 0.92 rad and lost the can it was carrying). If the straight-in
        # footprint is crowded and another side is clear, walk in from that side.
        if reanchor and os.environ.get("MISSION_SITE_CHECK", "1") == "1":
            _rx, _ry, _, _ = self.base_pose()
            _th0 = float(np.arctan2(target_w[1] - _ry, target_w[0] - _rx))
            _th, _clr, _nat, _nob = self._choose_heading(target_w, aim, _th0)
            _dth = float((_th - _th0 + np.pi) % (2 * np.pi) - np.pi)
            _od = self._obst_desc(target_w, self.obstacles_near(target_w, 1.6,
                                                                exclude_xy=target_w))
            if _nob and _nat < 0.15 and abs(_dth) > np.radians(40) and _clr > _nat + 0.15:
                _u = np.array([np.cos(_th), np.sin(_th)])
                _ws, _ww = self._arc_of(np.asarray(target_w, float)[:2] - 1.3 * _u)
                self.log.ev(f"approach: {_nob} remembered objects crowd the crouch site "
                            f"(clearance {_nat:.2f} m straight in: {_od}) — walking in "
                            f"from {np.degrees(_dth):+.0f} deg (clearance {_clr:.2f} m) "
                            f"via arc {_ws:.1f} lat {_ww:+.2f}")
                self.goto_arc(_ws, lat_target=float(np.clip(_ww, -1.05, 1.05)))
                for _ in range(400):      # face the object before the position loop
                    _b = self.to_base_frame(target_w)
                    _bear = float(np.arctan2(_b[1], _b[0]))
                    if abs(_bear) < 0.15:
                        break
                    cmd = torch.zeros(1, 3, device=self.device)
                    cmd[0, 2] = float(np.clip(1.5 * _bear, -0.6, 0.6))
                    self._write_arm(self.travel_q, open_=False)
                    self._step(self.legs.step(cmd))
            elif _nob and _nat <= 0.0 and _clr <= 0.0:
                self.log.ev(f"approach abandoned: the crouch site is crowded on every "
                            f"side ({_nob} remembered objects: {_od}) — not squatting "
                            f"on top of them")
                return False, target_w
            elif _nob and _nat < 0.3:
                self.log.ev(f"approach: crouch site tight (clearance {_nat:.2f} m: {_od}) "
                            f"— going straight in")
        t0, best = self.step_i, 1e9
        turned = False
        # SQUARE-UP for elongated objects (MISSION_ALIGN_CLASSES, default lunchbox).
        # The policy is yaw-blind, so the fingers close along the robot's y axis
        # wherever the object happens to point; the lunchbox lid (85-105 mm across,
        # 170+ along) only fits the 100 mm jaws within ~25 deg of its short axis, and
        # the measured ~1/3 per-cycle lunchbox rate is just that angular window. Here
        # the last stage of the approach steers the HEADING onto the object's long
        # axis (the detector's blob principal axis, refined in stop_and_look) while the
        # position loop keeps the object at a standoff, then closes in straight.
        # ...and for the paper bag the useful heading is the OTHER one. Its carry
        # handle is a frame in the bag's long-axis/vertical plane (build_prop_usds.py:
        # uprights at local x +-45 mm, crossbar across them), so the window the fingers
        # thread through faces along the bag's SHORT axis. Squaring up to the long axis
        # — the lunchbox rule — leaves that window pointing at the robot's shoulder:
        # bag4 solved the entry pose to ep 41 mm and still had to give up because the
        # reach was lateral. MISSION_ALIGN_PERP_CLASSES turns the square-up 90 deg for
        # the named classes so the robot ends up facing the window.
        _perp_cls = self._perp_align_classes()
        psi_star = None
        _yr = det.get("yaw_ref", np.nan)
        _perp_align = det.get("cls") in _perp_cls
        align = ((det.get("cls") in self._align_classes() or _perp_align)
                 and np.isfinite(_yr) and reanchor)
        _align_tol = float(os.environ.get("MISSION_ALIGN_TOL", "0.15"))
        _align_stand = float(os.environ.get("MISSION_ALIGN_STANDOFF", "0.35"))
        timeout = self.p.approach_timeout + (400 if align else 0)
        # Re-anchor on a rolling MEDIAN of recent views, not on the latest single one.
        # Per-frame position noise is largely independent, and a 0.20 m error in the
        # committed estimate is enough to miss the grasp entirely ("never reached it",
        # TCP 0.14 m away) — a median over a handful of views cuts that tail.
        views = [np.asarray(det["pos_w"], float)]
        while self.step_i - t0 < timeout:
            if reanchor and (self.step_i - t0) % 12 == 0:
                # RE-ANCHOR ON THE TRACK, not on a fixed radius. The 0.45 m test was
                # calibrated when match_object guaranteed the first detection was within
                # 0.60 m of the article; without that crutch the first sighting can be a
                # metre out (measured: honest tracks sit a median 0.96 m from their
                # article) and a 0.45 m test then LOCKS the approach to that error —
                # every close-range view of the same article looks like a different
                # object and is thrown away. The tracker already merges by range-aware
                # gate and lets the nearest views set the position, so following the
                # track id is both honest and self-correcting: the closer the robot
                # gets, the better the estimate it is walking to.
                _tname = det.get("name")
                for d2 in self.scan():
                    _same = (d2.get("name") == _tname) if _tname else (
                        d2["cls"] == det["cls"]
                        and np.linalg.norm(d2["pos_w"][:2] - target_w[:2]) < 0.45)
                    if _same:
                        views.append(np.asarray(d2["pos_w"], float))
                        self.nav.object_fix(
                            self.to_base_frame(np.asarray(d2["pos_w"], float))[:2])
                        del views[:-7]
                        target_w = np.median(np.stack(views), axis=0)
                        break
            b = self.to_base_frame(target_w)
            e_psi = 0.0
            if align:
                _, _, _, psi = self.base_pose()
                if psi_star is None and float(np.hypot(b[0], b[1])) < 1.6:
                    # lock the end of the axis that costs the smaller turn
                    cands = ([_yr + np.pi / 2, _yr - np.pi / 2] if _perp_align
                             else [_yr, _yr - np.pi])
                    psi_star = min(cands, key=lambda c: abs(
                        (c - psi + np.pi) % (2 * np.pi) - np.pi))
                    self.log.ev(f"approach: squaring up "
                                f"{'PERPENDICULAR to' if _perp_align else 'to'} "
                                f"the {det['cls']} long axis "
                                f"({np.degrees(_yr):.0f} deg) — turning "
                                f"{np.degrees((psi_star - psi + np.pi) % (2 * np.pi) - np.pi):+.0f} deg")
                if psi_star is not None:
                    e_psi = float((psi_star - psi + np.pi) % (2 * np.pi) - np.pi)
            # LATERAL ABORT. No object is worth being outside the rope for. The
            # scatter puts articles at |lat| <= 1.15 and the robot stands ~0.5 m off
            # one, so a legitimate approach never needs |lat| past ~1.5; the boundary
            # line is 1.98 and the feet reach it from ~1.75. hs21e chased its last
            # redbag out to lat -2.13, took the -10, and ended "off the platform — no
            # way to the finish line from here" at step 77,099 with the whole task-1
            # crossing still unspent. Give the object up instead.
            _wlim = float(os.environ.get("MISSION_APPROACH_LAT_ABORT", "1.70"))
            if _wlim > 0 and abs(self.nav.arc_lat()[1]) > _wlim:
                self.log.ev(f"approach abandoned: the walk has reached lat "
                            f"{self.nav.arc_lat()[1]:+.2f}, past the {_wlim:.2f} abort "
                            f"line — no {det.get('cls')} is worth the boundary")
                self._recentre_after_pick(lim=1.2, goal=0.9)
                return False, target_w
            # while still turning, hold the object further out so the body swings
            # clear of it; the standoff collapses once the heading is nearly there
            _late = self.step_i - t0 > int(timeout * 0.7)
            standoff = _align_stand if (abs(e_psi) > 2 * _align_tol and not _late) else 0.0
            ex = b[0] - aim[0] - standoff
            ey = b[1] - aim[1]
            err = float(np.hypot(ex, ey))
            best = min(best, err)
            # A trotting gait settles around 0.15-0.17 m of residual on this aim point —
            # but the two axes are NOT equal. Per-trial comparison of the champion's 3/6
            # run against an 0/6 run (0826): every HELD grasp started at |y-err| <= 0.03
            # (one at x 0.24, DEEP inside the nominal foot-stomp zone), while six
            # consecutive starts at y -0.13..-0.16 all stalled 0.1-0.2 m short. Depth
            # error is free; LATERAL error kills. So demand the lateral component tight
            # (0.07) and keep the radial gate loose — with a time fallback so an
            # ill-conditioned lateral chase cannot eat the whole approach budget.
            _ey_ok = abs(ey) < 0.07 or _late
            _psi_ok = abs(e_psi) < _align_tol or _late
            if err < 0.15 and _ey_ok and _psi_ok and standoff == 0.0:
                if align and psi_star is not None:
                    self.log.ev(f"approach: squared up, heading error "
                                f"{np.degrees(e_psi):+.0f} deg")
                _blk = self._footprint_blocked(target_w) if reanchor else None
                if _blk is not None:
                    self.log.ev(f"approach abandoned: a remembered {_blk[0]} sits under "
                                f"the body at base ({_blk[1]:+.2f}, {_blk[2]:+.2f}) — "
                                f"crouching here would land on it")
                    return False, target_w
                for _ in range(25):     # hold so the gait settles before the squat
                    self._write_arm(self.travel_q, open_=False)
                    self._step(self.legs.step(torch.zeros(1, 3, device=self.device)))
                return True, target_w
            # Reversing is for trimming an overshoot of a few centimetres, never for
            # travelling. Once the target is properly BEHIND the robot, backing up at the
            # -0.35 m/s cap for the rest of the 900-step timeout walks it 6 m backwards —
            # measured: the robot ended at arc -22, off the back of the course, where the
            # search phase then could not push forward at all. Give up instead and let the
            # caller re-detect; goto_arc knows how to turn around, this loop does not.
            if b[0] < -0.35:
                # The old behaviour abandoned here and the caller notes "approach
                # timeout" — measured as THE dominant per-object loss (9 abandons
                # across s5600d/e/f, each writing off a live 20-point object). The
                # comment below already knew the cure; do it, once: loop around via
                # goto_arc to 2 m before the object and re-run the approach fresh.
                if turned or not reanchor:
                    self.log.ev(f"approach abandoned: target is {-b[0]:.2f} m behind "
                                f"the robot — re-detecting rather than reversing into it")
                    return False, target_w
                turned = True
                _tx, _tw = self._arc_of(target_w)
                self.log.ev(f"approach: target {-b[0]:.2f} m behind — looping around "
                            f"to 2 m before it (arc {_tx:.1f})")
                self.goto_arc(_tx - 2.0, lat_target=float(np.clip(_tw, -1.05, 1.05)))
                self.face_forward()
                t0 = self.step_i
                continue
            # ... and it must not walk off the course while doing it. The course runs
            # arc -21..+21 on a raised platform; a run backed off the start edge to
            # arc -22.0 and could not climb back on, so the search phase then spent the
            # rest of its budget printing "stalled near arc -22.0".
            ax_now, w_now = self.nav.arc_lat()
            if self.nav.honest and os.environ.get("NAV_MINIMAL") != "1" \
                    and (getattr(self.nav, "since_fix", 0) > 700
                         or (getattr(self.nav, "since_fix", 0) > 300
                             and abs(self.nav.lat()) > 1.2)):
                # the approach builds its own walk command, so walk_cmd's starvation
                # scan never runs here — and r41d's dense trace shows exactly this
                # phase walking blind across the boundary (est lat pinned +1.7 while
                # the true pose ran to +5.8). Stop, re-find the belt, retry.
                self.log.ev("approach paused — navigation starving; relocalizing")
                self.nav.reacquire(self)
                return False, target_w
            _x0m = 1.0 if self.nav.honest else 0.6
            # 1.6 -> _edge_lat (1.45): the rope line is 1.98 and the foot guard fired
            # at 1.77 (svF); abandon while the body is still clear of the rope
            if ax_now < self.nav.s_start() + _x0m or abs(w_now) > self._edge_lat() + 0.1:
                self.log.ev(f"approach abandoned: robot at arc {ax_now:.1f} lat "
                            f"{w_now:+.2f} is leaving the course")
                self._reenter_course()
                return False, target_w
            cmd = torch.zeros(1, 3, device=self.device)
            cmd[0, 0] = float(np.clip(0.9 * ex, -0.35, 0.55))
            cmd[0, 1] = float(np.clip(0.9 * ey, -0.25, 0.25))
            # Yaw must aim at the AIM POINT's bearing, not at zero bearing. Driving the
            # object to dead ahead while the position loop drives it to a laterally
            # offset aim point are contradictory goals, and the residual never fell below
            # tolerance — every approach ran to timeout, even with perfect perception.
            bearing = np.arctan2(b[1], max(b[0], 0.2))
            want = np.arctan2(aim[1], max(aim[0], 0.2))
            if align and psi_star is not None:
                # heading loop owns yaw; the position loop strafes the object back
                # to the aim point as the body turns (an orbit at the standoff)
                cmd[0, 2] = float(np.clip(1.5 * e_psi, -0.6, 0.6))
            else:
                cmd[0, 2] = float(np.clip(1.6 * (bearing - want), -0.6, 0.6))
            self._write_arm(self.travel_q, open_=False)
            self._step(self.legs.step(cmd))
        # out of time: still worth a squat if we ever got reasonably close
        b = self.to_base_frame(target_w)
        final = float(np.hypot(b[0] - aim[0], b[1] - aim[1]))
        self.log.ev(f"approach timed out (best {best:.3f} m, final {final:.3f} m)")
        return final < 0.25, target_w

    def _align_classes(self) -> set:
        return set(c for c in os.environ.get(
            "MISSION_ALIGN_CLASSES", "lunchbox").split(",") if c)

    def _perp_align_classes(self) -> set:
        """Classes squared up PERPENDICULAR to their long axis (see approach())."""
        return set(c for c in os.environ.get(
            "MISSION_ALIGN_PERP_CLASSES", "").split(",") if c)

    def _yaw_ref_classes(self) -> set:
        """Classes whose blob principal axis stop_and_look should estimate. Both
        square-up modes need it; without this the perpendicular mode silently never
        fired, because yaw_ref was only ever written for MISSION_ALIGN_CLASSES."""
        return self._align_classes() | self._perp_align_classes()

    @staticmethod
    def aim_y_off(cls) -> float:
        """Desired post-crouch lateral offset of the object from the bank anchor."""
        if cls in ("redbag", "purplebag"):
            return 0.0
        return float(os.environ.get("MISSION_AIM_Y", "0"))

    def _edge_lat(self) -> float:
        """|lat| beyond which a manoeuvre is walking into the boundary rope (the rope
        line sits at ~1.98; the foot guard fired at lat 1.77 in svF and the body was
        hooked at 1.8). MISSION_EDGE_LAT, default 1.45.

        CONFIDENCE-AWARE, because a fixed bar on a drifting estimate is not a guard.
        Measured over 2,622 NAVEST calibration lines this round the lateral estimate is
        excellent — median error 20 mm, p90 80 mm — but not in the one regime that
        matters: over the 56 samples with |true lat| > 1.2 it UNDERSTATES how far out
        the robot is 62 % of the time, p10 -0.58 m. The reason is visible in the fix
        itself: out near the boundary the posts are visible on ONE side only
        (last_fit["sides"] == 1), so the lateral is weakly constrained exactly where the
        guard leans on it. hs18f, hs25d and vh1 all walked off the deck believing they
        were 0.6-0.8 m further in than they were, each costing the whole run. The robot
        knows the geometry of its own fix, so spend that knowledge: pull the bar in
        while the fix is one-sided or stale.
        """
        base = float(os.environ.get("MISSION_EDGE_LAT", "1.45"))
        try:
            f = getattr(self.nav, "last_fit", None) or {}
            if int(f.get("sides", 2)) < 2 or getattr(self.nav, "since_fix", 0) > 200:
                base -= float(os.environ.get("MISSION_EDGE_LAT_UNSURE", "0.35"))
        except Exception:
            pass
        return base

    def bring_into_view(self, target_w, max_steps: int = 420) -> bool:
        """Put an estimated object position inside the head camera's field of view.

        The head cam sits 22 deg down with a 40 deg vertical FOV, so from a standing
        base the ground is visible only from ~0.85 m ahead, inside +-33 deg. Both
        places the mission re-measures an object from — the crouch spot after a failed
        attempt (0.5 m ahead) and a memory pick seen from the centreline (often 40+ deg
        off the nose) — are inside that blind cone: 45 % of all stop-and-looks over the
        s5600 record returned "no confirming view" and the approach then walked on the
        stale estimate (ctlA burned 5,700 steps on one can that way). Turn to face the
        estimate, and back off until it is MISSION_LOOK_DIST (1.25 m) away.
        MISSION_LOOK_DIST=0 disables the manoeuvre.
        """
        want = float(os.environ.get("MISSION_LOOK_DIST", "1.25"))
        if want <= 0:
            return True
        t0 = self.step_i
        _w_prev = self.nav.arc_lat()[1]
        for _ in range(max_steps):
            b = self.to_base_frame(np.asarray(target_w, float))
            dist = float(np.hypot(b[0], b[1]))
            bearing = float(np.arctan2(b[1], b[0]))
            # the reverse gait barely moves on mud: svC stood 1.24 m from a 1.25 m
            # goal for the full 420 steps. Past 150 steps, 0.2 m short is close enough.
            _want = want - (0.20 if self.step_i - t0 > 150 else 0.0)
            if abs(bearing) < 0.20 and dist >= _want:
                if self.step_i - t0 > 0:
                    self.log.ev(f"  brought the target into view: {dist:.2f} m ahead, "
                                f"bearing {np.degrees(bearing):+.0f} deg "
                                f"({self.step_i - t0} steps)")
                return True
            cmd = torch.zeros(1, 3, device=self.device)
            cmd[0, 2] = float(np.clip(1.5 * bearing, -0.6, 0.6))
            if dist < want and abs(bearing) < 0.6:
                # straight back, the object stays on the nose — unless that is walking
                # off the belt: svF backed away from a can that had bounced to lat
                # -1.77, hooked the boundary rope at lat -1.8 and never got free again
                # (6,000 steps of "could not re-enter the course").
                _, _w = self.nav.arc_lat()
                if abs(_w) > self._edge_lat() and abs(_w) >= abs(_w_prev):
                    self.log.ev(f"  look-back stopped at lat {_w:+.2f}: the course edge "
                                f"is behind the robot")
                    # ...and do not just STAND there: swB (mud, arc -4.5) stopped at
                    # -1.63, slid to -1.78 while scanning, -1.91 before the next
                    # look-back could even start, and went over the edge at -1.99.
                    # A stationary walk policy on mud drifts; walk back in.
                    self._walk_inboard(self._edge_lat() - 0.2)
                    return False
                # a nose-inboard robot backs OUTBOARD; from |lat| within 0.35 m of the
                # edge that back-off runs onto the rope/edge before it reaches 1.25 m
                # (swB above started at ~-1.3). Keep the turn, skip the back-off.
                if abs(_w) > self._edge_lat() - 0.35 and self._nose_inboard():
                    self.log.ev(f"  look-back skipped at lat {_w:+.2f}: backing off would "
                                f"walk outboard toward the edge")
                    return False
                _w_prev = _w
                cmd[0, 0] = -0.30
            self._write_arm(self.travel_q, open_=False)
            self._step(self.legs.step(cmd))
        b = self.to_base_frame(np.asarray(target_w, float))
        self.log.ev(f"  could not bring the target into view in {max_steps} steps "
                    f"({np.hypot(b[0], b[1]):.2f} m, bearing "
                    f"{np.degrees(np.arctan2(b[1], b[0])):+.0f} deg)")
        return False

    def stop_and_look(self, det, target_w, steps=70, want=6):
        """Halt and re-measure the target from a STATIONARY base.

        While walking, the camera's cached pose and its rendered frame belong to
        different moments of the gait cycle, which threw the unprojected positions
        0.2-0.3 m below ground. Standing still removes that coupling entirely, so every
        acquisition is refined here before the robot commits to an approach.
        """
        self.bring_into_view(target_w)
        self._walk_inboard(self._edge_lat())      # see re-acquire: never scan at the line
        zero = torch.zeros(1, 3, device=self.device)
        seen, yaws = [], []
        edges, views, hs = [], [], []      # shape, from the same stationary views
        # at an honest low detector rate, stand long enough to actually collect the
        # confirming views instead of giving up two frames in
        steps = max(steps, 20 + self.look_every * want)
        for k in range(steps):
            self._write_arm(self.travel_q, open_=False)
            self._step(self.legs.step(zero))
            if k < 20 or k % self.look_every:
                continue                      # let the gait damp out first
            for d2 in self.scan():
                if d2["cls"] == det["cls"] and \
                        np.linalg.norm(d2["pos_w"][:2] - target_w[:2]) < 0.7:
                    seen.append(np.asarray(d2["pos_w"], float))
                    if np.isfinite(d2.get("yaw", np.nan)) and d2.get("elong", 1.0) > 1.3:
                        yaws.append(float(d2["yaw"]))
                    # the shape of the thing, measured from the same still frames —
                    # the caller keeps the refined POSITION, so it must keep the
                    # matching near edge and height too or the two disagree
                    if d2.get("edge_w") is not None:
                        edges.append(np.asarray(d2["edge_w"], float))
                        views.append(np.asarray(d2["view_w"], float))
                    if np.isfinite(d2.get("h_est", np.nan) or np.nan):
                        hs.append(float(d2["h_est"]))
            if len(seen) >= want:
                break
        if seen:
            refined = np.median(np.stack(seen), axis=0)
            if edges:
                det["edge_w"] = np.median(np.stack(edges), axis=0)
                det["view_w"] = np.median(np.stack(views), axis=0)
            if hs:
                det["h_est"] = float(np.median(hs))
            self.log.ev(f"stop-and-look: {len(seen)} views, estimate moved "
                        f"{np.linalg.norm(refined[:2] - target_w[:2]):.3f} m to "
                        f"{np.round(refined, 2).tolist()}")
            if yaws and det["cls"] in self._yaw_ref_classes():
                # axial mean (mod pi) of the blob principal axes; the sim pose is read
                # ONLY to print the estimation error, never to steer
                yr = float(0.5 * np.arctan2(np.mean(np.sin(2 * np.array(yaws))),
                                            np.mean(np.cos(2 * np.array(yaws)))) % np.pi)
                det["yaw_ref"] = yr
                truth = ""
                for n, o in self.objects.items():
                    _oxy = self._S_of_true(o.data.root_pos_w[0, :2].cpu().numpy())[:2]
                    if self.obj_cls.get(n) == det["cls"] and \
                            np.linalg.norm(_oxy - refined[:2]) < 0.5:
                        ty = float(math_utils.euler_xyz_from_quat(
                            o.data.root_quat_w)[2][0]) % np.pi
                        e = (yr - ty + np.pi / 2) % np.pi - np.pi / 2
                        truth = f", true {np.degrees(ty):.0f} deg (err {np.degrees(e):+.0f})"
                        break
                self.log.ev(f"stop-and-look: long axis {np.degrees(yr):.0f} deg from "
                            f"{len(yaws)} views{truth}")
            return refined
        self.log.ev("stop-and-look: no confirming view, keeping the walking estimate")
        return target_w

    def settle(self, steps=45):
        zero = torch.zeros(1, 3, device=self.device)
        for _ in range(steps):
            self._write_arm(self.travel_q, open_=False)
            self._step(self.legs.step(zero))

    def crouch_and_check(self, section: str) -> bool:
        """Descend with the crouch policy, then gate on how solid the squat is."""
        self.legs.set_mode("crouch", blend_steps=40)
        for _ in range(30):                     # hold the stand command while blending
            self._write_arm(self.travel_q, open_=False)
            self._step(self.legs.step(self.crouch_cmd(0.0)))
        for _ in range(140):                    # descend
            self._write_arm(self.travel_q, open_=False)
            self._step(self.legs.step(self.crouch_cmd(1.0)))
        rolls, pitches, zs = [], [], []
        for _ in range(self.p.stability_steps):
            self._write_arm(self.travel_q, open_=False)
            self._step(self.legs.step(self.crouch_cmd(1.0)))
            r, p, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
            rolls.append(abs(float((r[0] + np.pi) % (2 * np.pi) - np.pi)))
            pitches.append(abs(float((p[0] + np.pi) % (2 * np.pi) - np.pi)))
            zs.append(self._stance_height())
        roll, pitch = float(np.mean(rolls)), float(np.mean(pitches))
        h, hstd = float(np.mean(zs)), float(np.std(zs))
        target_h = float(self.cfg_depth)
        ok = (roll < self.p.max_roll and pitch < self.p.max_pitch
              and abs(h - target_h) < self.p.max_height_err and hstd < 0.035)
        self.log.ev(f"crouch on {section}: roll={roll:.3f} pitch={pitch:.3f} "
                    f"h={h:.3f}(target {target_h:.2f}) std={hstd:.3f} -> "
                    f"{'STABLE' if ok else 'UNSTABLE'}")
        if ok:
            self.terrain_ok[section] = True
        else:
            self.terrain_strikes[section] = self.terrain_strikes.get(section, 0) + 1
        return ok

    def stand_up(self):
        # Ramp the crouch out instead of stepping the command to zero: c=0 in one
        # control step makes the crouch policy fire the base upward and the robot HOPS
        # (user-visible on the probe videos, object still in hand). MISSION_STAND_RAMP=0
        # restores the old step release for A/B.
        if os.environ.get("MISSION_STAND_RAMP", "1") != "0":
            n = 80
            for i in range(n):
                self._write_arm(self.travel_q, open_=False)
                self._step(self.legs.step(self.crouch_cmd(1.0 - (i + 1) / n)))
            for _ in range(40):
                self._write_arm(self.travel_q, open_=False)
                self._step(self.legs.step(self.crouch_cmd(0.0)))
        else:
            for _ in range(110):
                self._write_arm(self.travel_q, open_=False)
                self._step(self.legs.step(self.crouch_cmd(0.0)))
        self.legs.set_mode("walk", blend_steps=40)
        for _ in range(40):
            self._write_arm(self.travel_q, open_=False)
            self._step(self.legs.step(torch.zeros(1, 3, device=self.device)))
        # honest-nav relocalization ritual: a grasp cycle is minutes of belt
        # occlusion + contact slip; r38h navigated 8,500 steps cleanly and then
        # died on the FIRST post-grasp walk-off with a corrupted lateral estimate.
        # Re-find the belt before walking anywhere.
        if self.nav.honest and getattr(self.nav, "since_fix", 0) > 600 \
                and os.environ.get("NAV_MINIMAL") != "1":
            self.nav.reacquire(self)

    def _policy_pick(self, obj, gz: float, b: np.ndarray, dims: np.ndarray, cls: str = "",
                     det: dict | None = None):
        """Choose the pregrasp-bank start config by asking the grasp actor.

        Every kept bank config becomes a candidate step-0 observation (build_virtual, with
        the quaternion the sim produces for it); the actor's first position action is
        projected on the TCP-to-target direction ("toward"). Candidates must be inside
        the trained start distribution (|grasp vec| <= MISSION_PICK_DMAX, 0.16 m),
        command an approach (cos > 0.3), not have been used earlier in this squat, and
        keep the HAND clear of the object:
          * over the object's footprint (half the longer side + 6 cm — the lunchbox
            lid's seam lip is 205 x 115 mm on a 150 x 110 body) the LOWER fingertip pad
            must sit MISSION_PICK_CLEAR (3 cm, plus 3 cm per blocked ramp this squat)
            above the object's top. The pads hang +-GRIP_OPEN along the gripper's x
            axis (arm_joint7 slides along it), so a tilted gripper's low pad is what
            lands first; and the top uses the catalogue's origin-to-top offset where the
            origin is not the centre (the lunchbox's is its bottom face: +0.065, not
            +0.030). Measured on the policy-pick probe: every lunchbox ramp that ended
            0.17-0.45 rad off its config, fingers half shut and the box shoved 0.25 m,
            had a pad within 4 cm of the lid — the old half-height rule put the top
            3.5 cm too low and called that clear.
          * elsewhere the fingertips stay no lower than the grasp point (3 cm slack);
          * the low pad also stays MISSION_PICK_GROUND (2 cm) above the ground the robot
            is standing on (its own stance height), and the config must be in the ramp
            mask when one is loaded (MISSION_BANK_RAMP_MASK).
        Among the survivors the ones that START LIKE TRAINING win: reset_pregrasp_rsi
        dropped the object within +-3 cm of the gripper's xy every episode, so a TCP
        within MISSION_PICK_LAT (4 cm) laterally of the grasp point is the state the
        actor knows; the probe's holds from 0.10-0.15 m out were the actor's own
        close-lift-reopen retry loop getting lucky, and its misses (fingers shut on air
        6 cm from the can, then a climb to 0.4-0.6 m) were that loop diverging. Highest
        "toward" wins inside the preferred set (all survivors when none is that near);
        None = no candidate (the caller keeps the far pick)."""
        pos_b, quat_b = self.bank_pos_sim, self.bank_quat
        n = pos_b.shape[0]
        with torch.no_grad():
            obs = self.obs_builder.build_virtual(pos_b, quat_b, obj, gz)
            act = self.grasp(obs)
        gvec = obs[:, 7:10]
        dist = gvec.norm(dim=1)
        dir_b = math_utils.quat_apply(quat_b, gvec)              # TCP -> target, base frame
        u = dir_b / dist.clamp(min=1e-6).unsqueeze(1)
        a_xyz = act[:, :3]
        toward = (a_xyz * u).sum(dim=1)
        cosang = toward / a_xyz.norm(dim=1).clamp(min=1e-6)
        _ax = torch.tensor([[0.0, 0.0, TCP_OFFSET], [1.0, 0.0, 0.0]], device=self.device)
        tcp_b = pos_b + math_utils.quat_apply(quat_b, _ax[0:1].expand(n, 3))
        x_ax = math_utils.quat_apply(quat_b, _ax[1:2].expand(n, 3))
        pad_low = tcp_b[:, 2] - GRIP_OPEN * x_ax[:, 2].abs()
        est_b = tcp_b + dir_b
        lat = (est_b[:, :2] - tcp_b[:, :2]).norm(dim=1)
        _bt = torch.tensor(b, dtype=torch.float32, device=self.device)
        _cu = cls.upper()
        # Footprint in the object's OWN yaw: catalogue extents along its local x / y (the
        # convention _basket_axes uses), each padded 6 cm. The old circle of the longer
        # half-side was 2-5 cm too wide beside a lunchbox's short side, and that is where
        # the only starts that ever held one sit (probe p1: #322 x3 and #98, TCP 0.14-0.16 m
        # off to the side at lid height, all HELD; every over-the-lid start that reached
        # closed 27-38 mm on the lip's edge and lost it). Circle again when the object is
        # on its side (a local axis near-vertical).
        dxy = tcp_b[:, :2] - _bt[:2]
        if self.obj_honest:
            # Footprint yaw from the BLOB's long axis (the detector's own measurement,
            # decision frame -> base by the pose estimate) when the blob is elongated
            # enough for that axis to mean anything; a circle otherwise. The object's
            # simulator quaternion is not available on the robot.
            _yb, _el = np.nan, 1.0
            if det is not None:
                _yb = float(det.get("yaw_ref", det.get("yaw", np.nan)))
                _el = float(det.get("elong", 1.0) or 1.0)
            if np.isfinite(_yb) and _el > 1.3:
                _c = float(np.cos(_yb - self.base_pose()[3]))
                _s = float(np.sin(_yb - self.base_pose()[3]))
                _axx = torch.tensor([_c, _s], dtype=torch.float32, device=self.device)
                _axy = torch.tensor([-_s, _c], dtype=torch.float32, device=self.device)
                over = ((dxy @ _axx).abs() <= 0.5 * float(max(dims[0], dims[1])) + 0.06) \
                    & ((dxy @ _axy).abs() <= 0.5 * float(min(dims[0], dims[1])) + 0.06)
            else:
                over = dxy.norm(dim=1) < 0.5 * float(max(dims[0], dims[1])) + 0.06
        else:
            _rq_inv = math_utils.quat_inv(self.robot.data.root_quat_w[:1])
            _R = math_utils.matrix_from_quat(math_utils.quat_mul(_rq_inv, obj.data.root_quat_w[:1]))[0]
            _axx, _axy = _R[:2, 0], _R[:2, 1]
            if float(_axx.norm()) > 0.7 and float(_axy.norm()) > 0.7:
                _axx, _axy = _axx / _axx.norm(), _axy / _axy.norm()
                over = ((dxy @ _axx).abs() <= 0.5 * float(dims[0]) + 0.06) \
                    & ((dxy @ _axy).abs() <= 0.5 * float(dims[1]) + 0.06)
            else:
                over = dxy.norm(dim=1) < 0.5 * float(max(dims[0], dims[1])) + 0.06
        _top_off = self.obs_builder.top_off
        if self.obj_honest:
            # b[2] is the detection's GROUND contact, not the object's centre, so the
            # top is the class's whole resting height above it.
            top_z = float(b[2]) + self._det_top_height(cls, det or {})
        else:
            top_z = float(b[2]) + (float(_top_off) if _top_off is not None else 0.5 * float(dims[2]))
        clear = float(os.environ.get(f"MISSION_PICK_CLEAR_{_cu}",
                                     os.environ.get("MISSION_PICK_CLEAR", "0.03"))) + self._pick_clear_bump
        z_ok = torch.where(over, pad_low >= top_z + clear, tcp_b[:, 2] >= est_b[:, 2] - 0.03)
        # The bank was built at a fixed 0.24 m squat; the honest squat settles anywhere in
        # 0.18-0.24 m, so a config whose low pad was 2 cm off the ground when the bank was
        # made can be IN the ground now. Ground = the robot's own stance height (leg
        # kinematics), no map. The no-object sweep's first blocked config was exactly this:
        # hand pointing down, low finger 0.16 m under the base with the base 0.21 m up.
        g_clear = float(os.environ.get("MISSION_PICK_GROUND", "0.02"))
        ground_ok = pad_low >= -self._stance_height() + g_clear
        dmax = float(os.environ.get("MISSION_PICK_DMAX", "0.16"))
        ok = (dist <= dmax) & z_ok & ground_ok & (cosang > 0.3)
        if self.bank_ramp_ok is not None:
            ok &= self.bank_ramp_ok
        for i_used in self._pick_used:
            ok &= (pos_b - pos_b[i_used]).norm(dim=1) > 0.03
        n_ok = int(ok.sum())
        if n_ok == 0:
            _in = dist <= dmax
            self.log.ev(f"  bank pick (policy): no start config qualifies — {int(_in.sum())} "
                        f"within {dmax:.2f} m, {int((_in & z_ok).sum())} of those clear "
                        f"the object (top {top_z:+.3f}, clearance {clear:.2f}), "
                        f"{int((_in & z_ok & ground_ok).sum())} also the ground, "
                        + (f"{int((_in & z_ok & ground_ok & self.bank_ramp_ok).sum())} also "
                           f"ramp-reachable, " if self.bank_ramp_ok is not None else "")
                        + f"{len(self._pick_used)} used — far start")
            return None
        # Per-class window on the start's lateral offset (MISSION_PICK_LATMIN_<CLS> ..
        # MISSION_PICK_LAT_<CLS>): the RSI-like over-the-object start for what the fingers
        # can span, a SIDE start for the lunchbox (nothing to close on from above).
        lat_max = float(os.environ.get(f"MISSION_PICK_LAT_{_cu}",
                                       os.environ.get("MISSION_PICK_LAT", "0.04")))
        lat_min = float(os.environ.get(f"MISSION_PICK_LATMIN_{_cu}",
                                       os.environ.get("MISSION_PICK_LATMIN", "0.0")))
        near = ok & (lat >= lat_min) & (lat <= lat_max)
        n_near = int(near.sum())
        pool = near if n_near else ok
        score = torch.where(pool, toward, torch.full_like(toward, -1e9))
        order = torch.argsort(score, descending=True)[:3]
        rows = []
        for i in order.tolist():
            rows.append(f"#{i} d {float(dist[i]):.3f} lat {float(lat[i]):.3f} toward "
                        f"{float(toward[i]):+.2f} a_z {float(a_xyz[i, 2]):+.2f} "
                        f"a_g {float(act[i, 6]):+.2f} tcp z {float(tcp_b[i, 2]):+.3f} "
                        f"pad {float(pad_low[i]):+.3f}")
        pick = int(order[0])
        self._pick_used.append(pick)
        self._pick_pred = obs[pick].clone()
        self.log.ev(f"  bank pick (policy): {n_ok} candidates within {dmax:.2f} m, {n_near} "
                    f"{lat_min:.2f}-{lat_max:.2f} m laterally (top {top_z:+.3f}, clearance "
                    f"{clear:.2f}, ground {-self._stance_height():+.3f}); " + " | ".join(rows))
        return pick

    def run_grasp(self, name: str, det) -> bool:
        """Champion stage-2 policy, started from a bank pregrasp pose as in training."""
        if self.obj_cls[name] in ("redbag", "purplebag") \
                and os.environ.get("MISSION_BAG_IK", "1") != "0":
            if self.obj_honest:
                # The bag pinch is scripted straight onto the handle bar's simulator
                # pose (a 25 mm bar the camera never resolves) — there is no honest
                # version of it, and bags have never converted anyway.
                self.log.ev(f"grasp skipped {name}: the bag handle IK needs the "
                            f"simulator pose (MISSION_OBJ_HONEST=1)")
                return False
            return self.bag_grasp_ik(name, det)
        obj = self._sim_obj(name)
        self._trace_name = name
        self._pick_pred = None
        # Clear the previous attempt's action, or the trace marks the NEXT object's whole
        # approach as "already in the grasp policy" -- the mud bag showed in_grasp=1 from
        # step 0, which is what a stale _last_a7 looks like.
        self._last_a7 = None
        dims = np.array(OBJ_DIMS[self.obj_cls[name]])
        true_p = obj.data.root_pos_w[0].cpu().numpy()
        det_p = np.asarray(det["pos_w"], dtype=float)
        # sim bookkeeping: the camera error is measured in WORLD (the detection is in
        # the decision frame — S when honest)
        cam_err = (self.nav.true_of_S(det_p) if self.nav.honest else det_p) - true_p
        # MISSION_GRASP_DET_SEED=1: seed the attempt from what the CAMERA reported instead
        # of from object truth. On hardware there is no true_p and therefore no cam_err —
        # the camera's error is already inside the estimate, so the honest substitution is
        # target := detection, cam_err := 0. Everything downstream (the bank-drift gate,
        # the grasp observation, the ground lookup) then runs on a quantity the robot can
        # actually produce. Off by default: the champion policies were trained against a
        # bias distribution clipped to PN_BIAS (12 mm) and the real error is wider.
        self.obs_builder.target_b_fn = None
        self._grasp_target_S = None
        if self.obj_honest:
            # RE-SEE FROM THE CROUCH, the last honest moment before the arm blocks the
            # view. run_grasp is called AFTER crouch_and_check, and the crouch shoves
            # the body ~0.20 m forward relative to where the approach framed the object
            # (measured: post-crouch alignment error [0.20, -0.12] even in the
            # teleport probe). The old target was frozen from the pre-crouch approach
            # detection, so it was already 0.2 m stale before the first finger moved —
            # and honesty means it can never be refreshed again, because from here on
            # the arm occludes the head camera. This one scan, taken stationary and
            # crouched with the object dead ahead at 0.5-0.6 m and the arm still
            # tucked, is the honest equivalent of the live truth refresh the OBJ=0
            # policy got for free (which is the whole 6/6-vs-1/8 gap). Take the
            # nearest same-class detection to the current estimate and re-seed from it.
            if os.environ.get("MISSION_GRASP_RESEE", "1") == "1":
                try:
                    _e0 = np.asarray(det["pos_w"], float)[:2]
                    _best, _bd = None, 1e9
                    # SCOPED NEAR_CUT. 0.40 m exists to reject the robot's own arm,
                    # but right here the arm is still tucked in travel_q and the
                    # object sits 0.3-0.6 m from the head camera — the very pixels
                    # NEAR_CUT throws away. Lowering it for THIS scan only (restored
                    # in finally) is safe: no arm in frame, and no other scan ever
                    # sees the loosened bound.
                    _pcls = type(self.percep)
                    _nc_saved = _pcls.NEAR_CUT
                    _pcls.NEAR_CUT = float(os.environ.get(
                        "MISSION_RESEE_NEARCUT", "0.22"))
                    try:
                        _dets2 = self.scan()
                    finally:
                        _pcls.NEAR_CUT = _nc_saved
                    for _d2 in _dets2:
                        if _d2["cls"] != det["cls"]:
                            continue
                        _dd = float(np.linalg.norm(
                            np.asarray(_d2["pos_w"], float)[:2] - _e0))
                        if _dd < _bd:
                            _best, _bd = _d2, _dd
                    # the pre-crouch estimate is 0.2-0.4 m stale (measured post-crouch
                    # alignment error), so the accept radius must cover that spread
                    _rmax = float(os.environ.get("MISSION_GRASP_RESEE_MAX", "0.55"))
                    if _best is not None and _bd < _rmax:
                        det = {**det, **_best}
                        self.log.ev(f"  re-see from the crouch: {name} refined "
                                    f"{_bd*1000:.0f} mm to "
                                    f"{np.round(_best['pos_w'], 3).tolist()}")
                    else:
                        self.log.ev(f"  re-see from the crouch: no {det['cls']} within "
                                    f"{_rmax:.2f} m of the estimate — keeping the "
                                    f"approach fix")
                except Exception as _e:
                    self.log.ev(f"  re-see skipped: {_e}")
            # OBJECT-HONEST TARGET. The detection is the blob's ground CONTACT LINE —
            # the near edge of the footprint — so the centre is half the object's
            # depth further along the line of sight, and the top is the class's
            # resting height above the ground the robot stands on (a can read as
            # standing when its footprint is round). Frozen for the attempt in the
            # decision frame; every observation re-expresses it through the pose
            # estimate. The sim pose is read below ONLY for the [TRUTH] error line.
            cam_err = np.zeros(3)
            det_p = np.asarray(det["pos_w"], float)
            true_p = self._det_centre_S(self.obj_cls[name], det)
            b = self.to_base_frame(true_p)
            _tb = self.to_base_frame_true(obj.data.root_pos_w[0].cpu().numpy())
            # THE CLEAN ERROR, both sides in the SAME frame. The line below compares a
            # point mapped S->base through the ESTIMATE against one mapped world->base
            # through TRUTH, so it carries the pose estimate's heading error times the
            # range: at 0.40 rad (the measured worst case) and 2 m that is 0.8 m of pure
            # artefact, which is exactly the size of the "detector error" it appears to
            # show. The robot never acts in world — it detects in S and walks in S, and
            # those two errors cancel. So express the article in S as well and take the
            # difference there: this is the only number that says whether the ROBOT is
            # wrong rather than the comparison.
            try:
                _sT = self.nav.S_of_true(obj.data.root_pos_w[0].cpu().numpy())
                _eS = np.asarray(true_p, float)[:2] - np.asarray(_sT, float)[:2]
                self.log.ev(f"  [TRUTH] S-frame target error [{_eS[0]*1000:+.0f}, "
                            f"{_eS[1]*1000:+.0f}] mm (|{float(np.hypot(*_eS))*1000:.0f}| mm) "
                            f"— estimate and article in the SAME frame")
            except Exception:
                pass
            self.log.ev(f"  [TRUTH] detection-seeded centre vs object base-frame "
                        f"{np.round(b - _tb, 3).tolist()} "
                        f"({'near-edge' if det.get('edge_w') is not None else 'contact-point'} form)")
        elif os.environ.get("MISSION_GRASP_DET_SEED") == "1":
            true_p = det_p
            cam_err = np.zeros(3)
            b = self.to_base_frame(true_p)          # decision frame -> body
        else:
            b = self.to_base_frame_true(true_p)
        drift = float(np.linalg.norm(b[:2] - self.grasp_anchor[:2]))
        self.log.ev(f"grasp start {name}: object base-frame {np.round(b, 3).tolist()} "
                    f"vs bank anchor {np.round(self.grasp_anchor, 3).tolist()} "
                    f"(drift {drift:.3f} m), cam err {np.linalg.norm(cam_err):.3f} m")
        # The pregrasp bank only spans about +-0.33 m laterally around the anchor, so an
        # object this far off was never going to be reachable — it means the approach
        # never converged, usually because it was chasing a false detection (seen with
        # cam err 1.96 m). Running the 320-step policy anyway wastes budget and often
        # bats the real object away, so bail out before touching anything.
        if drift > 0.40:
            self.log.ev(f"grasp skipped {name}: {drift:.2f} m outside the pregrasp bank "
                        f"— the approach did not converge")
            # Nothing inside this squat can fix an out-of-bank object, so the retry loop
            # must not spin on it. Measured on the venue run: a lunchbox was lifted 0.21 m
            # and dropped, which batted it 0.50 m away, and the next FOUR retries logged
            # this identical line without touching anything. Only a fresh approach helps.
            self._grasp_out_of_bank = True
            return False
        # The grasp target (and the ground under it) is seeded BEFORE the start pose is
        # chosen: the policy pick below scores candidate starts on the very observation
        # the attempt will open with, and an attempt's first observation must be built
        # from the same seeded point. (This used to sit after the joint ramp; nothing in
        # the ramp reads it.)
        self.obs_builder.reset_target(true_p, dims, cam_err, self.obj_cls[name])
        # ground under the object: the height map under the oracle; the robot's own
        # stance height when honest (it is squatting next to the object)
        gz = self._ground_z() if self.nav.honest else ground_z_world(true_p[0], true_p[1])
        if self.obj_honest:
            # ONE ground for the whole attempt. The detection's z is the ground the
            # detector was handed when the object was SIGHTED, metres back up the
            # course; everything from here on uses the ground the robot is standing on
            # NOW, at the crouch. _policy_pick reads b[2] for its clearance test, so
            # the two have to be the same number or the pick clears the object against
            # one ground and the policy aims at the other.
            b = np.array([b[0], b[1],
                          float(self.to_base_frame(np.array([true_p[0], true_p[1], gz]))[2])])
            # The grasp point, built the way _grasp_point_w builds it but from the
            # CATALOGUE resting height instead of the object's measured half-height:
            # top of the object minus the class's grip depth, never below the ground.
            # Frozen in the decision frame for the attempt; to_base_frame re-expresses
            # it every step, so the policy still tracks it while the base rocks.
            _top = self._det_top_height(self.obj_cls[name], det)
            _tz = max(gz + _top - float(self.obs_builder.grip_depth), gz + 0.010)
            self._grasp_target_S = np.array([true_p[0], true_p[1], _tz], float)
            # HOLD IT IN THE ODOMETRY FRAME, not in the navigation estimate. The
            # estimate is corrected by belt fixes, and a correction lands as a JUMP:
            # a target frozen in S teleports several centimetres mid-attempt and the
            # policy sees the object move. Odometry is the robot's own smooth,
            # never-corrected dead reckoning — over the ~10 s of a grasp its drift is
            # negligible, and it is exactly what the bin memory already rides on. The
            # height rides on the leg kinematics instead: the target sits a fixed
            # distance above the ground the robot is standing on.
            self._grasp_target_h = _tz - gz
            _tb0 = self.to_base_frame(self._grasp_target_S)
            if self.nav.honest:
                from course_mission.navest import _rot2 as _r2
                _o0 = self.nav.odo.copy()
                self._grasp_target_odo = _o0[:2] + _r2(_o0[2], _tb0[:2])

                def _target_b(_r2=_r2):
                    _o = self.nav.odo
                    _xy = _r2(-_o[2], self._grasp_target_odo - _o[:2])
                    _vh = np.array([_xy[0], _xy[1],
                                    -self._stance_height() + self._grasp_target_h])
                    # yaw-aligned, gravity-levelled -> body, through the IMU attitude
                    _q = self.robot.data.root_quat_w
                    _qt = math_utils.quat_mul(
                        math_utils.quat_inv(math_utils.yaw_quat(_q)), _q)
                    return math_utils.quat_apply_inverse(
                        _qt, torch.tensor(_vh, dtype=torch.float32,
                                          device=self.device).reshape(1, 3))
                self.obs_builder.target_b_fn = _target_b
            else:
                self.obs_builder.target_b_fn = lambda: torch.tensor(
                    self.to_base_frame(self._grasp_target_S), dtype=torch.float32,
                    device=self.device).reshape(1, 3)
            _gp = self.obs_builder._grasp_point_w(obj, gz)[0].cpu().numpy()
            self.log.ev(f"  [TRUTH] honest grasp point vs truth grasp point "
                        f"{np.round(self.to_base_frame(self._grasp_target_S) - self.to_base_frame_true(_gp), 3).tolist()} "
                        f"(top {_top:.3f} m from h_est "
                        f"{float(det.get('h_est', float('nan')) or float('nan')):.3f}, "
                        f"true {float(obj.data.root_pos_w[0, 2]) - gz + 0.5 * float(dims[2]):.3f}, "
                        f"grip depth {self.obs_builder.grip_depth:.3f} m)")
        # start pose: a bank config whose fingertips are NOT already on the object
        d_bank = np.linalg.norm(self.bank_ee[:, :2] - b[:2], axis=1)
        far = np.nonzero(d_bank > 0.15)[0]
        pick = int(far[np.argmin(d_bank[far])]) if len(far) else int(np.argmin(d_bank))
        _pick_mode = os.environ.get("MISSION_BANK_PICK", "far")
        # MISSION_BANK_PICK=policy: let the ACTOR choose its own start. Every reachable
        # bank config is scored through the grasp policy on the step-0 observation it
        # would produce (GraspObsBuilder.build_virtual), and the attempt starts from the
        # config whose first action points most directly at the object. Measured on 20
        # honest probe attempts from the "far" pick: the 11 flails all began with the
        # actor commanding motion AWAY from the object (approach score -0.4..-3.9) from
        # 0.17-0.34 m out, while the 7 holds began at 0.08-0.14 m with +3.4..+7.6 —
        # the far start is simply outside what reset_pregrasp_rsi ever trained.
        if _pick_mode == "policy" and self.bank_quat is not None:
            _pp = self._policy_pick(obj, gz, b, dims, self.obj_cls[name], det)
            if _pp is not None:
                pick = _pp
        elif _pick_mode == "policy":
            self.log.ev("  bank pick: MISSION_BANK_PICK=policy needs MISSION_BANK_EE "
                        "(the bank quaternion dump) — far start")
        # MISSION_BANK_PICK=over: start with the fingertips OVER the object instead.
        # Training (graspenv reset_pregrasp_rsi) begins EVERY episode that way — a random
        # bank config, the object dropped under its fingertips within +-3 cm — and the
        # far-anchor variant (GRASP_RSI_FAR) was never switched on, so a start 0.15 m
        # off the object is a state the policy has only ever reached by its own drift.
        # That matches the flail signature (saturated actions from step 1, TCP never
        # inside 0.20 m, 17 of 44 honest attempts; a second try from the same start
        # flails again 59 %). The config is the xy-nearest one whose gripper base still
        # clears the object top by the TCP offset (+3 cm) so the joint ramp lowers the
        # open fingers onto the object rather than sweeping through it; when nothing
        # clears (a standing bottle) the far pick stays.
        if os.environ.get("MISSION_BANK_PICK", "far") == "over":
            _need = float(b[2]) + 0.5 * float(dims[2]) + 0.13 + 0.03
            _over = np.nonzero((d_bank < 0.05) & (self.bank_ee[:, 2] >= _need))[0]
            if len(_over):
                _cost = d_bank[_over] + 0.1 * (self.bank_ee[_over, 2] - _need)
                pick = int(_over[np.argmin(_cost)])
                self.log.ev(f"  bank pick OVER the object: d_xy {d_bank[pick]:.3f} m, "
                            f"gripper base z {self.bank_ee[pick, 2]:+.3f} (needs "
                            f">= {_need:+.3f}), {len(_over)} candidates")
            else:
                self.log.ev(f"  bank pick: nothing clears the object top (needs z >= "
                            f"{_need:+.3f}) — far start d_xy {d_bank[pick]:.3f} m")
        q0 = torch.tensor(self.bank_q[pick], dtype=torch.float32, device=self.device)
        # where the object sat before the hand came down on it (the ramp-blocked check)
        _pre_xy = obj.data.root_pos_w[0, :2].cpu().numpy().copy()

        # COME IN FROM ABOVE, SLOWLY, ON IK -- do not sweep there in joint space.
        # This ramp used to interpolate joint angles straight from wherever the arm was to
        # the bank pose over 60 steps. Joint-space interpolation says nothing about where
        # the HAND goes in between, and on a 27 cm standing paper bag it goes through it:
        # measured, the grass bag stood at 1.2 deg for all 1,040 steps of the approach and
        # was flat 16 steps after this handover, with the nearest FOOT 145 mm away and
        # receding. The arm knocked it over, not the feet, and a toppled bag turns its
        # mouth sideways so nothing can be lowered into it.
        #
        # So: lift the TCP to a clear height first, then descend onto the bank pose's own
        # EE position with the same differential IK the grasp policy uses. Both legs of that
        # are straight lines in CARTESIAN space, which is the space the collision is in.
        # MISSION_APPROACH_CLEAR=0 restores the old joint-space ramp.
        # DEFAULT 0 = the original joint-space ramp. The Cartesian descent was measured
        # across 4 configurations x 4-6 trials and it did not fix the bag (still 0) while
        # the classes that DID convert stopped converting: can 0/6, 0/6, 0/6 across the
        # settle sweep against a 61% first-attempt hold on the old ramp. It also refutes
        # the null-space hypothesis: settling onto the bank JOINT configuration after the
        # descent (MISSION_BANK_SETTLE 40/90) recovered nothing. Whatever the policy needs
        # from its start state, "same position, then same configuration" does not provide
        # it -- most plausibly it needs the DYNAMIC state of the ramp it was trained
        # behind, not any static pose. Kept behind the flag for future experiments.
        _clear = float(os.environ.get("MISSION_APPROACH_CLEAR", "0"))
        _slow = int(os.environ.get("MISSION_APPROACH_STEPS", "150"))
        if _clear > 0.0:
            ee_b = np.asarray(self.bank_ee[pick], float)
            top_b = float(b[2]) + 0.5 * float(dims[2])
            hi = max(top_b, ee_b[2]) + _clear
            here = self.arm.frame_pose()[0][0].cpu().numpy()
            self.servo_tcp_to([here[0], here[1], hi], steps=_slow // 3, grip_open=True)
            self.servo_tcp_to([ee_b[0], ee_b[1], hi], steps=_slow // 2, grip_open=True)
            self.servo_tcp_to(ee_b, steps=_slow, grip_open=True)
            # SETTLE ONTO THE BANK'S OWN JOINT CONFIGURATION.
            # The IK servo above puts the TCP at the bank pose's position, but IK has a
            # null space: the same fingertip position is reachable with the elbow anywhere
            # on a circle, and the policy was trained starting from bank CONFIGURATIONS,
            # not merely from bank positions. Measured after the Cartesian rewrite, the
            # dominant failure moved from "the arm knocked it over" to "never reached it"
            # (3 of 4 trials), which is what a policy started outside its training
            # distribution looks like.
            # Blending the last stretch in joint space is safe HERE and was not safe
            # before: the hand is already at the target, so this resolves the elbow rather
            # than sweeping the hand across the object.
            _settle = int(os.environ.get("MISSION_BANK_SETTLE", "40"))
            for k in range(_settle):
                a = (k + 1) / max(_settle, 1)
                self._write_arm(self.arm.current_q6() * (1 - a) + q0 * a, open_=True)
                self._step(self.legs.step(self.crouch_cmd(1.0)))
        else:
            for k in range(60):
                a = (k + 1) / 60
                self._write_arm(self.arm.current_q6() * (1 - a) + q0 * a, open_=True)
                self._step(self.legs.step(self.crouch_cmd(1.0)))
            # MISSION_RAMP_SETTLE=N: hold the bank target up to N more steps until the
            # arm has actually arrived (max joint error < 0.05 rad). The probe traces
            # show 7 of 19 attempts opening 0.10-0.24 rad off their bank config after
            # the 60-step ramp — the outstretched configs sag and the hand meets the
            # ground — and the policy pick scored the config the arm was SUPPOSED to be
            # in. 0 (default) keeps the fixed ramp.
            for _ in range(int(os.environ.get("MISSION_RAMP_SETTLE", "0"))):
                if float((self.arm.current_q6() - q0).abs().max()) < 0.05:
                    break
                self._write_arm(q0, open_=True)
                self._step(self.legs.step(self.crouch_cmd(1.0)))
        if self._pick_pred is not None:
            # how far the real first observation is from the one the pick was scored on
            # (built from the LIVE hand pose the same way the candidates were, so the
            # difference is the ramp's tracking error and nothing else)
            _d = self.robot.data
            _pb, _qb = math_utils.subtract_frame_transforms(
                _d.root_pos_w, _d.root_quat_w, _d.body_pos_w[:, self.arm.body_idx],
                _d.body_quat_w[:, self.arm.body_idx])
            with torch.no_grad():
                _o = self.obs_builder.build_virtual(_pb, _qb, obj, gz)
            _pe = float((_o[0, 0:3] - self._pick_pred[0:3]).norm())
            _ge = float((_o[0, 7:10] - self._pick_pred[7:10]).norm())
            _qe = float((_o[0, 3:7] - self._pick_pred[3:7]).abs().max())
            _ja = float((self.arm.current_q6() - q0).abs().max())
            self.log.ev(f"  bank pick check: arm {_ja:.3f} rad off the config, obs vs "
                        f"scored — ee pos {_pe:.3f} m, ee quat {_qe:.3f}, grasp vec "
                        f"{_ge:.3f} m (|gvec| {float(_o[0, 7:10].norm()):.3f})")
            # RAMP BLOCKED: the hand met the object on the way down. A ramp that tracks
            # ends 0.03-0.05 rad off its config (j2 gravity sag); one that lands on the
            # lunchbox lid stops 0.15-0.45 rad short with the fingers pushed half shut,
            # and the policy then opens from a state it was never scored on, on an
            # object it has already shoved (0.25 m on the probe). Nothing in the RL loop
            # recovers that, so give the steps to the next start config, demanding 3 cm
            # more clearance over the object from every later pick in this squat, and
            # let the retry loop re-approach when the object was actually moved.
            _ramp_fail = float(os.environ.get("MISSION_RAMP_FAIL", "0.12"))
            if _ramp_fail > 0.0 and _ja > _ramp_fail:
                _mv = float(np.linalg.norm(obj.data.root_pos_w[0, :2].cpu().numpy() - _pre_xy))
                self._pick_clear_bump += 0.03
                if self.obj_honest:
                    # No object truth to measure the shove with: the fingers themselves
                    # say whether the hand came down ON something. An open gripper that
                    # was pushed below 60 mm met the object, and whatever it met is no
                    # longer where the detection put it — re-approach and re-detect.
                    self._grasp_displaced = float(self.arm.gripper_width()[0, 0]) < 0.06
                else:
                    self._grasp_displaced = _mv > float(os.environ.get("MISSION_GRASP_MOVED", "0.20"))
                self.log.ev(f"grasp ramp blocked {name}: arm {_ja:.3f} rad off config #{pick} "
                            f"(width {float(self.arm.gripper_width()[0, 0]):.3f}), object moved "
                            f"{_mv:.2f} m — skipping the attempt, next pick needs "
                            f"{self._pick_clear_bump:.2f} m more clearance"
                            + (" (displaced: re-approach)" if self._grasp_displaced else ""))
                return False
        rest_z = float(obj.data.root_pos_w[0, 2])
        self._trace_rest_z = rest_z
        rest_xy = obj.data.root_pos_w[0, :2].cpu().numpy().copy()
        rest_up = abs(float(math_utils.matrix_from_quat(
            obj.data.root_quat_w[0].unsqueeze(0))[0][2, 2]))   # 1 = standing on its end
        held = 0
        # object-honest bookkeeping: how many steps the fingers were LOADED (a real
        # width against real effort) and how far the TCP rose since they first were
        self._hold_z_load = None
        loaded_n, sensed_lift, _load_run, _empty_run = 0, 0.0, 0, 0
        _gw_prev = float(self.arm.gripper_width()[0, 0])
        # Failure bookkeeping. Returning a bare False makes every failure look the same,
        # but "never reached the object", "reached it and the fingers stayed open" and
        # "closed on it and it slipped" need completely different fixes.
        best_reach, best_lift, closed_near = 9.9, -9.9, 0
        # Narrowest the fingers ever got while commanded shut. This is THE missing
        # number: the whole thin-feature argument turns on whether there was any
        # material between the fingers, and nothing recorded it -- mission.py
        # computes the width every step and throws it into the observation vector.
        # If it settles near the feature's thickness the bite was real; if it goes
        # to ~0 the fingers closed on air, whatever the reach said.
        gw_min = 9.9
        # INSTRUMENTATION for the visible hand tremble. Every magnitude quoted about this
        # policy so far came from RECONSTRUCTED observations, never from a live grasp:
        # offline the deployed checkpoint commands |a_xyz| 3.7-6.2 and |a_rot| 19-28,
        # which through `action6 * scale` (scale 0.02, mission.py) is 75-124 mm and 22-32
        # deg of pose delta per 20 ms step — against a training contract of "2 cm/step".
        # If that holds on-policy it is the tremble, and it is not sensor noise (measured
        # at ~2% of the command). Record it so the cap values are chosen from real data.
        amag, arot, adel, dpw = [], [], [], []
        a_prev, pw_prev, flips = None, None, 0
        # JOINT VELOCITY / TORQUE. This is the safety number, not a cosmetic one: the URDF
        # rates these joints at 10 rad/s (a2_piperx.urdf <limit ... velocity="10">) but the
        # sim config raises velocity_limit_sim to 100 (a2pro_rl/a2pkg/a2piper.py), so the
        # simulator will happily swing the arm ten times faster than the hardware allows
        # and never complain. On the real robot that is where the driver faults out. Record
        # the peaks so we know whether the commanded motion is actually reachable.
        jv_peak, jt_peak = 0.0, 0.0
        # Portable verdict signals (see the sampling block in the loop): narrowest width
        # and peak finger effort over EVERY commanded-shut step, not just the near ones.
        gw_shut, ge_peak = 9.9, 0.0
        _aj = self.arm.joint_ids
        # MISSION_GRASP_STEPS: attempt-length override for the deploy profile. With the
        # arm rate limiter and 1-step command latency on (both mandatory on hardware) the
        # arm tracks the same trajectory SLOWER, and the profile's 340 steps were tuned
        # with neither. Extra steps cost ~2 s each of the 30-min budget and nothing else.
        # EARLY ABORT for a flailing attempt. 61% of failed lunchbox attempts (and a
        # third of all failures) end "never reached it" with the TCP 0.17-0.75 m from
        # the object for the whole 500 steps — the policy is lost from step one, and
        # the remaining ~300 steps buy nothing but clock. If the TCP has not come within
        # MISSION_GRASP_STALL_REACH by MISSION_GRASP_STALL_AT steps, give the retry
        # (a re-approach, or the other policy) the time instead. 0 disables.
        _stall_at = int(os.environ.get("MISSION_GRASP_STALL_AT", "220"))
        _stall_reach = float(os.environ.get("MISSION_GRASP_STALL_REACH", "0.20"))
        # EARLIER STILL, from the policy's OWN estimate. A flail is decided in the first
        # second: on the probe traces the estimated TCP-to-target distance at step 40 is
        # 0.45-0.90 m for a diverging attempt (two at 0.21-0.24) and at most 0.13 m for
        # every attempt that went on to hold. Nothing that is 0.20 m out at step 40 has
        # ever recovered, so hand the remaining ~180 steps to the next start config.
        # Reads the observation the policy itself gets (no sim truth); the best_reach
        # guard only keeps a near miss that is already in contact from being cut. 0 off.
        _div_at = int(os.environ.get("MISSION_GRASP_DIVERGE_AT", "40"))
        _div = float(os.environ.get("MISSION_GRASP_DIVERGE", "0.20"))
        # HOVER, the case both aborts miss: the policy parks 0.12-0.15 m over the object
        # and never closes (probe: best reach 0.14 for all 500 steps, fingers open
        # throughout). Inside the stall radius, so never "stalled"; past step 40 with a
        # believable estimate, so never "diverged". If the policy's own estimate has not
        # once been inside MISSION_GRASP_HOVER (0.10 m) by MISSION_GRASP_HOVER_AT steps
        # it is not going to descend — a hold has the estimate under 0.05 m by step ~60.
        _hov_at = int(os.environ.get("MISSION_GRASP_HOVER_AT", "150"))
        _hov = float(os.environ.get("MISSION_GRASP_HOVER", "0.10"))
        # tactile probe state: how many neighbourhood probes this attempt has spent and
        # when the next "did the fingers find anything" check falls due (see below)
        _probe_i, _probe_due = 0, _stall_at
        est_min = 9.9
        for _k in range(int(os.environ.get("MISSION_GRASP_STEPS", "0")) or self.p.grasp_steps):
            # LIVE HONEST REFRESH (0909). Two days were spent teaching policies to
            # accept a frozen target, because "the arm occludes the camera at grasp
            # range". Both halves of that were over-frozen: the crouched head camera
            # DOES see the object (the re-see proved it at 69-96 mm refinements, with
            # NEAR_CUT scoped to 0.22), and the occlusion argument only bites once
            # the arm actually crosses the view — at which point the DETECTION
            # DISAPPEARS and this block freezes on its own. So while a same-class
            # detection keeps appearing near the current estimate, keep re-seeding
            # the frozen target from it — the sensor-honest version of the live
            # tracking the champion was trained on (6/6 refreshed vs 1/8 frozen).
            if (self.obj_honest and _k > 0
                    and _k % int(os.environ.get("MISSION_GRASP_LIVE_EVERY", "25")) == 0
                    and os.environ.get("MISSION_GRASP_LIVE", "0") == "1"):
                try:
                    _pcls9 = type(self.percep)
                    _nc9 = _pcls9.NEAR_CUT
                    _pcls9.NEAR_CUT = float(os.environ.get(
                        "MISSION_RESEE_NEARCUT", "0.22"))
                    try:
                        _dets9 = self.scan()
                    finally:
                        _pcls9.NEAR_CUT = _nc9
                    _cur9 = np.asarray(self._grasp_target_S, float)[:2]
                    _bb9, _bd9 = None, 0.30
                    for _d9 in _dets9:
                        if _d9["cls"] != self.obj_cls[name]:
                            continue
                        _dd9 = float(np.linalg.norm(
                            np.asarray(_d9["pos_w"], float)[:2] - _cur9))
                        if _dd9 < _bd9:
                            _bb9, _bd9 = _d9, _dd9
                    if _bb9 is not None:
                        _np9 = self._det_centre_S(self.obj_cls[name], _bb9)
                        _sz9 = float(self._grasp_target_S[2])
                        self._grasp_target_S = np.array(
                            [float(_np9[0]), float(_np9[1]), _sz9], float)
                        if getattr(self, "_grasp_target_odo", None) is not None:
                            # the obs reads the ODO carrier live — recompute it from
                            # the new S point exactly the way the setup did (height
                            # model unchanged; xy from the fresh sighting)
                            from course_mission.navest import _rot2 as _r29
                            _o9 = self.nav.odo
                            _tb9 = self.to_base_frame(self._grasp_target_S)
                            self._grasp_target_odo = (
                                _o9[:2] + _r29(_o9[2], _tb9[:2]))
                        if _k % 100 == 0:
                            self.log.ev(f"  live refresh @{_k}: target moved "
                                        f"{_bd9*1000:.0f} mm (camera still sees it)")
                except Exception:
                    pass
            if _stall_at and _k == _stall_at and best_reach > _stall_reach:
                self.log.ev(f"grasp stalled: TCP never within {_stall_reach:.2f} m of "
                            f"{name} in {_k} steps (best {best_reach:.2f}) — aborting early")
                break
            # ...and the honest twin of that abort: the hand DID arrive where it aimed
            # and found nothing there. Under object-honesty "best_reach" is the
            # distance to the estimate, so an attempt converging on a wrong estimate
            # can never trip the stall test above — the fingers are what notice.
            if self.obj_honest and _stall_at and _k == _probe_due \
                    and best_reach <= _stall_reach and loaded_n == 0 and gw_min > 0.005:
                # TACTILE PROBE instead of walking away. "Arrived on its target and
                # closed on nothing" is 6 of 8 honest failures, and the reason is
                # measured: over 62 honest attempts the detection-seeded centre sits a
                # median 42 mm from the article laterally (dx +26 mm, dy -3 mm, p90
                # 386 mm). 42 mm is already outside a 66 mm can's capture window in a
                # 100 mm jaw, and there is no close-range sensor to correct it with —
                # the arm occludes the head camera at grasp range, which is why a
                # FROZEN target scores 1/8 even on perfect truth while a live one
                # scores 6/6. What the robot does have is its fingers: closing on air
                # is an honest measurement that the article is not here.
                # So spend a few hundred steps sweeping the neighbourhood the error
                # actually lives in, rather than a ~4,000-step re-approach cycle that
                # re-measures the same estimate from further away. The pattern is a
                # 45 mm box — the measured median — walked in the odometry frame the
                # target already rides on. The tail (p90 386 mm) is NOT reachable this
                # way and is not meant to be: that population is ghost tracks, and
                # reacquire's ghost write-off is what handles it.
                _pmax = int(os.environ.get("MISSION_GRASP_PROBES", "4"))
                if _probe_i < _pmax:
                    _pd = float(os.environ.get("MISSION_GRASP_PROBE_D", "0.045"))
                    _pat = [(1, 0), (0, 1), (-1, 0), (0, -1),
                            (1, 1), (-1, -1), (1, -1), (-1, 1)]
                    _dx, _dy = _pat[_probe_i % len(_pat)]
                    _off = np.array([_dx * _pd, _dy * _pd], float)
                    # shift both carriers so either target_b_fn branch follows
                    if getattr(self, "_grasp_target_odo", None) is not None:
                        self._grasp_target_odo = self._grasp_target_odo + _off
                    if getattr(self, "_grasp_target_S", None) is not None:
                        self._grasp_target_S = np.asarray(
                            self._grasp_target_S, float) + np.array([_off[0], _off[1], 0.0])
                    _probe_i += 1
                    _probe_due = _k + max(60, _stall_at // 2)
                    best_reach = 9.9
                    gw_min = 9.9
                    self.log.ev(f"  grasp probe {_probe_i}/{_pmax} on {name}: fingers "
                                f"found nothing at the estimate — stepping the target "
                                f"[{_off[0]*1000:+.0f}, {_off[1]*1000:+.0f}] mm and "
                                f"trying again from here")
                    continue
                self.log.ev(f"grasp found nothing: TCP reached its target "
                            f"({best_reach:.2f} m) but the fingers never took a load in "
                            f"{_k} steps, and {_probe_i} probes around it found nothing "
                            f"either — aborting for the next start config")
                break
            obs = self.obs_builder.build(obj, gz)
            est_min = min(est_min, float(obs[0, 7:10].norm()))
            if _div_at and _k == _div_at and best_reach > 0.15 \
                    and float(obs[0, 7:10].norm()) > _div:
                self.log.ev(f"grasp diverged: estimated TCP-target distance "
                            f"{float(obs[0, 7:10].norm()):.2f} m at step {_k} (best reach "
                            f"{best_reach:.2f}) — aborting for the next start config")
                break
            if _hov_at and _k == _hov_at and est_min > _hov and closed_near == 0:
                self.log.ev(f"grasp hovering: estimated TCP-target distance never under "
                            f"{_hov:.2f} m in {_k} steps (min {est_min:.2f}, best reach "
                            f"{best_reach:.2f}, never closed near it) — aborting for the "
                            f"next start config")
                break
            with torch.no_grad():
                a7 = self.grasp(obs)
            amag.append(float(a7[0, :3].norm()))
            arot.append(float(a7[0, 3:6].norm()))
            if a_prev is not None:
                adel.append(float((a7[:, :6] - a_prev[:, :6]).norm()))
                if (float(a7[0, 6]) >= 0) != (float(a_prev[0, 6]) >= 0):
                    flips += 1
            a_prev = a7.clone()
            pw = self.tcp_w()[0].clone()
            if pw_prev is not None:
                dpw.append(float((pw - pw_prev).norm()))
            pw_prev = pw
            _jv = self.robot.data.joint_vel[0, _aj]
            jv_peak = max(jv_peak, float(_jv.abs().max()))
            _jt = getattr(self.robot.data, "applied_torque", None)
            if _jt is not None:
                jt_peak = max(jt_peak, float(_jt[0, _aj].abs().max()))
            self.obs_builder.last_action = a7.clone()
            self.arm.ik_step(a7[:, :6])
            self.arm.set_gripper(bool(a7[0, 6] >= 0))
            self.arm.write_gripper()
            self._step(self.legs.step(self.crouch_cmd(1.0)))
            if self.obj_honest:
                # "reach" without the object: the distance the policy itself is
                # working with (target minus TCP, both in the base frame), and "lift"
                # from the TCP's own rise since the fingers first took a load. Same
                # meaning for every downstream test, no simulator pose.
                _tb = self.obs_builder.target_b_fn()[0].cpu().numpy()
                reach = float(np.linalg.norm(
                    self.arm.frame_pose()[0][0].cpu().numpy() - _tb))
                _ld, _gw_now, _ = self._sensed_load()
                _held_now = False
                _strong_hold = False
                # A LOAD IS A JAMMED FINGER, not a finger passing through a band. The
                # jaw sweeps the whole [3, 99.5] mm band on its way shut every time the
                # policy flips it — and MISSION_GRIP_RATE is 0.02 m PER STEP, so a
                # closing ramp crosses the band in a couple of steps and can hold a
                # saturated effort the entire way. Counting band membership scored
                # "loads" on attempts whose fingers provably shut on air (probe_ohD:
                # 14 loaded steps at gw_min 0.1 mm) and lifts of 0.24 m on objects
                # that never left the ground (hs9c). What an object actually does is
                # STOP the fingers: in-band, commanded shut, and the width not moving.
                _shut = bool(a7[0, 6] < 0)
                _still = abs(_gw_now - _gw_prev) < 0.001
                _gw_prev = _gw_now
                _load_run = (_load_run + 1) if (_ld and _shut and _still) else 0
                _empty_run = 0 if _ld else (_empty_run + 1)
                if _load_run >= 3:
                    loaded_n += 1
                    _z = float(self.arm.frame_pose()[0][0, 2])
                    if self._hold_z_load is None:
                        self._hold_z_load = _z
                    sensed_lift = max(sensed_lift, _z - self._hold_z_load)
                    _held_now = _z - self._hold_z_load >= self._hold_lift
                elif _load_run == 0 and _empty_run >= 3:
                    # DEBOUNCE. The finger encoder crosses out of the band for a step
                    # or two when the object shifts in the jaw; dropping the reference
                    # height on the first miss restarts the lift measurement and a real
                    # hold never reaches its 25 steps (probe_ohD: sensed lift 0.184 m
                    # over 30 loaded steps, and the attempt still ran on and lost it).
                    self._hold_z_load = None
                best_lift = max(best_lift, sensed_lift)
                # STRONG-EVIDENCE HOLD. The consecutive counter below demands ~25
                # unbroken held steps, and a real object shifting in the jaw flickers
                # the load band enough that vm1@4704 lifted a can +0.345 m over 24
                # loaded steps and was still scored MISS (the retry then knocked it
                # away) — 15 such provably-lifted attempts were thrown away across the
                # vm/vr/vp rounds. The INTEGRATED evidence is unfakeable by the
                # closed-on-air modes this file guards against: air-shut fingers read
                # gw ~0.1 mm (this wants >8 mm of real width in the jaw), ground/lid
                # pinches never rise (this wants 3x _hold_lift while loaded), and
                # band-sweep phantoms never bank 15 loaded steps. _ld keeps it live:
                # a hold that just slipped stops qualifying at once.
                if (os.environ.get("MISSION_HOLD_STRONG", "1") == "1"
                        and loaded_n >= int(os.environ.get(
                            "MISSION_HOLD_LOADED_N", "15"))
                        and sensed_lift >= float(os.environ.get(
                            "MISSION_HOLD_LIFT_STRONG", "0.12"))
                        and _gw_now > 0.008 and _ld):
                    _strong_hold = True
            else:
                reach = float((obj.data.root_pos_w[0] - self.tcp_w()[0]).norm())
                best_lift = max(best_lift, float(obj.data.root_pos_w[0, 2]) - rest_z)
            best_reach = min(best_reach, reach)
            if reach < 0.10 and bool(a7[0, 6] < 0):
                closed_near += 1
                gw_min = min(gw_min, float(self.arm.gripper_width()[0, 0]))
            # PORTABLE GRASP VERDICT, recorded on every attempt. held() is object truth
            # and cannot exist on hardware; width and finger effort can. Sampling both
            # whenever the fingers are commanded shut lets the width/effort decision bands
            # be fitted here — the sim half of what would otherwise be a bench-only task.
            if bool(a7[0, 6] < 0):
                gw_shut = min(gw_shut, float(self.arm.gripper_width()[0, 0]))
                _ge = self.arm.grip_effort()
                if _ge is not None:
                    ge_peak = max(ge_peak, _ge)
            # Per-step trace for the overlay video. Off unless someone attached a sink, so
            # a mission run pays nothing: the point is to SHOW what the numbers do rather
            # than argue from a single summary line at the end.
            self._last_a7 = a7
        # (row builder below keeps the hot loop readable)

            if (_held_now or _strong_hold) if self.obj_honest else self.held(obj, rest_z):
                held += 1
                if held >= 25 or (self.obj_honest and _strong_hold):
                    self._log_grasp_motion(name, amag, arot, adel, dpw, flips,
                                           jv_peak, jt_peak, gw_min=gw_min)
                    # The VERDICT line is the width/effort CALIBRATION record, so its
                    # "truth=" column must always be the simulator's answer — in honest
                    # mode `held` is the sensed verdict, and writing that here would
                    # make the record agree with itself by construction.
                    self._log_verdict(name, self.held(obj, rest_z) if self.obj_honest
                                      else True, gw_shut, ge_peak, gw_min,
                                      (float(obj.data.root_pos_w[0, 2]) - rest_z)
                                      if self.obj_honest else best_lift)
                    return True
            elif self.obj_honest:
                # tolerate flicker rather than restarting the 25-step count: a grip that
                # shifts in the jaw for two steps is still a grip
                held = max(0, held - 3)
            else:
                held = 0
        if self.obj_honest:
            # The honest "reach" is the distance to the ESTIMATE, so a policy that
            # converges on a wrong estimate reads as "reached it". What separates the
            # cases without the object is the FINGERS: nothing in the load band means
            # the hand arrived where it aimed and closed on air; a settled load that
            # ended empty means it had it and lost it.
            if best_reach > 0.12:
                why = "never reached its target"
            elif loaded_n == 0:
                why = "arrived on its target and closed on nothing"
            else:
                why = "closed on it but it slipped"
        elif best_reach > 0.12:
            why = "never reached it"
        elif closed_near < 20:
            why = "reached but never closed"
        else:
            why = "closed on it but it slipped"
        # If the gripper MOVED the object, every remaining retry in this squat is aimed
        # at where it used to be: det["pos_w"] is pinned before the retry loop and never
        # refreshed inside it. Measured over the whole record, retry k=0 converts 21/52
        # (40%) while k>=1 converts 4/38 (11%), and after a slip the object travels
        # 0.44-1.48 m — which is also why 51% of all lunchbox grasp-starts are
        # out-of-bank skips. Re-approach instead; the caller re-detects.
        # ... and the lift test alone misses the STANDING can. Seed 0's grass can
        # (trash_1) is batted along the ground, not lifted: attempt 1 gets within
        # 0.03-0.13 m and tips it, the can rolls 0.3-0.5 m, and attempts 2-5 of the
        # same squat "never reach it" at 0.2-0.7 m (svY..swI: 10 squats, the pattern in
        # every one; swC's re-acquire then found it 0.44 m away). Those four dead tries
        # are 1,500-2,000 steps per squat, so the horizontal travel of the object
        # counts as displacement too. Sim truth, like best_lift and held() — the
        # hardware analogue is the re-detection the caller does anyway. 0.20 m, not
        # tighter: the probe's mud can was nudged 0.128 m and the NEXT try in the same
        # squat held it (the bank spans +-0.33 m), so a small nudge stays in-squat.
        if self.obj_honest:
            # Honest "the object is no longer where the detection said": the fingers
            # took a real load at some point (so they had it, and dropped it) or the
            # hand spent long enough shut around the target to have shoved it. Both
            # are gripper signals; the caller re-detects on the re-approach.
            # Displacement, honestly: the fingers took a SETTLED load (so they had the
            # object and lost it) or lifted it. `closed_near` is dropped — it counts
            # steps the hand spent shut near its own ESTIMATE, which says nothing about
            # whether the object moved, and it was sending every near miss back for a
            # fresh approach.
            self._grasp_displaced = (sensed_lift > 0.03 or loaded_n >= int(
                os.environ.get("MISSION_DISP_LOADED_N", "5")))
            _mv = float(np.linalg.norm(obj.data.root_pos_w[0, :2].cpu().numpy() - rest_xy))
            self.log.ev(f"grasp FAILED {name}: est TCP-target {best_reach:.3f} m, "
                        f"sensed lift {sensed_lift:+.3f} m, {loaded_n} loaded steps, "
                        f"{closed_near} steps closed within 10 cm -> {why}"
                        + (" (displaced: re-approach)" if self._grasp_displaced else "")
                        + f"  [TRUTH] object moved {_mv:.2f} m")
            self._log_grasp_motion(name, amag, arot, adel, dpw, flips, jv_peak, jt_peak,
                                   gw_min=gw_min)
            self._log_verdict(name, self.held(obj, rest_z), gw_shut, ge_peak, gw_min,
                              float(obj.data.root_pos_w[0, 2]) - rest_z)
            return False
        moved_xy = float(np.linalg.norm(obj.data.root_pos_w[0, :2].cpu().numpy() - rest_xy))
        # A standing can that TOPPLED is displaced whatever the centre travelled: swC's
        # video (100-140 s) shows it tipping toward the robot and lying at the
        # front-right foot, inside the legs, where no retry from this squat reaches
        # it (0.25-0.47 m for the next four) — re-approach.
        end_up = abs(float(math_utils.matrix_from_quat(
            obj.data.root_quat_w[0].unsqueeze(0))[0][2, 2]))
        toppled = rest_up > 0.7 and end_up < 0.5
        self._grasp_displaced = (best_lift > 0.05 or toppled
                                 or moved_xy > float(os.environ.get("MISSION_GRASP_MOVED", "0.20")))
        self.log.ev(f"grasp FAILED {name}: closest TCP-object {best_reach:.3f} m, "
                    f"max lift {best_lift:+.3f} m, moved {moved_xy:.2f} m"
                    f"{' (toppled)' if toppled else ''}, {closed_near} "
                    f"steps closed within 10 cm -> {why}")
        # Pass the REAL peaks. This call omitted them, so jv_peak/jt_peak fell back to
        # their 0.0 defaults and every failure line reported "joint vel peak 0.0 rad/s |
        # torque peak 0 N.m". Those zeros are a default argument, not a measurement, and
        # they were read as evidence that the arm was never driven at all.
        self._log_grasp_motion(name, amag, arot, adel, dpw, flips, jv_peak, jt_peak,
                               gw_min=gw_min)
        self._log_verdict(name, False, gw_shut, ge_peak, gw_min, best_lift)
        return False

    def _log_verdict(self, name, truth: bool, gw_shut, ge_peak, gw_min, lift):
        """One line per attempt pairing the SIM-TRUTH verdict with the signals a real
        robot would have. Feeding these into a confusion matrix is how the width/effort
        decision bands get fitted without the PiPER-X bench (DEPLOY.md, WP13).

        PEAKS ARE NOT ENOUGH — measured on the first probe: an attempt that lifted the can
        0.231 m and then lost it recorded effort=45.00 (saturated) exactly like a success.
        "Did the fingers ever load up" and "are the fingers loaded NOW" are different
        questions, and only the second one can gate carrying an object to the bin. So the
        line also carries the state at the moment the attempt ends.
        """
        gw_now = float(self.arm.gripper_width()[0, 0])
        ge_now = self.arm.grip_effort()
        self.log.ev(f"[VERDICT] {name} truth={'HELD' if truth else 'MISS'} "
                    f"gw_shut={gw_shut*1000:.1f}mm gw_min={gw_min*1000:.1f}mm "
                    f"effort_peak={ge_peak:.2f} "
                    f"gw_end={gw_now*1000:.1f}mm effort_end="
                    f"{-1.0 if ge_now is None else ge_now:.2f} lift={lift:+.3f}")

    def _trace_row(self, name, obj, reach, a7, rest_z):
        """One row of the per-step grasp trace, including the OBJECT's own state.

        The object's tilt is in here because the failure this is meant to show is the bag
        being knocked over during the approach: once it topples, its mouth faces sideways
        and a top-down gripper cannot enter it. Tilt is the angle between the object's own
        up-axis and world up, so 0 is upright and 90 is flat on its side.
        """
        q = obj.data.root_quat_w[0]
        w, x, y, z = (float(v) for v in q)
        up_z = 1.0 - 2.0 * (x * x + y * y)          # world z of the body's local +z
        tilt = float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))
        op = obj.data.root_pos_w[0]
        feet = self.robot.data.body_pos_w[0, self._foot_idx] if self._foot_idx is not None \
            else None
        foot_d = float((feet[:, :2] - op[:2]).norm(dim=1).min()) if feet is not None else -1.0
        # a7 is None until the grasp policy starts stepping; the approach still records
        # everything that describes the OBJECT, which is the part in question.
        return {
            "step": self.step_i, "name": name,
            "reach": reach,
            "width": float(self.arm.gripper_width()[0, 0]),
            "cmd_close": bool(a7[0, 6] < 0) if a7 is not None else False,
            "a_rot": float(a7[:, 3:6].norm()) if a7 is not None else float("nan"),
            "a_xyz": float(a7[:, :3].norm()) if a7 is not None else float("nan"),
            "in_grasp": 1.0 if a7 is not None else 0.0,
            "lift": float(op[2]) - rest_z,
            "tilt": tilt,
            "foot_d": foot_d,
        }

    def _log_grasp_motion(self, name, amag, arot, adel, dpw, flips,
                          jv_peak=0.0, jt_peak=0.0, gw_min=None):
        """One line of what the arm was actually commanded to do during a grasp."""
        if not amag:
            return
        import numpy as _np
        self.log.ev(
            f"grasp motion {name}: |a_xyz| med {_np.median(amag):.2f} p90 "
            f"{_np.percentile(amag, 90):.2f} -> {1000 * 0.02 * _np.median(amag):.0f} mm/step | "
            f"|a_rot| med {_np.median(arot):.2f} -> "
            f"{_np.degrees(0.02 * _np.median(arot)):.0f} deg/step | "
            f"d|a| med {(_np.median(adel) if adel else 0):.2f} | "
            f"TCP path {1000 * sum(dpw):.0f} mm over {len(dpw)} steps "
            f"({1000 * (_np.median(dpw) if dpw else 0):.1f} mm/step) | grip flips {flips} | "
            f"joint vel peak {jv_peak:.1f} rad/s (URDF limit 10) "
            f"{'OVER HARDWARE LIMIT' if jv_peak > 10.0 else 'ok'} | "
            f"torque peak {jt_peak:.0f} N.m (limit 100)"
                  + (f" | finger width at closure {gw_min*1000:.1f} mm"
                     if gw_min is not None and gw_min < 9.0 else " | never closed"))

    def tcp_w(self):
        d = self.robot.data
        return d.body_pos_w[:, self.arm.body_idx] + math_utils.quat_apply(
            d.body_quat_w[:, self.arm.body_idx],
            torch.tensor([[0.0, 0.0, 0.135]], device=self.device))

    def held(self, obj, rest_z: float) -> bool:
        """Lifted clear of its resting height AND moving with the gripper."""
        p = obj.data.root_pos_w[0]
        if float(p[2]) - rest_z < 0.05:
            return False
        d = self.robot.data
        ee_p = d.body_pos_w[:, self.arm.body_idx]
        ee_q = d.body_quat_w[:, self.arm.body_idx]
        tcp = ee_p + math_utils.quat_apply(
            ee_q, torch.tensor([[0.0, 0.0, 0.135]], device=self.device))
        if float((p.unsqueeze(0) - tcp).norm()) > 0.14:
            return False
        ee_v = d.body_lin_vel_w[:, self.arm.body_idx]
        return float((obj.data.root_lin_vel_w - ee_v).norm()) < 0.25

    def play_traj(self, waypoints, grip_open: bool, steps_each=220):
        q = self.arm.current_q6()
        for wp in waypoints:
            tgt = torch.tensor(wp, dtype=torch.float32, device=self.device)
            for s in range(steps_each):
                a = (s + 1) / steps_each
                self._write_arm(q * (1 - a) + tgt * a, open_=grip_open)
                self._step(self.legs.step(self.crouch_cmd(1.0) if self.legs.mode == "crouch"
                                          else torch.zeros(1, 3, device=self.device)))
            q = tgt

    def deposit_to_basket(self, name: str) -> bool:
        t = self.deposit
        # A shallow lid pinch (lunchbox, 8 mm) holds well enough to lift but not to
        # survive the swing at normal speed: the box was grasped, then found on the
        # ground 0.43 m behind the basket. Fragile grips get a slower trajectory so
        # inertia does not peel them out of the fingers.
        _, depth = grasp_hint(self.obj_cls[name], OBJ_DIMS[self.obj_cls[name]])
        # Rough ground gets the slow swing too: s5600j lost the SAME can twice on
        # pebble mid-swing (found at rel x +0.48 on the ground, "MISS" both cycles)
        # while the identical deposit banks fine on grass/mud.
        try:
            if self.nav.honest:
                # no section map: the IMU's attitude-jitter EMA says "rough ground"
                _rough_ground = self.nav.rough() > float(
                    os.environ.get("NAVEST_ROUGH_ON", "0.004"))
            else:
                _rough_ground = section_of(self.nav.arc_lat()[0]) in ("pebble", "mixed")
        except Exception:
            _rough_ground = False
        each = 380 if (depth < 0.012 or _rough_ground) else 220
        # MISSION_DEPOSIT_SCALE < 1 shortens every leg of the swing (ctlA: 1,000-1,480
        # steps per deposit, 3-5 deposits a run); the solid can/rind grips (gw 50-65
        # mm at effort 32) have never peeled out on flat ground
        _ds = float(os.environ.get("MISSION_DEPOSIT_SCALE", "1.0"))
        each = int(each * _ds)
        # SPREAD the landings. With 3+ aboard the objects cluster within ~8 cm of the
        # fixed release point and jam each other's retrieval (measured: 2-aboard
        # retrieval 7/7, 3-aboard 3/6 — both failures had a neighbour at 8 cm). A
        # per-slot twist of the release joint 1 moves the TCP (armkin, z unchanged):
        # +0.20 -> (-0.026,-0.271), -0.20 -> (-0.117,-0.177), max slot spacing 13 cm.
        # ...but slot by CLASS, not by count. The 200 mm rind is gripped near one end
        # and hangs ~9 cm forward of the TCP, so the forward slot (+0.20, TCP x -0.026)
        # lands it on the FRONT wall (x +0.074): svA rel x +0.063 rim-perched at z 0.258
        # and fell off within the first metres; svE rel x +0.111, on the ground. ctlA's
        # two rinds went in from the aft/middle slots. Cans are not immune: svF's can,
        # gripped off-centre after a 0.21 m sideways grasp, clipped the same wall and
        # bounced out to x +0.23. Tally over all runs — aft slot 5/5 IN, forward slot
        # cans 6/7, rinds 0/2 — so EVERY first object goes aft (protrusion is toward
        # the basket centre there); only a second can takes the forward slot, which
        # keeps the 13 cm spread that retrieval needs. A second rind goes mid instead.
        # ...and the FORWARD slot is out of RETRIEVAL reach, which the spread argument
        # above never priced in. Slot +0.20 releases at TCP y -0.271, an object then
        # settles at y -0.30, and the arm's in-basket window ends near y -0.26.
        # Measured across hs16a/hs16b: every basket pick from y -0.19..-0.23 converted
        # (residual 5-9 mm), every one from y -0.27..-0.31 stalled at 44-121 mm and was
        # abandoned ("outside the arm's in-basket envelope") — three objects, 60 points,
        # all scored `none` while sitting in the basket. A 7 cm spread that can be
        # picked beats a 13 cm spread that cannot, so the forward slot is now the last
        # resort for both classes instead of the second choice for cans.
        _pref = (-0.20, 0.0, 0.11, 0.20)
        _used = {self._basket_slot.get(n) for n, _ in self.carried}
        _slot_dq1 = next((s for s in _pref if s not in _used),
                         _pref[len(self.carried) % 4])
        self._basket_slot[name] = _slot_dq1
        _q_rel = list(t["q_basket"])
        _q_rel[0] = float(_q_rel[0]) + _slot_dq1
        # ...and release LOW. From the stock 0.313 m TCP a can falls ~15 cm and ROLLS to
        # the outboard wall every time (slot 0/1 of run basket10 aimed 9 cm apart and
        # landed 2.4 cm apart, both on the wall line y -0.34..-0.35 — rolling erases the
        # aim). Shoulder +0.16 rad lowers the TCP to 0.260 (armkin; all four slot poses
        # stay inside the v2 opening), halving the drop and the roll-out.
        _q_rel[1] = float(_q_rel[1]) + 0.16
        # One leg at a time, checking the load between legs: a can that peeled out at
        # the staging lift (svA: found 0.8 m off to the side, "MISS") used to ride the
        # whole 1,000-step swing, release and return before anyone noticed. Bail at
        # the first empty hand and say WHICH leg lost it.
        obj = self._sim_obj(name)
        for _ in range(30):     # let the just-lifted object stop swinging first
            self._write_arm(open_=False)
            self._step(self.legs.step(self.crouch_cmd(1.0)))
        for _leg, _wp in (("staging", t["q_staging"]), ("mid", t["q_mid"]),
                          ("basket", _q_rel)):
            self.play_traj([_wp], grip_open=False, steps_each=each)
            if self.obj_honest:
                # the fingers themselves say whether the load survived the leg
                _ld, _gw, _ge = self._sensed_load()
                if not _ld:
                    self.log.ev(f"  deposit swing lost {name} at the {_leg} leg "
                                f"(width {_gw * 1000:.0f} mm, effort "
                                f"{-1.0 if _ge is None else _ge:.1f})")
                    self.play_traj([t["q_staging"]], grip_open=True,
                                   steps_each=int(180 * _ds))
                    self.log.ev(f"basket deposit {name}: hand empty at the {_leg} leg "
                                f"-> MISS")
                    return False
                continue
            _ob = self.to_base_frame_true(obj.data.root_pos_w[0].cpu().numpy())
            _tb = self.to_base_frame_true(self.tcp_w()[0].cpu().numpy())
            _gap = float(np.linalg.norm(_ob - _tb))
            if _gap > 0.22:
                self.log.ev(f"  deposit swing lost {name} at the {_leg} leg (TCP gap "
                            f"{_gap:.2f} m, object base-frame {np.round(_ob, 2).tolist()})")
                self.play_traj([t["q_staging"]], grip_open=True,
                               steps_each=int(180 * _ds))
                rel = self.to_base_frame_true(obj.data.root_pos_w[0].cpu().numpy())
                self.log.ev(f"basket deposit {name}: rel={np.round(rel, 3).tolist()} -> MISS")
                return False
        _tb = self.to_base_frame_true(self.tcp_w()[0].cpu().numpy())
        # Where the hand let go, from the arm's own FK — the only address the robot has
        # for an object inside its basket, and what the blind retrieval aims at.
        self._basket_slot_p[name] = self.arm.frame_pose()[0][0].cpu().numpy().copy()
        self.log.ev(f"  release {name} from slot {_slot_dq1:+.2f}: TCP base-frame "
                    f"{np.round(_tb, 3).tolist()} (FK "
                    f"{np.round(self._basket_slot_p[name], 3).tolist()}, aboard "
                    f"{len(self.carried)})")
        for _ in range(int(70 * _ds)):                         # release
            self._write_arm(open_=True)
            self._step(self.legs.step(self.crouch_cmd(1.0)))
        for _ in range(int(90 * _ds)):                         # let it settle
            self._write_arm(open_=True)
            self._step(self.legs.step(self.crouch_cmd(1.0)))
        if os.environ.get("MISSION_DEPOSIT_TAMP", "1") == "1":
            # TAMP (0908). Three deliveries died the same way — deposit read
            # "IN (sensed)", and at the bins the basket was empty with the object
            # 8.5 m back on the course (fv20/fv31; fv22 likewise): a rim/lip percher
            # passes the loaded-release verdict and rides off at the first jolt.
            # The arm is still directly over the opening, so spend ~220 steps making
            # the case impossible: press the open hand 10 cm down into the opening
            # (below rim level anything is on the sloped floor's catchment), sweep
            # inboard toward the near wall the slope drains to, and withdraw.
            _sp = self._basket_slot_p[name]
            self.servo_tcp_to(_sp + np.array([0.0, 0.0, -0.10]), steps=90,
                              grip_open=True)
            self.servo_tcp_to(_sp + np.array([0.0, 0.04, -0.10]), steps=60,
                              grip_open=True)
            self.servo_tcp_to(_sp + np.array([0.0, 0.0, 0.06]), steps=70,
                              grip_open=True)
        self.play_traj([t["q_staging"]], grip_open=True, steps_each=int(180 * _ds))
        obj = self._sim_obj(name)
        rel = self.to_base_frame_true(obj.data.root_pos_w[0].cpu().numpy())
        bc = t["basket_center_b"]
        if self.obj_honest:
            # Nothing on the robot can look into its own basket (the head camera sees
            # forward and down, the basket rides behind the shoulders), so the honest
            # verdict is the bookkeeping one: the hand was still loaded at the slot and
            # the jaw opened there. A landing that then bounces out is caught later by
            # the retrieval miss, not by a pose read.
            _ok_h = bool(abs(rel[0] - bc[0]) < 0.17 and abs(rel[1] - bc[1]) < 0.13
                         and 0.04 < rel[2] < 0.22)
            self.log.ev(f"basket deposit {name}: released loaded at slot "
                        f"{_slot_dq1:+.2f} -> IN (sensed)  [TRUTH] rel="
                        f"{np.round(rel, 3).tolist()} {'IN' if _ok_h else 'MISS'}")
            return True
        # z: floor 0.09, wall top 0.24 in the base frame; a can lying on the floor
        # reads 0.13-0.16, a rind 0.11, a can on a can 0.19. svA's rind read 0.258 —
        # balanced on the rim — was called IN and was gone 3 m later. Above 0.22 it is
        # on top of the wall, not inside.
        _in_xy = bool(abs(rel[0] - bc[0]) < 0.17 and abs(rel[1] - bc[1]) < 0.13)
        ok = bool(_in_xy and 0.04 < rel[2] < 0.22)
        # LEANING, NOT OUT. svP trash_3 read z 0.221 inside the footprint (a can propped
        # on end against the wall) — 1 mm over the bar, called MISS, and the robot then
        # spent 2,000 steps hunting for it on the ground, wrote it off "knocked 1.4 m
        # away" and hauled it, unlisted, to the bins. Inside the footprint and below
        # the wall-top band (0.22..0.27) it is ABOARD but unsettled: press it down with
        # the open jaw, re-read, and only then decide. Above 0.27 it really is on the rim.
        if not ok and _in_xy and 0.22 <= rel[2] < 0.27 \
                and int(os.environ.get("MISSION_DEPOSIT_NUDGE", "1")):
            _rw = [float(v) for v in os.environ.get(
                "MISSION_BASKET_REACH", "-0.15,0.05,-0.34,-0.19").split(",")]
            _tg = rel.copy()
            _tg[0] = float(np.clip(_tg[0], _rw[0], _rw[1]))
            _tg[1] = float(np.clip(_tg[1], _rw[2], _rw[3]))
            self.log.ev(f"  {name} sits high at z {rel[2]:.3f} inside the basket "
                        "footprint — pressing it down with the open jaw")
            # enter the basket the way pick_from_basket does (its q_mid/q_pick are the
            # reachability-checked in-basket waypoints), press, and leave the same way
            _r = self.retrieve if self.retrieve is not None else {"q_mid": t["q_mid"],
                                                                  "q_pick": t["q_mid"]}
            self.play_traj([_r["q_mid"], _r["q_pick"]], grip_open=True,
                           steps_each=int(150 * _ds))
            self.servo_tcp_to(_tg + np.array([0.0, 0.0, 0.08]), steps=90, grip_open=True)
            self.servo_tcp_to(_tg + np.array([0.0, 0.0, -0.04]), steps=90, grip_open=True)
            self.servo_tcp_to(_tg + np.array([0.0, 0.0, 0.10]), steps=60, grip_open=True)
            self.play_traj([_r["q_mid"], t["q_staging"]], grip_open=True,
                           steps_each=int(150 * _ds))
            rel = self.to_base_frame_true(obj.data.root_pos_w[0].cpu().numpy())
            ok = bool(abs(rel[0] - bc[0]) < 0.17 and abs(rel[1] - bc[1]) < 0.13
                      and 0.04 < rel[2] < 0.22)
            self.log.ev(f"  after the nudge: rel={np.round(rel, 3).tolist()}")
        self.log.ev(f"basket deposit {name}: rel={np.round(rel, 3).tolist()} -> "
                    f"{'IN' if ok else 'MISS'}")
        return ok

    # ---------------------------------------------------------------- bin phase
    def _resight_bin(self, colour: str, tgt_arc: float, tgt_lat: float = 0.0) -> bool:
        """Back out to the landmark window and look at the bin row again.

        Everything about a drop — the stance the approach walks to and the re-centre at
        the rim — rests on ONE camera fix of the row, carried forward by raw odometry.
        Standing at the bins to fish the next object out of the basket costs 1,500-3,000
        steps and 2-4 m of shuffling, and by the second drop that fix has expired: the
        fallback is the row's position in the S frame, which carries the pose estimate's
        lateral error, and correcting from it is a coin flip (hs12a's +89 mm would have
        rescued a drop, hs13a's +44 mm pushed one onto the rim). The bins cannot be seen
        from the 0.6 m drop stand-off — one bin fills the frame — so the only honest
        answer is to walk back out to 2.4-3.5 m and look. ~900 steps against 15 points.
        """
        if not self.nav.honest or not int(os.environ.get("MISSION_BIN_RESIGHT", "1")):
            return False
        _ax0, _ = self.nav.arc_lat()
        if (tgt_arc - _ax0) > 2.2:
            return False                      # already outside; the walk-in will look
        self.log.ev(f"  {colour} bin sighting is stale "
                    f"({self.nav.bin_obs_age(colour)} steps since it was SEEN) — "
                    f"backing out to look at the row again before the drop")
        # Back off at 0.45, not 0.32: this walk policy marches in place below ~0.3 m/s,
        # and the first attempt spent its whole 900-step budget covering 2 m — the
        # bins fix it was waiting for landed 57 steps AFTER the loop gave up
        # (hs13a @49093 "re-sight done: 3143 steps old", @49150 "[NAVFIX] bins ...
        # rng 2.7"). Exit on the FIX, which is the thing being waited for, not on a
        # distance-plus-fix conjunction.
        for _ in range(int(os.environ.get("MISSION_BIN_RESIGHT_STEPS", "1200"))):
            _ax, _ = self.nav.arc_lat()
            _gap = tgt_arc - _ax
            if self.nav.bin_obs_age(colour) < 150:
                break
            # HOLD THE BIN'S OWN LANE. Steering to the course centre while backing out
            # walks the target bin out of the +-33 deg field of view and re-sights the
            # NEIGHBOURS instead: hs15b backed off the blue bin, drifted lat 0.91 ->
            # 0.21, and came back with a fresh fix on green and red while blue stayed
            # 4,606 steps old.
            # STAND IN THE DETECTOR'S WINDOW, not past it. detect_bins only accepts a
            # row inside NAVEST_BIN_NEAR_RANGE (3.5 m), and backing off until "_gap >
            # 3.2" put the robot AT that edge: twice the loop timed out and the fix
            # landed 60-80 steps later, during the walk back in (hs15b @21550,
            # hs16b @20350). Hold 2.6-3.0 m, where the row is comfortably inside.
            if _gap < 2.6:
                _vx = -0.45                    # still too close: keep backing
            elif _gap > 3.0:
                _vx = 0.25                     # overshot the window: ease back in
            else:
                _vx = 0.0                      # in the window: stand and look
            cmd, _, _ = self.walk_cmd(float(np.clip(tgt_lat, -1.05, 1.05)), vx=_vx)
            self._write_arm(self.travel_q, open_=False)
            self._step(self.legs.step(cmd))
        _fresh = self.nav.bin_rel_body(colour) is not None
        self.log.ev(f"  re-sight done: {self.nav.bin_obs_age(colour)} steps old now "
                    f"({'body-relative fix live' if _fresh else 'still no live fix'})")
        return _fresh

    def goto_bin(self, colour: str) -> bool:
        # The carry-speed cap must not throttle THIS approach (see walk_cmd._bin_final).
        self._bin_final = True
        try:
            return self._goto_bin_body(colour)
        finally:
            self._bin_final = False

    def _goto_bin_body(self, colour: str) -> bool:
        # abort only when the lost item IS the (sole) cargo. r7: after a wreck
        # emptied `carried`, the vacuous any() kept aborting every goto_bin —
        # including the finish-line dash, which uses this routine.
        _cl = getattr(self, "_carry_lost", None)
        if _cl is not None and [n for n, _ in self.carried] == [_cl]:
            self.log.ev(f"  hand cargo {_cl} lost mid-carry and nothing "
                        "else aboard — aborting the bin trip")
            return False
        if _cl is not None and _cl not in [n for n, _ in self.carried]:
            self._carry_lost = None          # stale flag from a cleaned-up wreck
        tgt = self.nav.bin_pos(colour)
        if tgt is None:
            # honest mode, bins never sighted yet: walk the course forward until the
            # trio comes into detect_bins range (1.6-4 m band, measured 7/7). "The
            # bins are at the end of the course" is an event rule, not map knowledge.
            self.log.ev("  bins not yet sighted — walking the course forward to find them")
            # ...FORWARD, not whichever way the robot happens to be pointing. This loop
            # commands 0.5 m/s along the BODY axis: hs5d entered it with the nose 138
            # deg off the tangent, drove that way while the yaw loop unwound, and was
            # over the side 395 steps later with three objects aboard (60 points).
            # hs4d and hs3d simply walked BACKWARD down the course logging "forward".
            # Come in before turning, like every other spin in this file: a spin sweeps
            # the feet ~0.3 m around the base.
            self._walk_inboard(self._edge_lat())
            self.face_forward()
            for _ in range(4000):
                if self.nav.bin_pos(colour) is not None:
                    break
                if self.off_platform:      # nothing below is meaningful any more
                    break
                cmd, _, _ = self.walk_cmd(0.0, vx=0.5)
                self._write_arm(self.travel_q, open_=False)
                self._step(self.legs.step(cmd))
            tgt = self.nav.bin_pos(colour)
            if tgt is None:
                self.log.ev("  bins never came into view — giving up the trip")
                return False
        # the drop pose sits tcp_drop_b[1] to the robot's right; stand that much to the
        # other side so the released object is over the bin's centreline, not its wall
        _drop_dy = float(self.retrieve["tcp_drop_b"][1]) if self.retrieve is not None else 0.0
        # The bins stand INSIDE the belts (event rule), so a lateral target beyond the
        # corridor is a sighting error, never a place to walk to. And a FAR sighting
        # is a tangent-line projection — the honest S frame is the local tangent, and
        # across the curving course the centreline itself sits 1.2 m off that line at
        # 7 m. hs3b: "blue bin at w +1.8" seen from the stairs (true +0.9), the robot
        # chased it to the deck edge (OOB at lat 1.78), then hung there for 3,000
        # steps and shook the hand-carried lunchbox out. Beyond 3 m hold the course
        # centre and let the sighting refresh on the way in.
        _lat_cap = self._edge_lat() - 0.35          # 1.10: the blue stance is 0.98-1.05
        _far_m = float(os.environ.get("MISSION_BIN_FAR", "3.0"))

        def _aim(tg, ax_now):
            if self.nav.honest and (tg[0] - ax_now) > _far_m:
                return (tg[0], 0.0)
            return (tg[0], float(np.clip(tg[1] - _drop_dy, -_lat_cap, _lat_cap)))
        tgt = _aim(tgt, self.nav.arc_lat()[0])
        # (see _resight_bin: the walk-in itself refreshes the fix when the robot is
        # still outside the landmark window, so this only fires when goto_bin is
        # called from close in — the second bin of a stop)
        if self.nav.honest and (self.nav.bin_rel_body(colour) is None
                                or self.nav.bin_obs_age(colour) > int(
                                    os.environ.get("MISSION_BIN_RESIGHT_AGE", "2000"))):
            if self._resight_bin(colour, tgt[0], tgt[1]):
                _bp2 = self.nav.bin_pos(colour)
                if _bp2 is not None:
                    tgt = _aim(_bp2, self.nav.arc_lat()[0])
        t0 = self.step_i
        stall_ax, stall_at = 1e9, self.step_i
        # The first bin can be 30 m away (the last pick happens mid-course), and a flat
        # 2500-step cap = 25 m of walking made the robot give up short of the green bin.
        budget = int(1500 + 260 * abs(tgt[0] - self.nav.arc_lat()[0]))
        while self.step_i - t0 < budget:
            if self.nav.honest and (self.step_i - t0) % 50 == 0:
                # the sighted bin moves with every fresh view (navest.bin_pos dead-
                # reckons the latest one); a target frozen at the trip's start is the
                # 8 m sighting the first hs1 drop fell 0.31 m short on
                _bp = self.nav.bin_pos(colour)
                if _bp is not None:
                    tgt = _aim(_bp, self.nav.arc_lat()[0])
            if (self.step_i - t0) % 200 == 0 and \
                    getattr(self, "_carry_lost", None) is not None and \
                    [n for n, _ in self.carried] == [self._carry_lost]:
                self.log.ev(f"  hand cargo {self._carry_lost} lost mid-walk — "
                            "aborting the bin trip")
                return False
            ax, w = self.nav.arc_lat()
            # Geometry (all MEASURED): the robot cannot stand closer than ~0.5 m from a
            # bin centre — its front feet reach the bin's base plate — and the retrieve
            # drop pose carries the TCP 0.59 m ahead of the base, so standing 0.42-0.68 m
            # short of the centre puts the TCP over the 0.40 x 0.26 m opening.
            lat_err = abs(w - tgt[1])
            # ALONG and ACROSS are not the same problem. The 0.16 m band was sized on
            # the along axis, where the opening is 0.40 m wide and the arc window
            # (0.12 m deep) absorbs the rest; across, the opening is only 0.26 m, so a
            # 0.16 m stop error is already 0.03 m from the rim before the drop
            # trajectory adds anything. Measured over hs4a-hs4e: every stop within
            # 0.09 m of the believed lat scored correct, both stops past 0.20 m scored
            # `area`. Hold out for a tight lateral stop while there is budget, then
            # take what we can get rather than walk away with the cargo.
            _lat_tol = float(os.environ.get("MISSION_BIN_STOP_LAT", "0.07"))
            if (self.step_i - t0) > 0.6 * budget:
                _lat_tol = 0.16
            if (tgt[0] - 0.62) < ax < (tgt[0] - 0.50) and lat_err < _lat_tol:
                self.settle(60)
                self.log.ev(f"reached the {colour} bin at arc {ax:.2f} lat {w:.2f}")
                return True
            # Moving BETWEEN bins (0.9 m apart laterally) cannot be done from inside the
            # drop band — the bins block the strafe — so back out to a stand-off first.
            # If the robot is no longer on the course there is nothing to approach: the
            # bin-approach loop once kept issuing commands with lat = -207 m, having left
            # the platform entirely. Stop and let the caller record the failure.
            # ...and the bin approach is also where a run ends up at the deck edge with
            # the cargo aboard: the old test only fired at 2.5 m, half a metre OUTSIDE
            # the rope, which is a sanity check on the estimator and not an edge guard.
            # Come in first; only give the bin up if the walk-in cannot get back.
            if abs(w) > float(os.environ.get("MISSION_BIN_ABORT_LAT", "1.85")):
                self._walk_inboard(1.20, max_steps=400)
                _, w = self.nav.arc_lat()
                if abs(w) > float(os.environ.get("MISSION_BIN_ABORT_LAT", "1.85")):
                    self.log.ev(f"abandoning the {colour} bin: robot is {abs(w):.1f} m "
                                f"off the course centreline and will not come back in")
                    return False
                continue
            if self.step_i % (self.detect_every_eff * 2) == 0:
                self.scan()          # free: populates self.seen, issues no sim step
            goal_arc = (tgt[0] - 1.45) if lat_err > 0.25 else (tgt[0] - 0.56)
            if (self.step_i - t0) % 250 == 0:
                self.log.ev(f"  approaching {colour} bin: arc={ax:.2f}->{goal_arc:.2f} "
                            f"lat={w:.2f}->{tgt[1]:.2f}"
                            f"{' (backing off to strafe)' if lat_err > 0.25 else ''}")
            derr = goal_arc - ax
            if derr < -0.12:
                cmd, _, _ = self.walk_cmd(tgt[1], vx=-0.32)   # back off
            else:
                # the walk policy was trained at 0.75 m/s and marches in place below about
                # 0.3 m/s, so creeping in at 0.2 stalled the robot 0.7 m short of the bin
                vx = 0.5 if derr > 0.9 else 0.35
                if (self.step_i - stall_at) > 400:
                    if abs(ax - stall_ax) < 0.15:
                        vx = 0.65                            # unstick
                    stall_ax, stall_at = ax, self.step_i
                cmd, _, _ = self.walk_cmd(tgt[1], vx=vx)
            self._write_arm(self.travel_q, open_=False)
            self._step(self.legs.step(cmd))
        ax, w = self.nav.arc_lat()
        self.log.ev(f"gave up on the {colour} bin at arc={ax:.2f} lat={w:.2f}")
        return False

    def orbit_to(self, center_w, stance_w, radius=0.80, tol=0.16, timeout=2600):
        """Walk AROUND center_w to the stance point without ever crossing inside
        `radius` of it.

        Exists because every straight-line reposition around the paper bag TRAMPLED it:
        the robot stands right beside the object it needs to circle, and approach() knows
        nothing about keeping clear -- measured, a "reposition onto the mouth" ended with
        the mouth BEHIND the robot at base [-0.12,-0.36], i.e. the walk went through the
        bag. This follows the circle of `radius` toward the stance angle and only cuts
        straight once the angular error is small, then turns to face the centre.
        """
        c = np.array([float(center_w[0]), float(center_w[1])])
        t = np.array([float(stance_w[0]), float(stance_w[1])])
        t0, ok = self.step_i, False
        while self.step_i - t0 < timeout:
            X, Y, _, yaw = self.base_pose_true()
            pos = np.array([X, Y])
            if np.linalg.norm(t - pos) < tol:
                ok = True
                break
            rc = pos - c
            d = max(np.linalg.norm(rc), 1e-6)
            ang_p = np.arctan2(rc[1], rc[0])
            ang_t = np.arctan2((t - c)[1], (t - c)[0])
            dang = (ang_t - ang_p + np.pi) % (2 * np.pi) - np.pi
            if abs(dang) < 0.25:
                wp = t
            else:
                r_go = max(radius, min(d, np.linalg.norm(t - c)))
                a_go = ang_p + np.clip(dang, -0.45, 0.45)
                wp = c + r_go * np.array([np.cos(a_go), np.sin(a_go)])
                # clamp the waypoint's LAT to the deck: a lat-1.1 centre plus a 0.8-0.9
                # swing routinely put the base at 2.0+ (r24: two OOBs from orbit legs,
                # then the wreck) — walk a squashed arc instead of leaving the course
                if not self.nav.honest:
                    _axw, _ww = arc_of(float(wp[0]), float(wp[1]))
                    if abs(_ww) > 1.65:
                        _exw, _eyw = ct.curve_map(_axw,
                                                  float(np.clip(_ww, -1.65, 1.65)))
                        wp = np.array([float(_exw), float(_eyw)])
            # COURSE GUARD. This loop had none, and it is the only walking loop in the
            # runner without one: approach() checks every iteration (runner.py:971), the
            # traverse checks against LAT_GUARD, the shuttle checks — orbit_to could walk
            # 2600 steps (52 s) around a centre that may be a stale or phantom estimate.
            # Measured, run videos_venue/0824_1315_dep_footguard: the bag estimate went
            # unconfirmed ("no fresh view within 2.5 m — keeping the old estimate"), the
            # orbit chased it, and the robot ended at arc -19.0 lat +54.75 -- 54 metres off
            # a 4 m wide course, with the mission still issuing walk commands. That run
            # scored 0/160 and lost half of task 1 as well. The orbit is already the
            # lowest-yield routine in the stack (0 held in 60+ attempts); it must not be
            # able to end a run.
            ax_o, w_o = self.nav.arc_lat()
            if self.nav.honest and os.environ.get("NAV_MINIMAL") != "1" \
                    and (getattr(self.nav, "since_fix", 0) > 700
                         or (getattr(self.nav, "since_fix", 0) > 300
                             and abs(self.nav.lat()) > 1.2)):
                self.log.ev("orbit paused — navigation starving; relocalizing")
                self.nav.reacquire(self)
                return False
            # 2.2 -> 1.75: r6 fell OFF THE SIDE at lat -2.2 — the abandon fired at
            # the platform edge itself, too late to walk back. Abandon while both
            # feet still have deck under them.
            if abs(w_o) > self._edge_lat() + 0.1 or ax_o < self.nav.s_start() + 0.3:
                self.log.ev(f"orbit abandoned: robot at arc {ax_o:.1f} lat {w_o:+.2f} "
                            f"is leaving the course")
                self._reenter_course()
                return False
            to_wp = wp - pos
            head = np.arctan2(to_wp[1], to_wp[0])
            err = (head - yaw + np.pi) % (2 * np.pi) - np.pi
            cmd = torch.zeros(1, 3, device=self.device)
            cmd[0, 0] = 0.32 if abs(err) < 0.8 else 0.04
            cmd[0, 2] = float(np.clip(1.4 * err, -0.8, 0.8))
            self._write_arm(self.travel_q, open_=False)
            self._step(self.legs.step(cmd))
        # face the centre so the pinch happens in the arm's forward workspace
        for _ in range(260):
            X, Y, _, yaw = self.base_pose_true()
            head = np.arctan2(c[1] - Y, c[0] - X)
            err = (head - yaw + np.pi) % (2 * np.pi) - np.pi
            if abs(err) < 0.10:
                break
            cmd = torch.zeros(1, 3, device=self.device)
            cmd[0, 0] = 0.02
            cmd[0, 2] = float(np.clip(1.5 * err, -0.8, 0.8))
            self._write_arm(self.travel_q, open_=False)
            self._step(self.legs.step(cmd))
        return ok

    def servo_tcp_pose(self, target_w, jaw_dir_w=None, steps=140, tol=0.010,
                       grip_open=True, stand=False, decision=False):
        """Position servo like servo_tcp_to, plus FINGER-AXIS alignment.

        jaw_dir_b: the base-frame horizontal direction the finger-OPENING axis should
        point along (i.e. the normal of the sheet the fingers are meant to straddle).
        The RL policy handles this alignment implicitly for the objects it was trained
        on; a scripted pinch has to do it explicitly or the fingers arrive parallel to
        the wall and push it instead of straddling it. Mechanism borrowed from
        for_claude/act/wb_grasp.py: the opening axis is the EE frame's local Y, and the
        correction is a pure roll about the approach axis (local Z), which cannot
        disturb the position solution.
        """
        # WORLD-frame target, converted to the base frame EVERY step. The first version
        # took base-frame targets computed once up front, and the crouched legs drift
        # several centimetres over a few hundred settling steps -- the hand faithfully
        # tracked a target that was itself sliding across the world, which showed up as a
        # 37-53 mm jaw-axis miss against a 10 mm servo residual.
        tgt_w = torch.tensor(np.asarray(target_w, float), dtype=torch.float32,
                             device=self.device)
        jd_w = None
        if jaw_dir_w is not None:
            jd_w = torch.tensor(np.asarray(jaw_dir_w, float), dtype=torch.float32,
                                device=self.device)
            jd_w = jd_w / jd_w.norm().clamp(min=1e-9)
        # decision=True: the target (and the jaw direction) are given in the DECISION
        # frame — S when navigating honestly — and are re-expressed through the pose
        # ESTIMATE every step instead of the simulator's root pose. Both maps are
        # affine, so a direction transforms as the difference of two mapped points.
        _to_b = self.to_base_frame if decision else self.to_base_frame_true
        for _ in range(steps):
            base_R = math_utils.matrix_from_quat(self.robot.data.root_quat_w)[0]
            tgt = torch.tensor(_to_b(tgt_w.cpu().numpy()),
                               dtype=torch.float32, device=self.device)
            if jd_w is None:
                jd = None
            elif decision:
                _o = _to_b(np.zeros(3))
                jd = torch.tensor(_to_b(jd_w.cpu().numpy()) - _o,
                                  dtype=torch.float32, device=self.device)
            else:
                jd = base_R.T @ jd_w
            p, q = self.arm.frame_pose()
            err = tgt - p[0]
            a = torch.zeros(1, 6, device=self.device)
            done_pos = float(err.norm()) < tol
            if not done_pos:
                a[0, :3] = (err / self.arm.scale).clamp(-1.0, 1.0)
            done_rot = True
            if jd is not None:
                R = math_utils.matrix_from_quat(q)[0]
                jaw = R[:, 1].clone(); jaw[2] = 0.0
                n = jaw.norm().clamp(min=1e-9); jaw = jaw / n
                tgtj = jd.clone(); tgtj[2] = 0.0
                tgtj = tgtj / tgtj.norm().clamp(min=1e-9)
                # a sheet has no front/back: aligning to -n is as good as +n
                if float((jaw * tgtj).sum()) < 0.0:
                    tgtj = -tgtj
                cross = jaw[0] * tgtj[1] - jaw[1] * tgtj[0]
                ang = float(torch.asin(cross.clamp(-1.0, 1.0)))
                done_rot = abs(ang) < 0.06
                if not done_rot:
                    axis = R[:, 2] / R[:, 2].norm().clamp(min=1e-9)
                    a[0, 3:6] = (axis * ang / self.arm.scale).clamp(-1.0, 1.0)
            if done_pos and done_rot:
                break
            self.arm.ik_step(a)
            if grip_open is not None:      # None = keep the caller's grip_target
                self.arm.set_gripper(grip_open)
            self.arm.write_gripper()
            if stand:
                self._step(self.legs.step(torch.zeros(1, 3, device=self.device)))
            else:
                self._step(self.legs.step(self.crouch_cmd(1.0)))
        p, _ = self.arm.frame_pose()
        tgt = torch.tensor(_to_b(tgt_w.cpu().numpy()),
                           dtype=torch.float32, device=self.device)
        return float((tgt - p[0]).norm())

    def _bag_handle_q(self, tcp_b, yaw_bar, pitch=0.0, z_tool=None, y_tool=None,
                      q0=None):
        """armkin 6D DLS: joint pose putting the TCP at tcp_b with the tool VERTICAL and
        the jaw across the bar. The online position servo left the tool 18 deg off
        vertical, and a tilted jaw descending over an 8 mm strap leads with one finger —
        measured on video, it shoves the 100 g bag over instead of straddling the bar."""
        import importlib.util as _ilu
        if not hasattr(self, "_ak"):
            _spec = _ilu.spec_from_file_location(
                "armkin", str(Path(__file__).resolve().parent / "armkin.py"))
            self._ak = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(self._ak)
        ak = self._ak
        zt = np.array([0.0, 0.0, -1.0])
        yt = np.array([np.cos(yaw_bar + np.pi / 2), np.sin(yaw_bar + np.pi / 2), 0.0])
        if z_tool is not None:
            # EXPLICIT TOOL FRAME for the loop thread. `pitch` rotates zt about yt, so
            # it can only ever tilt the tool along the bar — it cannot lay the tool
            # onto the handle window's normal, which is what threading needs. The
            # caller passes the axes directly: z_tool = the direction the fingers
            # point (into the window), y_tool = the closing axis (along the bar, so
            # the two fingers sit side by side and the crossbar lands across them).
            zt = np.asarray(z_tool, float)
            zt = zt / max(1e-9, float(np.linalg.norm(zt)))
            yt = np.asarray(y_tool, float) if y_tool is not None else yt
            yt = yt - zt * float(np.dot(yt, zt))          # orthogonalise
            yt = yt / max(1e-9, float(np.linalg.norm(yt)))
        elif pitch != 0.0:
            # tilt about the CLOSING axis: every clean catch to date happened with the
            # tool pitched (far-reach trials); vertical-tool descents shove the bag
            # (+84 mm creep on video). Fingers stay level, so this is not the
            # one-finger-leads roll the docstring measured as harmful.
            c_, s_ = np.cos(pitch), np.sin(pitch)
            K = np.array([[0, -yt[2], yt[1]], [yt[2], 0, -yt[0]], [-yt[1], yt[0], 0]])
            Rp = np.eye(3) + s_ * K + (1 - c_) * (K @ K)
            zt = Rp @ zt
        xt = np.cross(yt, zt)
        Rt = np.stack([xt, yt, zt], axis=1)
        # SEED MATTERS. The DLS is local: bag8's attempt 1 solved this pose to 1 mm
        # from the q_mid posture and attempt 2, seeded from wherever the failed attempt
        # left the arm, stalled at 345 mm on a window 0.47 m in front of the robot.
        # q0 lets the caller retry from a known mid-range posture.
        q = (np.asarray(q0, float)[:6] if q0 is not None
             else self.arm.current_q6().cpu().numpy().astype(float)[:6])
        for _ in range(180):
            pcur, Rc, _, _ = ak.fk(q)
            e_pos = np.asarray(tcp_b, float) - pcur
            Rd = Rt @ Rc.T
            c = np.clip((np.trace(Rd) - 1) / 2, -1, 1)
            ang = np.arccos(c)
            e_rot = (np.zeros(3) if ang < 1e-7 else
                     ang / (2 * np.sin(ang)) * np.array([Rd[2, 1] - Rd[1, 2],
                                                         Rd[0, 2] - Rd[2, 0],
                                                         Rd[1, 0] - Rd[0, 1]]))
            if np.linalg.norm(e_pos) < 0.004 and np.linalg.norm(e_rot) < 0.06:
                return q, float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))
            J = ak.jacobian(q)
            dq = ak.dls_step(J, np.concatenate([e_pos, 0.6 * e_rot]), lam=0.05)
            q = np.clip(q + np.clip(dq, -0.15, 0.15),
                        [l for l, _ in ak.JOINT_LIMITS],
                        [h for _, h in ak.JOINT_LIMITS])
        pcur, Rc, _, _ = ak.fk(q)
        Rd = Rt @ Rc.T
        c = np.clip((np.trace(Rd) - 1) / 2, -1, 1)
        ang = float(np.arccos(c))
        return q, float(np.linalg.norm(np.asarray(tcp_b) - pcur)), ang

    def bag_handle_pinch(self, name: str) -> bool:
        """Pinch the STANDING bag's carry handle and lift (MISSION_BAG_HANDLE=1).

        v3: joint-space, armkin-planned. The bar centre sits at bag-local +z 0.199
        (build_prop_usds handle geometry); both the hover and the pinch pose are solved
        offline as full 6D targets (tool vertical, jaw across the bar) and played as
        joint ramps — the online position-only servo could not control the tool tilt
        and toppled the bag twice on video. Between tries the arm resets through
        q_mid: retry from a diverged pose stayed diverged (residual 0.41 x3).
        """
        obj = self._sim_obj(name)
        BAR_Z = 0.205   # bar centre: hz 0.135 + uprights 0.060 + bar_h/2 0.010
        # STAY CROUCHED. Ten versions of the standing pinch all lost to the same noise
        # source: the standing base drifts through every servo leg (aim scatter
        # +-30 mm, straps swept, bags toppled), while the crouch holds still to
        # std 0.002 — the platform the whole ground-grasp stack already trusts.
        # armkin solves the vertical bar pinch at crouch height (base-z +0.095) to
        # 1-2 mm across the entire x 0.45-0.60 approach window.
        zero = torch.zeros(1, 3, device=self.device)
        # SCOPED Jacobian frame fix. The descent/servo stalls (+33..+159 mm) happen
        # only at x-close, where the tool is most vertical — exactly where the known
        # wrong-frame TCP-offset Jacobian (ArmDriver._jacobian) is worst. The RL grasp
        # loop keeps the buggy plant it was trained on; the scripted pinch gets the
        # correct transform for its duration only.
        import course_mission.mission as _msn
        _ikff_saved = _msn.IK_FRAME_FIX
        _msn.IK_FRAME_FIX = True
        try:
            return self._bag_handle_pinch_body(obj, name, zero)
        finally:
            _msn.IK_FRAME_FIX = _ikff_saved

    def _bag_roll_over(self, obj, name: str) -> bool:
        """Tip a handle-down bag over so its carry bar comes clear of the ground.

        This is the bag failure: of every bag attempt on record, the commonest exit is
        "bar under/at ground", twice in a row, and then a permanent write-off — the
        pinch never gets to run at all. Nothing about that pose fixes itself, but the
        bag is a 100 g paper sack, and the arm can simply push it over: put the CLOSED
        jaw against its far side at bar height and sweep toward the robot. One attempt,
        bounded, and the caller re-reads the bar afterwards; if the bag lands
        handle-down again the existing hit counter writes it off as before.
        MISSION_BAG_ROLL=0 disables.
        """
        if not int(os.environ.get("MISSION_BAG_ROLL", "1")):
            return False
        pw = obj.data.root_pos_w[0].cpu().numpy()
        b = self.to_base_frame_true(pw)
        _u = b[:2] / max(float(np.linalg.norm(b[:2])), 1e-6)   # robot -> bag, horizontal
        _gz_b = -self._stance_height()
        _h = float(os.environ.get("MISSION_BAG_ROLL_H", "0.13"))
        _far = np.array([b[0] + _u[0] * 0.16, b[1] + _u[1] * 0.16, _gz_b + _h])
        _near = np.array([b[0] - _u[0] * 0.10, b[1] - _u[1] * 0.10, _gz_b + _h])
        self.log.ev(f"  bag roll: pushing {name} over from the far side "
                    f"(base {np.round(_far, 2).tolist()} -> {np.round(_near, 2).tolist()})")
        self.play_traj([self.retrieve["q_mid"]], grip_open=False, steps_each=120)
        self.servo_tcp_to(_far + np.array([0.0, 0.0, 0.10]), steps=90, grip_open=False)
        self.servo_tcp_to(_far, steps=90, grip_open=False)
        self.servo_tcp_to(_near, steps=140, grip_open=False)
        self.servo_tcp_to(_near + np.array([0.0, 0.0, 0.12]), steps=80, grip_open=False)
        self.play_traj([self.retrieve["q_mid"]], grip_open=True, steps_each=120)
        _p2 = obj.data.root_pos_w[0].cpu().numpy()
        self.log.ev(f"  bag roll: {name} moved "
                    f"{float(np.linalg.norm(_p2[:2] - pw[:2])) * 1000:.0f} mm")
        return True

    def _bag_thread_entry(self, obj):
        """Base-frame entry pose for threading the CLOSED jaw through the bag's carry
        handle, or None if the arm cannot hold one.

        Geometry, read off a2pro_rl/scripts/build_prop_usds.py: two 8 mm uprights at
        bag-local x +-45 mm rise from the mouth (local z 0.135) to an 8 x 20 mm
        crossbar centred at 0.205. The window is therefore 82 mm along the bar by
        60 mm through, centred at local z 0.165, with its normal along the bag's
        local y.

        THIS IS THE MANOEUVRE FOR THE 93 % CASE. Across every log in the record 399 of
        428 bag encounters were LYING, and 270 of those died at "bar under/at ground"
        without one attempt — the pinch needs the bar clear of the deck, and a bag on
        its face has the bar 27 mm up. But that same pose turns the handle window
        FACING UP: a hole the vertical tool can simply drop into, then lift, and the
        bar catches across the fingers. The 39 attempts that did reach a pinch scored
        9 HELD, so the grab is not the bottleneck — getting to one is.

        Returns (q, ep, er, n_b, win_b, depth, bar_dir_b). n_b points OUT of the window
        on the side the hand comes from; depth is how far past the window plane the
        fingers may travel before the deck stops them; bar_dir_b is the bag's local +z
        in the base frame, i.e. the direction from the window centre TO the crossbar
        (40 mm away), which is where the fingers have to slide to hook it.
        """
        pw = obj.data.root_pos_w[0].cpu().numpy()
        Rw = math_utils.matrix_from_quat(
            obj.data.root_quat_w[0].unsqueeze(0))[0].cpu().numpy()
        _, _, _, base_yaw = self.base_pose_true()
        _c, _s = np.cos(-base_yaw), np.sin(-base_yaw)

        def _dir_b(v):
            return np.array([_c * v[0] - _s * v[1], _s * v[0] + _c * v[1], v[2]])

        win_b = self.to_base_frame_true(pw + Rw[:, 2] * 0.165)
        ny = _dir_b(Rw[:, 1])
        ny = ny / max(1e-9, float(np.linalg.norm(ny)))
        bd = _dir_b(Rw[:, 0])
        bd = bd / max(1e-9, float(np.linalg.norm(bd)))
        # the hand arrives from the shoulder, so enter through whichever face points
        # that way: back at the robot for a standing bag, straight up for a lying one
        _home = np.array([0.15, 0.0, 0.35])
        n = ny if float(np.dot(ny, _home - win_b)) > 0 else -ny
        depth = 0.03
        if n[2] > 0.5:
            # window facing up: the deck is what limits the plunge, and the fingers
            # only have to clear the 8 mm bar to hook it
            _gz_b = self.to_base_frame_true(np.array([
                pw[0], pw[1],
                ground_z_arc(arc_of(float(pw[0]), float(pw[1]))[0])]))[2]
            depth = float(min(0.03, max(0.0, (win_b[2] - _gz_b) - 0.012)))
            if depth < 0.010:
                self.log.ev(f"  thread entry: window only "
                            f"{(win_b[2] - _gz_b) * 1000:.0f} mm off the deck — no room "
                            f"to get a finger under the bar")
                return None
        # ENTER OFF-NORMAL IF THAT IS WHAT REACHES. A standing bag whose window faces
        # sideways needs a lateral reach the arm cannot hold dead-on (bag4: ep 41 mm
        # but er 0.50); the window is 82 x 60 mm against ~55 mm of finger, so a slanted
        # entry still clears. No-op when the normal is vertical.
        # DEGENERATE FRAME GUARD. y_tool is orthogonalised against z_tool, so a bar
        # direction that has rolled onto the entry axis leaves a near-zero vector and
        # the "closing axis" becomes numerical noise — an arbitrary target orientation
        # the arm cannot reach. bag9's attempt 2 came back "best of 7 axes ep 334 mm
        # er 1.44" on a window 0.47 m in front of the robot, right after the failed
        # attempt 1 had shoved the bag 47 mm and turned it. Any perpendicular will do
        # for a closed jaw.
        if abs(float(np.dot(bd, n))) > 0.9:
            _alt = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            bd = _alt - n * float(np.dot(_alt, n))
            bd = bd / max(1e-9, float(np.linalg.norm(bd)))
            self.log.ev("  thread entry: bar axis has rolled onto the entry axis — "
                        "using a substitute closing axis")
        _seeds = [None]
        if self.retrieve is not None and "q_mid" in self.retrieve:
            _seeds.append(list(self.retrieve["q_mid"]))
        best = None
        for _yo in (0.0, 0.35, -0.35, 0.6, -0.6, 0.9, -0.9):
            _cy, _sy = np.cos(_yo), np.sin(_yo)
            nt = np.array([_cy * n[0] - _sy * n[1], _sy * n[0] + _cy * n[1], n[2]])
            for _q0 in _seeds:
                q, ep, er = self._bag_handle_q(win_b + nt * 0.10, 0.0,
                                               z_tool=-nt, y_tool=bd, q0=_q0)
                if best is None or (ep + 0.05 * er) < (best[1] + 0.05 * best[2]):
                    best = (q, ep, er, nt)
                if ep <= 0.03 and er <= 0.25:
                    break
            if best[1] <= 0.03 and best[2] <= 0.25:
                break
        q, ep, er, nt = best
        # The DLS residual is ADVISORY now, not a gate. It was rejecting windows the
        # servo can reach perfectly well (bagE: "ep 357 mm" on a window 0.49 m in front
        # of a robot that routinely grasps at 0.62 m) because the solver kept landing on
        # an arm branch the drive cannot hold — see the servo note in the thread branch.
        # What actually decides feasibility is whether the window is inside the arm's
        # working envelope at all.
        _r = float(np.hypot(win_b[0], win_b[1]))
        if _r > 0.78 or _r < 0.20 or win_b[2] < -0.30:
            self.log.ev(f"  thread entry out of range: window "
                        f"{np.round(win_b, 3).tolist()} (reach {_r:.2f} m)")
            return None
        if ep > 0.05 or er > 0.45:
            self.log.ev(f"  thread entry: DLS could not pose it (best of 7 axes "
                        f"ep {ep*1000:.0f} mm er {er:.2f}) — servoing in anyway, "
                        f"window {np.round(win_b, 3).tolist()} reach {_r:.2f} m")
        return q, ep, er, nt, win_b, depth, _dir_b(Rw[:, 2])

    def _bag_handle_pinch_body(self, obj, name: str, zero) -> bool:
        BAR_Z = 0.205

        def _bar_now():
            _pwn = obj.data.root_pos_w[0]
            _Rmn = math_utils.matrix_from_quat(
                obj.data.root_quat_w[0].unsqueeze(0))[0]
            return self.to_base_frame_true((_pwn + _Rmn[:, 2] * BAR_Z).cpu().numpy())

        for attempt in range(2):
            pw = obj.data.root_pos_w[0]
            Rm = math_utils.matrix_from_quat(obj.data.root_quat_w[0].unsqueeze(0))[0]
            bar_w = (pw + Rm[:, 2] * BAR_Z).cpu().numpy()
            # The pinch maths is frame-generic — a LYING bag's bar juts out sideways
            # and takes the same vertical-tool, jaw-across grab. All that matters is
            # that the bar sits clear of the ground (handle-down bags excluded).
            _gz = ground_z_arc(arc_of(float(pw[0]), float(pw[1]))[0])
            # ...and the LOW-BAR write-off only applies to the pinch. A bag on its
            # face has the bar 27 mm up (below the 50 mm gate) but its handle window
            # facing straight up, which the thread drops into. 270 of 428 logged bag
            # encounters died right here without an attempt; if the thread has a pose,
            # take it instead of writing the object off.
            _thread_first = os.environ.get("MISSION_BAG_THREAD_FIRST", "1") == "1"
            _thr = self._bag_thread_entry(obj) if _thread_first else None
            if bar_w[2] - _gz < 0.05 and _thr is not None:
                self.log.ev(f"  bag handle: bar only "
                            f"{(bar_w[2] - _gz) * 1000:.0f} mm up, but the handle "
                            f"window is reachable (normal {np.round(_thr[3], 2).tolist()}, "
                            f"depth {_thr[5] * 1000:.0f} mm) — threading instead")
            elif bar_w[2] - _gz < 0.05 and not getattr(self, "_bag_rolled", set()) \
                    .__contains__(name):
                # ...before writing it off, tip it over: the bar is only unreachable
                # because of the pose the bag happens to be lying in.
                self._bag_rolled = getattr(self, "_bag_rolled", set()) | {name}
                if self._bag_roll_over(obj, name):
                    pw = obj.data.root_pos_w[0]
                    Rm = math_utils.matrix_from_quat(
                        obj.data.root_quat_w[0].unsqueeze(0))[0]
                    bar_w = (pw + Rm[:, 2] * BAR_Z).cpu().numpy()
                    _gz = ground_z_arc(arc_of(float(pw[0]), float(pw[1]))[0])
                    self.log.ev(f"  bag roll: bar now {(bar_w[2] - _gz) * 1000:.0f} mm "
                                f"above the ground (needs 50)")
            if bar_w[2] - _gz < 0.05 and _thr is None:
                self._bag_down_hits[name] = self._bag_down_hits.get(name, 0) + 1
                if self._bag_down_hits[name] >= 2:
                    # handle-down twice: this pose will not fix itself, and the
                    # persist re-sweep kept resurrecting it (b33 burned 8+ cycles on
                    # one bag). Permanent blacklist — persist does not clear this one.
                    self.perma_blacklist.append(pw.cpu().numpy()[:2])
                    self.log.ev("  bag handle: bar under/at ground twice — "
                                "permanently written off")
                else:
                    self.log.ev("  bag handle: bar under/at ground — falling back")
                return False
            bar_b = self.to_base_frame_true(bar_w)
            yaw_bar_w = float(np.arctan2(float(Rm[1, 0]), float(Rm[0, 0])))
            _, _, _, base_yaw = self.base_pose_true()
            yaw_bar = yaw_bar_w - base_yaw
            z0 = float(pw[2])
            _perp = np.array([np.cos(yaw_bar + np.pi / 2),
                              np.sin(yaw_bar + np.pi / 2), 0.0])
            _hov_b = bar_b + _perp * 0.040 + np.array([0, 0, 0.06])
            q_h, ep_h, er_h = self._bag_handle_q(_hov_b, yaw_bar)
            self.log.ev(f"  bag handle plan: hover ep {ep_h*1000:.0f} mm er {er_h:.2f}")
            # LET THE TOOL TILT BEFORE GIVING UP. The vertical-tool constraint is what
            # makes this pose hard to reach, and the record is full of near misses that
            # were thrown away for it: "hover ep 60 mm er 0.04" is 15 mm outside the
            # gate with the ORIENTATION already solved. _bag_handle_q takes a pitch
            # about the closing axis — the docstring's own note is that "every clean
            # catch to date happened with the tool pitched" — so sweep a few before
            # declaring the bar unreachable. The jaw stays across the bar either way.
            if ep_h > 0.045 or er_h > 0.35:
                for _p in (0.3, -0.3, 0.6, -0.6):
                    _q2, _e2, _r2 = self._bag_handle_q(_hov_b, yaw_bar, pitch=_p)
                    if _e2 <= 0.045 and _r2 <= 0.35:
                        q_h, ep_h, er_h = _q2, _e2, _r2
                        self.log.ev(f"  bag handle plan: vertical tool unreachable — "
                                    f"pitched {np.degrees(_p):+.0f} deg solves it "
                                    f"(ep {_e2*1000:.0f} mm er {_r2:.2f})")
                        break
            # The pose gate is a PINCH requirement — jaw across the bar, tool near
            # vertical. The loop-thread below needs none of it: it pokes a CLOSED jaw
            # through the 90 x 60 mm handle opening with a position-only servo. Probe
            # bag1 threw away exactly that chance: attempt 0 aborted on creep, the bag
            # had shifted, attempt 1's pinch plan came back er 1.68 and the whole bag
            # was written off without the thread ever running. Unreachable pinch pose
            # now means "thread only", not "give up".
            _reach = not (ep_h > 0.045 or er_h > 0.35)
            if not _reach:
                self.log.ev("  bag handle: no reachable hook pose at any tool pitch — "
                            "loop-thread only")
            # THREAD BEFORE PINCH. The pinch presses a jaw DOWN onto the handle of a
            # ~100 g paper bag and the bag simply leaves: probe bag1 measured -32 mm
            # of creep with the jaw wide (tips on the shoulders) and bag2 -44 mm with
            # it narrowed to 40 mm (tips on the straps), both on a cleanly STANDING
            # bag with the hover pose solved to 22-30 mm. Downward contact is the
            # problem, not the aim. The thread enters SIDEWAYS at loop height with the
            # jaw shut — a 10 mm blade through a 90 x 60 mm opening — which is also
            # what every HELD in the record was at heart. MISSION_BAG_THREAD_FIRST=0
            # restores pinch-first.
            # A LOW bar does NOT mean the thread. When the bag lies on its face the
            # handle frame is horizontal and the bar presents a 98 x 20 mm strip 29 mm
            # off the deck, 8 mm thick — and _bag_handle_q's standard frame is already
            # right for it: yt = perpendicular to the bar is the strip's 20 mm width,
            # zt = straight down. That is an ordinary top-down straddle, the same shape
            # as the lid pinch that works on lunchboxes, and it does not have the
            # standing bag's problem (nothing to topple — the bag is already down).
            # The thread has to squeeze a finger UNDER a bar with 29 mm of clearance,
            # which bagJ measured as marginal: after the deepest plunge the deck allows
            # the bar still sat 14-17 mm above the fingertip and the shaft pushed it
            # (bag dragged 39-45 mm on every hook). So: bar low AND handle facing up ->
            # PINCH, with the descent clamped so the fingers stop above the deck.
            _lowbar = bar_w[2] - _gz < 0.05
            _flat = _thr is not None and float(_thr[3][2]) > 0.5
            _do_thread = ((not _reach) or ((attempt == 0) == _thread_first)
                          or _lowbar) and not (_lowbar and _flat)
            self.play_traj([self.retrieve["q_mid"]], grip_open=True, steps_each=140)
            if _reach and not _do_thread:
                self.play_traj([q_h.tolist()], grip_open=True, steps_each=150)
            bar_b = _bar_now()
            depth = -0.035
            if _lowbar:
                # DECK-LIMITED DESCENT. -35 mm under a bar that is only 29 mm off the
                # deck is 6 mm underground: the servo drives the fingers into the floor,
                # stalls, and the descent gate then aborts the attempt. Go as deep as
                # the deck allows and no deeper — the bar is 8 mm thick, so even 12 mm
                # of clearance puts the fingers below it.
                depth = -max(0.008, (bar_w[2] - _gz) - 0.012)
                self.log.ev(f"  bag pinch: bar {(bar_w[2] - _gz)*1000:.0f} mm off the "
                            f"deck — clamping the descent to {depth*1000:.0f} mm")
            self.servo_tcp_to(bar_b + np.array([0, 0, 0.06]),
                              steps=90, grip_open=not _do_thread)

            if _do_thread:
                # LOOP THREAD WITH THE TOOL LAID HORIZONTAL.
                # Handle geometry, read off a2pro_rl/scripts/build_prop_usds.py: two
                # 8 mm uprights at bag-local x +-45 mm rise 60 mm from the mouth
                # (local z 0.135) to an 8 x 20 mm crossbar centred at 0.205. The
                # window is therefore 82 mm ALONG the bar by 60 mm tall, its centre at
                # local z 0.165, and its NORMAL is the bag's local y.
                # A hand pointing straight down cannot enter a window whose plane is
                # vertical, and _bag_handle_q's `pitch` cannot fix that — it rotates
                # about the closing axis, so it only ever tilts the tool along the
                # bar. bag3 ran the old thread from a tool-down pose and bulldozed the
                # bag 1.1 m ("rose -0.106, gap 1.100 -> MISS"). Solve the entry pose
                # explicitly: fingers pointing along the entry direction, closing axis
                # along the bar so the two fingers sit side by side and the crossbar
                # drops across both of them when the arm lifts.
                # _bag_thread_entry solves the whole entry (window centre, which face
                # to come in through, the off-normal cone and how deep the deck lets
                # the fingers go). Recomputed here when the low-bar branch above did
                # not need it, e.g. pinch-first with the thread as attempt 1.
                _te = _thr if _thr is not None else self._bag_thread_entry(obj)
                if _te is None:
                    self.log.ev("  loop-thread: no reachable entry pose on the whole "
                                "cone — skipping the thread")
                    self.play_traj([self.retrieve["q_mid"]], grip_open=True,
                                   steps_each=150)
                    continue
                q_t, ep_t, er_t, _n, _win_b, _tdepth, _zb = _te
                self.log.ev(f"  loop-thread entry: window {np.round(_win_b, 3).tolist()} "
                            f"axis {np.round(_n, 2).tolist()} "
                            f"(ep {ep_t*1000:.0f} mm er {er_t:.2f}, "
                            f"plunge {_tdepth*1000:.0f} mm)")
                # SERVO IN, DO NOT JUMP TO THE DLS SOLUTION. _bag_handle_q solves the
                # entry pose to 0-1 mm, but the arm will not HOLD what it returns:
                # bagB/bagE commanded [-0.20, 2.35, -1.33, 0.65, 0.01, 2.78], held it
                # for 330 steps, and the joints sat at [-0.33, 2.32, -0.63, 0.79, 0.16,
                # 2.90] the whole time — TCP 0.35 m short of the window. armkin agrees
                # with the robot to 2 mm on the pose it IS in, so the chain is right and
                # the pose is legal; the solver has simply landed on an arm branch the
                # drive cannot support. The differential servo never leaves the branch
                # it is already on, and it is the same servo every scored grasp uses to
                # reach ground level at x 0.62 — so drive the TCP down instead of
                # teleporting the joints. For a bag on its face the window faces UP, so
                # "tool down" is already the orientation the thread wants and nothing
                # extra has to be imposed.
                # START FROM A PREGRASP BANK CONFIGURATION. Servoing out of q_mid only
                # got the TCP to x 0.31 of a 0.51 target (bagF) — the carry pose is a
                # folded branch and the differential IK cannot walk out of it. The bank
                # holds 745 configurations the arm demonstrably HOLDS while reaching
                # forward (every scored grasp starts from one and then drives down to
                # the object at base z -0.23), so drop into the nearest of those first
                # and let the servo do the rest.
                # ...and pick a LOW one. Selecting on xy alone is not enough: bagH
                # started from cfg #247 (EE z -0.002) and the plunge closed to 10 mm,
                # bagI from cfg #147 (EE z +0.072) and the same plunge stalled 90 mm
                # short. The servo has to give away height it does not have to climb
                # down, so pay for every centimetre the start sits above the window.
                _bxy = np.linalg.norm(self.bank_ee[:, :2] - _win_b[:2], axis=1)
                _bi = int(np.argmin(_bxy + 0.6 * np.maximum(
                    0.0, self.bank_ee[:, 2] - _win_b[2])))
                self.play_traj([np.asarray(self.bank_q[_bi], float)[:6].tolist()],
                               grip_open=False, steps_each=180)
                self.servo_tcp_to(_win_b + _n * 0.12, steps=150, grip_open=False)
                _pe, _ = self.arm.frame_pose()
                self.log.ev(f"  thread[entered] tcp "
                            f"{np.round(_pe[0].cpu().numpy(), 3).tolist()} "
                            f"want {np.round(_win_b + _n * 0.12, 3).tolist()} "
                            f"(bank cfg #{_bi} at "
                            f"{np.round(self.bank_ee[_bi], 3).tolist()} + servo)")
                # straight in along the tool axis, orientation pinned: the free-drift
                # servo lets the wrist fall back toward vertical over 130 mm of travel
                _thru = _win_b - _n * _tdepth
                _res = {}

                def _stage(tag):
                    """Where the fingertip actually got to, and where the bar is from
                    it. The thread has failed twice with the entry solving to 0-1 mm
                    and the lift reading rose 0.000, so the loss is somewhere between
                    the plunge and the hook — print every stage rather than guess."""
                    _p, _ = self.arm.frame_pose()
                    _p = _p[0].cpu().numpy()
                    _b = _bar_now()
                    self.log.ev(f"  thread[{tag}] tcp {np.round(_p, 3).tolist()} "
                                f"bar-tcp {np.round(_b - _p, 3).tolist()} "
                                f"(|d| {float(np.linalg.norm(_b - _p)):.3f})")

                # ...and DON'T pin the wrist while pushing down. hold_rot adds a
                # rotation term to the same damped solve as the translation, and the
                # two compete: bagG stalled the plunge 95 mm above target at TCP z
                # -0.107 on every trial, while the RL grasp — same ik_step, no rotation
                # term — reaches z -0.22 at the same reach. The bank configuration
                # already has the tool pointing down, which is all the thread needs.
                _res["plunge"] = self.servo_tcp_to(_thru, steps=170, grip_open=False)
                if _res["plunge"] > 0.030:
                    # A stalled plunge is the difference between a hook and a shove;
                    # give the solver a second run at it from where it got stuck (the
                    # IK is local, so the retry starts on a different part of the
                    # branch and often walks the rest of the way).
                    _res["plunge"] = self.servo_tcp_to(_thru, steps=140,
                                                       grip_open=False)
                _stage("plunged")
                # HOOK UNDER THE BAR — threading by itself lifts nothing. bag8 put the
                # fingers cleanly through (entry ep 1 mm, plunge 22 mm) and the lift
                # still read "rose -0.000 m, gap 0.872": a smooth prong sitting in the
                # middle of an 82 x 60 mm hole touches nothing on the way out. The
                # crossbar is 40 mm from the window centre along the bag's own +z
                # (window at local 0.165, bar at 0.205), so slide the fingers across
                # to sit UNDER it before going up. Same move in both attitudes —
                # sideways for a bag on its face, upward for one standing.
                # ...and slide onto the bar WHERE IT ACTUALLY IS, not 45 mm along the
                # bag's nominal +z. bagH plunged to within 12 mm of the bar and the
                # fixed slide then carried the fingers 44 mm PAST it, dragging the bag
                # 39 mm sideways: by then the bag had been shoved around by the earlier
                # attempts and the nominal window-to-bar offset no longer described it.
                # The bar is measured every step anyway — steer by that.
                _hk = float(os.environ.get("MISSION_BAG_HOOK_SLIDE", "0.045"))
                _pn0, _ = self.arm.frame_pose()
                _pn0 = _pn0[0].cpu().numpy()
                _bn0 = _bar_now()
                _dxy = _bn0[:2] - _pn0[:2]
                _dn = float(np.hypot(*_dxy))
                if _dn > 0.005:
                    _hkv = np.array([_dxy[0], _dxy[1], 0.0]) / _dn * min(_dn, _hk)
                else:
                    _hkv = _zb * _hk                    # already under it; nudge along
                _res["hook"] = self.servo_tcp_to(_pn0 + _hkv, steps=110,
                                                 grip_open=False)
                self.log.ev(f"  thread hook: bar was {_dn*1000:.0f} mm away in xy — "
                            f"sliding {np.round(_hkv * 1000, 0).tolist()} mm")
                _stage("hooked")
                # ...and take the weight straight up, before the q_mid joint swing
                # (which travels back over the torso and would drag the bag sideways)
                _pn, _ = self.arm.frame_pose()
                _res["lift"] = self.servo_tcp_to(
                    _pn[0].cpu().numpy() + np.array([0.0, 0.0, 0.12]),
                    steps=130, grip_open=False)
                _stage("lifted")
                self.log.ev(f"  thread servo residuals: plunge "
                            f"{_res.get('plunge', -1)*1000:.0f} mm, hook "
                            f"{_res.get('hook', -1)*1000:.0f} mm, lift "
                            f"{_res.get('lift', -1)*1000:.0f} mm")
                _bt = _bar_now()
                self.log.ev(f"  loop-thread through: bar moved [{(_bt[0]-bar_b[0])*1000:+.0f}, "
                            f"{(_bt[1]-bar_b[1])*1000:+.0f}, "
                            f"{(_bt[2]-bar_b[2])*1000:+.0f}] mm "
                            f"(hook slide {_hk*1000:.0f} mm along "
                            f"{np.round(_zb, 2).tolist()})")
            else:
                # ---- CENTRED STAGED DESCENT + CAGE (attempt 0) ----
                # STAGED with a creep watch: the open descent can shove the bag
                # (+84 mm on video, mud). Check the bar between stages; on creep
                # retreat, re-centre once; twice — abort with the bag standing.
                # DESCENT JAW WIDTH, measured both ways and left WIDE. The theory was
                # that GRASP_GRIP_OPEN=0.05 puts the tips +-50 mm out — outside the
                # 90 x 60 mm handle opening, on the bag's shoulders — so probe bag2
                # tried +-20 mm. It was worse, not better: creep went from (-8, -32)
                # and (-7, -27) mm wide to (-40, -44) and (-97, -318) mm narrow, the
                # last of which put the bag on its side. Narrow tips ride the STRAPS,
                # which converge on the bar, so they hit sooner and with more leverage.
                # Knob kept for the record; the real answer was to stop pressing down
                # at all (MISSION_BAG_THREAD_FIRST above).
                _jaw = float(os.environ.get("MISSION_BAG_JAW", "0.05"))
                if _lowbar:
                    # ...but NARROW it on a bag lying on its face, which is the opposite
                    # geometry. There the handle sticks out sideways from the mouth: one
                    # side of the bar is free air and the other is the bag's own body, so
                    # a finger 50 mm out lands on the body and presses the whole bag into
                    # the deck — bagL measured the bar sinking 12, 23, 33 and 36 mm during
                    # the descent, and the last of those put it 11 mm off the ground where
                    # nothing can reach it. The bar is 20 mm across in the closing
                    # direction; +-20 mm straddles it and touches neither side.
                    _jaw = float(os.environ.get("MISSION_BAG_JAW_LOW", "0.020"))
                self.arm.grip_target = torch.tensor([_jaw, -_jaw], device=self.device)
                _creeped, _aborted = 0, False
                for _stage in range(3):
                    zt_ = 0.06 + (depth - 0.06) * (_stage + 1) / 3
                    # 45 steps per stage is not enough to ARRIVE on a low bar: with the
                    # corrections switched off (they wreck the alignment, see below)
                    # bagP held xy to [+2, -7] mm and still finished 30 mm high, purely
                    # because the descent ran out of steps. Spend them here, where the
                    # servo is not fighting a moving target, instead of afterwards.
                    self.servo_tcp_to(bar_b + np.array([0, 0, zt_]),
                                      steps=110 if _lowbar else 45, grip_open=None)
                    _bn = _bar_now()
                    if float(np.hypot(_bn[0] - bar_b[0], _bn[1] - bar_b[1])) > 0.020:
                        _creeped += 1
                        self.log.ev(f"  bag creeping at stage {_stage} "
                                    f"({(_bn[0]-bar_b[0])*1000:+.0f}, "
                                    f"{(_bn[1]-bar_b[1])*1000:+.0f} mm) — "
                                    f"{'re-centring' if _creeped == 1 else 'aborting'}")
                        if _creeped >= 2:
                            break
                        self.servo_tcp_to(bar_b + np.array([0, 0, 0.08]), steps=60,
                                          grip_open=None)
                        bar_b = _bar_now()
                        self.servo_tcp_to(bar_b + np.array([0, 0, 0.06]), steps=60,
                                          grip_open=None)
                if _creeped >= 2:
                    self.play_traj([self.retrieve["q_mid"]], grip_open=True,
                                   steps_each=150)
                    continue
                # DESCENT GATE: never rise off a stalled descent (physical
                # obstruction — the solver stalls died with the scoped frame fix).
                _p_now, _ = self.arm.frame_pose()
                ez = float(_p_now[0, 2].cpu()) - (bar_b[2] + depth)
                if abs(ez) > 0.030:
                    self.servo_tcp_to(bar_b + np.array([0.006 if ez > 0 else -0.006,
                                                        0.0, depth]),
                                      steps=100, grip_open=None)
                    _p_now, _ = self.arm.frame_pose()
                    ez = float(_p_now[0, 2].cpu()) - (bar_b[2] + depth)
                if abs(ez) > 0.030:
                    self.log.ev(f"  bag handle: descent stalled {ez*1000:+.0f} mm — "
                                "aborting attempt with the bag upright")
                    self.servo_tcp_to(bar_b + np.array([0, 0, 0.08]), steps=90,
                                      grip_open=True)
                    self.play_traj([self.retrieve["q_mid"]], grip_open=True,
                                   steps_each=150)
                    continue
                # FRESH BAR + XY VERIFY before closing: cage H closed 24 mm off in y
                # (outside the +-15 mm cage) and knocked the bag down. Correct, then
                # verify; if the jaw is still off, do NOT close — leave the bag
                # standing for the loop-thread attempt.
                _bar3 = _bar_now()
                _mv = _bar3 - bar_b
                self.log.ev(f"  bar moved during descent: [{_mv[0]*1000:+.0f}, "
                            f"{_mv[1]*1000:+.0f}, {_mv[2]*1000:+.0f}] mm")
                bar_b = _bar3
                # The correction loop exits on XY ALONE, so a jaw that is dead-centre
                # but 36 mm too high breaks out on the first pass and gets refused by
                # the z gate below without one attempt at closing the gap. bagM hit
                # exactly that: "jaw off bar by [+4, -3, z +36]". Judge both axes, and
                # give the descent more room — the low bar is where the servo stalls.
                # ...and on a low bar, DON'T CORRECT AT ALL. Every correction presses
                # on a 100 g bag that is already under the hand, and the bag answers by
                # sliding: bagN's z-only fix pulled a dead-centre jaw out to [-34, -46]
                # mm, bagO's to [-55, -41]. The one lift this manoeuvre has produced
                # (bagL, +601 mm) was a single clean descent with no chasing. Judge what
                # the descent achieved; if it is off, the attempt loop and then a fresh
                # approach are the honest retries — they re-establish the alignment
                # instead of grinding it away.
                for _try in range(0 if _lowbar else 2):
                    _p_now, _ = self.arm.frame_pose()
                    _exy = _p_now[0].cpu().numpy()[:2] - bar_b[:2]
                    _ezl = float(_p_now[0, 2].cpu()) - (bar_b[2] + depth)
                    if float(np.hypot(*_exy)) <= 0.012 and abs(_ezl) <= 0.015:
                        break
                    if _lowbar:
                        # Z ONLY. Re-aiming at the bar in xy chases an object the hand
                        # is pressing on: bagN's z corrections dragged a dead-centre jaw
                        # out to [-34, -46] mm while the bar itself slid and sank to
                        # 12 mm. Keep the alignment already earned and just close the
                        # height — the successful lift (bagL) was a single clean descent
                        # with no lateral chasing at all.
                        _tz = np.array([float(_p_now[0, 0].cpu()),
                                        float(_p_now[0, 1].cpu()), bar_b[2] + depth])
                        self.servo_tcp_to(_tz, steps=90, grip_open=None)
                    else:
                        self.servo_tcp_to(bar_b + np.array([0, 0, depth]), steps=50,
                                          grip_open=None)
                    bar_b = _bar_now()
                _p_now, _ = self.arm.frame_pose()
                _exy = _p_now[0].cpu().numpy()[:2] - bar_b[:2]
                _ez3 = float(_p_now[0, 2].cpu()) - (bar_b[2] + depth)
                # z belongs in the verify too: cage I closed +32 mm above the bar
                # (xy was fine) and the q_mid swing swept the bag down.
                # ...but on a LOW bar the z target itself is unreachable and the gate
                # then refuses every attempt forever. bagK put the jaw within 1-2 mm of
                # the bar in xy — the best alignment this manoeuvre has ever produced,
                # and the bag moved only 8 mm during the descent, so the toppling that
                # ruins a standing bag simply does not happen to one on its face — and
                # was still refused for "z +43 mm", because the bar SANK 23-33 mm under
                # the hand and the deck stopped the fingers following it. Re-derive the
                # reachable depth from the bar's fresh height and judge z against that.
                if _lowbar:
                    _fresh = float(bar_b[2] - self.to_base_frame_true(np.array([
                        pw.cpu().numpy()[0], pw.cpu().numpy()[1], _gz]))[2])
                    _ez3 = float(_p_now[0, 2].cpu()) - (bar_b[2]
                                                        - max(0.008, _fresh - 0.012))
                    self.log.ev(f"  bag pinch: bar now {_fresh*1000:.0f} mm off the "
                                f"deck, jaw {_ez3*1000:+.0f} mm from the reachable depth")
                # xy tolerance: the cage closes to a 30 mm window around a bar that is
                # 20 mm wide in the closing direction on a lying bag, so anything inside
                # ~22 mm still ends up caged. 15 mm was fitted on the STANDING bag,
                # where the bar is 8 mm across. bagL's second trial was refused at
                # 16 mm — one millimetre outside a bound that does not apply to it.
                _xytol = 0.022 if _lowbar else 0.015
                # z tolerance on a low bar: the last centimetre is not available. With
                # the chasing switched off the descent lands consistently 26-30 mm high
                # (45 steps a stage and 110 differ by 4 mm, so it is not step count —
                # the fingertips reach the deck while the TCP is still short), and the
                # one lift on record closed from +15 mm. Refusing at 22 mm throws away
                # the band in between; the cage costs 60 steps and the lift test is the
                # real verdict, so try it rather than never closing at all.
                _ztol = 0.030 if _lowbar else 0.022
                if float(np.hypot(*_exy)) > _xytol or abs(_ez3) > _ztol:
                    self.log.ev(f"  jaw off bar by [{_exy[0]*1000:+.0f}, "
                                f"{_exy[1]*1000:+.0f}, z {_ez3*1000:+.0f}] mm — not closing")
                    self.servo_tcp_to(bar_b + np.array([0, 0, 0.08]), steps=90,
                                      grip_open=True)
                    self.play_traj([self.retrieve["q_mid"]], grip_open=True,
                                   steps_each=150)
                    continue
                # CAGE: half-close to a 30 mm cage around the 8 mm bar — it cannot
                # exit sideways — the q_mid joint ramp below does the actual lifting
                # (the task-space rise servo measurably sinks; see cage D telemetry).
                # Through grip_target, NOT set_joint_position_target: the direct write
                # jumps the fingers 35 mm in one control step and the PD answers with
                # 6.5 m/s on arm_joint8 against a 1.0 m/s rating (torque audit, 0.34 %
                # of steps, 306 Nm demanded on a 45 Nm joint). write_gripper()'s
                # MISSION_GRIP_RATE ramp is the only thing that keeps the fingers
                # inside their rating.
                self.arm.grip_target = torch.tensor([0.015, -0.015], device=self.device)
                for k in range(60):
                    self.arm.write_gripper()
                    self._step(self.legs.step(self.crouch_cmd(1.0)
                                              if self.legs.mode == "crouch" else zero))
                _p_now, _ = self.arm.frame_pose()
                _err = _p_now[0].cpu().numpy() - (bar_b + np.array([0, 0, depth]))
                self.log.ev(f"  bag handle at-hook: err [{_err[0]*1000:+.0f}, "
                            f"{_err[1]*1000:+.0f}, {_err[2]*1000:+.0f}] mm")

            _gw = float((self.robot.data.joint_pos[0, self.arm.grip_ids]).abs().sum()) \
                if hasattr(self.arm, "grip_ids") else -1.0
            self.log.ev(f"  bag handle caged: gw {_gw:.3f}")
            # q_mid IS the lift; jaw closes on the way so the bar ends pinched or
            # hooked against the hand for transport.
            self.play_traj([self.retrieve["q_mid"]], grip_open=False, steps_each=190)
            for _ in range(40):
                self._write_arm(open_=False)
                self._step(self.legs.step(self.crouch_cmd(1.0)
                                          if self.legs.mode == "crouch" else zero))
            _gw2 = float((self.robot.data.joint_pos[0, self.arm.grip_ids]).abs().sum())
            self.log.ev(f"  after q_mid lift: gw {_gw2:.3f}")
            now_w = obj.data.root_pos_w[0]
            rose = float(now_w[2]) - z0
            p_tcp, _ = self.arm.frame_pose()
            gap = float(np.linalg.norm(self.to_base_frame_true(now_w.cpu().numpy())
                                       - p_tcp[0].cpu().numpy()))
            # The gap bound is a "did it come with the hand" test, and a HOOKED bag
            # hangs further out than a pinched one: the handle window sits 165 mm from
            # the bag's centre and the body swings under it, so a good thread reads
            # 0.25-0.40 m where the cage pinch reads ~0.21. `rose` is the load-bearing
            # half of the test anyway — 100 mm of altitude is not something a bag left
            # on the ground can fake.
            held = rose > 0.10 and gap < 0.40
            self.log.ev(f"  bag handle pinch try {attempt + 1}: rose {rose:+.3f} m, "
                        f"gap {gap:.3f} m -> {'HELD' if held else 'MISS'}")
            if held:
                return True
            for _ in range(30):
                self._write_arm(open_=True)
                self._step(self.legs.step(self.crouch_cmd(1.0)
                                          if self.legs.mode == "crouch" else zero))
            self.play_traj([self.retrieve["q_mid"]], grip_open=True, steps_each=150)
        return False

    def attempt_grasp(self, name: str, det) -> bool:
        """One crouch's full attempt budget: the RL policy first, then the scripted IK
        pinch for the rigid classes once the policy has burned MISSION_RL_TRIES (default
        2) attempts without converting. Shared by the mission FSM and probe_manip so the
        probe measures the same machine the venue runs. Bails out early when the object
        left the bank or was knocked away — that needs a re-approach, not another try at
        a stale position."""
        rl_tries = max(1, int(os.environ.get("MISSION_RL_TRIES", "2")))
        budget = self.p.grasp_retries + 1
        # propped-object gate (see _perched); the cycle loop checks before the
        # approach, this catches an object knocked onto something mid-cycle
        self._grasp_perched = False
        if self._perched(name, "grasp skipped"):
            self._grasp_perched = True
            return False
        if self.obj_cls[name] in ("redbag", "purplebag"):
            # Time-box the bags: the lying-bag pinch has never converted (0/50 history,
            # and s5600c burned ~20 attempts on one bag for 0 points while a 20-point
            # lunchbox went unconverted). Two tries per crouch, then move on.
            budget = min(budget, int(os.environ.get("MISSION_BAG_TRIES", "2")))
        # Lunchbox economics (0903 record, 14 crouch cycles): RL attempt 1 held 4/14,
        # attempt 2 2/9, attempt 3 1/9, the IK pinch 0/9 — while a FRESH cycle
        # (stand, re-look, re-approach, re-crouch) is back at ~50 %. A 5-attempt
        # crouch with two IK calls costs ~5,000 steps (svH: 7,098 for one cycle, and
        # the OBJ_CAP then wrote the object off after that single cycle). Cap the
        # lunchbox at MISSION_LBOX_TRIES per crouch and spend the steps on cycles.
        if self.obj_cls[name] == "lunchbox" and os.environ.get("MISSION_LBOX_TRIES"):
            _lb = max(1, int(os.environ["MISSION_LBOX_TRIES"]))
            rl_tries = min(rl_tries, _lb)
            # last cycle with the IK-last-resort switch on: RL tries + ONE IK call
            _ik_tail = (os.environ.get("MISSION_IK_LAST_CYCLE", "0") == "1"
                        and getattr(self, "_last_cycle", False)
                        and "lunchbox" in os.environ.get("MISSION_RIGID_IK_CLASSES",
                                                         "can,rind,lunchbox").split(","))
            budget = min(budget, _lb + (1 if _ik_tail else 0))
        # STANDING CAN (MISSION_UPCAN_RL_TRIES=N, off unless set). Seed 0's grass can
        # lands on its end (identity spawn quaternion, 20 cm drop onto grass) and the
        # box-trained, yaw-blind policy converts it 7/44 across svY..swI — the usual
        # miss is "TCP never within 0.20 m". A standing 66 mm cylinder straddled 25 mm
        # under its crown is the geometry the scripted pinch was written for, and the
        # lying-can IK record (1/208, pinch point at ground level) says nothing about
        # it. N RL tries, then the pinch, in EVERY cycle. Uprightness is re-read per
        # attempt: an RL miss that topples the can hands it back to the policy.
        _upcan_n = os.environ.get("MISSION_UPCAN_RL_TRIES")
        got = False
        # start configs already tried in THIS squat (MISSION_BANK_PICK=policy): a retry
        # goes to the next-best one, not back to the config that just failed
        self._pick_used = []
        self._pick_clear_bump = 0.0
        for k in range(budget):
            self._grasp_out_of_bank = False
            self._grasp_displaced = False
            _upcan = False
            if _upcan_n and self.obj_cls[name] == "can":
                if self.obj_honest:
                    # honest: the can's attitude is the CAMERA's measured height —
                    # 0.115 m standing against 0.066 m on its side (see _det_top_height)
                    _upcan = self._det_top_height("can", det) > 0.09
                else:
                    _R = math_utils.matrix_from_quat(
                        self._sim_obj(name).data.root_quat_w[0].unsqueeze(0))[0]
                    _upcan = abs(float(_R[2, 2])) > 0.7
                if _upcan and k == 0:
                    _how = (f"measured height {self._det_top_height('can', det):.3f} m"
                            if self.obj_honest else f"axis·z {abs(float(_R[2, 2])):.2f}")
                    self.log.ev(f"{name} is standing on its end ({_how}) — "
                                f"{max(0, int(_upcan_n))} RL tries, then the scripted "
                                f"straddle")
            # MISSION_RIGID_IK_CLASSES narrows the fallback: over 27 missions the
            # scripted pinch converted 1 of 208 calls (109 "hover unreachable", 34
            # jaw-axis misses) — for can/rind it is pure time, ~600 steps a call
            _ik_cls = os.environ.get("MISSION_RIGID_IK_CLASSES", "can,rind,lunchbox").split(",")
            # MISSION_IK_LAST_CYCLE=1: the IK fallback only in the object's LAST
            # cycle (it is the last resort, not a 900-step tax on every crouch)
            _ik_ok = (os.environ.get("MISSION_IK_LAST_CYCLE", "0") != "1"
                      or getattr(self, "_last_cycle", True))
            # N=0 puts the straddle FIRST: the probe (upcan1) showed the pinch is
            # pointless AFTER an RL miss — the miss has already rolled the can away
            # (hover unreachable 643 mm / object moved 128 mm) — so its only chance
            # is on an untouched can
            _k_ik = max(0, int(_upcan_n)) if _upcan else rl_tries
            # standing can: ONE straddle at attempt N, the policy gets the rest
            _ik_turn = (k == _k_ik) if _upcan else (k >= _k_ik)
            if (_ik_turn and os.environ.get("MISSION_RIGID_IK", "1") != "0"
                    and self.obj_cls[name] in ("can", "rind", "lunchbox")
                    and ((self.obj_cls[name] in _ik_cls and _ik_ok) or _upcan)):
                self.log.ev(f"grasp attempt {k + 1}: RL burned {_k_ik} tries — "
                            f"switching to the scripted IK pinch")
                got = self.rigid_grasp_ik(name, det)
            else:
                if self._grasp_alt is not None:
                    self.grasp = self._grasp_pri if k % 2 == 0 else self._grasp_alt
                    if k == 1:
                        self.log.ev("grasp attempt 2: alternate-seed policy "
                                    "(retry-diversity ensemble)")
                got = self.run_grasp(name, det)
            if got:
                break
            if self._grasp_out_of_bank or self._grasp_displaced:
                self.log.ev(f"grasp attempt {k + 1}: object is out of the bank "
                            f"or was moved by the attempt — re-approaching "
                            f"instead of retrying at a stale position")
                break
            self.log.ev(f"grasp attempt {k + 1} failed")
        return got

    def rigid_grasp_ik(self, name: str, det) -> bool:
        """Scripted top-down IK pinch for the rigid classes (can/rind/lunchbox) — the
        fallback once the RL policy has burned its tries. Same machinery as the bag
        pinch: hover above the pinch point, descend, trim the jaw-axis miss against the
        LIVE pose, staged close, two-stage lift, held() verdict.
        MISSION_RIGID_IK=0 disables the fallback.

        Geometry: the fingers straddle the smallest roughly-horizontal object axis at
        the widest-chord depth. When no horizontal extent fits the jaws (the lunchbox
        lid is 110+ mm on both axes against the 90 mm span), the pinch moves to the
        NEAR top edge along that axis — the scripted cousin of the shallow lid pinch
        the policy learned.

        Runs under the SCOPED TCP frame fix, exactly like bag_handle_pinch: the RL
        loop keeps the buggy plant it was trained on, the scripted servos get the
        correct transform for their duration only (the global MISSION_IK_FRAME_FIX=1
        round measured hover errors of 226-561 mm — it breaks everything else)."""
        import course_mission.mission as _msn
        _ikff_saved = _msn.IK_FRAME_FIX
        _msn.IK_FRAME_FIX = True
        try:
            if self.obj_honest:
                return self._rigid_grasp_ik_honest(name, det)
            return self._rigid_grasp_ik_body(name, det)
        finally:
            _msn.IK_FRAME_FIX = _ikff_saved

    def _rigid_grasp_ik_honest(self, name: str, det) -> bool:
        """The scripted pinch with the CAMERA in place of the simulator.

        Same manoeuvre as `_rigid_grasp_ik_body` — hover, pre-tilt about the jaw axis,
        descend, staged close, two-stage lift — but every geometric input comes from
        the detection: the centre from the blob's ground contact line plus half the
        class's depth, the top from the class's resting height above the ground the
        robot is standing on, and the straddle direction across the blob's long axis
        (its own line of sight when the blob is too round to have one). The target
        lives in the decision frame and is re-expressed through the pose estimate each
        servo step (servo_tcp_pose(decision=True)); the live-pose trim has no honest
        counterpart and is dropped. The verdict is the fingers': a real width against
        a working drive after the lift.
        """
        cls = self.obj_cls[name]
        exts = list(OBJ_DIMS[cls])
        hint_w, depth = grasp_hint(cls, exts)
        centre = self._det_centre_S(cls, det)               # decision frame
        gz = self._ground_z() if self.nav.honest else ground_z_world(centre[0], centre[1])
        top_z = gz + self._det_top_height(cls, det)
        # Straddle direction: across the blob's long axis when it has one, else across
        # the line of sight (a round footprint has no preferred side, and the wrist is
        # already square to the approach).
        _yaw = float(det.get("yaw_ref", det.get("yaw", np.nan)) or np.nan)
        _el = float(det.get("elong", 1.0) or 1.0)
        if not (np.isfinite(_yaw) and _el > 1.3):
            _v = det.get("view_w")
            if _v is not None:
                _yaw = float(np.arctan2(float(_v[1]), float(_v[0])))
            else:
                _cb = self.to_base_frame(centre)
                _yaw = float(np.arctan2(_cb[1], _cb[0]) + self.base_pose()[3])
        jaw = np.array([np.cos(_yaw + np.pi / 2), np.sin(_yaw + np.pi / 2), 0.0])
        ext = float(min(exts[0], exts[1]))
        edge_bite = ext > GRIPPER_MAX_WIDTH - 0.005
        lid_pinch = False
        if cls == "lunchbox" and os.environ.get("MISSION_LBOX_IK", "lid") == "lid":
            edge_bite, lid_pinch = False, True
            depth = float(os.environ.get("MISSION_LBOX_IK_DEPTH", "0.006"))
            ext = 0.085 + 0.020 * min(1.0, depth / 0.028)
        bite = depth if (edge_bite or lid_pinch) else max(depth, min(ext / 2, 0.033))
        point = centre.copy()
        point[2] = top_z - depth if (edge_bite or lid_pinch) else max(top_z - bite, gz + 0.01)
        if edge_bite:
            # bite the NEAR top edge along the straddle axis (the far one is out of reach)
            _e = self.to_base_frame(centre + jaw * (ext / 2 - 0.008))
            _f = self.to_base_frame(centre - jaw * (ext / 2 - 0.008))
            point = point + (jaw if _e[0] < _f[0] else -jaw) * (ext / 2 - 0.008)
        close_w = hint_w if lid_pinch else (min(hint_w, ext) if edge_bite
                                            else max(ext - 0.006, 0.010))
        rb = self.to_base_frame(point)
        self.log.ev(f"  rigid IK {name} (honest): {cls} pinch at base "
                    f"[{rb[0]:.2f},{rb[1]:.2f},{rb[2]:.2f}], jaw ext {ext*1000:.0f} mm"
                    f"{' (edge bite)' if edge_bite else ''}"
                    f"{f' (lid pinch {depth*1000:.0f} mm)' if lid_pinch else ''}"
                    f", close to {close_w*1000:.0f} mm, straddle yaw "
                    f"{np.degrees(_yaw + np.pi / 2):.0f} deg")
        if rb[0] > 0.85 or rb[0] < 0.15 or abs(rb[1]) > 0.40:
            self.log.ev("  rigid IK: pinch point outside the reach window — the RL "
                        "attempt moved it; re-approach instead of retrying in place")
            self._grasp_out_of_bank = True
            return False
        hover = point.copy()
        hover[2] = top_z + 0.11
        e1 = self.servo_tcp_pose(hover, jaw_dir_w=jaw, steps=170, tol=0.02, decision=True)
        if e1 > 0.05:
            self.play_traj([self.retrieve["q_mid"]], grip_open=True, steps_each=90)
            e1 = self.servo_tcp_pose(hover, jaw_dir_w=jaw, steps=170, tol=0.02,
                                     decision=True)
        if e1 > 0.05:
            self.log.ev(f"  rigid IK: hover unreachable (err {e1*1000:.0f} mm)")
            return False
        tilt = float(os.environ.get("MISSION_RIGID_IK_TILT", "0.6"))
        if abs(tilt) > 1e-6:
            n_tilt = 45
            _o = self.to_base_frame(np.zeros(3))
            for _ in range(n_tilt):
                jb = torch.tensor(self.to_base_frame(jaw) - _o, dtype=torch.float32,
                                  device=self.device)
                a = torch.zeros(1, 6, device=self.device)
                a[0, 3:6] = (jb * (tilt / n_tilt) / self.arm.scale).clamp(-1.0, 1.0)
                self.arm.ik_step(a)
                self.arm.set_gripper(True)
                self.arm.write_gripper()
                self._step(self.legs.step(self.crouch_cmd(1.0)))
        # The truth version trims the last few millimetres against the object's live
        # pose; there is no honest counterpart, so the DESCENT itself has to be
        # accurate — a 50 mm rind straddled with a 12-16 mm miss is a miss. Spend the
        # steps here instead: the trim they replace cost 70 of them.
        e2 = self.servo_tcp_pose(point, jaw_dir_w=jaw,
                                 steps=int(os.environ.get("MISSION_IK_SERVO_STEPS", "300")),
                                 tol=float(os.environ.get("MISSION_IK_SERVO_TOL", "0.008")),
                                 decision=True)
        self.log.ev(f"  rigid IK: servo errs {e1*1000:.0f}/{e2*1000:.0f} mm (no live "
                    f"trim — the object's pose is not observable from here)")
        _ld8, _, _ = self._sensed_load()
        if e2 > 0.040 and _ld8:
            # press detector on the FIRST descent too (see the probe-loop twin)
            _tz8 = float(self.arm.frame_pose()[0][0, 2])
            _pn8, _ = self.arm.frame_pose()
            self.servo_tcp_to(_pn8[0].cpu().numpy() + np.array([0.0, 0.0, 0.07]),
                              steps=60, grip_open=True)
            point = point.copy()
            point[2] = max(_tz8 - 0.025, gz + 0.008)
            e2 = self.servo_tcp_pose(point, jaw_dir_w=jaw, steps=160, tol=0.008,
                                     decision=True)
            self.log.ev(f"  rigid IK: first descent stalled ON the object — top at "
                        f"TCP z, grip re-aimed 25 mm below (servo {e2*1000:.0f} mm)")
        _leg = lambda: self.crouch_cmd(1.0)  # noqa: E731
        half = min(close_w, 2 * 0.05 - 0.004) / 2
        # TACTILE SWEEP. fv2/fv3 measured the pinch's one failure mode precisely:
        # servo errs 30/8 mm (the ARM goes exactly where told) yet "closed to 59.9 mm
        # against a 60.0 mm command" — the estimate's residual error (30-80 mm even
        # after the crouch re-see) exceeds the can's ±17 mm capture margin, and the
        # old pinch closed ONCE on air and walked away. Fingers closing on air are an
        # honest measurement that the object is elsewhere, so sweep the neighbourhood:
        # reopen, rise 70 mm, shift ±MISSION_IK_PROBE_D along the jaw axis / across
        # it, re-descend, re-close. Each probe reads the load PRE-lift (~180 steps);
        # the full three-stage lift is paid only when the fingers report a catch.
        _pmaxik = int(os.environ.get("MISSION_IK_PROBES", "4"))
        _pdik = float(os.environ.get("MISSION_IK_PROBE_D", "0.035"))
        # Z-DROP built into the sweep (fv4/fv5): pinches with a 13 mm-accurate
        # estimate still closed on air FOUR probes running — an xy pattern cannot
        # miss that way, but the HEIGHT can: the pinch z rides on _det_top_height's
        # standing-vs-lying call, and a can read standing while lying flat puts the
        # close plane ~50 mm above the real top. So probe 1 re-closes straight DOWN
        # (same xy, z - PROBE_DZ) and every later xy shift stays at the lowered z.
        _pdz = float(os.environ.get("MISSION_IK_PROBE_DZ", "0.030"))
        _perp = np.array([-jaw[1], jaw[0], 0.0])
        _pat = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (-1, -1), (1, -1), (-1, 1)]
        if os.environ.get("MISSION_IK_PROBE_PATTERN", "") == "los":
            # 1-D sweep along the LINE OF SIGHT only (0909): after the re-see and
            # the forward calibration the jaw-axis (lateral) error is ~±5 mm — the
            # remaining spread lives along x/LOS. Sweeping the jaw axis is what
            # manufactured the side-jams; LOS-only closes keep the pinch line on
            # the object's centreline, so every close is a real pad-pinch chance.
            _pat = [(0, 0), (0, 1), (0, -1), (0, 2), (0, -2)]
        _gap_ik = float(os.environ.get("MISSION_IK_HELD_GAP", "0.004"))
        _eff_ik = float(os.environ.get("MISSION_IK_HELD_EFFORT", "15"))
        _caught = False
        _pt_used = point.copy()
        for _pk in range(1 + _pmaxik):
            if _pk:
                for _ in range(30):
                    self.arm.grip_target = torch.tensor([0.05, -0.05],
                                                        device=self.device)
                    self.arm.write_gripper()
                    self._step(self.legs.step(_leg()))
                _pn, _ = self.arm.frame_pose()
                self.servo_tcp_to(_pn[0].cpu().numpy() + np.array([0.0, 0.0, 0.07]),
                                  steps=60, grip_open=True)
                _da, _dp = _pat[(_pk - 1) % len(_pat)]
                _pt2 = point + jaw * (_da * _pdik) + _perp * (_dp * _pdik)
                _pt2[2] = max(float(point[2]) - _pdz, gz + 0.008)
                _pt_used = _pt2
                _e3 = self.servo_tcp_pose(_pt2, jaw_dir_w=jaw, steps=200, tol=0.008,
                                          decision=True)
                self.log.ev(f"  rigid IK probe {_pk}/{_pmaxik}: fingers met nothing — "
                            f"shifted [{_da*_pdik*1000:+.0f}, {_dp*_pdik*1000:+.0f}] mm "
                            f"jaw/perp, z {-_pdz*1000:.0f} mm, re-descending "
                            f"(servo {_e3*1000:.0f} mm)")
                # PRESS DETECTOR (0909). Every "catch" that then died at the lift
                # shares one trail: descent servo stalled 40-160 mm HIGH with a load
                # already reading — the hand is sitting ON TOP of an object taller
                # than the height model believed (a standing can read as lying), and
                # the "grip" it goes on to report is fingertips squeezing the top rim,
                # which sheds at liftoff every time. Contact + a high stall IS a
                # height measurement: the object's top is at the stalled TCP. Re-aim
                # the pinch 25 mm below it and approach from above.
                _ld9, _, _ = self._sensed_load()
                if _e3 > 0.040 and _ld9:
                    _tz = float(self.arm.frame_pose()[0][0, 2])
                    _pn9, _ = self.arm.frame_pose()
                    self.servo_tcp_to(_pn9[0].cpu().numpy() + np.array([0.0, 0.0, 0.07]),
                                      steps=60, grip_open=True)
                    _pt2 = _pt2.copy()
                    _pt2[2] = max(_tz - 0.025, gz + 0.008)
                    _e3 = self.servo_tcp_pose(_pt2, jaw_dir_w=jaw, steps=160,
                                              tol=0.008, decision=True)
                    self.log.ev(f"  rigid IK probe: descent stalled ON the object — "
                                f"top measured at TCP z, re-aimed grip 25 mm below "
                                f"(servo {_e3*1000:.0f} mm)")
            for k in range(50):
                w = 0.05 - (0.05 - half) * (k + 1) / 50
                self.arm.grip_target = torch.tensor([w, -w], device=self.device)
                self.arm.write_gripper()
                self._step(self.legs.step(_leg()))
            for _ in range(40 if _pk < _pmaxik else 75):
                self.arm.write_gripper()
                self._step(self.legs.step(_leg()))
            _ld0, _gw0, _ge0 = self._sensed_load()
            try:
                # diagnostics only: where IS the hand relative to the article at the
                # moment the fingers finish closing? 74 sweep probes with a 13 mm
                # estimate all met air — this is the number that says why.
                _tp = self.tcp_w()[0].cpu().numpy()
                _op = self._sim_obj(name).data.root_pos_w[0].cpu().numpy()
                self.log.ev(f"  [TRUTH] IK close {_pk}: TCP-object "
                            f"[{(_tp[0]-_op[0])*1000:+.0f}, {(_tp[1]-_op[1])*1000:+.0f}, "
                            f"{(_tp[2]-_op[2])*1000:+.0f}] mm, width {_gw0*1000:.1f}, "
                            f"ld={_ld0} effort={-1.0 if _ge0 is None else _ge0:.1f}")
            except Exception:
                pass
            # WIDTH IS THE PHYSICAL STOP SIGNAL. fv12 close 2 stopped at 66.9 mm on a
            # 60.0 command — a can diameter, the can IS in the jaws — and was thrown
            # away because the effort gate wants >=15 while real reads run 0.1-6 and
            # the load band decays over the 40-75 settle steps as the drive relaxes.
            # A stop margin of MISSION_IK_WIDE_GAP (8 mm) cannot be faked by the
            # hs11c ground pinch (its false HELD had a 0.3 mm margin): fingers parked
            # 8+ mm wider than commanded have something object-sized between them.
            _stop0 = _gw0 > close_w + _gap_ik
            _wide0 = _gw0 > close_w + float(os.environ.get("MISSION_IK_WIDE_GAP",
                                                           "0.008"))
            _clamp0 = _ge0 is None or _ge0 >= _eff_ik
            if _stop0 and (_ld0 or _wide0) and (_clamp0 or _wide0):
                _caught = True
                break
        if not _caught:
            self.log.ev(f"  rigid IK: every probe closed on air "
                        f"(last width {_gw0*1000:.1f} mm vs command "
                        f"{close_w*1000:.1f} mm) — no lift, giving the cycle back")
            return False
        if (os.environ.get("MISSION_IK_REGRIP", "1") == "1"
                and not (lid_pinch or edge_bite)):
            # regrip is STRADDLE geometry (drop 22 mm to grip below a round body's
            # equator). A lid/edge pinch holds a 6 mm bite on a face — dropping onto
            # it would ram the lid; those grips keep the single-close verdict.
            # A CATCH IS A LOCALISATION EVENT: the jaws are on the object to ~1 cm.
            # But a catch at the swept height pinches ABOVE the equator of a round
            # body — width 67-72 mm and effort 15-21 at rest, then the curved surface
            # squeezes out within the first lift centimetres (fv24-29: 22 catches,
            # 22 ejections, final width back at the command, effort 0). So use the
            # measurement: open, drop 22 mm at the position the catch just proved
            # (floor-clamped), close again — a standing can is now gripped further
            # below its midline, a lying can at its equator. Both strictly better.
            # CAGED re-grip (0908b): fully opening to 0.05 let the object roll free
            # during the descent — fv33's three re-grips all re-closed on air. Open
            # only to the caught width + 12 mm: the fingers stay around the object
            # while the hand drops, so it cannot escape sideways.
            _cw = min(0.05, _gw0 / 2 + 0.006)
            for _ in range(25):
                self.arm.grip_target = torch.tensor([_cw, -_cw], device=self.device)
                self.arm.write_gripper()
                self._step(self.legs.step(_leg()))
            _rg = _pt_used.copy()
            _rg[2] = max(float(_pt_used[2]) - float(os.environ.get(
                "MISSION_IK_REGRIP_DZ", "0.022")), gz + 0.008)
            _e4 = self.servo_tcp_pose(_rg, jaw_dir_w=jaw, steps=90, tol=0.008,
                                      decision=True, grip_open=None)
            for k in range(50):
                w = 0.05 - (0.05 - half) * (k + 1) / 50
                self.arm.grip_target = torch.tensor([w, -w], device=self.device)
                self.arm.write_gripper()
                self._step(self.legs.step(_leg()))
            for _ in range(40):
                self.arm.write_gripper()
                self._step(self.legs.step(_leg()))
            _ld0, _gw0, _ge0 = self._sensed_load()
            self.log.ev(f"  rigid IK re-grip: dropped "
                        f"{(float(_pt_used[2])-_rg[2])*1000:.0f} mm at the caught spot "
                        f"(servo {_e4*1000:.0f} mm) -> width {_gw0*1000:.1f} mm, "
                        f"ld={_ld0} effort={-1.0 if _ge0 is None else _ge0:.1f}")
        p_now, _ = self.arm.frame_pose()
        here = p_now[0].cpu().numpy()
        # hold_rot=True ON THE LIFT (0908c). The relative-pose IK re-anchors its zero
        # rotation every step, so the wrist drifts as it climbs — the bag round
        # measured the tool axis going -0.94 -> -0.58 over one servo. fv34 showed a
        # can STILL IN HAND after the caged re-grip (width 68.7, ld=True, effort 13)
        # and gone 26 steps into this climb: a rotating jaw unscrews the object it
        # holds. Pin the wrist for these two servos only; the other call sites keep
        # the drifting form they were tuned on.
        self.servo_tcp_to(here + np.array([0.0, 0.0, 0.035]), steps=100,
                          grip_open=None, hold_rot=True)
        self.servo_tcp_to(here + np.array([0.0, 0.0, 0.09]), steps=100,
                          grip_open=None, hold_rot=True)
        # VERDICT AT +12.5 cm, not +20. fv14/15: the probes caught real loaded grips
        # (width 50-53 on a 44 command, ld=True, effort 9-13) and every one arrived
        # at the final verdict empty — the tall third stage swings marginal
        # edge-pinches off before they are ever credited. 12.5 cm of ride is the same
        # evidence bar the strong-hold verdict uses; the last 11 cm are hoisted only
        # AFTER the catch is banked, where a drop is a carry-loss the carry watch
        # already knows how to handle (re-grasp from the deck) instead of a silent
        # write-off of a grasp that physically happened.
        _ld, _gw, _ge = self._sensed_load()
        # THE FINGERS MUST HAVE BEEN STOPPED SHORT. This clamp is commanded to a width
        # UNDER the object on purpose, so an object between the jaws leaves them wider
        # than the command and the drive stalled against it; fingers that reached the
        # commanded width met nothing. Without this the pinch called a rind HELD at
        # 44.3 mm against a 44.0 mm command and effort 5.9 while the object was still
        # flat on the ground (hs11c @16918).
        _stopped = _gw > close_w + float(os.environ.get("MISSION_IK_HELD_GAP", "0.004"))
        _clamped = _ge is None or _ge >= float(os.environ.get("MISSION_IK_HELD_EFFORT", "15"))
        # width-wide acceptance, mirroring the probe gate above: a post-LIFT width
        # still 8+ mm over the command means the object rode the lift up between the
        # jaws — the strongest honest signal there is, and the effort/load-band pair
        # was measured throwing exactly that away (fv12: can-diameter stops, caught 0)
        _wide = _gw > close_w + float(os.environ.get("MISSION_IK_WIDE_GAP", "0.008"))
        _ok = bool(_stopped and (_ld or _wide) and (_clamped or _wide))
        _obj = self._sim_obj(name)
        _lift_t = float(_obj.data.root_pos_w[0, 2]) - float(gz)
        self.log.ev(f"  rigid IK: closed to {_gw*1000:.1f} mm against a {close_w*1000:.1f} "
                    f"mm command at effort {-1.0 if _ge is None else _ge:.1f} -> "
                    f"{'HELD' if _ok else 'not held'} (sensed"
                    f"{'' if _stopped else ', fingers met nothing'}"
                    f"{'' if _clamped else ', no clamp force'})  [TRUTH] object z above "
                    f"the stance ground {_lift_t:+.3f} m")
        if _ok:
            # bank the catch, THEN hoist the last stage toward the carry pose
            _pn2, _ = self.arm.frame_pose()
            self.servo_tcp_to(_pn2[0].cpu().numpy() + np.array([0.0, 0.0, 0.11]),
                              steps=140, grip_open=None)
        return _ok

    def _rigid_grasp_ik_body(self, name: str, det) -> bool:
        obj = self._sim_obj(name)
        cls = self.obj_cls[name]
        exts = list(OBJ_DIMS[cls])
        hint_w, depth = grasp_hint(cls, exts)
        pw = obj.data.root_pos_w[0]
        R = math_utils.matrix_from_quat(obj.data.root_quat_w[0].unsqueeze(0))[0]
        rest_z = float(pw[2])
        # Top height: the lunchbox USD's origin is its BOTTOM face (TOP_ABOVE_ORIGIN);
        # everything else is centred, so the top is the rotated half-extent stack.
        if cls in TOP_ABOVE_ORIGIN:
            top_z = rest_z + TOP_ABOVE_ORIGIN[cls]
        else:
            top_z = rest_z + sum(abs(float(R[2, i])) * exts[i] / 2 for i in range(3))
        cands = [i for i in range(3) if abs(float(R[2, i])) < 0.6]
        if not cands:
            cands = [int(np.argmin(exts))]
        jaw_i = min(cands, key=lambda i: exts[i])
        jaw = R[:, jaw_i]
        ext = exts[jaw_i]
        edge_bite = ext > GRIPPER_MAX_WIDTH - 0.005
        lid_pinch = False
        if cls == "lunchbox" and os.environ.get("MISSION_LBOX_IK", "lid") == "lid":
            # The edge bite never held a lunchbox (0/all): its inner finger lands on the
            # solid lid top and blocks the descent. What DOES hold it is the policy's
            # slope pinch — a centred straddle of the lid taper (build_lunchbox_usd:
            # 85 mm across the top widening to 105 mm at 28 mm down, so the 100 mm
            # finger span clears it only in the top ~12 mm) and a firm clamp on the two
            # sloped faces. Scripted here with the same geometry; the depth is the one
            # free parameter (MISSION_LBOX_IK_DEPTH, default 6 mm = 5.7 mm of clearance
            # per finger against the servo's ~10 mm settling error).
            edge_bite, lid_pinch = False, True
            depth = float(os.environ.get("MISSION_LBOX_IK_DEPTH", "0.006"))
            ext = 0.085 + 0.020 * min(1.0, depth / 0.028)
        # Bite depth: the class hint is tuned for the RL policy's shallow top pinch; a
        # scripted straddle wants the WIDEST chord (a can gripped 15 mm below its crown
        # presents a 55 mm chord that squirts out; at centre depth it presents all 66).
        # Never below the object's centre plane — that is where the ground begins to
        # matter and the extra depth buys nothing.
        if lid_pinch:
            bite = depth
        else:
            bite = depth if edge_bite else max(depth, min(ext / 2, 0.033))
        point = pw.clone()
        point[2] = top_z - depth if (edge_bite or lid_pinch) else max(top_z - bite, rest_z)
        if edge_bite:
            base_p = self.robot.data.root_pos_w[0]
            sgn = 1.0 if float(((pw + jaw * (ext / 2)) - base_p).norm()) \
                < float(((pw - jaw * (ext / 2)) - base_p).norm()) else -1.0
            point = point + sgn * jaw * (ext / 2 - 0.008)
        rb = self.to_base_frame_true(point.cpu().numpy())
        if lid_pinch:
            close_w = hint_w          # stall the PD well under the taper = the clamp
        else:
            close_w = min(hint_w, ext) if edge_bite else max(ext - 0.006, 0.010)
        self.log.ev(f"  rigid IK {name}: {cls} pinch at base "
                    f"[{rb[0]:.2f},{rb[1]:.2f},{rb[2]:.2f}], jaw ext {ext*1000:.0f} mm"
                    f"{' (edge bite)' if edge_bite else ''}"
                    f"{f' (lid pinch {depth*1000:.0f} mm)' if lid_pinch else ''}"
                    f", close to {close_w*1000:.0f} mm")
        if rb[0] > 0.85 or rb[0] < 0.15 or abs(rb[1]) > 0.40:
            self.log.ev("  rigid IK: pinch point outside the reach window — the RL "
                        "attempt moved it; re-approach instead of retrying in place")
            self._grasp_out_of_bank = True
            return False
        hover = point.clone()
        hover[2] = top_z + 0.11
        jaww = jaw.cpu().numpy()
        e1 = self.servo_tcp_pose(hover.cpu().numpy(), jaw_dir_w=jaww, steps=170, tol=0.02)
        if e1 > 0.05:
            # A failed RL attempt sometimes parks the arm where the DLS servo cannot
            # recover (hover errors 150-474 mm measured). Pass through the validated
            # mid waypoint ONLY then — an unconditional q_mid detour made hover WORSE
            # (the folded pose sits near a descent singularity: 14-52 mm errors without
            # the detour became 150-363 mm with it).
            self.play_traj([self.retrieve["q_mid"]], grip_open=True, steps_each=90)
            e1 = self.servo_tcp_pose(hover.cpu().numpy(), jaw_dir_w=jaww,
                                     steps=170, tol=0.02)
        if e1 > 0.05:
            self.log.ev(f"  rigid IK: hover unreachable (err {e1*1000:.0f} mm)")
            return False
        # PRE-TILT about the jaw axis before descending. The position servo holds
        # orientation (rotation channels commanded zero), and with the tool vertical
        # the wrist pitch binds ~13-15 cm above the ground — measured e2 104-166 mm
        # with the TCP not descending at all, tool tilt pinned at 0-3 deg. Rotating
        # about the JAW axis pitches the tool without disturbing the straddle
        # direction; the RL policy grasps these objects with a tilted tool too.
        tilt = float(os.environ.get("MISSION_RIGID_IK_TILT", "0.6"))
        if abs(tilt) > 1e-6:
            n_tilt = 45
            for _ in range(n_tilt):
                base_R = math_utils.matrix_from_quat(self.robot.data.root_quat_w)[0]
                jb = base_R.T @ jaw
                a = torch.zeros(1, 6, device=self.device)
                a[0, 3:6] = (jb * (tilt / n_tilt) / self.arm.scale).clamp(-1.0, 1.0)
                self.arm.ik_step(a)
                self.arm.set_gripper(True)
                self.arm.write_gripper()
                self._step(self.legs.step(self.crouch_cmd(1.0)))
        e2 = self.servo_tcp_pose(point.cpu().numpy(), jaw_dir_w=jaww, steps=200, tol=0.012)
        # Trim against the LIVE pose: the descend can nudge the object, and a stale
        # centre makes that displacement invisible (the bag pinch's measured lesson).
        pw2 = obj.data.root_pos_w[0]
        R2 = math_utils.matrix_from_quat(obj.data.root_quat_w[0].unsqueeze(0))[0]
        jaw2 = R2[:, jaw_i]
        tcp_now = self.tcp_w()[0]
        miss = float(torch.dot(tcp_now - pw2, jaw2))
        if edge_bite:
            miss -= float(torch.dot(point - pw, jaw))    # keep the deliberate edge offset
        moved = float((pw2 - pw).norm())
        self.log.ev(f"  rigid IK: servo errs {e1*1000:.0f}/{e2*1000:.0f} mm, jaw-axis "
                    f"miss {miss*1000:+.0f} mm, object moved {moved*1000:.0f} mm")
        if abs(miss) > 0.008:
            trim = (tcp_now - jaw2 * miss).cpu().numpy()
            self.servo_tcp_pose(trim, jaw_dir_w=jaw2.cpu().numpy(), steps=70, tol=0.006)
        rest_z = float(obj.data.root_pos_w[0, 2])
        # Close to slightly UNDER the straddled width and HOLD there. Commanding full
        # close on a rigid body drives the PD straight through it and squirts it out —
        # measured on this round's cans: "closed to 0.0 mm", can ejected. A target a few
        # mm under contact stalls the PD ON the object, which IS the clamp force. (Full
        # close remains correct for crushable sheets — that is the bags' path, not this
        # one.) The edge bite grips a wedge, not the full extent, so it keeps the class
        # hint width instead.
        _leg = lambda: self.crouch_cmd(1.0)  # noqa: E731
        half = min(close_w, 2 * 0.05 - 0.004) / 2
        for k in range(50):
            w = 0.05 - (0.05 - half) * (k + 1) / 50
            self.arm.grip_target = torch.tensor([w, -w], device=self.device)
            self.arm.write_gripper()
            self._step(self.legs.step(_leg()))
        for _ in range(75):
            self.arm.write_gripper()
            self._step(self.legs.step(_leg()))
        # Two-stage lift: break the object free slowly before travelling. grip_open=None
        # keeps the stalled clamp target — False would reset it to full close and squeeze
        # a rigid body back out (this round's cans: closed 60 mm, ended 0.2 mm, ejected).
        p_now, _ = self.arm.frame_pose()
        here = p_now[0].cpu().numpy()
        self.servo_tcp_to(here + np.array([0.0, 0.0, 0.035]), steps=100, grip_open=None)
        self.servo_tcp_to(here + np.array([0.0, 0.0, 0.09]), steps=100, grip_open=None)
        self.servo_tcp_to(here + np.array([0.0, 0.0, 0.20]), steps=140, grip_open=None)
        lift = float(obj.data.root_pos_w[0, 2]) - rest_z
        gw = float(self.arm.gripper_width()[0, 0])
        ok = self.held(obj, rest_z)
        self.log.ev(f"  rigid IK: closed to {gw*1000:.1f} mm, lift {lift*1000:+.0f} mm "
                    f"-> {'HELD' if ok else 'not held'}")
        return bool(ok)

    def bag_grasp_ik(self, name: str, det) -> bool:
        """Scripted IK pinch for the paper bags -- the RL policy never converts them.

        Measured basis: ~50 instrumented bag trials, 0 holds. The policy's close decision
        is geometric (no contact sensing), its clamp on a 3 mm sheet is ~2 N, and its own
        first movements knock a standing bag over. A paper bag is picked the way a hand
        does it: put one finger INSIDE the mouth, one outside a wall, and pinch the single
        sheet between them. That is a purely geometric manoeuvre -- exactly what IK is for.

        Two cases by the bag's measured attitude:
          STANDING  descend into the mouth from above so the fingers straddle the wall
                    nearest the robot, then close.
          LYING     the mouth faces sideways and its roof blocks any descent into the
                    interior, so straddle the VERTICAL side wall's plane just beyond the
                    mouth (over open ground), then SLIDE the hand horizontally so the
                    sheet enters the finger gap, then close.
        MISSION_BAG_IK=0 falls back to the RL policy.
        """
        obj = self._sim_obj(name)
        L, W, H = OBJ_DIMS[self.obj_cls[name]]
        wall = 0.003
        pw = obj.data.root_pos_w[0]
        qw = obj.data.root_quat_w[0]
        R = math_utils.matrix_from_quat(qw.unsqueeze(0))[0]
        base_p = self.robot.data.root_pos_w[0]
        up_z = float(R[2, 2])
        standing = abs(up_z) > 0.7
        rest_z = float(pw[2])
        self.log.ev(f"bag IK grasp {name}: {'STANDING' if standing else 'LYING'} "
                    f"(up_z {up_z:+.2f})")
        if (os.environ.get("MISSION_BAG_HANDLE") == "1"
                and self.bag_handle_pinch(name)):
            return True

        def to_b(v_w):
            return self.to_base_frame_true(v_w.cpu().numpy() if torch.is_tensor(v_w) else v_w)

        def refresh():
            nonlocal base_p, pw, R, rest_z
            base_p = self.robot.data.root_pos_w[0]
            pw = obj.data.root_pos_w[0]
            R = math_utils.matrix_from_quat(obj.data.root_quat_w[0].unsqueeze(0))[0]
            rest_z = float(pw[2])

        def reposition_onto(point_w, label, stance_dir_w=None):
            if os.environ.get("MISSION_BAG_ORBIT", "1") == "1" and stance_dir_w is not None:
                self.log.ev(f"  bag IK: orbiting to the {label} side")
                self.stand_up()
                # BACK STRAIGHT OFF FIRST. The robot is crouched right beside the bag;
                # standing up and turning in place is exactly the move that kicked it
                # over on video (the user caught this: "crouch 후 집기 시도를 안 하고
                # 갑자기 일어나면서 발로 차서 넘어뜨리고 다시 crouch해"). Reversing along
                # the current heading opens clearance before any turning happens.
                for _ in range(70):
                    cmd = torch.zeros(1, 3, device=self.device)
                    cmd[0, 0] = -0.28
                    self._write_arm(self.travel_q, open_=False)
                    self._step(self.legs.step(cmd))
                sd = stance_dir_w / stance_dir_w.norm().clamp(min=1e-9)
                # stance distance MUST exceed the orbit's keep-out radius, or the goal is
                # inside the circle the walk refuses to enter -- the first version put the
                # stance at 0.655 m from the centre against a 0.80 m radius, and arrival
                # became a lottery decided by the final straight cut (hover errors of
                # 326-550 mm on the losing tickets).
                # TWO-STAGE: orbit lands DELIBERATELY LONG (0.80 m, outside the walk's
                # 100-250 mm scatter), then approach() closes the last stretch walking
                # FORWARD onto the mouth -- a path that comes from beyond the mouth toward
                # the bag and so cannot cross it. One-stage versions could not hold the
                # 0.40-0.64 m window: stance error 99 mm was enough to miss it.
                stance = (point_w + sd * 0.80).cpu().numpy()
                if not self.orbit_to(pw.cpu().numpy(), stance, radius=0.70):
                    self.log.ev("  bag IK: orbit timed out")
                    return False
                X, Y, _, _ = self.base_pose_true()
                self.log.ev(f"  bag IK: orbit done, stance error "
                            f"{np.hypot(stance[0] - X, stance[1] - Y)*1000:.0f} mm")
                det2 = dict(det)
                det2["pos_w"] = (point_w.cpu().numpy()
                                 if torch.is_tensor(point_w) else np.asarray(point_w))
                ok2, _ = self.approach(det2, reanchor=False)
                if not ok2:
                    self.log.ev("  bag IK: forward trim onto the mouth failed")
                    return False
                if not self.crouch_and_check(det.get("section", "grass")):
                    return False
                refresh()
                return True
            if os.environ.get("MISSION_BAG_REPOSITION", "0") == "0":
                # Repositioning mid-grasp TRAMPLES the bag: the robot stands beside it and
                # every walking path to the far side crosses it. Measured: after a
                # "reposition onto the mouth" the mouth ended up BEHIND the robot at base
                # [-0.12,-0.36] -- the walk went through the bag. Until there is an orbit
                # manoeuvre that keeps a clearance radius, an out-of-window pinch is
                # reported honestly as out of reach instead of made worse.
                self.log.ev(f"  bag IK: {label} out of the reach window — skipping "
                            f"(no safe reposition)")
                return False
            """Stand up and use the NORMAL walking approach to put point_w at the bank
            anchor. Every slimmer variant of this failed in measurement: walking while
            crouched (the legs were in the wrong mode), approach() in reverse (it can
            only trim a few cm), and a raised crouch (0.40 is outside the crouch
            policy's trained 0.20-0.27 band -- it staggered 308 mm off target)."""
            self.log.ev(f"  bag IK: repositioning onto {label}")
            self.stand_up()
            det2 = dict(det)
            det2["pos_w"] = (point_w.cpu().numpy()
                             if torch.is_tensor(point_w) else np.asarray(point_w))
            ok2, _ = self.approach(det2, reanchor=False)
            if not ok2:
                self.log.ev("  bag IK: reposition failed")
                return False
            if not self.crouch_and_check(det.get("section", "grass")):
                return False
            refresh()
            return True

        if standing:
            # Rim pinch on a standing bag, executed STANDING: the rim is at ~0.30 m and
            # the crouched arm cannot hover above it. stand_up() puts the legs in plain
            # walk mode, which holds height without asking the crouch policy for a depth
            # it was never trained on.
            zsign = 1.0 if up_z > 0 else -1.0
            n_w = R[:, 1]
            sgn = 1.0 if float(((pw + n_w * (W / 2)) - base_p).norm()) \
                < float(((pw - n_w * (W / 2)) - base_p).norm()) else -1.0
            wall_c = pw + sgn * n_w * (W / 2 - wall / 2)
            rim_w = wall_c + zsign * R[:, 2] * (H / 2)
            rb = to_b(rim_w)
            if rb[0] > 0.85 or rb[0] < 0.15 or abs(rb[1]) > 0.40:
                if not reposition_onto(rim_w, "rim", stance_dir_w=sgn * n_w):
                    return False
                zsign = 1.0 if float(R[2, 2]) > 0 else -1.0
                n_w = R[:, 1]
                sgn = 1.0 if float(((pw + n_w * (W / 2)) - base_p).norm()) \
                    < float(((pw - n_w * (W / 2)) - base_p).norm()) else -1.0
                wall_c = pw + sgn * n_w * (W / 2 - wall / 2)
            self.stand_up()
            pinch_w = wall_c + zsign * R[:, 2] * (H / 2 - 0.035)
            hover_w = wall_c + zsign * R[:, 2] * (H / 2 + 0.11)
            jaww = (sgn * n_w).cpu().numpy()
            e1 = self.servo_tcp_pose(hover_w.cpu().numpy(), jaw_dir_w=jaww,
                                     steps=190, tol=0.02, stand=True)
            if e1 > 0.05:
                self.log.ev(f"  bag IK: hover unreachable (err {e1*1000:.0f} mm)")
                return False
            e2 = self.servo_tcp_pose(pinch_w.cpu().numpy(), jaw_dir_w=jaww,
                                     steps=170, tol=0.012, stand=True)
            tcp_now = self.tcp_w()[0]
            miss = float(torch.dot(tcp_now - wall_c, sgn * n_w))
            self.log.ev(f"  bag IK: servo errs {e1*1000:.0f}/{e2*1000:.0f} mm, "
                        f"jaw-axis miss {miss*1000:+.0f} mm")
            if abs(miss) > 0.008:
                trim = (tcp_now - (sgn * n_w) * miss).cpu().numpy()
                self.servo_tcp_pose(trim, jaw_dir_w=jaww, steps=70, tol=0.006, stand=True)
            self._bag_stand = True
        else:
            def lying_frames():
                x_w = R[:, 0]
                sg = 1.0 if float(((pw + x_w * (L / 2)) - base_p).norm()) \
                    < float(((pw - x_w * (L / 2)) - base_p).norm()) else -1.0
                md = R[:, 2] if float(R[2, 2]) < 0.3 else -R[:, 2]
                md = md / md.norm().clamp(min=1e-9)
                wc = pw + sg * x_w * (L / 2 - wall / 2)
                return x_w, sg, md, wc

            x_w, sg, mouth_dir, wall_c = lying_frames()
            mouth_w = pw + mouth_dir * (H / 2)
            mb = to_b(mouth_w)
            self.log.ev(f"  bag IK: mouth end at base [{mb[0]:.2f},{mb[1]:.2f}]")
            # ALWAYS try from the crouch we are already in. The window test used to gate
            # this pre-emptively, and the "fix" was worse than the miss: the robot stood
            # up next to the bag, kicked it over on the first orbit step, and then pinched
            # the wreck. The arm's own hover servo is the true reachability test -- let it
            # fail before paying for any repositioning.
            if mb[0] > 0.85 or mb[0] < 0.15 or abs(mb[1]) > 0.40:
                if not reposition_onto(mouth_w, "mouth", stance_dir_w=mouth_dir):
                    return False
                x_w, sg, mouth_dir, wall_c = lying_frames()
            over_w = wall_c + mouth_dir * (H / 2 + 0.03)
            in_w = wall_c + mouth_dir * (H / 2 - 0.030)
            grip_z = rest_z + 0.025
            over_w = over_w.clone(); over_w[2] = grip_z
            in_w = in_w.clone(); in_w[2] = grip_z
            hover_w = over_w.clone(); hover_w[2] = rest_z + W + 0.12
            jaww = (sg * x_w).cpu().numpy()
            hb, ob, ib = to_b(hover_w), to_b(over_w), to_b(in_w)
            self.log.ev(f"  bag IK: targets base hover[{hb[0]:.2f},{hb[1]:.2f},{hb[2]:.2f}]"
                        f" over[{ob[0]:.2f},{ob[1]:.2f},{ob[2]:.2f}]"
                        f" in[{ib[0]:.2f},{ib[1]:.2f},{ib[2]:.2f}]")
            e1 = self.servo_tcp_pose(hover_w.cpu().numpy(), jaw_dir_w=jaww,
                                     steps=170, tol=0.02)
            if e1 > 0.05:
                self.log.ev(f"  bag IK: hover unreachable (err {e1*1000:.0f} mm) "
                            f"— repositioning and retrying once")
                if not reposition_onto(mouth_w, "mouth", stance_dir_w=mouth_dir):
                    return False
                x_w, sg, mouth_dir, wall_c = lying_frames()
                over_w = wall_c + mouth_dir * (H / 2 + 0.03)
                in_w = wall_c + mouth_dir * (H / 2 - 0.030)
                grip_z = rest_z + 0.025
                over_w = over_w.clone(); over_w[2] = grip_z
                in_w = in_w.clone(); in_w[2] = grip_z
                hover_w = over_w.clone(); hover_w[2] = rest_z + W + 0.12
                jaww = (sg * x_w).cpu().numpy()
                e1 = self.servo_tcp_pose(hover_w.cpu().numpy(), jaw_dir_w=jaww,
                                         steps=170, tol=0.02)
                if e1 > 0.05:
                    self.log.ev(f"  bag IK: hover unreachable after reposition "
                                f"(err {e1*1000:.0f} mm)")
                    return False
            e2 = self.servo_tcp_pose(over_w.cpu().numpy(), jaw_dir_w=jaww,
                                     steps=150, tol=0.012)
            e3 = self.servo_tcp_pose(in_w.cpu().numpy(), jaw_dir_w=jaww,
                                     steps=120, tol=0.012)
            tcp_now = self.tcp_w()[0]
            # Measure against the bag's LIVE pose, not the frames captured before the
            # servos. The slide-in runs a finger through the bag's interior, and if it rubs
            # a wall the whole bag shifts -- against a STALE wall_c that displacement is
            # invisible: two trials measured miss -0 and -4 mm and still closed on air.
            pw_live = obj.data.root_pos_w[0]
            R_live = math_utils.matrix_from_quat(obj.data.root_quat_w[0].unsqueeze(0))[0]
            x_live = R_live[:, 0]
            wall_live = pw_live + sg * x_live * (L / 2 - wall / 2)
            drift_bag = float((pw_live - pw).norm())
            miss = float(torch.dot(tcp_now - wall_live, sg * x_live))
            self.log.ev(f"  bag IK: servo errs {e1*1000:.0f}/{e2*1000:.0f}/{e3*1000:.0f} mm"
                        f", jaw-axis miss {miss*1000:+.0f} mm"
                        f", bag moved {drift_bag*1000:.0f} mm during manoeuvre")
            wall_c = wall_live
            x_w = x_live
            if abs(miss) > 0.008:
                trim = (tcp_now - (sg * x_w) * miss).cpu().numpy()
                self.servo_tcp_pose(trim, jaw_dir_w=jaww, steps=70, tol=0.006)
                miss2 = float(torch.dot(self.tcp_w()[0] - wall_c, sg * x_w))
                self.log.ev(f"  bag IK: post-trim miss {miss2*1000:+.0f} mm")
            self._bag_stand = False

        # close, settle, lift, verify. The lift is TWO stages: paper held by a pinch
        # tears free under acceleration, and the measured failure was a clean 86 mm of
        # lift followed by a drop -- so break the object free slowly before travelling.
        # HOOK, THEN PINCH. Measured on seed-1 trash_12: a finger inside the mouth
        # lifted the bag 65 mm with the gripper fully OPEN -- the finger presses the roof
        # from inside and the bag simply hangs on it. Every ground-level close, staged or
        # not, lost the sheet (the free bag slides ahead of whichever pad touches first),
        # but a HANGING bag presses its roof sheet against the finger by its own weight,
        # so closing then cannot miss. Sequence: lift open 80 mm; if the bag came along,
        # close on the loaded sheet; if it did not, fall back to the ground-level slot
        # close (the standing-bag rim pinch still wants that path).
        _leg = ((lambda: torch.zeros(1, 3, device=self.device))
                if getattr(self, "_bag_stand", False) else (lambda: self.crouch_cmd(1.0)))
        hooked = False
        if not getattr(self, "_bag_stand", False):
            p0, _ = self.arm.frame_pose()
            here0 = p0[0].cpu().numpy()
            self.servo_tcp_to(here0 + np.array([0.0, 0.0, 0.08]), steps=110,
                              grip_open=True)
            hooked = (float(obj.data.root_pos_w[0, 2]) - rest_z) > 0.030
            self.log.ev(f"  bag IK: hook lift -> bag "
                        f"{(float(obj.data.root_pos_w[0, 2]) - rest_z)*1000:+.0f} mm "
                        f"({'HOOKED' if hooked else 'no hook'})")
            if not hooked:
                self.servo_tcp_to(here0, steps=90, grip_open=True)
        for k in range(50):
            wtgt = 0.05 - (0.05 - 0.012) * (k + 1) / 50
            self.arm.grip_target = torch.tensor([wtgt, -wtgt], device=self.device)
            self.arm.write_gripper()
            self._step(self.legs.step(_leg()))
        for _ in range(25):
            self.arm.write_gripper()
            self._step(self.legs.step(_leg()))
        for k in range(45):
            wtgt = 0.012 * (1.0 - (k + 1) / 45)
            self.arm.grip_target = torch.tensor([wtgt, -wtgt], device=self.device)
            self.arm.write_gripper()
            self._step(self.legs.step(_leg()))
        for _ in range(40):
            self.arm.set_gripper(False)
            self.arm.write_gripper()
            self._step(self.legs.step(_leg()))
        gw = float(self.arm.gripper_width()[0, 0])
        p_now, _ = self.arm.frame_pose()
        here = p_now[0].cpu().numpy()
        if not hooked:
            self.servo_tcp_to(here + np.array([0.0, 0.0, 0.035]), steps=100,
                              grip_open=False)
            self.servo_tcp_to(here + np.array([0.0, 0.0, 0.09]), steps=100,
                              grip_open=False)
        self.servo_tcp_to(here + np.array([0.0, 0.0, 0.24]), steps=140, grip_open=False)
        lift = float(obj.data.root_pos_w[0, 2]) - rest_z
        held = self.held(obj, rest_z)
        self.log.ev(f"  bag IK: closed to {gw*1000:.1f} mm, lift {lift*1000:+.0f} mm "
                    f"-> {'HELD' if held else 'not held'}")
        return bool(held)

    def servo_tcp_to(self, target_b, steps=140, tol=0.010, grip_open=True,
                     hold_rot=None):
        """Drive the TCP to a base-frame point with the SAME differential IK the grasp
        policy uses.

        grip_open=None leaves grip_target UNTOUCHED (only re-writes it): the clamp on a
        rigid body is a stalled position target a few mm under the object width, and
        grip_open=False here would reset that to full close every step — which squeezed
        the can back out during the lift (measured: staged close to 60 mm, final width
        0.2 mm, object ejected).

        The retrieve trajectory's fixed `q_pick` waypoint aims at the middle of the
        basket, but the object settles wherever it lands — measured 7.6 cm to the side,
        which is outside the gripper's +-5 cm finger span, so the hand closed on nothing
        and the object never left the basket. Servoing onto the measured position fixes
        that.

        hold_rot: the relative-pose IK re-anchors its "zero rotation" to the CURRENT
        tool orientation every step, so whatever the damped solve and the PD lag let
        the wrist drift by is never pulled back — probe p5 measured the tool axis
        going from straight down (z -0.94) to z -0.58 over one hover servo and to
        -0.03 (horizontal) after a lift-out, and the corner pick then stalled at
        0.065 m where the tool-down entry reached 0.009 m. True = command the
        rotation delta back to the orientation the servo STARTED with (<= 0.06 rad
        per step). None = MISSION_SERVO_HOLD_ROT (default 0: every scoring run
        used the drifting form and the other 36 call sites were tuned on it).
        """
        zero = torch.zeros(1, 3, device=self.device)
        tgt = torch.tensor(np.asarray(target_b, float), dtype=torch.float32,
                           device=self.device)
        err = torch.zeros(3, device=self.device)
        if hold_rot is None:
            hold_rot = os.environ.get("MISSION_SERVO_HOLD_ROT", "0") == "1"
        q_hold = self.arm.frame_pose()[1].clone() if hold_rot else None
        # hold the crouch when crouched — a zero command tells the crouch policy to
        # RISE, which is how a crouched servo used to stand the robot up mid-move
        _legs = (lambda: self.crouch_cmd(1.0)) if self.legs.mode == "crouch" \
            else (lambda: zero)
        for _ in range(steps):
            p, _q = self.arm.frame_pose()
            err = tgt - p[0]
            if float(err.norm()) < tol:
                break
            a = torch.zeros(1, 6, device=self.device)
            a[0, :3] = (err / self.arm.scale).clamp(-1.0, 1.0)
            if q_hold is not None:
                # delta such that delta (x) q_now = q_hold (apply_delta_pose
                # left-multiplies), as an axis-angle, capped per step
                dq = math_utils.quat_mul(q_hold, math_utils.quat_inv(_q))[0]
                if float(dq[0]) < 0.0:
                    dq = -dq
                s = float(dq[1:].norm())
                if s > 1e-6:
                    ang = 2.0 * float(np.arctan2(s, float(dq[0])))
                    a[0, 3:6] = dq[1:] / s * min(ang, 0.06) / self.arm.scale
            self.arm.ik_step(a)
            if grip_open is not None:
                self.arm.set_gripper(grip_open)
            self.arm.write_gripper()
            self._step(self.legs.step(_legs()))
        return float(err.norm())

    def servo_tcp_free(self, target_b, steps=80, tol=0.010, grip_open=True):
        """servo_tcp_to() with the tool orientation FREE (ArmDriver.ik_step_pos).

        Second stage for a pick point the fixed-orientation servo cannot close on;
        the caller decides when. Same per-step translation cap as the pose servo
        (arm.scale = 0.02 m), same leg handling. Returns the final residual.
        """
        zero = torch.zeros(1, 3, device=self.device)
        tgt = torch.tensor(np.asarray(target_b, float), dtype=torch.float32,
                           device=self.device)
        err = torch.zeros(3, device=self.device)
        _legs = (lambda: self.crouch_cmd(1.0)) if self.legs.mode == "crouch" \
            else (lambda: zero)
        for _ in range(steps):
            p, _q = self.arm.frame_pose()
            err = tgt - p[0]
            if float(err.norm()) < tol:
                break
            self.arm.ik_step_pos((err / self.arm.scale).clamp(-1.0, 1.0).reshape(1, 3)
                                 * self.arm.scale)
            if grip_open is not None:
                self.arm.set_gripper(grip_open)
            self.arm.write_gripper()
            self._step(self.legs.step(_legs()))
        return float(err.norm())

    def tool_down_z(self) -> float:
        """z-component of the tool axis in the base frame (-1 = pointing straight down)."""
        d = self.robot.data
        ee_q_b = math_utils.quat_mul(math_utils.quat_inv(d.root_quat_w),
                                     d.body_quat_w[:, self.arm.body_idx])
        ax = math_utils.quat_apply(ee_q_b, torch.tensor([[0.0, 0.0, 1.0]],
                                                        device=self.device))
        return float(ax[0, 2])

    def trace(self, name: str, tag: str):
        """Object + TCP + bin geometry at one stage of the unload (diagnostic)."""
        p = self._sim_obj(name).data.root_pos_w[0].cpu().numpy()
        rel = self.to_base_frame_true(p)
        d = self.robot.data
        ee_p = d.body_pos_w[:, self.arm.body_idx]
        ee_q = d.body_quat_w[:, self.arm.body_idx]
        tcp = (ee_p + math_utils.quat_apply(
            ee_q, torch.tensor([[0.0, 0.0, 0.135]], device=self.device)))[0].cpu().numpy()
        self.log.ev(f"  [{tag}] object world={np.round(p, 2).tolist()} "
                    f"base={np.round(rel, 3).tolist()} | TCP world={np.round(tcp, 2).tolist()}")
        return p, tcp

    def settle_payload(self, cycles: int = 2) -> None:
        """Slide the payload to the FRONT of the basket, where the arm can reach it.

        MEASURED: only 54% of the basket floor is inside the arm's in-basket reach
        (x -0.15..0.05 of an interior running back to -0.21), and walking forward is
        exactly what pushes the load into the unreachable rear 6 cm — a lunchbox ended
        there and four pick attempts closed on nothing 0.41 m away. Braking does the
        opposite: the load keeps going and slides forward. So before unloading, walk a
        short distance and stop hard, twice.
        """
        zero = torch.zeros(1, 3, device=self.device)
        cmd = torch.zeros(1, 3, device=self.device)
        cmd[0, 0] = 0.55
        for _ in range(cycles):
            for _ in range(45):                   # brief roll forward
                self._write_arm(self.travel_q, open_=False)
                self._step(self.legs.step(cmd))
            for _ in range(60):                   # hard stop: payload slides forward
                self._write_arm(self.travel_q, open_=False)
                self._step(self.legs.step(zero))

    def pick_from_basket(self, name: str, retries: int = 3) -> bool:
        """Grip the object inside the basket and CONFIRM it before swinging to the bin.

        The single-shot pick missed often (the object rolls to a new spot each time), and
        the arm then carried nothing to the bin. Each try re-measures where the object
        actually is, servos onto it, closes, lifts, and verifies the object came up with
        the hand; only a confirmed grip proceeds.
        """
        obj = self._sim_obj(name)
        zero = torch.zeros(1, 3, device=self.device)
        self._pick_hover_e0 = None       # per-retrieval: the last hover stall residual
        self._pick_nohover = False       # ...and whether the hover proved unreachable
        for attempt in range(retries + 1):
            if self.obj_honest:
                # OBJECT-HONEST retrieval: the robot cannot see into its own basket, so
                # it goes back to the SLOT it released the object over — the TCP pose at
                # release, from the arm's own FK — and drops to the basket floor plus a
                # half-height. A retry walks the target a few cm along the slot's short
                # axis, because an object that rolled did so across the basket.
                _sp = self._basket_slot_p.get(name)
                if _sp is None:
                    self.log.ev(f"  payload {name}: no release slot recorded — "
                                f"cannot retrieve blind")
                    return False
                _fl = float(os.environ.get("MISSION_BASKET_FLOOR_B", "0.09"))
                _hh = 0.5 * float(min(OBJ_DIMS.get(self.obj_cls.get(name, ""),
                                                   (0.07, 0.07, 0.07))))
                obj_b = np.array([_sp[0], _sp[1] + (0.0, 0.04, -0.04, 0.07)[attempt % 4],
                                  _fl + _hh], float)
                _tr = self.to_base_frame_true(obj.data.root_pos_w[0].cpu().numpy())
                self.log.ev(f"  basket pick try {attempt + 1}: aiming at the release "
                            f"slot {np.round(obj_b, 3).tolist()}  [TRUTH] object at "
                            f"{np.round(_tr, 3).tolist()}")
                in_basket = True
            else:
                obj_b = self.to_base_frame_true(obj.data.root_pos_w[0].cpu().numpy())
            # Is the payload even still aboard? A can bounced out during the 30 m walk to
            # the bins, fell off the course, and was at z = -175 m by the time the robot
            # tried to unload it — the loop still ran four "picks" against it and printed
            # servo residuals of 2819 m. Anything this far from the basket is gone.
            # Against the basket's ACTUAL bounds, not a loose sphere. The old test
            # (|z| > 0.6 or |xy| > 1.0) passed an object lying on the ground at
            # base-frame z = -0.35: the reach clamp only limits x and y, so the arm was
            # sent 0.35 m below its own base, converged on that impossible target
            # (servo residual 0.010 m) and reported a 0.43 m gap to the object, four
            # times over, before scoring `area` instead of `correct`.
            if not self.obj_honest:
                in_basket = (abs(obj_b[0] + 0.07) < 0.20 and abs(obj_b[1] + 0.27) < 0.15
                             and 0.04 < obj_b[2] < 0.40)
            if not in_basket:
                self.log.ev(f"  payload {name} is no longer in the basket "
                            f"(base-frame {np.round(obj_b, 2).tolist()}) — lost in transit")
                return False
            # In-basket reach window (scripts/solve_retrieve.py --scan): the arm can
            # only put a tool-down TCP inside this box; clamp instead of chasing an
            # unreachable point and closing on nothing. v1 basket: x -0.15..0.05,
            # y -0.34..-0.19. The v2 (wider, torso-hugging) basket re-scan opens it to
            # x -0.15..0.10, y -0.39..-0.16 — pass MISSION_BASKET_REACH="x0,x1,y0,y1"
            # alongside MISSION_BASKET_USD so window and geometry always travel together.
            _rw = [float(v) for v in os.environ.get(
                "MISSION_BASKET_REACH", "-0.15,0.05,-0.34,-0.19").split(",")]
            reach = obj_b.copy()
            reach[0] = float(np.clip(reach[0], _rw[0], _rw[1]))
            reach[1] = float(np.clip(reach[1], _rw[2], _rw[3]))
            if np.linalg.norm(reach[:2] - obj_b[:2]) > 0.005:
                self.log.ev(f"  try {attempt + 1}: object at "
                            f"{np.round(obj_b, 3).tolist()} is outside the in-basket "
                            f"reach — clamping to {np.round(reach, 3).tolist()}")
            z0 = float(obj_b[2])
            # HOVER FROM INSIDE THE REACH. The +0.07 hover over an object at the aft
            # end of the window (x -0.126, the aft deposit slot) was itself out of
            # reach — svG/svJ: "hover blocked" at residual 0.057-0.070, 3-4 tries
            # per can — while the pick point 7 cm lower servoed fine: the scan's aft
            # limit shrinks with height. Hover no further aft than MISSION_HOVER_XMIN
            # and let the descent add the last few cm diagonally.
            hov = reach + np.array([0.0, 0.0, 0.07])
            hov[0] = max(float(hov[0]),
                         float(os.environ.get("MISSION_HOVER_XMIN", "-0.10")))
            # tool-down held through both basket servos (MISSION_PICK_HOLD_ROT=0 for
            # the drifting form) — see servo_tcp_to(hold_rot)
            _hr = os.environ.get("MISSION_PICK_HOLD_ROT", "1") == "1"
            if getattr(self, "_pick_nohover", False):
                # this retrieval has proved the hover unreachable — the descent target
                # is 7 cm lower and inside the envelope, so aim there directly
                e0 = 0.0
            else:
                e0 = self.servo_tcp_to(hov, steps=110, hold_rot=_hr)
            if e0 > 0.04 and os.environ.get("MISSION_PICK_FREE", "1") == "1":
                # The hover itself is out of the tool-down reach (svP trash_2 at
                # (-0.10, -0.32): "hover blocked" 0.04-0.07 three times, then two
                # descents at 0.08-0.10 — six tries, can left in the basket). Before
                # re-entering, let the wrist tilt: the free servo reaches spots the
                # pose-locked one cannot (see ArmDriver.ik_step_pos).
                _az0 = self.tool_down_z()
                _e0f = self.servo_tcp_free(hov, steps=60)
                self.log.ev(f"  basket pick try {attempt + 1}: hover servo stalled at "
                            f"{e0:.3f} m — free-orientation servo -> {_e0f:.3f} m "
                            f"(tool axis z {_az0:+.2f} -> {self.tool_down_z():+.2f})")
                e0 = _e0f
            if e0 > 0.04 and attempt < retries \
                    and os.environ.get("MISSION_PICK_REHOME", "1") == "1":
                # The hover point is 7 cm above the object; a servo that cannot get
                # within 4 cm of it is wedged (a neighbour under the fingers, or a
                # bad elbow branch from the previous try's lift-out) and the close-
                # and-lift that follows is 450 steps of confirming a miss (svC: six
                # in a row at residual 0.04-0.14). Re-enter the basket from the
                # validated pick pose and try again from there instead.
                # ...but a re-entry only helps when the arm was WEDGED. When the point
                # is simply outside the arm's in-basket envelope, every try stalls at
                # the same residual and the re-entries are pure clock: hs11a's can had
                # rolled to (-0.102, -0.295) and six tries stalled at 0.106-0.121 m,
                # 2,600 steps, before scoring `none`. Two identical stalls (within
                # 15 mm of each other) mean the geometry, not the elbow — stop and let
                # the caller spend the time on something reachable.
                _prev_e0 = getattr(self, "_pick_hover_e0", None)
                _same = _prev_e0 is not None and abs(e0 - _prev_e0) < 0.015
                self._pick_hover_e0 = e0
                if _same and attempt + 1 >= int(
                        os.environ.get("MISSION_PICK_HOVER_GIVEUP", "2")):
                    # The HOVER is what is out of reach, not the object: the scan's aft
                    # limit shrinks with height, and the code's own note records the
                    # pick point 7 cm lower servoing fine where the hover stalled. So
                    # before giving the retrieval up, drop the hover and go straight
                    # down onto the object from the validated q_pick entry — hs16b's
                    # can sat at (-0.119, -0.252), inside the reach window, and five
                    # tries stalled 44-64 mm above it.
                    if not getattr(self, "_pick_nohover", False):
                        self._pick_nohover = True
                        self.log.ev(f"  basket pick try {attempt + 1}: hover blocked at "
                                    f"{e0:.3f} m again — the HOVER is out of reach, not "
                                    f"the object; going straight down onto it instead")
                    else:
                        self.log.ev(f"  basket pick try {attempt + 1}: blocked at "
                                    f"{e0:.3f} m with and without the hover — the point "
                                    f"is outside the arm's in-basket envelope; giving up")
                        return False
                self.log.ev(f"  basket pick try {attempt + 1}: hover blocked (servo "
                            f"residual {e0:.3f} m) — re-entering from the pick pose")
                self.play_traj([self.retrieve["q_mid"], self.retrieve["q_pick"]],
                               grip_open=True, steps_each=120)
                continue
            # ALTERNATE THE CLOSING AXIS. A can is a 66 mm-wide, 122 mm-long cylinder
            # lying at a random yaw: when its long axis happens to line up with the
            # fixed closing axis, 122 > the 100 mm finger span and the hand closes on
            # air AT the exact target — measured four times in a row at servo residual
            # 0.008-0.010 m on a lone, dead-centre can (basket11 trash_1), while the
            # same residuals GRIP whenever the can lies crosswise. The position-only
            # servo never commands rotation, so a wrist roll set here persists: give
            # every second try the perpendicular grip axis.
            # ...except against the v3 basket's rear lip (MISSION_BASKET_LIP_X = its
            # front face): fingers spread along x would land ON the lip for an
            # object resting against it, so those keep the crosswise grip.
            _lip = float(os.environ.get("MISSION_BASKET_LIP_X", "-9.0"))
            # ...and since the object's long axis IS known at this point (the same
            # truth the servo target comes from), roll by need rather than by turn:
            # a can/rind/bottle gets the closing axis SQUARED to its long axis NOW,
            # by the exact signed angle (probe p5: the corner can was closed on at
            # residual 0.009-0.010 twice with the axes parallel before the odd-try
            # roll finally gripped it; p6: a blind 90 deg roll left the grip at 64
            # deg and a fingertip on the can — see _basket_roll_delta). Blind
            # alternation stays as the fallback for an axis-agnostic miss (lunchbox)
            # and for MISSION_PICK_YAW=0.
            _roll = attempt % 2 == 1
            _cls = self.obj_cls.get(name)
            _ang = self._basket_axis_angle(obj, _cls)
            _dq6 = None
            if os.environ.get("MISSION_PICK_YAW", "1") == "1":
                _dq6 = self._basket_roll_delta(obj, _cls)
                if _dq6 is not None:
                    _roll = abs(_dq6) > np.radians(8.0)
            # AT THE LIP the perpendicular grip puts a finger over the lip — but when
            # the long axis is within ~25 deg of the fixed closing axis the unrolled
            # grip cannot close either (probe p7, can at (-0.121,-0.297) against the
            # v4 lip, axes 20 deg apart: two z-only stalls at 0.024/0.032 m, fingers
            # on the can, never gripped). Roll anyway there and shift the reach
            # target forward (MISSION_LIP_ROLL_X past the lip face) so the rear
            # finger comes down clear of the lip; its descent on the can's flank
            # nudges the can off the lip.
            _at_lip = float(obj_b[0]) <= _lip + 0.07
            if _roll and _at_lip and _dq6 is not None and _ang is not None \
                    and _ang < float(os.environ.get("MISSION_LIP_ROLL_ANG", "65")):
                _xmin = _lip + float(os.environ.get("MISSION_LIP_ROLL_X", "0.055"))
                _deep = (_xmin - reach[0]) > float(
                    os.environ.get("MISSION_LIP_DEEP", "0.02"))
                if reach[0] < _xmin and not _deep:
                    self.log.ev(f"  basket pick try {attempt + 1}: object against the "
                                f"rear lip with the axes {_ang:.0f} deg apart — rolling "
                                f"anyway, reach x {reach[0]:+.3f} -> {_xmin:+.3f}")
                    reach[0] = _xmin
                elif reach[0] < _xmin:
                    # DEEP AGAINST THE LIP (swB trash_1: a can the stair climb pushed
                    # back to x -0.141, 3.6 cm past the lip-roll target, six misses).
                    # Shifting the reach forward there puts the aft finger ON the
                    # can's top, not beside it (probe p10: reach -0.105 for a can at
                    # -0.130 stalled 0.009 m high and closed on air; a forward sweep
                    # of the hand rolled it 4 mm DEEPER and undid the wrist roll,
                    # 88 -> 50 deg). Reach at the object's own x instead: at grasp
                    # height the fingertips (z ~0.15) clear the 20 mm v4 lip (top
                    # z 0.118) by 3 cm and the crosswise aft finger (up to 5 cm
                    # aft of the TCP) stays 2 cm off the rear wall (-0.21) — p10's
                    # retry gripped that same can at reach -0.136, 81 deg, first
                    # descent, residual 0.009 m.
                    self.log.ev(f"  basket pick try {attempt + 1}: object {_xmin - reach[0]:.3f} m "
                                f"deeper than the lip-roll target with the axes {_ang:.0f} deg "
                                f"apart — rolling and reaching at its own x {reach[0]:+.3f} "
                                f"(fingertips clear the low lip)")
                _at_lip = False
            if _roll and not _at_lip:
                if _dq6 is None:
                    _q6 = float(self.arm.current_q6()[5])
                    self._roll_wrist(1.5708 if _q6 < 0.0 else -1.5708)
                else:
                    self._roll_wrist(_dq6)
                _ang2 = self._basket_axis_angle(obj, _cls)
                if _dq6 is not None and _ang2 is not None and _ang2 < 70.0:
                    # the sign convention above is derived from the URDF; if the
                    # measured result contradicts it, go the other way (and say so)
                    self._roll_wrist(-2.0 * _dq6)
                    _ang3 = self._basket_axis_angle(obj, _cls)
                    self.log.ev(f"  basket pick try {attempt + 1}: wrist roll "
                                f"{np.degrees(_dq6):+.0f} deg went the wrong way "
                                f"({-1.0 if _ang is None else _ang:.0f} -> {_ang2:.0f} deg) — reversed to "
                                f"{-1.0 if _ang3 is None else _ang3:.0f} deg")
                    _ang2 = _ang3
                self.log.ev(f"  basket pick try {attempt + 1}: wrist rolled "
                            f"{'blind' if _dq6 is None else f'{np.degrees(_dq6):+.0f} deg'}"
                            f" for the crosswise grip (long axis vs closing axis "
                            f"{-1.0 if _ang is None else _ang:.0f} -> "
                            f"{-1.0 if _ang2 is None else _ang2:.0f} deg)")
            e = self.servo_tcp_to(reach + np.array([0.0, 0.0, 0.005]), steps=100,
                                  hold_rot=_hr)
            if e > 0.03 and os.environ.get("MISSION_PICK_FREE", "1") == "1":
                # The fixed-orientation servo stalled short of the object (the
                # front-outboard dead zone, see ArmDriver.ik_step_pos): let the
                # wrist tilt to close the gap instead of closing the hand on air.
                _e1, _az0 = e, self.tool_down_z()
                e = self.servo_tcp_free(reach + np.array([0.0, 0.0, 0.005]), steps=80)
                self.log.ev(f"  basket pick try {attempt + 1}: pose servo stalled at "
                            f"{_e1:.3f} m — free-orientation servo -> {e:.3f} m "
                            f"(tool axis z {_az0:+.2f} -> {self.tool_down_z():+.2f})")
            # pre-close state for the verdict line: WHERE the residual is (a z-only
            # miss is the fingers on the floor/object; an x/y miss is a wall or a
            # joint limit), how the tool points, and the axis alignment
            _pp, _ = self.arm.frame_pose()
            _ev = (reach + np.array([0.0, 0.0, 0.005])) - _pp[0].cpu().numpy()
            _tz = self.tool_down_z()
            _q6r = float(self.arm.current_q6()[5])
            _ang = self._basket_axis_angle(obj, self.obj_cls.get(name))
            for _ in range(70):                                   # close
                self._write_arm(open_=False)
                self._step(self.legs.step(zero))
            # Lift OUT along the validated joint waypoint, not by servoing straight up:
            # deep inside the basket the IK cannot raise the TCP at all (measured: servo
            # residual 0.008 m, TCP gap 0.004 m — the hand sat perfectly on the object —
            # yet it rose only 0.005 m because the arm never moved). q_mid is a
            # reachability-checked pose high over the torso.
            self.play_traj([self.retrieve["q_mid"]], grip_open=False, steps_each=170)
            # confirmation: did the object come up with the hand, and is it still between
            # the fingers?
            now_b = self.to_base_frame_true(obj.data.root_pos_w[0].cpu().numpy())
            p, _q = self.arm.frame_pose()
            gap = float(np.linalg.norm(now_b - p[0].cpu().numpy()))
            rose = float(now_b[2]) - z0
            ok = rose > 0.10 and gap < 0.12
            if self.obj_honest:
                # GRIPPED = the fingers came out of the basket loaded: a width inside
                # the band an object leaves and the drive still working against it.
                # q_mid is high over the torso, so "it came up with the hand" is the
                # trajectory's job, not a measurement's.
                _ld, _gw, _ge = self._sensed_load()
                self.log.ev(f"  basket pick try {attempt + 1}: width {_gw * 1000:.0f} mm, "
                            f"effort {-1.0 if _ge is None else _ge:.1f} -> "
                            f"{'GRIPPED' if _ld else 'MISSED'} (sensed)  [TRUTH] rose "
                            f"{rose:+.3f} m, TCP gap {gap:.3f} m -> "
                            f"{'GRIPPED' if ok else 'MISSED'}")
                ok = _ld
            self.log.ev(f"  basket pick try {attempt + 1}: object at "
                        f"{np.round(obj_b, 3).tolist()}, servo residual {e:.3f} m "
                        f"(base-frame miss {np.round(_ev, 3).tolist()}, tool axis z "
                        f"{_tz:+.2f}, wrist q6 {_q6r:+.2f}, long axis vs closing axis "
                        f"{-1.0 if _ang is None else _ang:.0f} deg), "
                        f"object rose {rose:+.3f} m, TCP gap {gap:.3f} m -> "
                        f"{'GRIPPED' if ok else 'MISSED'}")
            if ok:
                return True
            # Release, then RE-ENTER FROM THE PICK POSE for the next try. The old
            # pose-locked lift (servo +0.16 m from q_mid) kept q_mid's nearly
            # horizontal tool (axis z -0.23), so every try after the first hovered and
            # descended with the fingers pointing sideways — probe p5 measured tool
            # axis z -0.18 / -0.03 on try 2 and the corner servo stalled at 0.065 m
            # where the tool-down try 1 had reached 0.009 m. q_mid -> q_pick is the
            # validated tool-down entry and costs the same ~120 steps.
            for _ in range(30):
                self._write_arm(open_=True)
                self._step(self.legs.step(zero))
            self.play_traj([self.retrieve["q_pick"]], grip_open=True, steps_each=120)
        return False

    def _basket_axis_angle(self, obj, cls):
        """Horizontal angle (deg, 0..90) between the object's long axis and the
        gripper's closing axis; None for a class without a dominant axis.

        Both in the base frame. The closing axis is gripper_base local x (URDF joint7/
        joint8 both slide along it); the object's long axis is the local axis with the
        largest OBJ_DIMS extent. Can 66x115, rind 50x200: with the two axes parallel the
        fingers land ON the object instead of around it (100 mm span).
        """
        v = self._basket_axes(obj, cls)
        if v is None:
            return None
        a, c, _tz = v
        cosang = abs(float(np.dot(a, c)))
        return float(np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0))))

    def _basket_axes(self, obj, cls):
        """(long axis, closing axis, tool-axis z) — the first two as horizontal unit
        vectors in the base frame; None for a class without a dominant axis or when
        either axis is near-vertical."""
        if cls not in OBJ_DIMS or self.obj_honest:
            # object-honest: the object's orientation inside the basket is unobservable
            # (no camera looks in there), so every axis-aware branch falls back to the
            # blind alternation the callers already implement
            return None
        exts = OBJ_DIMS[cls]
        i_long = int(np.argmax(exts))
        srt = sorted(exts)
        if srt[-1] < 1.3 * srt[-2]:                  # no dominant axis (lunchbox-like)
            return None
        d = self.robot.data
        rq_inv = math_utils.quat_inv(d.root_quat_w[:1])
        R_obj = math_utils.matrix_from_quat(
            math_utils.quat_mul(rq_inv, obj.data.root_quat_w[:1]))[0].cpu().numpy()
        R_ee = math_utils.matrix_from_quat(
            math_utils.quat_mul(rq_inv, d.body_quat_w[:1, self.arm.body_idx]))[0].cpu().numpy()
        a = R_obj[:2, i_long]
        c = R_ee[:2, 0]
        na, nc = np.linalg.norm(a), np.linalg.norm(c)
        if na < 0.3 or nc < 0.3:                     # one of them is near-vertical
            return None
        return a / na, c / nc, float(R_ee[2, 2])

    def _basket_roll_delta(self, obj, cls):
        """Wrist-roll (q6) delta that squares the closing axis to the object's long
        axis, the short way round; None without a dominant axis.

        Joint 6 spins about the tool axis (URDF arm_joint6 axis 0 0 1 = gripper_base
        z): +dq6 turns the closing axis by +dq6 about the tool axis, which is -dq6
        about base z while the tool points down. Squareness matters to a few degrees:
        the can's shadow along the closing axis is 2*(57 sin t + 33 cos t) for a
        misalignment t from perpendicular — 66 mm square, 101 mm at 20 deg, past the
        100 mm finger span (probe p6: a 64 deg grip put a fingertip ON the can, servo
        stalled 41 mm high; the 200 mm rind tolerates ~13 deg).
        """
        v = self._basket_axes(obj, cls)
        if v is None:
            return None
        a, c, tz = v
        phi = float(np.degrees(np.arctan2(a[0] * c[1] - a[1] * c[0], np.dot(a, c))))
        if phi > 90.0:                               # the long axis is undirected
            phi -= 180.0
        elif phi <= -90.0:
            phi += 180.0
        d_head = (90.0 if phi >= 0.0 else -90.0) - phi
        return float(np.radians(-d_head if tz < 0.0 else d_head))

    def _roll_wrist(self, dq6, steps=60):
        """Turn joint 6 by dq6 (rad), taking the other perpendicular (dq6 -/+ pi, the
        closing axis being a line) when that would run into the +-3.14 joint limit."""
        q = self.arm.current_q6().clone()
        new = float(q[5]) + dq6
        if abs(new) > 2.9:
            new -= np.pi * np.sign(dq6)
        q[5] = new
        self.play_traj([q.tolist()], grip_open=True, steps_each=steps)

    def goto_arc(self, target_arc: float, lat_target: float = 0.0) -> bool:
        """Walk to an arc position, turning around when the target is behind us.

        Used to shuttle back to the search point after emptying the basket at the bins.
        The walk policy only goes forward, so reversing means facing along the reversed
        course tangent (the ping-pong trick from course_sim/scripts/walk_course.py).
        """
        t0 = self.step_i
        # Never aim past either end of the course. The platform runs COURSE_X0..+21 and
        # walking off the back leaves the robot on the ground below it, unable to climb
        # back on — a run sat at arc -22.0 spending its whole remaining budget printing
        # "stalled near arc -22.0".
        # 4.0 m of margin. 2.0 was NOT enough and it cost two whole runs: the shuttle
        # stops on target at -18.5, the face_forward turn-in-place that follows drifts
        # ~2 m further back, and the traverse stall-push (vx 0.9, up to 300 steps) adds
        # more — both the v4 control run and the v5 'far' run ended up at arc -22.0, off
        # the back of the platform, printing "stalled near arc -22.0; off the course"
        # until the watchdog killed the search. 'far' quit at 38,321 steps with 16.6
        # minutes of its budget unspent. The robot cannot climb back on, so the only cure
        # is never to get there. Objects live between arc -18 and +13 and the earliest
        # ones are usually excluded anyway, so the last 4 m of walkable course is free.
        if self.nav.honest:
            target_arc = float(np.clip(target_arc, self.nav.s_start() + 2.8,
                                       self._course_end() - 0.8))
        else:
            target_arc = float(np.clip(target_arc, ct.COURSE_X0 + 2.8,
                                       ct.COURSE_X0 + ct.COURSE_LEN - 0.8))
        ax0 = self.nav.arc_lat()[0]
        stall_ax, stall_at = 1e9, self.step_i      # progress watchdog for the re-aim below
        stuck = 0                                  # consecutive fruitless re-aims
        wrong_way = 0                              # times the gap to target grew
        dir_sign_hint = 1.0 if target_arc > ax0 else -1.0
        budget = int(1800 + 300 * abs(target_arc - ax0))
        self.log.ev(f"shuttling from arc {ax0:.1f} to {target_arc:.1f} "
                    f"({'back' if target_arc < ax0 else 'forward'})")
        # TURN FIRST, THEN WALK. The loop below drives at vx=0.6 while correcting heading
        # at a 0.7 rad/s yaw cap, so a reversal sweeps a wide arc: ~2.7 m of forward travel
        # for the turn alone, and more in practice. MEASURED — all three runs of 0817_1504
        # died of this. The oracle run asked to shuttle from arc 14.5 back to -10.6 and
        # went FORWARDS, 14.5 -> 19.0 -> 22.0, straight off the +21 end of the course,
        # where it re-aimed for the rest of its budget; the camera run swept the same turn
        # sideways and ended 124 m off the centreline. face_forward turns in place with no
        # forward command and converges in a couple of seconds, which is exactly what is
        # needed when there are only 6.5 m of course left ahead.
        if dir_sign_hint < 0:
            self.face_forward(steps=250, dir_sign=-1.0)
        while self.step_i - t0 < budget:
            if (self.step_i - t0) % 200 == 0 and \
                    getattr(self, "_carry_lost", None) is not None and \
                    [n for n, _ in self.carried] == [self._carry_lost]:
                self.log.ev(f"  hand cargo {self._carry_lost} lost mid-walk — "
                            "aborting the bin trip")
                return False
            ax, w = self.nav.arc_lat()
            # LATERAL GUARD, same as traverse. goto_arc had NO lateral test at all — only
            # goto_bin did (|w| > 2.5) — so a backward shuttle could drift off the side of
            # the platform unwatched. MEASURED: run c2 shuttled arc 9.3 -> -0.7 normally
            # and then reported arc -22.0 with lat -56.83, growing to -70.61: it had left
            # the platform entirely and arc_of was projecting a point 57-70 m away. By the
            # time traverse noticed, nothing could walk it back. Catch it at 1.8 m, while
            # the robot is still ON the course, and steer to the centreline before
            # resuming; give up on the shuttle rather than walking further off.
            if abs(w) > LAT_GUARD:
                if abs(w) > 4.0:
                    # hopelessly off the deck — r13 racked up 32 strikes walking at
                    # lat +104 in the void; nothing out there scores
                    self.log.ev(f"shuttle at lat {w:+.1f} — off the deck, ending it")
                    self.off_platform = True
                    return False
                self.lat_strikes += 1
                self.log.ev(f"shuttle off the side at arc {ax:.1f} lat {w:+.2f} — "
                            f"straightening up (strike {self.lat_strikes})")
                if self.lat_strikes >= 6:
                    self.log.ev("shuttle cannot hold the centreline — abandoning it")
                    return False
                # STRIKE = THE FRAME IS LYING (fv34: strike 3 at est arc +15.9 while
                # truly at +11.7 — a 4.2-arc drift steering the shuttle at a
                # centreline that is not there, straight off the deck for -10 x 2).
                # Straightening against the same bad frame repeats the walk-off, so
                # from the second strike STOP and stand for a landmark re-fix before
                # walking on: fixes are what shrink since_fix, not reflexes.
                if (self.nav.honest and self.lat_strikes >= 2
                        and os.environ.get("MISSION_SHUTTLE_REFIX", "1") == "1"
                        and self.nav.since_fix > 400):
                    self.log.ev(f"  strike {self.lat_strikes} with since_fix "
                                f"{self.nav.since_fix} — standing for a re-fix "
                                f"before continuing the shuttle")
                    zero3 = torch.zeros(1, 3, device=self.device)
                    for _k in range(240):
                        self._write_arm(self.travel_q, open_=False)
                        self._step(self.legs.step(zero3))
                        if self.nav.since_fix < 50:
                            break
                # Same perpendicular reflex as the traverse: turn IN PLACE until facing
                # the centreline, then walk STRAIGHT in until actually inside. The old
                # walk_cmd steering here is what lost u/r/1: it advances while the
                # heading still points outboard, so each "recovery" ended further out
                # (u: banked 100 points, then drifted -19/-22.7 and lost the finish).
                _th = self._tan_yaw(ax)
                _want = _th + (np.pi / 2 if w < 0 else -np.pi / 2)
                for _ in range(300):
                    _, _, _, _yaw = self.base_pose()
                    _e = (_want - _yaw + np.pi) % (2 * np.pi) - np.pi
                    if abs(_e) < 0.4:
                        break
                    c = torch.zeros(1, 3, device=self.device)
                    c[0, 2] = float(np.clip(1.5 * _e, -0.7, 0.7))
                    self._write_arm(self.travel_q, open_=False)
                    self._step(self.legs.step(c))
                for _ in range(900):
                    _, _, _, _yaw = self.base_pose()
                    _, _w2 = self.nav.arc_lat()
                    if abs(_w2) < 1.2:
                        break
                    _e = (_want - _yaw + np.pi) % (2 * np.pi) - np.pi
                    c = torch.zeros(1, 3, device=self.device)
                    c[0, 0] = 0.4
                    c[0, 2] = float(np.clip(1.2 * _e, -0.5, 0.5))
                    self._write_arm(self.travel_q, open_=False)
                    self._step(self.legs.step(c))
                self.face_forward()
                continue
            # Reaching the target means passing it, not landing within 45 cm of it.
            # Overshooting a backward shuttle walked the robot 2.5 m past the course
            # start, onto the ground below the platform, which it cannot climb back onto.
            if abs(ax - target_arc) < 0.45 or \
                    (dir_sign_hint < 0 and ax <= target_arc) or \
                    (dir_sign_hint > 0 and ax >= target_arc):
                return True
            if ax < self.nav.s_start() + 0.3 and target_arc > ax:
                dir_sign = 1.0            # already at the back edge: only ever forward
            else:
                dir_sign = 1.0 if target_arc > ax else -1.0
            # Not moving? Re-aim in place instead of grinding. A shuttle logged
            # "arc=8.4 -> -19.5" unchanged for thousands of steps because the robot was
            # pointed across the course and walk_cmd could not turn it while driving.
            # Wrong-way guard: if the gap to the target has GROWN by more than 2 m since
            # the shuttle started, the robot is being driven away from where it is meant
            # to go and no amount of further walking fixes that — turn in place instead.
            if abs(ax - target_arc) > abs(ax0 - target_arc) + 2.0:
                wrong_way += 1
                self.log.ev(f"  shuttle going the wrong way: arc {ax:.1f}, target "
                            f"{target_arc:.1f}, started at {ax0:.1f} — turning in place "
                            f"({wrong_way}/3)")
                if wrong_way >= 3:
                    self.log.ev("  shuttle abandoned: cannot get pointed the right way")
                    return False
                self.face_forward(steps=250, dir_sign=(1.0 if target_arc > ax else -1.0))
                continue
            if self.step_i - stall_at >= 600:
                if abs(ax - stall_ax) < 0.4:
                    stuck += 1
                    # Re-aiming only helps if the robot can still walk. Once it is off
                    # the platform it cannot, and goto_arc used to keep trying until its
                    # whole budget expired — measured on the oracle run: pinned at arc
                    # 22.0 (past the +21 end of the course), it printed "shuttle stuck"
                    # and re-aimed a dozen times against an 11,600-step budget, 13% of the
                    # venue allowance, with the arc value never moving off 22.0. Bound it:
                    # four fruitless re-aims means walk back is not happening, and the
                    # caller can at least deliver what it is holding.
                    if stuck >= 4:
                        self.log.ev(f"  shuttle abandoned: arc {ax:.1f} has not moved "
                                    f"after {stuck} re-aims")
                        return False
                    self.log.ev(f"  shuttle stuck at arc {ax:.1f} — re-aiming "
                                f"({stuck}/4)")
                    self.face_forward(steps=250, dir_sign=dir_sign)
                else:
                    stuck = 0
                stall_ax, stall_at = ax, self.step_i
            if self.step_i % (self.detect_every_eff * 2) == 0:
                self.scan()                 # populates self.seen; result unused here
            cmd, _, _ = self.walk_cmd(lat_target, dir_sign=dir_sign, vx=0.6)
            self._write_arm(self.travel_q, open_=False)
            self._step(self.legs.step(cmd))
            if (self.step_i - t0) % 400 == 0:
                self.log.ev(f"  shuttle: arc={ax:.1f} -> {target_arc:.1f}{self.payload_state()}")
        self.log.ev(f"shuttle to arc {target_arc:.1f} timed out")
        return False

    def payload_state(self) -> str:
        """Where every carried object is, in the base frame.

        Raising the basket walls 0.12 -> 0.15 m did NOT stop objects escaping (a rind
        deposited at base-frame [-0.054, -0.271, 0.131] was 0.49 m away by the time the
        robot reached the bin), so the escape has to be watched rather than guessed at:
        this prints the payload's position as the robot walks.
        """
        if not self.carried:
            return ""
        FLOOR, TOP = 0.098, 0.248          # basket floor and wall top, base frame
        out = []
        for name, _cls in self.carried:
            b = self.to_base_frame_true(self._sim_obj(name).data.root_pos_w[0].cpu().numpy())
            if self.p.carry_in_gripper:
                # held, not stowed: it should be near the TCP, well above the ground
                tcp = self.to_base_frame_true(self.tcp_w()[0].cpu().numpy())
                held = float(np.linalg.norm(b - tcp)) < 0.20
                tag = "" if held else " DROPPED"
            else:
                inside = (abs(b[0] + 0.07) < 0.15 and abs(b[1] + 0.27) < 0.11
                          and FLOOR - 0.03 < b[2] < TOP + 0.10)
                tag = "" if inside else " OUT"
            out.append(f"{name}{np.round(b, 2).tolist()}{tag}")
        return "  payload: " + " ".join(out)

    def deliver_payload(self) -> None:
        """Take everything in the basket to its colour-matched bin, then empty the list."""
        if not self.carried:
            return
        if self.off_platform:
            self.log.ev(f"off the platform with {[c for _, c in self.carried]} aboard — "
                        f"the bins are unreachable (left in the report as carried)")
            return
        # IS IT ACTUALLY ABOARD? hs1: the can was ejected by a tip-over (payload_state
        # printed OUT at the departure), the trip went ahead with a phantom, and
        # "lost in transit" was only declared after six basket-grip tries at the bin.
        # A basket item that is nowhere near the basket is not carried: forget it here
        # (it stays detectable on the ground, and its pick spot is un-blacklisted).
        self._verify_payload("departure")
        if not self.carried:
            self.log.ev("nothing aboard after the payload check — no trip")
            self._hand_item = None
            self.travel_q = self.travel_q_empty.clone()
            return
        self.log.ev(f"delivering {[c for _, c in self.carried]} to the bins"
                    f"{self.payload_state()}")
        # settle_payload() — a deliberate accelerate-then-brake to slide the load forward —
        # is NOT called. It was written for a problem that does not exist: every deposit
        # measured so far lands inside the arm's in-basket reach (base-frame x -0.03..-0.08,
        # y -0.20..-0.30). What actually goes wrong is the load LEAVING the basket in
        # transit, and hard braking with an open-top basket is the best way to help it.
        # LAST IN, FIRST OUT: the most recently deposited object sits on top and nearest
        # the arm's reachable centre.
        by_colour = {}
        for name, cls in reversed(self.carried):
            by_colour.setdefault(BIN_OF[cls], []).append(name)
        # The gripper item blocks the arm: its bin comes first, and it leads that list.
        _order = sorted(by_colour.items(),
                        key=lambda kv: 0 if self._hand_item in kv[1] else 1)
        for colour, names in _order:
            if self._hand_item in names:
                names.remove(self._hand_item)
                names.insert(0, self._hand_item)
            if not self.goto_bin(colour):
                _lost = getattr(self, "_carry_lost", None)
                if _lost in names:
                    self.log.ev(f"  {_lost} dropped mid-carry — written off, "
                                "resuming the sweep (it stays detectable)")
                    self.carried = [(n, c) for n, c in self.carried if n != _lost]
                    if self._hand_item == _lost:
                        self._hand_item = None
                    self._carry_lost = None
                    self.travel_q = self.travel_q_empty.clone()
                    continue
                self.log.ev(f"could not reach the {colour} bin")
                # THE GUIDE PAYS 25 % FOR THE APRON. An object placed in the general
                # placement area around the bins scores 5 (SCORE["area"] above, guide
                # 376) -- but "area" only ever existed as an OUTCOME of classify_landing,
                # never as an ACTION. Failing to reach the bin therefore fell through to
                # self.carried.clear() below and wrote the object out of the books while
                # it was still in the gripper: a guaranteed 0 where 5 was available.
                # MISSION_APRON_FALLBACK=1 puts it down where the robot is standing, which
                # is inside the placement zone whenever goto_bin got close and gave up.
                if os.environ.get("MISSION_APRON_FALLBACK") == "1":
                    for name in names:
                        if self.deliver_to_apron(name):
                            for a in self.log.attempts:
                                if a.name == name:
                                    # MUST be one of report()'s rank keys ("correct",
                                    # "wrong", "area") -- anything else KeyErrors at
                                    # runner.py:2659 and the run's report json, the only
                                    # structured record it produces, is never written.
                                    a.binned = "area"
                continue
            self.log.ev(f"at the {colour} bin, unloading {len(names)} object(s)")
            # MOST-ISOLATED FIRST, not LIFO: a grip attempt fails when a neighbour sits
            # within finger reach, and every successful removal widens the gaps for the
            # rest. The hand item (if any) stays first — it is already in the gripper.
            _aboard = [n for _ns in by_colour.values() for n in _ns]
            def _isolation(n):
                if self.obj_honest:
                    # the slots the objects were RELEASED over (arm FK bookkeeping) —
                    # the robot's only knowledge of where its cargo sits
                    pn = self._basket_slot_p.get(n)
                    if pn is None:
                        return 9.9
                    d = [float(np.linalg.norm(pn[:2] - self._basket_slot_p[m][:2]))
                         for m in _aboard if m != n and m in self._basket_slot_p]
                    return min(d) if d else 9.9
                pn = self._sim_obj(n).data.root_pos_w[0].cpu().numpy()[:2]
                d = [float(np.linalg.norm(
                        pn - self._sim_obj(m).data.root_pos_w[0].cpu().numpy()[:2]))
                     for m in _aboard if m != n]
                return min(d) if d else 9.9
            # Two objects of one colour are each other's only neighbour, so isolation
            # ties and LIFO decided: svC took the OUTER can (base y -0.32) first while
            # the inner one (y -0.24) lay under the arm's entry path — six tries, servo
            # residual 0.04-0.14 m every time (the fingers descending onto the
            # neighbour), 3,800 steps, and the can stayed in the basket. The arm
            # enters the basket from the body side, so on a tie take the INNERMOST
            # (largest base-frame y) first and clear the path to the outer one.
            def _order_key(n):
                if self.obj_honest:
                    _sp = self._basket_slot_p.get(n)
                    yb = float(_sp[1]) if _sp is not None else -9.9
                else:
                    yb = float(self.to_base_frame_true(
                        self._sim_obj(n).data.root_pos_w[0].cpu().numpy())[1])
                return (round(_isolation(n), 2), yb)
            while names:
                if names[0] == self._hand_item:
                    name = names.pop(0)
                else:
                    name = max(names, key=_order_key)
                    names.remove(name)
                _aboard.remove(name)
                self._release_believed = False
                res = self.unload_one(name, colour)
                cls_of = self.obj_cls.get(name)
                # The quota is spent on what the robot BELIEVES it put in the bin (the
                # jaw opened inside the SIGHTED opening with the load) whenever it is
                # not entitled to the surveyed answer — under object-honesty, and under
                # map-free navigation too: classify_landing is built from the surveyed
                # bin row on the surveyed curve (ct.SECTION_CENTERS["bins"], curve_map,
                # curve_theta, BIN_TOP), so letting it spend the quota would let the
                # course map decide which class the robot hunts for the rest of the run.
                # `res` is the judge's answer and stays in the report.
                _credit = (self._release_believed
                           if (self.obj_honest or self.nav.honest)
                           else res == "correct")
                if _credit and cls_of:
                    self.binned_ok_per_cls[cls_of] = \
                        self.binned_ok_per_cls.get(cls_of, 0) + 1
                for a in self.log.attempts:
                    if a.name == name:
                        a.binned = res
                self.log.ev(f"unload {name} -> {res}")
                self.done_names.add(name)         # never a target again, any result
                self.seen.pop(name, None)         # (the rescue scan may have re-added it)
        # whatever the trip did with them, they are off the books now
        for name, _c in self.carried:
            self.done_names.add(name)
            self.seen.pop(name, None)
        self.carried.clear()
        self._hand_item = None
        self.travel_q = self.travel_q_empty.clone()   # hand is empty again either way

    def _verify_payload(self, when: str) -> None:
        """Drop from `carried` every BASKET item that is not in (or right by) the
        basket. Hand items are the carry-watch's business (their own gap test)."""
        if not self.carried:
            return
        if self.obj_honest:
            # Nothing looks into the basket, so an item is aboard until the RETRIEVAL
            # says otherwise (pick_from_basket returns False after its tries, and the
            # caller writes it off there). The truth line is kept for the record.
            _tl = []
            for name, cls in self.carried:
                if name == self._hand_item or self.p.carry_in_gripper:
                    continue
                _b = self.to_base_frame_true(
                    self._sim_obj(name).data.root_pos_w[0].cpu().numpy())
                if float(np.hypot(_b[0] + 0.07, _b[1] + 0.27)) > 0.32 \
                        or _b[2] < 0.0 or _b[2] > 0.45:
                    _tl.append(f"{name} {np.round(_b, 2).tolist()}")
            if _tl:
                self.log.ev(f"  payload check ({when}) [TRUTH] not in the basket: "
                            + ", ".join(_tl) + " — kept aboard (unobservable)")
            return
        keep = []
        for name, cls in self.carried:
            if name == self._hand_item or self.p.carry_in_gripper:
                keep.append((name, cls))
                continue
            b = self.to_base_frame_true(self._sim_obj(name).data.root_pos_w[0].cpu().numpy())
            # basket box (payload_state) with a generous margin: only a clear miss
            # counts — an item wedged on the wall top is still coming along
            gone = (float(np.hypot(b[0] + 0.07, b[1] + 0.27)) > 0.32 or b[2] < 0.0
                    or b[2] > 0.45)
            if not gone:
                keep.append((name, cls))
                continue
            self.log.ev(f"  payload check ({when}): {name} ({cls}) is not in the basket "
                        f"(base-frame {np.round(b, 2).tolist()}) — not carrying it; "
                        "it stays detectable on the ground")
            self.picked_per_cls[cls] = max(0, self.picked_per_cls.get(cls, 0) - 1)
            spot = self._pick_spot.pop(name, None)
            if spot is not None:
                self.blacklist = [q for q in self.blacklist
                                  if float(np.hypot(*(np.asarray(q, float)[:2] - spot))) > 1e-6]
            for a in self.log.attempts:
                if a.name == name and a.binned is None:
                    a.binned = "none"
        self.carried = keep

    def deliver_to_apron(self, name: str) -> bool:
        """Salvage a carried object for the 25 % score when the bin cannot be reached.

        "Release where we stand" was the first version and it was wrong twice over: the
        placement zone that pays 5 points is only +-0.46 m along x +-1.29 m across around
        the bin row (terrain.py:339-341), and guide 135 voids anything done while out of
        bounds -- exactly where a failed bin approach tends to leave the robot. So walk to
        the zone first, bounded, and release only once actually inside it.
        """
        cx = self._bins_arc()
        _, w0 = self.nav.arc_lat()
        # aim just short of the row centre, on whichever side we are already on
        lat_aim = float(np.clip(w0, -1.0, 1.0))
        self.log.ev(f"apron fallback: walking {name} to the placement zone "
                    f"(arc {cx - 0.2:.1f} lat {lat_aim:+.2f})")
        self.goto_arc(cx - 0.2, lat_aim)
        ax, w = self.nav.arc_lat()
        if not (abs(ax - cx) < ct.PLACEMENT_HALF_ALONG + 0.25
                and abs(w) < ct.PLACEMENT_HALF_CROSS):
            # Releasing outside the zone scores 0 and (out of bounds) may not count at
            # all -- and the object is being written off either way, so do not spend
            # 600 more steps putting it down ceremonially.
            self.log.ev(f"apron fallback abandoned: at arc {ax:.1f} lat {w:+.2f}, "
                        f"outside the placement zone")
            return False
        if self.release_here(name):
            self.log.ev(f"apron fallback: released {name} in the placement zone "
                        f"(arc {ax:.2f} lat {w:+.2f})")
            return True
        return False

    def release_here(self, name: str) -> bool:
        """Put the carried object down on the ground in front of the robot and open.

        Deliberately dumb: no bin geometry, no servo, no verification -- the situation it
        exists for is "the bin could not be reached and the alternative is scoring zero".
        Uses the same lower-then-open sequence as the bin release so a shallow lid pinch
        is not peeled out by the motion.
        """
        if not (self.p.carry_in_gripper or name == self._hand_item):
            return False
        self._hand_item = None
        try:
            self.legs.set_mode("crouch", blend_steps=60)
            for _ in range(120):
                self._step(self.legs.step(self.crouch_cmd(1.0)))
            q = self.arm.current_q6().clone()
            q[1] += 0.35                       # shoulder down, lowers the TCP
            self.play_traj([q.tolist()], grip_open=False, steps_each=160)
            for _ in range(70):
                self._write_arm(open_=True)
                self._step(self.legs.step(self.crouch_cmd(1.0)))
            self.play_traj([self.travel_q_empty.tolist()], grip_open=True, steps_each=160)
            self.stand_up()
            return True
        except Exception as e:                 # never let a fallback end the run
            self.log.ev(f"apron fallback failed for {name}: {e}")
            return False

    def unload_one(self, name: str, colour: str) -> str:
        """Reach into the basket, lift the object out and drop it over the bin."""
        if name != self._hand_item and name not in [n for n, _ in self.carried]:
            # written off mid-trip (carry-watch loss) — the trip's unload plan was
            # built before the write-off; don't spend a basket-pick sequence on it.
            # ...UNLESS it can only have gone into the basket. The carry-watch fires on
            # an empty gripper, and the hand-carry pose hangs directly over the basket
            # opening, so "the fingers emptied" and "it fell in the basket" are the same
            # event most of the time (hs16b trash_7, hs22v5 trash_5 — 20 points each,
            # written off at the bin with the object riding along). One look costs ~800
            # steps and the robot is already standing at the bin.
            if int(os.environ.get("MISSION_HAND_TO_BASKET", "1")) \
                    and self.retrieve is not None \
                    and name in getattr(self, "_lost_in_hand", set()):
                self.log.ev(f"  {name} was written off mid-trip — looking in the "
                            f"basket, which the carry pose hangs over")
                r = self.retrieve
                self.play_traj([r["q_mid"], r["q_pick"]], grip_open=True,
                               steps_each=200)
                if self.pick_from_basket(name, retries=1) and self._sensed_load()[0]:
                    self.log.ev(f"  {name}: FOUND IT in the basket — delivering after all")
                    self._carry_lost = None
                    self._lost_in_hand.discard(name)
                    return self.release_over_bin(name, colour)
                self.log.ev(f"  {name}: not in the basket either")
            else:
                self.log.ev(f"  {name} was written off mid-trip — skipping its unload")
            return "none"
        if self.retrieve is None:
            self.log.ev("no retrieve_traj.json — leaving the object in the basket")
            return "area"
        r = self.retrieve
        bx, blat = bin_targets()[colour]
        BX, BY = ct.curve_map(bx, blat)
        self.log.ev(f"unloading {name} into the {colour} bin at world "
                    f"({float(BX):.2f}, {float(BY):.2f})")
        self.trace(name, "before")
        # coarse: get the hand into the basket; fine: servo onto where the object actually
        # settled (the fixed waypoint misses by up to 8 cm)
        if self.p.carry_in_gripper or name == self._hand_item:
            self._hand_item = None
            return self.release_over_bin(name, colour)   # already in hand
        self.play_traj([r["q_mid"], r["q_pick"]], grip_open=True, steps_each=200)
        if not self.pick_from_basket(name, retries=self.p.basket_retries):
            self.log.ev(f"  could not get a grip on {name} after "
                        f"{self.p.basket_retries + 1} tries — leaving it in the basket")
            self.play_traj([r["q_mid"], self.travel_q.tolist()], grip_open=False,
                           steps_each=160)
            # RESCUE: the failed tries usually EJECT the object onto the pad next to
            # the bin (two missions in a row: the rind found at z ~0.01 m, 0.7-1.0 m
            # from the bin, scored none). That is a plain ground object now — walk
            # onto it, re-grasp and finish the delivery. MISSION_BIN_RESCUE=0 disables.
            # Already landed? Two rinds stacked in the basket went into the green bin
            # TOGETHER with the first pick (ctlA @21746: trash_4 at z 0.05 inside the
            # bin), and the "lost in transit" path then spent 1,300 steps crouching and
            # grasping at a bin floor before the landing check called it correct anyway.
            _pre = self.classify_landing(name, colour)
            # (the same surveyed verdict, deciding here whether to spend ~3,500 steps
            # walking onto an ejected object — the honest branch below asks the camera)
            if _pre == "correct" and not (self.obj_honest or self.nav.honest):
                self.log.ev(f"  {name} is already inside the {colour} bin — no rescue needed")
                return _pre
            if self.nav.honest or self.obj_honest:
                # the robot only knows what the camera shows: look for the same class
                # on the pad around it (an ejected object lands 0.7-1.0 m from the bin)
                _ob, _bd = None, 9.9
                _X, _Y, _, _ = self.base_pose()
                self._allow_pad = True            # the pad is exactly where to look
                try:
                    _dets = self.scan()
                finally:
                    self._allow_pad = False
                for _d in _dets:
                    if _d["cls"] != self.obj_cls[name]:
                        continue
                    _dd = float(np.hypot(_d["pos_w"][0] - _X, _d["pos_w"][1] - _Y))
                    if _dd < _bd:
                        _ob, _bd = np.asarray(_d["pos_w"], dtype=float), _dd
                if _ob is None or _bd > 1.8:
                    self.log.ev(f"  {name} is not visible on the pad — no rescue")
                    return self.classify_landing(name, colour)
                _ax, _aw = float(_ob[0]), float(_ob[1])
            else:
                _ob = self._sim_obj(name).data.root_pos_w[0].cpu().numpy()
                _ax, _aw = arc_of(float(_ob[0]), float(_ob[1]))
            if (float(_ob[2]) < 0.25 and abs(_aw) < 1.6
                    and abs(_ax - self._bins_arc()) < 3.0
                    and self.steps_left() > 3500
                    and os.environ.get("MISSION_BIN_RESCUE", "1") != "0"):
                self.log.ev(f"  {name} was ejected onto the pad — ground re-grasp rescue")
                det2 = {"cls": self.obj_cls[name], "pos_w": _ob, "section": "bins"}
                ok2, _t = self.approach(det2, reanchor=False)
                if ok2 and self.crouch_and_check("bins"):
                    got2 = self.attempt_grasp(name, det2)
                    self.stand_up()
                    if got2:
                        return self.release_over_bin(name, colour)
                else:
                    self.stand_up()
            return self.classify_landing(name, colour)
        self.trace(name, "gripped")
        return self.release_over_bin(name, colour)

    def carry_pose(self, cls: str):
        """Joint pose the arm holds while walking with `cls` in the gripper."""
        return self.carry_q_cls.get(cls, self.carry_q)

    def release_over_bin(self, name: str, colour: str) -> str:
        # whatever happens below, the hand is empty afterwards
        """Lower the held object into the bin and let go.

        Shared by both delivery modes: the basket route reaches this with the
        object just lifted out of the basket, the carry-in-gripper route with the
        object held since the grasp.
        """
        r = self.retrieve
        # the "did it ride back up on the fingers" verdict belongs to THIS release; a
        # bag (which is never tested) would otherwise inherit the previous object's
        self._last_stuck = False
        # DON'T SWING AN EMPTY HAND. ~1,500 steps of carry pose -> mid -> drop -> lower
        # -> open -> withdraw are spent whatever is (or is not) in the gripper, and the
        # release then credits a delivery that never happened (hs11a trash_8: the box
        # had slipped into the basket, the arm swung over the bin empty and the class
        # quota was spent on it). The fingers know.
        _ld0, _gw0, _ge0 = self._sensed_load()
        if not _ld0 and int(os.environ.get("MISSION_RELEASE_LOAD_CHECK", "1")):
            # ...but an empty hand does not mean the object is gone: the hang carry
            # pose sits DIRECTLY OVER THE BASKET, so a slip on the last few metres
            # drops it in. hs16b trash_7 emptied the hand somewhere in the bin
            # approach and was scored `none` while lying in the basket 0.7 m from the
            # blue bin — and, worse, it landed on top of trash_1, whose own basket
            # pick then stalled five times at 64-121 mm ("outside the arm's in-basket
            # envelope") for a second `none`. 40 points, both recoverable by looking
            # in the obvious place. Two tries, ~800 steps, and the fetch also clears
            # the intruder for whatever is underneath it.
            _got = False
            if int(os.environ.get("MISSION_HAND_TO_BASKET", "1")) and r is not None:
                self.log.ev(f"  release {name}: hand empty (width {_gw0*1000:.0f} mm) "
                            f"— looking in the basket, which the carry pose hangs over")
                self.play_traj([r["q_mid"], r["q_pick"]], grip_open=True,
                               steps_each=200)
                _got = self.pick_from_basket(name, retries=1)
                if _got:
                    _ld0, _gw0, _ge0 = self._sensed_load()
                    _got = bool(_ld0)
                self.log.ev(f"  release {name}: basket rescue "
                            f"{'FOUND IT — carrying on with the drop' if _got else 'came up empty'}")
            if not _got:
                self._release_believed = False
                self.log.ev(f"  release {name}: the hand is EMPTY (width {_gw0*1000:.0f} mm, "
                            f"effort {-1.0 if _ge0 is None else _ge0:.1f}) — skipping the "
                            f"swing over the {colour} bin")
                self.travel_q = self.travel_q_empty.clone()
                return self.classify_landing(name, colour)
        bx, blat = bin_targets()[colour]
        BX, BY = ct.curve_map(bx, blat)
        # LOOK AGAIN IF THE SIGHTING HAS DIED — here, not earlier. goto_bin is called
        # once per COLOUR and the basket pick that follows it burns 1,000-2,000 steps,
        # so a fix that was alive when the unload began is dead by the time the aim
        # matters (hs14b: the check passed in unload_one, and 1,700 steps later the
        # drop logged "stale (1000000000)" and landed the can on the rim). This is the
        # last moment before the swing, with the object already in the gripper.
        if self.nav.honest and int(os.environ.get("MISSION_BIN_RESIGHT", "1")) \
                and (self.nav.bin_rel_body(colour) is None
                     or self.nav.bin_obs_age(colour) > int(
                         os.environ.get("MISSION_RECENTRE_MAX_AGE", "2500"))):
            # age test added (0908): fv14 dropped 1 m off believing IN — goto_bin's
            # fix was fresh on arrival, the basket-pick sequence burned 1,500+ steps,
            # and this pre-swing check only asked "is there a sighting at all". Test
            # the SAME age bound the release gate uses, so no drop is ever attempted
            # on a fix the release itself would refuse to check against.
            _bp = self.nav.bin_pos(colour)
            if _bp is not None:
                _moved = self._resight_bin(colour, float(_bp[0]), float(_bp[1]))
                # WALK BACK WHETHER OR NOT THE LOOK WORKED. The back-out moves the
                # robot 2-3 m off the bin; returning only on success left hs15b
                # standing 3 m away and releasing there (the refusal line asked for
                # +3,051 mm and the object scored `none`).
                if not self.goto_bin(colour):
                    self.log.ev(f"  could not walk back to the {colour} bin after the "
                                f"re-sight — dropping from where we stand")
                elif _moved:
                    self.log.ev(f"  back at the {colour} bin with a live sighting")
        _cq = self.carry_pose(self.obj_cls[name])
        _pre = [_cq] if (_cq is not None and _cq != r["q_mid"]) else []
        self.play_traj(_pre + [r["q_mid"], r["q_drop"]], grip_open=False, steps_each=220)
        _, tcp = self.trace(name, "at-drop")
        self.log.ev(f"  TCP vs {colour} bin centre: dx={float(tcp[0] - BX):+.2f} "
                    f"dy={float(tcp[1] - BY):+.2f} (opening is +-0.20 x +-0.13 m)")
        # Descend to just above the 0.55 m rim before letting go. Releasing from the
        # full drop height bounced a rind off the rim and out of the bin even though the
        # TCP was inside the opening.
        p_now, _ = self.arm.frame_pose()
        # centre the OBJECT (not the TCP) over the opening for every class: the item
        # can hang up to ~0.10 m off the TCP inside the jaw (r20: can at dy +0.21 with
        # the TCP at +0.11 -> rim bounce -> "none")
        _binc_b0 = self.nav.bin_base(colour)
        if self.obj_honest:
            # Where the load hangs inside the jaw is not observable, so centre what IS:
            # the TCP over the sighted bin centre. (The old shift chased the object's
            # own offset, which only the simulator knows.)
            _obj_b0 = p_now[0].cpu().numpy()
        else:
            _obj_b0 = self.to_base_frame_true(
                self._sim_obj(name).data.root_pos_w[0].cpu().numpy())
        _want0 = (_binc_b0[:2] - _obj_b0[:2]) if _binc_b0 is not None else np.zeros(2)
        _sh0 = np.clip(_want0, -0.12, 0.12)
        # REFUSE A SATURATED CORRECTION. The approach has already parked the body so the
        # drop pose sits over the bin the robot believes in; a re-centre asking for more
        # than MISSION_RECENTRE_MAX on top of that is not a refinement, it is the two
        # beliefs disagreeing — and when they disagree it is the bin memory that is
        # stale, not the pose the robot walked to. Measured (hs4b/c/d, second drop of a
        # stop): every saturated shift turned a dead-centre TCP into `area`, while every
        # shift under 90 mm landed the object in the bin.
        # 0.09 was calibrated when "fresh" only meant "age < 2500 steps", which the
        # odometry-distance gate could still fail — hs9b's 1,093-step-old sighting had
        # already been dropped and its "fresh" correction was really the drifted
        # S-frame memory. That hole is now closed below (bin_rel_body is None -> stale),
        # so a correction that reaches this branch is a LIVE body-relative fix, and
        # refusing it costs real points: hs16b's last drop asked +115 mm, was refused
        # for being 25 mm over the bar, and the rind landed 150 mm from the green bin
        # for `area` instead of `correct`. The detector's own bin fix is good to
        # ~50-60 mm, so trust it out to 140 mm and keep the refusal for genuine
        # disagreements beyond that.
        _rmax = float(os.environ.get("MISSION_RECENTRE_MAX", "0.14"))
        _rage = int(os.environ.get("MISSION_RECENTRE_MAX_AGE", "2500"))
        _age = self.nav.bin_obs_age(colour) if self.nav.honest else 0
        # AGE IS NOT THE WHOLE TEST. bin_base falls back to the S-frame memory — which
        # carries the pose estimate's lateral error — whenever the body-relative
        # sighting is rejected for ANY reason (age or odometry distance). Ask the
        # question that decides which of the two bin_base actually returned: hs9b's
        # sighting was 1,093 steps old, passed the age gate, and had still been
        # dropped by the distance gate, so the "fresh" re-centre it authorised was
        # built on the drifted memory and pushed the drop 0.26 m out.
        if self.nav.honest and self.nav.bin_rel_body(colour) is None:
            _age = 10 ** 9
        # Only a SIGHTED bin can be stale. Under the oracle `bin_base` is the surveyed
        # position and the correction it asks for is exactly right however large it is,
        # so the refusal must not touch that path.
        # ...but TRIM the stale correction rather than throwing it away. A refusal is
        # the right answer to a belief that is grossly wrong (hs9b asked +120 mm on a
        # dead-centre TCP and scored `area`), and the wrong answer to one that is
        # merely uncertain: hs12a's second drop arrived at dy +0.15 — already outside
        # the +-0.13 opening — and the refused correction was +89 mm, which would have
        # brought it to +0.06 and scored. Both cases are covered by keeping the
        # DIRECTION and bounding the MAGNITUDE: capped at 60 mm, a wrong belief can
        # move a centred load no further than half the opening, and a right one still
        # recovers most of an approach error.
        _stale0 = self.nav.honest and _age > _rage
        if self.nav.honest and float(np.max(np.abs(_want0))) > _rmax and not _stale0:
            self.log.ev(f"  load re-centre REFUSED: the bin memory asks for "
                        f"[{_want0[0]*1000:+.0f}, {_want0[1]*1000:+.0f}] mm against a "
                        f"FRESH sighting — the two beliefs disagree by more than either "
                        f"is worth; keeping the approach's aim")
            _sh0 = np.zeros(2)
        elif _stale0:
            # A stale correction is a COIN FLIP, not a weak signal. Trimming it to 60 mm
            # and applying it was tried: hs12a's +89 mm would have rescued a drop that
            # arrived at dy +0.15, and hs13a's +44 mm pushed a drop that arrived at
            # dy +0.09 out to the rim (`area`). Same magnitude, opposite value — the
            # S-frame fallback carries the estimate's lateral error, whose SIGN is not
            # knowable at the drop. Expected value zero, variance positive: don't take
            # the bet. The answer to staleness is to go and look again (_resight_bin).
            self.log.ev(f"  load re-centre REFUSED: the bin sighting is stale "
                        f"({_age} steps since it was SEEN) and the S-frame fallback "
                        f"asks for [{_want0[0]*1000:+.0f}, {_want0[1]*1000:+.0f}] mm — "
                        f"keeping the approach's aim")
            _sh0 = np.zeros(2)
        elif float(np.hypot(*_sh0)) > 0.03:
            self.log.ev(f"  bin sighting is {_age} steps old — re-centring by "
                        f"[{_sh0[0]*1000:+.0f}, {_sh0[1]*1000:+.0f}] mm")
        if float(np.hypot(*_sh0)) > 0.03:
            _tg0 = p_now[0].cpu().numpy()
            _tg0[0] += _sh0[0]
            _tg0[1] += _sh0[1]
            self.servo_tcp_to(_tg0, steps=70, grip_open=False)
            self.log.ev(f"  load re-centred by [{_sh0[0]*1000:+.0f}, "
                        f"{_sh0[1]*1000:+.0f}] mm over the opening")
            p_now, _ = self.arm.frame_pose()
        if self.obj_cls.get(name) == "redbag":
            # the bag hangs ~0.35 m below the pinched bar: the standard 0.28 m lower
            # released it AT rim height (z 0.49 vs inside gate 0.42) and it settled
            # ACROSS the 0.26 m opening -> scored "none" twice. Feed it down until the
            # bag's own centre is below the rim, then let go inside the mouth.
            # centre the BAG, not the TCP: the bag hangs up to 0.10 m sideways from
            # the pinched bar (cage G video: TCP dead-centre, bag over the near rim,
            # fed down the OUTSIDE face -> "area"). Shift the hand so the bag's own
            # centre sits over the opening before feeding down.
            _binc_b = self.nav.bin_base(colour)
            _obj_b = self.to_base_frame_true(
                self._sim_obj(name).data.root_pos_w[0].cpu().numpy())
            _shift = (np.clip(_binc_b[:2] - _obj_b[:2], -0.15, 0.15)
                      if _binc_b is not None else np.zeros(2))
            if float(np.hypot(*_shift)) > 0.03:
                p_now, _ = self.arm.frame_pose()
                _tgt = p_now[0].cpu().numpy()
                _tgt[0] += _shift[0]
                _tgt[1] += _shift[1]
                self.servo_tcp_to(_tgt, steps=80, grip_open=False)
                self.log.ev(f"  bag re-centred by [{_shift[0]*1000:+.0f}, "
                            f"{_shift[1]*1000:+.0f}] mm over the opening")
            _bz_prev = 1e9
            _jig = 0
            for _ in range(8):
                p_now, _ = self.arm.frame_pose()
                self.servo_tcp_to(p_now[0].cpu().numpy() - np.array([0.0, 0.0, 0.11]),
                                  steps=60, grip_open=False)
                _bz = float(self._sim_obj(name).data.root_pos_w[0, 2].cpu())
                if _bz < 0.30:
                    break
                if _bz > _bz_prev - 0.02:
                    # stalled: below the rim = resting, stop; AT/above the rim = the
                    # bag caught the edge (r20: stalled z 0.49, toppled out on
                    # release) — jiggle toward the centre and keep feeding
                    if _bz < ct.BIN_HEIGHT - 0.03 or _jig >= 2:
                        break
                    _jig += 1
                    _ob = self.to_base_frame_true(
                        self._sim_obj(name).data.root_pos_w[0].cpu().numpy())
                    _bc = self.nav.bin_base(colour)
                    _dj = (np.clip(_bc[:2] - _ob[:2], -0.03, 0.03)
                           if _bc is not None else np.zeros(2))
                    p_j, _ = self.arm.frame_pose()
                    _tj = p_j[0].cpu().numpy()
                    _tj[0] += _dj[0]
                    _tj[1] += _dj[1]
                    self.servo_tcp_to(_tj, steps=40, grip_open=False)
                    self.log.ev(f"  bag caught the rim at z {_bz:.2f} — jiggling in")
                _bz_prev = _bz
            self.log.ev(f"  bag fed down to z {_bz:.2f} (rim {ct.BIN_HEIGHT:.2f})")
            for _ in range(50):                     # let the swing settle before opening
                self._write_arm(open_=False)
                self._step(self.legs.step(torch.zeros(1, 3, device=self.device)))
        else:
            # 0.28 -> 0.36: r18 landed two cans at dy +0.3 from the blue bin centre —
            # released near rim height they bounce off the rim and roll out ("area").
            # A deeper release cuts the free fall below bounce-out energy; the TCP
            # stays inside the 0.40 x 0.26 opening at the measured +-0.04 aim scatter.
            low = p_now[0].cpu().numpy() - np.array([0.0, 0.0, 0.36])
            self.servo_tcp_to(low, steps=150, grip_open=False)
        self.trace(name, "lowered")
        # WHAT THE ROBOT BELIEVES IT DID: the jaw opened with the TCP inside the
        # sighted bin's opening. That belief — not the object's landing, which only
        # the simulator (and at the venue, the judges) can see — is what tells the
        # sweep whether this class still needs a pick.
        _binc_r = self.nav.bin_base(colour)
        _p_rel_b = self.arm.frame_pose()[0][0].cpu().numpy()
        # ...and don't ask a memory we just refused to trust. When the sighting is
        # stale the drop keeps the aim the APPROACH built (which was made against a
        # fresh one), so the honest belief is that aim — testing it against the stale
        # memory would fail the very drops the refusal was there to protect.
        _stale_r = self.nav.honest and (
            self.nav.bin_rel_body(colour) is None
            or self.nav.bin_obs_age(colour) > int(
                os.environ.get("MISSION_RECENTRE_MAX_AGE", "2500")))
        self._release_believed = bool(
            _binc_r is None or _stale_r
            or (abs(_p_rel_b[0] - _binc_r[0]) < 0.20
                and abs(_p_rel_b[1] - _binc_r[1]) < 0.13))
        for _ in range(60):                                     # release over the bin
            self._write_arm(open_=True)
            self._step(self.legs.step(torch.zeros(1, 3, device=self.device)))
        for _ in range(120):
            self._write_arm(open_=True)
            self._step(self.legs.step(torch.zeros(1, 3, device=self.device)))
        self.trace(name, "released")
        # DID IT ACTUALLY LET GO? The lunchbox's lid lip hooks on an open finger: svP
        # trash_9 was lowered into the blue bin, the jaw opened for 180 steps, and the
        # box rode back UP with the hand (object at the TCP, z 0.75, 0.26 m from the
        # bin -> "none"), then sat on the fingers and blocked all six basket picks of
        # the can behind it. Shake it off while still inside the mouth: wrist-roll
        # flicks, a re-grip/re-open, then a sideways jiggle, checking the gap each time.
        # (the bag hangs 0.35 m under its pinched handle, so the gap test means nothing
        # for it — a bag caught on a finger is a different problem, not handled here)
        _chk = self.obj_cls.get(name) != "redbag"
        if _chk and self._release_stuck(name):
            self._shake_off(name, colour)
        # Retreat STRAIGHT UP out of the bin before folding the arm away. Playing the
        # travel trajectory directly from inside the bin swept the just-released object
        # sideways and dragged it back out (landed 0.35 m behind the bin, on the ground).
        p_rel, _ = self.arm.frame_pose()
        self.servo_tcp_to(p_rel[0].cpu().numpy() + np.array([0.0, 0.0, 0.32]),
                          steps=120, grip_open=True)
        self.trace(name, "withdrawn")
        # ...and if it came up with the hand anyway, go back down and try once more —
        # a box on the fingers scores nothing and wrecks every pick that follows.
        if _chk and self._release_stuck(name) and int(os.environ.get("MISSION_SHAKE_OFF", "1")):
            self.log.ev(f"  {name} rode up out of the bin on the open fingers — "
                        "lowering it back in for another shake")
            self.servo_tcp_to(p_rel[0].cpu().numpy(), steps=120, grip_open=True)
            self._shake_off(name, colour)
            p_rel, _ = self.arm.frame_pose()
            self.servo_tcp_to(p_rel[0].cpu().numpy() + np.array([0.0, 0.0, 0.32]),
                              steps=120, grip_open=True)
            # LAST RESORT, hand high: still hooked after two in-bin sessions means the
            # box is probably not over the opening at all (hs1 F3: lowered outside the
            # tapered wall). Tip it off wherever the hand is — a box on the pad scores
            # "area" at worst and, above all, leaves the hand free for the picks that
            # follow (F3 -> F4: the box on the fingers wrecked all six rind picks).
            if self._release_stuck(name) and float(os.environ.get("MISSION_SHAKE_TILT", "60")) > 0:
                _ph, _ = self.arm.frame_pose()
                _ph = _ph[0].cpu().numpy()
                for _el in (np.radians(75.0), np.radians(-75.0)):
                    _got = self._tilt_jaw(_ph, _el, steps=100)
                    for _ in range(30):
                        self._write_arm(open_=True)
                        self._step(self.legs.step(torch.zeros(1, 3, device=self.device)))
                    if not self._release_stuck(name):
                        self.log.ev(f"  {name} tipped off the fingers at the withdrawn "
                                    f"height (jaw {np.degrees(_got):+.0f} deg)")
                        break
                self._tilt_jaw(_ph, 0.0, steps=90)
                if self._release_stuck(name):
                    self.log.ev(f"  {name} is STILL on the fingers — the basket picks "
                                "that follow are at risk")
        self.travel_q = self.travel_q_empty.clone()
        self.play_traj([self.travel_q.tolist()], grip_open=True, steps_each=150)
        if self.obj_honest or self.nav.honest:
            # an object that rode back up on the open fingers was not delivered, however
            # the release looked (obj-honest knows this from its own test-close; with
            # only the map taken away, the gap test is still allowed)
            _stuck = (getattr(self, "_last_stuck", False) if self.obj_honest
                      else (self.obj_cls.get(name) != "redbag"
                            and self._release_stuck(name)))
            self._release_believed = bool(self._release_believed and not _stuck)
            _lp = self.classify_landing(name, colour)
            self.log.ev(f"  release {name} over the {colour} bin: believed "
                        f"{'IN' if self._release_believed else 'NOT in'} the opening "
                        f"(TCP {np.round(_p_rel_b[:2], 3).tolist()} vs bin "
                        f"{'—' if _binc_r is None else np.round(_binc_r[:2], 3).tolist()})"
                        f"  [TRUTH] {_lp}")
            return _lp
        return self.classify_landing(name, colour)

    def _release_stuck(self, name: str) -> bool:
        """Is the just-released object still hanging on the hand?

        A released object drops away: the TCP sits ~0.36 m under the rim and the bin
        floor is 0.5 m further down, so anything at rest reads a gap >= 0.25 m (svP:
        a can on the floor 0.38 m, a box on a can 0.25 m). Within 0.15 m it is riding
        the fingers (svP trash_9: gap 0.03 m, object AT the TCP).
        """
        if self.obj_honest:
            # TEST-CLOSE, the robot's own version of the question: shut the jaw for a
            # moment and see whether anything stops it. A released object is gone and
            # the fingers meet each other (width -> 0, no effort); one still between
            # them reads a real width against a working drive. Then re-open — the
            # caller is mid-release and the hand must stay open.
            zero = torch.zeros(1, 3, device=self.device)
            for _ in range(int(os.environ.get("MISSION_STUCK_CLOSE", "45"))):
                self._write_arm(open_=False)
                self._step(self.legs.step(zero))
            _ld, _gw, _ge = self._sensed_load()
            for _ in range(30):
                self._write_arm(open_=True)
                self._step(self.legs.step(zero))
            _ob = self.to_base_frame_true(self._sim_obj(name).data.root_pos_w[0].cpu().numpy())
            _tb = self.to_base_frame_true(self.tcp_w()[0].cpu().numpy())
            self.log.ev(f"  release check {name}: test-close width {_gw * 1000:.0f} mm, "
                        f"effort {-1.0 if _ge is None else _ge:.1f} -> "
                        f"{'STILL ON THE HAND' if _ld else 'clear'}  [TRUTH] gap "
                        f"{float(np.linalg.norm(_ob - _tb)):.3f} m")
            self._last_stuck = bool(_ld)
            return bool(_ld)
        _ob = self.to_base_frame_true(self._sim_obj(name).data.root_pos_w[0].cpu().numpy())
        _tb = self.to_base_frame_true(self.tcp_w()[0].cpu().numpy())
        return float(np.linalg.norm(_ob - _tb)) < 0.15

    def _tilt_jaw(self, hold_b, elev: float, steps: int = 90) -> float:
        """Roll the finger-opening axis to `elev` rad above horizontal (0 = level)
        while holding the TCP at base-frame `hold_b`, jaw open. Same relative-pose IK
        path as servo_tcp_pose, but the correction is the minimal rotation taking the
        current jaw direction onto the target one, so the yaw of the jaw is kept and
        only its elevation changes. Returns the elevation actually reached (rad)."""
        tgt = torch.tensor(np.asarray(hold_b, float), dtype=torch.float32, device=self.device)
        got = 0.0
        for _ in range(steps):
            p, q = self.arm.frame_pose()
            R = math_utils.matrix_from_quat(q)[0]
            jaw = R[:, 1] / R[:, 1].norm().clamp(min=1e-9)
            got = float(torch.asin(jaw[2].clamp(-1.0, 1.0)))
            psi = float(torch.atan2(jaw[1], jaw[0]))
            want = torch.tensor([np.cos(psi) * np.cos(elev), np.sin(psi) * np.cos(elev),
                                 np.sin(elev)], dtype=torch.float32, device=self.device)
            axis = torch.cross(jaw, want, dim=0)
            s = float(axis.norm())
            ang = float(np.arctan2(s, float((jaw * want).sum())))
            a = torch.zeros(1, 6, device=self.device)
            err = tgt - p[0]
            done_pos = float(err.norm()) < 0.012
            if not done_pos:
                a[0, :3] = (err / self.arm.scale).clamp(-1.0, 1.0)
            done_rot = abs(ang) < 0.05
            if not done_rot and s > 1e-6:
                a[0, 3:6] = (axis / s * min(ang, 0.06) / self.arm.scale).clamp(-1.0, 1.0)
            if done_pos and done_rot:
                break
            self.arm.ik_step(a)
            self.arm.set_gripper(True)
            self.arm.write_gripper()
            self._step(self.legs.step(torch.zeros(1, 3, device=self.device)))
        return got

    def _shake_off(self, name: str, colour: str) -> bool:
        """Free an object hooked on the open fingers, inside the bin mouth.

        Three escalating moves, each followed by a gap check: (1) wrist-roll flicks
        (joint 6 +-1.2 rad — the lid lip caught on one finger slides off when the jaw
        rotates under it), (2) re-grip and re-open (fresh finger contact), (3) a
        sideways jiggle of the TCP. Every move keeps the gripper OPEN and the TCP
        inside the opening (+-0.04 m) so a freed object still falls into the bin.
        MISSION_SHAKE_OFF=0 disables (stock behaviour: withdraw blind).
        """
        if not int(os.environ.get("MISSION_SHAKE_OFF", "1")):
            return False
        zero = torch.zeros(1, 3, device=self.device)
        self.log.ev(f"  {name} still on the open fingers after the release — shaking it off")
        q0 = self.arm.current_q6().clone()
        p0, _ = self.arm.frame_pose()
        p0 = p0[0].cpu().numpy()
        # (0) TIP THE JAW. hs1 F3 (video): the tool points straight DOWN at the drop
        # pose, so the joint-6 "flicks" below only spin the box about the vertical —
        # the 110 mm lid flange stays hooked on both fingertips of the 100 mm-open jaw
        # through two full rounds. Rolling the finger-opening axis 55-70 deg out of
        # horizontal puts one fingertip well below the other; the flange slides down
        # the tilted rails (needs tan(tilt) > mu) off the upper finger, and the box
        # then pivots off the lower one. Either sign, then level again.
        _tilt = float(os.environ.get("MISSION_SHAKE_TILT", "60"))
        if _tilt > 0.0:
            for _el in (np.radians(_tilt), -np.radians(_tilt)):
                _got = self._tilt_jaw(p0, _el, steps=90)
                for _ in range(30):                # let the box slide / swing
                    self._write_arm(open_=True)
                    self._step(self.legs.step(zero))
                if not self._release_stuck(name):
                    self.log.ev(f"  shake-off: {name} slid off on the jaw tilt "
                                f"({np.degrees(_el):+.0f} deg, reached {np.degrees(_got):+.0f})")
                    self._tilt_jaw(p0, 0.0, steps=90)
                    return True
                self.log.ev(f"  shake-off: jaw tilted {np.degrees(_got):+.0f} deg "
                            f"(asked {np.degrees(_el):+.0f}) — {name} still on the fingers")
            self._tilt_jaw(p0, 0.0, steps=90)
        for rnd in range(2):
            # (1) wrist-roll flicks
            for _d in (1.2, -1.2, 0.0):
                _q = q0.clone()
                _q[5] = float(q0[5]) + _d
                self.play_traj([_q.tolist()], grip_open=True, steps_each=45)
                if not self._release_stuck(name):
                    self.log.ev(f"  shake-off: {name} came free on a wrist flick ({_d:+.1f} rad)")
                    return True
            # (2) re-grip, re-open
            for _ in range(50):
                self._write_arm(open_=False)
                self._step(self.legs.step(zero))
            for _ in range(80):
                self._write_arm(open_=True)
                self._step(self.legs.step(zero))
            if not self._release_stuck(name):
                self.log.ev(f"  shake-off: {name} came free on the re-grip")
                return True
            # (3) sideways jiggle, back to the start point
            for _dx, _dy in ((0.04, 0.0), (-0.04, 0.0), (0.0, 0.04), (0.0, -0.04), (0.0, 0.0)):
                self.servo_tcp_to(p0 + np.array([_dx, _dy, 0.0]), steps=35, grip_open=True)
                if not self._release_stuck(name):
                    self.log.ev(f"  shake-off: {name} came free on the jiggle")
                    return True
            self.log.ev(f"  shake-off round {rnd + 1}: {name} still on the fingers")
        return False

    def classify_landing(self, name: str, colour: str) -> str:
        """Where did the object end up? -> 'correct' | 'wrong' | 'area' | 'none'."""
        p = self._sim_obj(name).data.root_pos_w[0].cpu().numpy()
        # Test the BIN's real footprint, not a circle: the opening is 0.40 x 0.26 m with
        # its long axis along the course tangent, and its floor plate top sits at z=0.04.
        # A circular 0.26 m test plus an absolute z>0.05 gate scored a rind that was
        # genuinely inside the bin (0.14 m along, 0.06 m across, z=0.04) as merely "area".
        best, bd, inside = None, 1e9, False
        for c, (bx, blat) in bin_targets().items():
            X, Y = ct.curve_map(bx, blat)
            th = float(ct.curve_theta(bx))
            dx, dy = float(p[0] - X), float(p[1] - Y)
            along = dx * np.cos(th) + dy * np.sin(th)
            cross = -dx * np.sin(th) + dy * np.cos(th)
            d = float(np.hypot(along, cross))
            if d < bd:
                best, bd = c, d
                inside = (abs(along) < ct.BIN_TOP[0] / 2 and abs(cross) < ct.BIN_TOP[1] / 2
                          and 0.03 < float(p[2]) < ct.BIN_HEIGHT - 0.05)   # below the rim:
                # an object still hanging in the gripper over the bin scored
                # 'correct' at z=0.91 before this bound was added
        ax, lat = arc_of(float(p[0]), float(p[1]))
        # 일반 배치 구역 (guide line 295): a rectangle 26 cm outside the three bins' OUTER
        # edges -- roughly 0.92 x 2.58 m. This used to award the 5 points for landing
        # ANYWHERE in the 6 x 4 m bins cell, ~10x the real area, and it also caught objects
        # that were never placed at all. Guide line 274 additionally requires the object to
        # be COMPLETELY inside ("완전히 들어간"), so the object's own half-extent is taken
        # off the zone before testing its centre.
        _cls = self.obj_cls.get(name)
        half = 0.5 * max(OBJ_DIMS[_cls][:2]) if _cls in OBJ_DIMS else 0.0
        in_zone = (abs(ax - ct.SECTION_CENTERS["bins"]) < ct.PLACEMENT_HALF_ALONG - half
                   and abs(lat) < ct.PLACEMENT_HALF_CROSS - half
                   and float(p[2]) < 0.20)          # resting on the pad, not held aloft
        self.log.ev(f"landing check {name}: world={np.round(p, 2).tolist()} arc={ax:.2f} "
                    f"lat={lat:+.2f} section={section_of(ax)} nearest bin={best}@{bd:.2f} m "
                    f"inside={inside} in_zone={in_zone}")
        if inside:
            return "correct" if best == colour else "wrong"
        if in_zone:
            return "area"
        return "none"

    # ---------------------------------------------------------------- main
    def run(self, crouch_depth: float):
        self.cfg_depth = crouch_depth
        self.env.reset()
        self.legs.set_mode("walk", blend_steps=1)
        # objects spawn 15-25 cm above the ground; scanning before they land would
        # localise them mid-fall (the first smoke run did exactly that)
        for _ in range(130):                      # let the robot AND the trash settle
            self._write_arm(self.travel_q, open_=False)
            self._step(self.legs.step(torch.zeros(1, 3, device=self.device)))

        # End of the last section that can hold objects, whatever it is called.
        # This named "mixed" outright, which does not exist on the Hong Kong course.
        _last = max(OBJ_SECTIONS, key=lambda s_: ct.SECTION_CENTERS[s_])
        search_end = ct.SECTION_CENTERS[_last] + ct.SECTION_LEN / 2 - 0.5
        if self.nav.honest:
            # no map: the search runs until the bins come into view (the collection
            # zone ends at the bin row, an event rule) or the IMU says stairs; the
            # nominal-length fallback only matters if the bins are never sighted
            search_end = self._bins_arc() - 1.5
        handled, trips = 0, 0
        passes = 1
        self._barren_sweeps = 0        # consecutive persist-sweeps that converted nothing
        self._sweep_handled = 0
        while handled < self.p.max_picks:
            det = self.traverse(search_end)
            if det is None:
                if (passes >= self.p.max_passes or trips >= self.p.max_trips
                        or self.step_i > self.p.search_budget
                        or self.must_head_home() or self.off_platform):
                    # SPEND THE SURPLUS, DO NOT BANK IT. Guide 125: once the finish line
                    # is crossed, collection is over — so budget left at this point is
                    # worth exactly zero afterwards. Measured (run 0824_1426_deploy_strict):
                    # under deploy constraints every object failed out its 2 cycles, the
                    # search ended at step 39k and the robot crossed at 13.0 min with 17
                    # minutes unspent. At even a 7 % per-cycle grasp rate that surplus is
                    # ~1 expected extra conversion (+20 pts). MISSION_PERSIST_BUDGET=1:
                    # when the ONLY stop reason is the pass/attempt caps and ample budget
                    # remains, wipe the failure records and sweep again instead.
                    # Reserve the FINISH, not just the walk home. Measured, run
                    # 0824_1536_deploy_strict: the +14k threshold left no budget for the
                    # stairs traverse + finish crossing, which carries 20 checkpoint
                    # points — the run banked nothing from its extra sweeps (grasping was
                    # at its 0 Hz floor), never crossed, and lost 20+20 vs just stopping.
                    # And if the LAST sweep converted nothing, one more identical sweep
                    # is the definition of expecting a different result: allow exactly
                    # one barren retry, then stop.
                    # ...and the sweep margin is a knob, because the runs that stop are
                    # not stopping for want of a reserve: the step ledger over five
                    # honest runs shows a mean 12,328 unused steps at the whistle
                    # (hs4b 17,655 — 4.5 fresh pick cycles at ~3,900 steps each, with a
                    # lunchbox copy it had never engaged still on the course), all of it
                    # under the old flat +14,000 bar. The finish itself is protected by
                    # home_reserve() + _crossed_reserve either way.
                    _crossed_reserve = 8000        # stairs + finish lane + margin
                    if (os.environ.get("MISSION_PERSIST_BUDGET") == "1"
                            and not self.off_platform and not self.must_head_home()
                            and self.step_i <= self.p.search_budget
                            and trips < self.p.max_trips
                            # a sweep for objects that cannot score is pure edge
                            # exposure (swE: 120 banked, then 4 more 20 m sweeps)
                            and self.needed_classes()
                            # Barren-streak cap, GRADED. The cap exists to stop the
                            # strict-mode pathology (0 % grasp rate, identical sweeps
                            # forever) — but at the deploy profile's ~30-50 % per-sweep
                            # conversion, two barren sweeps in a row are ordinary variance
                            # and cutting there cost a replicate 40 points (final run 20
                            # vs iter-2/3's 60, which converted on rounds 4-6). If the
                            # run has EVER converted, the mechanism demonstrably works:
                            # allow 5. The finish reserve bounds the time risk either way.
                            and self._barren_sweeps < (2 if handled == 0 else 5)
                            and self.steps_left() > self.home_reserve()
                                + _crossed_reserve
                                + int(os.environ.get("MISSION_PERSIST_MARGIN", "8000"))):
                        if self._sweep_handled == handled:
                            self._barren_sweeps += 1
                        else:
                            self._barren_sweeps = 0
                        self._sweep_handled = handled
                        passes = 1
                        self.obj_fails.clear()
                        self.blacklist.clear()      # gave-up spots become visible again
                        self.log.ev(f"budget surplus ({self.steps_left()} steps left) — "
                                    f"clearing failure records for another sweep "
                                    f"(barren streak {self._barren_sweeps})")
                    else:
                        break
                # COMBINE THE TRIP when it is cheap: if the basket still has room,
                # the hand is free, and something remembered sits a short hop back,
                # fetch it BEFORE walking to the bins — it rides in the gripper on top
                # of the basket load instead of costing its own bins round trip
                # (measured: trash_9 sat 0.8 m behind the sweep end while the robot
                # walked an 18 m round trip to bank 2 cans, then walked back). Bounded:
                # a LONG backward shuttle with a loaded basket is the transit-loss
                # case, so beyond an 8 m hop bank the load first as before.
                _rc = self.recall_target()
                if _rc is not None and len(self.carried) < self.p.max_carry \
                        and self._hand_item is None:
                    if (self.nav.arc_lat()[0] - (_rc[3] - 2.5)) < 8.0:
                        name, cls, xy, ax, w = _rc
                        self.recall_tries[name] = self.recall_tries.get(name, 0) + 1
                        self._recalling = True
                        self.log.ev(f"nothing new ahead — fetching the {cls} ({name}) "
                                    f"at arc {ax:.1f} BEFORE the bins trip (combined "
                                    f"load; recall {self.recall_tries[name]})")
                        self.goto_arc(ax - 2.5,
                                      lat_target=float(np.clip(w, -1.05, 1.05)))
                        self.face_forward()
                        continue
                self.deliver_payload()          # do not carry anything back
                if not self.needed_classes():
                    self.log.ev("every scoring class is at its cap — nothing left to "
                                "earn, heading for the finish line")
                    break
                # Prefer going straight to something already seen over another blind
                # sweep: the forward-only search wastes a whole traverse re-finding an
                # object it walked past an hour ago.
                recall = self.recall_target()
                if recall is not None:
                    name, cls, xy, ax, w = recall
                    self.recall_tries[name] = self.recall_tries.get(name, 0) + 1
                    self._recalling = True
                    self.log.ev(f"nothing new ahead — returning to the {cls} ({name}) "
                                f"remembered at arc {ax:.1f} lat {w:+.2f} "
                                f"(recall {self.recall_tries[name]})")
                    # stand off to the SAME side it was seen on, so the forward cone
                    # covers it on the way past instead of sweeping by 0.8 m away
                    self.goto_arc(ax - 2.5, lat_target=float(np.clip(w, -1.05, 1.05)))
                    self.face_forward()
                    continue
                passes += 1
                self.log.ev(f"end of the course with {handled} picked and nothing "
                            f"remembered — sweeping again (pass {passes}/{self.p.max_passes})")
                self.goto_arc(self.pass_start_arc())
                self.face_forward()
                continue
            name, dd = self.match_object(det)
            if name is None:
                self.rej["unmatched"] += 1
                self.log.ev(f"phantom {det['cls']} at {np.round(det['pos_w'], 2).tolist()} "
                            f"(nearest same-class object {dd:.2f} m) — blacklisted")
                self.blacklist.append(det["pos_w"][:2])
                continue
            if self._perched(name, "skipping"):
                # before the approach, which is where most of the loss was
                self.obj_fails[name] = self._attempts_for(det["cls"])
                if not self.nav.honest:      # honest: only the estimate is known
                    self.blacklist.append(self._sim_obj(name).data.root_pos_w[0, :2].cpu().numpy())
                self.blacklist.append(np.asarray(det["pos_w"])[:2])
                self.seen.pop(name, None)
                continue
            # Persist on ONE object: walk on only after object_attempts full
            # approach->squat->grasp cycles. Returning to the search after a single
            # miss left the robot metres past an object it could still have picked
            # up — the probe gets a lunchbox about three times in four, but only if
            # it is allowed more than one try.
            att = None
            # as many cycles as the class is allowed (hand classes get one more than
            # the profile's object_attempts), less any already spent on this object
            # from an earlier target. The loop used to run object_attempts (2) cycles
            # while _attempts_for said 3, so the lunchbox's 3rd cycle came as a fresh
            # target with _cycle 0 — never flagged as the last one, so the IK tail
            # (MISSION_IK_LAST_CYCLE) never ran: svK's grass lunchbox, 6 RL attempts,
            # no IK, written off.
            _cycles = max(1, self._attempts_for(det["cls"]) - self.obj_fails.get(name, 0))
            for _cycle in range(_cycles):
                _t_cycle = self.step_i
                if _cycle > 0:
                    # A cycle on an outboard object leaves the robot close to the
                    # edge (svM: lunchbox at lat -1.16 pushed to -1.4, robot at -1.6
                    # after cycle 2); the re-acquire's look-back and pan then walked
                    # the feet over the line. Come back inside first — the next
                    # approach is then from the inboard side, which keeps it there.
                    self._recentre_after_pick(
                        lim=float(os.environ.get("MISSION_CYCLE_RECENTRE_LAT", "1.3")),
                        goal=1.0)
                    det = self.reacquire(det, name)
                    if det.get("stale"):
                        # Same 0.6 m rule the memory planner applies to a knocked
                        # object: if it is no longer where the estimate says AND the
                        # camera cannot find it, the cycle would be spent crouching at
                        # an empty spot (ctlA/svA: 5 stalled attempts, 3,000+ steps,
                        # both on the same mud can). Write it off and spend the time on
                        # the other copies instead.
                        if self.nav.honest:
                            # the robot cannot know where the object really went: a
                            # second fruitless re-acquire (straight look, pan, back-off,
                            # pan) on the same object is the write-off signal
                            self._stale_hits[name] = self._stale_hits.get(name, 0) + 1
                            _here = np.asarray(det["pos_w"], dtype=float)
                            _mv = 9.9 if self._stale_hits[name] >= 2 else 0.0
                        else:
                            _here = self._sim_obj(name).data.root_pos_w[0].cpu().numpy()
                            _mv = float(np.hypot(*(_here[:2] - np.asarray(det["pos_w"])[:2])))
                        if _mv > 0.60:
                            self.log.ev(f"{name} was knocked {_mv:.2f} m from the estimate "
                                        f"and is out of view — writing it off")
                            self.obj_fails[name] = self._attempts_for(det["cls"])
                            self.blacklist.append(_here[:2])
                            self.blacklist.append(np.asarray(det["pos_w"])[:2])
                            self.seen.pop(name, None)
                            self.face_forward()
                            break
                    # A re-found object that now lies at the belt edge is not worth a
                    # cycle: the fresh-detection path never engages beyond the corridor,
                    # but the retry path did — svF's can dropped from the basket
                    # swing to lat -1.77 (rope line 1.98), the approach walked the
                    # feet out of bounds (penalty) and the body onto the rope.
                    _rw = self._arc_of(det["pos_w"])[1]
                    if abs(_rw) > self._edge_lat() - 0.15:
                        self.log.ev(f"{name} now lies at lat {_rw:+.2f}, at the course "
                                    f"edge — writing it off")
                        self.obj_fails[name] = self._attempts_for(det["cls"])
                        self.blacklist.append(np.asarray(det["pos_w"])[:2])
                        self.seen.pop(name, None)
                        self.face_forward()
                        break
                att = Attempt(name=name, cls=det["cls"], section=det["section"])
                self._last_cycle = (self.obj_fails.get(name, 0)
                                    >= self._attempts_for(det["cls"]) - 1)
                self.log.ev(f"target {name} ({det['cls']}) on {det['section']} "
                            f"at {np.round(det['pos_w'], 2).tolist()} (assoc {dd:.2f} m)"
                            f"{'' if not self.obj_fails.get(name, 0) else f' — cycle {self.obj_fails[name] + 1}'}")
                # approach -> crouch -> verify the object is still where the grasp policy
                # expects it. Squatting shifts the base by 10-20 cm, which would otherwise
                # hand the policy an object outside its training distribution; one corrected
                # retry (pre-compensated by the measured shift) fixes that.
                ok_app, target_w = False, np.asarray(det["pos_w"], float)
                aligned = False
                _bias = self.aim_bias          # fixed prior; one-shot corrected on retry,
                                               # never persisted (the old EMA accumulated
                                               # noise across objects for the whole run)
                # MISSION_REAPP_MAX re-approaches (default 1). Mud is where the one
                # shot runs out: svG's can took two crouches to get inside the lateral
                # bar, and the mud lunchbox landed at base x 0.45-0.72 across 14
                # crouches (the pebble copy: 0.55-0.59) — the RL policy never reached
                # it from the far ones and the re-approach had already been spent.
                _reapp = max(0, int(os.environ.get("MISSION_REAPP_MAX", "1")))
                for corr in range(1 + _reapp):
                    ok_app, target_w = self.approach(det, _bias)
                    if not ok_app:
                        break
                    self.settle()
                    att.stable = self.crouch_and_check(det["section"])
                    if not att.stable:
                        break
                    b = self.to_base_frame(target_w)      # target is decision-frame
                    err = b[:2] - self.grasp_anchor[:2]
                    self.log.ev(f"post-crouch alignment error {np.round(err, 3).tolist()} m")
                    # Retry ONLY on the axes that measurably kill grasps. The champion
                    # HELD at x-errors down to -0.33 (object at 0.24!) as long as y was
                    # centred, and stalled at every y-err beyond -0.13 — so x-close is
                    # NOT a retry reason. Only physically-out-of-reach far (x > +0.20)
                    # and genuine lateral misses (|y| > 0.18) go around again.
                    # The +0.20 far bar was measured on the OLD champion. The
                    # plant-matched policy stalls on far objects earlier: s5600e's
                    # can sat at x-err +0.17, both RL attempts died 0.12-0.21 short
                    # and the retry never fired. +0.14 catches that band.
                    _far = float(os.environ.get("MISSION_FAR_RETRY", "0.14"))
                    # lateral bar tightenable (MISSION_LAT_RETRY): s5600ai's can at
                    # y-err -0.12/-0.14 burned 6 RL attempts, every HELD sat |y|<0.09
                    _lat = float(os.environ.get("MISSION_LAT_RETRY", "0.18"))
                    # the lateral bar is centred on the aimed sweet spot, not the anchor
                    _ey = err[1] - self.aim_y_off(det["cls"])
                    _bad = (err[0] > (0.32 if det["cls"] == "redbag" else _far)
                            or err[0] < -float(os.environ.get(
                                "MISSION_CLOSE_RETRY", "99"))
                            or abs(_ey) > _lat)
                    if not _bad or corr == _reapp:
                        aligned = True
                        break
                    if np.linalg.norm(err) < 0.6:
                        # PARTIAL correction (MISSION_REAPP_GAIN). 576 crouches: the
                        # crouch-induced shift repeats to only ~0.07-0.10 m laterally,
                        # so folding the whole measured error into the retry aim
                        # over-corrects — the re-approach crouches landed at y +0.04
                        # (std 0.11) vs the first crouch's -0.04 (std 0.07): no better,
                        # just on the other side.
                        _g = float(os.environ.get("MISSION_REAPP_GAIN", "1.0"))
                        _bias = np.clip(_bias + _g * np.array([err[0], _ey]),
                                        -0.45, 0.45)      # == aim_bias + g*err on the 1st retry
                    else:
                        self.log.ev(f"ignoring implausible alignment error "
                                    f"{np.round(err, 2).tolist()} m — not folding it into the aim")
                    self.log.ev(f"re-approaching with one-shot aim bias "
                                f"{np.round(_bias, 3).tolist()}")
                    self.stand_up()
                    # Open 0.3 m of space FIRST. The retry used to start with the robot
                    # already on top of the object; the approach then needed to BACK UP,
                    # which the forward-trained gait barely does, so it timed out in
                    # place and re-crouched even further forward (the measured -0.183
                    # median). Walking a deliberate reverse leg makes the re-approach a
                    # fresh forward run, which lands centred (measured).
                    # ...unless the reverse leg points at the edge. swF: crouched on the
                    # mud rind (lat +1.05) from its outboard side with the nose inboard,
                    # the 0.3 m back-out walked the base to lat +1.78 (feet over the
                    # 1.98 line, -10), the boundary post caught a leg and the robot
                    # could not walk back in. Here the reverse gait is the only thing
                    # that moves the base outboard on purpose, so predict where it ends.
                    # ...and reverse ALONG THE BODY AXIS. walk_cmd steers for the
                    # centreline with a forward-facing model: a robot that walked in
                    # around a crowd faces anywhere (hs3: -165 deg at lat -1.14), and
                    # walk_cmd(0.0, vx=-0.30) then spins it while reversing — the base
                    # swept 0.7 m outboard in 107 steps, feet over the line (-10) and
                    # 3,000 steps wedged on the boundary. Predict the straight reverse
                    # from the actual heading (S-frame: w grows to the left of the
                    # tangent, forward = (cos rel, sin rel)) and skip it near the edge.
                    _ax, _wb = self.nav.arc_lat()
                    _, _, _, _yaw = self.base_pose()
                    _rel = (_yaw - self._tan_yaw(_ax) + np.pi) % (2 * np.pi) - np.pi
                    _w_after = _wb - 0.3 * float(np.sin(_rel))
                    # 0.2 was not enough margin. hs18f predicted +0.88 -> +1.12, passed
                    # the gate, and came out at lat +1.79: the estimate was already
                    # 0.29 m short of the truth (+1.17) and the reverse gait curves
                    # rather than tracking the body axis. Feet over the line at 14,880,
                    # -10 at 14,920, off the deck at 14,995 — the whole run for one
                    # re-approach. Widen the predicted margin AND watch the lateral
                    # while reversing (below), since the open-loop prediction is the
                    # thing that has now been wrong twice.
                    if abs(_w_after) > self._edge_lat() - 0.35:
                        self.log.ev(f"skipping the 0.3 m back-out at lat {_wb:+.2f}: heading "
                                    f"{np.degrees(_rel):+.0f} deg off the tangent, reversing "
                                    f"would end at lat {_w_after:+.2f} by the edge — "
                                    f"re-approaching from here")
                    else:
                        self.log.ev(f"backing out 0.3 m along the body axis before the "
                                    f"re-approach (heading {np.degrees(_rel):+.0f} deg, "
                                    f"lat {_wb:+.2f} -> {_w_after:+.2f})")
                        c = torch.zeros(1, 3, device=self.device)
                        c[0, 0] = -0.30
                        _wstop = self._edge_lat() - 0.25
                        for _ in range(220):
                            self._write_arm(self.travel_q, open_=False)
                            self._step(self.legs.step(c))
                            if abs(self.nav.arc_lat()[1]) > _wstop:
                                self.log.ev(f"  back-out stopped at lat "
                                            f"{self.nav.arc_lat()[1]:+.2f}: the reverse "
                                            f"is heading for the edge")
                                break
                det = {**det, "pos_w": target_w}
                if not ok_app:
                    att.note = "approach timeout"
                    self.obj_fails[name] = self.obj_fails.get(name, 0) + 1
                    if self.obj_fails[name] >= self._attempts_for(det["cls"]):
                        self.blacklist.append(det["pos_w"][:2])
                    self.log.attempts.append(att)
                    self.stand_up()
                    # RE-AIM BEFORE RESUMING. A failed approach leaves the robot pointing
                    # at whatever it was chasing — including a phantom that read as
                    # "118.45 m behind the robot" — and traverse then issues vx>0 along
                    # the BODY axis without re-orienting. Facing backwards, "forward" walks
                    # down the course: a run crawled -16.5 -> -20.9 and fell off the back
                    # edge, which it cannot climb back onto. This is the upstream cause of
                    # the "off the course at arc -22.0" deaths, and 218 steps of turning
                    # is cheap insurance against losing the rest of the run.
                    self.face_forward()
                    break
                if not att.stable:
                    att.note = "unstable crouch — terrain downgraded to traverse-only"
                    self.blacklist.append(det["pos_w"][:2])
                    self.stand_up()
                    self.log.attempts.append(att)
                    self.face_forward()      # same re-aim as the approach-failure path
                    break
                got = self.attempt_grasp(name, det)
                att.grasped = got
                _hand_carry = self.p.carry_in_gripper or (
                    det["cls"] not in self.p.basket_classes)
                if got and _hand_carry:
                    # lift to the validated mid waypoint and hold THAT while walking
                    # a shallow lid pinch peels out under acceleration, so lift it slowly
                    _, _depth = grasp_hint(det["cls"], OBJ_DIMS[det["cls"]])
                    # lift through q_mid (validated clear of the object's resting spot)
                    # and then tuck to the carry pose; a shallow lid pinch peels out under
                    # acceleration, so both legs of that move go slowly.
                    _slow = 420 if _depth < 0.012 else 200
                    _cq = self.carry_pose(det["cls"])
                    _wp = [self.retrieve["q_mid"]]
                    if _cq is not None and _cq != self.retrieve["q_mid"]:
                        _wp.append(_cq)
                    self.play_traj(_wp, grip_open=False, steps_each=_slow)
                    # CONFIRM it survived the lift. A lunchbox held by its 8 mm lid pinch
                    # let go on the way up and the robot walked 30 m to the bin with an
                    # empty hand, scoring `none`; the object was still where it had been
                    # picked up. A dropped lift must count as a failed grasp so the retry
                    # machinery gets another go at it.
                    obj_b = self.to_base_frame_true(
                        self._sim_obj(name).data.root_pos_w[0].cpu().numpy())
                    tcp_b = self.to_base_frame_true(self.tcp_w()[0].cpu().numpy())
                    gap = float(np.linalg.norm(obj_b - tcp_b))
                    if self.obj_honest:
                        # the fingers say whether the lift survived: a real width
                        # against a working drive, not a distance to a pose
                        _ld, _gw, _ge = self._sensed_load()
                        self.log.ev(f"  lift check {name}: width {_gw * 1000:.0f} mm, "
                                    f"effort {-1.0 if _ge is None else _ge:.1f} -> "
                                    f"{'still held' if _ld else 'DROPPED'} (sensed)"
                                    f"  [TRUTH] TCP gap {gap:.2f} m")
                        gap = 0.0 if _ld else 9.9
                    if gap < 0.20:
                        self.travel_q = torch.tensor(_cq,
                                                     dtype=torch.float32,
                                                     device=self.device)
                        if det["cls"] == "redbag":
                            # the bag hangs ~0.37 m below the TCP: at the normal carry
                            # height its bottom cleared flat ground but caught a stair
                            # nose on the climb (r2: bag found resting ON the stairs at
                            # z 0.63, delivery scored none). Raise the shoulder so the
                            # hanging bag clears the steps.
                            self.travel_q[1] = float(max(-2.0,
                                                         self.travel_q[1] - 0.40))
                        att.deposited = True
                        # _hand_item always: unload routes on it OR the profile flag,
                        # and the carry-watch telemetry arms off it (it stayed silent
                        # through r2's stair loss because carry_in_gripper never set it)
                        self._hand_item = name
                        self._carry_lost_logged = False
                        # the sensed carry watch needs two consecutive empty samples;
                        # a count left over from the PREVIOUS hand item would write
                        # this one off on its first sample
                        self._carry_empty_n = 0
                        self.log.ev(f"carrying {name} in the gripper to the bin "
                                    f"(TCP gap {gap:.3f} m)")
                    else:
                        att.grasped = got = False
                        self.log.ev(f"{name} slipped out during the lift "
                                    f"(TCP gap {gap:.3f} m) — treating as a failed grasp")
                elif got:
                    att.deposited = self.deposit_to_basket(name)
                # bookkeeping is common to BOTH delivery modes
                if att.deposited:
                    self.carried.append((name, det["cls"]))
                    self.picked_per_cls[det["cls"]] = self.picked_per_cls.get(det["cls"], 0) + 1
                    handled += 1
                self.stand_up()
                self.log.attempts.append(att)
                self._recalling = False
                # Only give up on an object after object_attempts full cycles. Blacklisting on
                # the first failure meant a lunchbox — which the probe picks up about three
                # times in four — was written off after one miss.
                if att.deposited:
                    _spot = np.asarray(det["pos_w"], float)[:2].copy()
                    self.blacklist.append(_spot)
                    self._pick_spot[name] = _spot
                    self._forget_obstacle(_spot, det["cls"])   # no ghost under the next site
                    # It is aboard now — its remembered GROUND spot is stale, and recall
                    # walked back to the empty spot chasing a can that was riding in the
                    # robot's own basket (basket9, two wasted recalls).
                    self.seen.pop(name, None)
                else:
                    self.obj_fails[name] = self.obj_fails.get(name, 0) + 1
                    # PER-OBJECT CLOCK. s5600ai spent 7,500 steps on one can and 10,000
                    # on one rind (2 cycles x 5 attempts each, zero conversions) while
                    # two other copies of each class went unengaged — every class has
                    # three copies and only two score, so a copy that has eaten
                    # MISSION_OBJ_CAP steps is written off and the next copy gets the time.
                    self._obj_steps[name] = (self._obj_steps.get(name, 0)
                                             + self.step_i - _t_cycle)
                    _cap = int(os.environ.get("MISSION_OBJ_CAP", "0"))
                    if _cap and self._obj_steps[name] >= _cap \
                            and self.obj_fails[name] < self._attempts_for(det["cls"]):
                        self.log.ev(f"{name} has used {self._obj_steps[name]} steps "
                                    f"(cap {_cap}) — writing it off")
                        self.obj_fails[name] = self._attempts_for(det["cls"])
                    if getattr(self, "_grasp_perched", False):
                        # attempt_grasp already said why; make the write-off final
                        self.obj_fails[name] = self._attempts_for(det["cls"])
                    if self.obj_fails[name] >= self._attempts_for(det["cls"]):
                        # blacklist where it actually IS now, not where it was first seen
                        # (honest mode knows only the last estimate)
                        if not self.nav.honest:
                            here = self._sim_obj(name).data.root_pos_w[0].cpu().numpy()
                            self.blacklist.append(here[:2])
                        self.blacklist.append(det["pos_w"][:2])
                        self.seen.pop(name, None)      # and forget it, so recall cannot
                        self.log.ev(f"giving up on {name} after {self.obj_fails[name]} cycles")
                        # ...and actually stop: the cap/perch write-offs used to fall
                        # through into the next cycle anyway (svH: "giving up on
                        # trash_7" then "cycle 2" — reacquire backed the robot off
                        # the platform at lat -9.9 and the run ended with 0 points)
                        break
                    else:
                        self.log.ev(f"{name}: cycle {self.obj_fails[name]} failed — "
                                    f"will try again")
                if att is not None and att.deposited:
                    break
                if att is not None and (att.note or not att.stable):
                    break        # approach timeout / bad crouch: moving on is right
            self._recentre_after_pick()
            # basket full -> shuttle: deliver, then come back and carry on searching
            if ((len(self.carried) >= self.p.max_carry
                 or self._hand_item is not None) and handled < self.p.max_picks):
                # Resume PAST the object just taken, not on top of it. resume_arc is read
                # while standing over the pick, so the robot used to shuttle all the way
                # back and re-sweep a stretch it had already cleared — at 85 steps/m that
                # is pure loss, and the re-swept ground is by definition empty.
                resume_arc = self.nav.arc_lat()[0] + 1.5
                # ...but never step OVER something we still need. The three mixed-section
                # objects sit within 0.54 m of each other, so skipping 1.5 m past the one
                # just picked jumps the other two, and the pass then ends at search_end
                # 14.5. trash_8 (rind, mixed, arc 13.01) was targeted ZERO times in all 11
                # recorded runs for exactly this reason. Back the resume point up to just
                # before the nearest remembered object of a class still needed, when the
                # clock can afford it — 2.7 m of walking is ~233 steps against a 20-point
                # object sitting at the cheapest delivery distance on the course.
                rc = self.recall_target()
                if rc is not None:
                    rc_arc = rc[3]
                    # Bounded to a genuine STEP-OVER, not a journey. The intent is "do not
                    # walk past the neighbour of the object I just took" — the mixed
                    # cluster spans 0.54 m. Unbounded it produced "resume pulled back from
                    # arc 13.5 to -5.5", a 19 m BACKWARD shuttle, and backward shuttles are
                    # the single operation that has killed runs (turn-first helps, but the
                    # cheapest fix is not to ask for them). Anything further back is what
                    # the end-of-pass recall machinery is for.
                    PULLBACK_MAX = 6.0
                    if (resume_arc - PULLBACK_MAX) < (rc_arc - 1.5) < resume_arc \
                            and self.afford_object(rc_arc):
                        self.log.ev(f"resume pulled back from arc {resume_arc:.1f} to "
                                    f"{rc_arc - 1.5:.1f} for a remembered {rc[1]}")
                        resume_arc = rc_arc - 1.5
                trips += 1
                self.log.ev(f"carrying {len(self.carried)} — trip {trips}, will "
                            f"resume the search at arc {resume_arc:.1f}")
                self.deliver_payload()
                if (trips >= self.p.max_trips or self.step_i > self.p.search_budget
                        or self.must_head_home()):
                    break
                # NOTHING LEFT TO EARN: with every scoring class at its cap (and the
                # bags skipped) the score cannot change, yet swE shuttled 9 m back
                # from the bins at 120/160 with 37k steps on the clock to sweep for
                # objects it could not score — every extra minute near the edge is
                # only downside (svU: 120 banked, then off the deck orbiting a bag).
                if not self.needed_classes():
                    self.log.ev("every scoring class is at its cap — nothing left to "
                                "earn, heading for the finish line")
                    break
                # survey mode: nothing to see in the swept sections behind the
                # first pending station / remembered object — start there
                resume_arc = self._survey_resume_arc(resume_arc)
                # Empty-handed at the bins: only walk back if there is time to pick
                # something up once we get there.
                if not self.afford_object(resume_arc):
                    self.log.ev(f"venue clock: {self.steps_left()} steps left — not "
                                f"enough to service another object at arc "
                                f"{resume_arc:.1f}, stopping the search")
                    break
                self.goto_arc(resume_arc)
                self.face_forward()

        self.log.ev(f"search done after {trips} shuttle trip(s): "
                    f"carrying {[c for _, c in self.carried]}")
        self.deliver_payload()
        self.cross_finish()
        return self.report()

    def cross_finish(self):
        """Drive past the red finish line, which is what actually ends the core task.

        Guide line 123: after placing what it collected the robot keeps going and completes
        the core task by CROSSING the line; line 125 stops the clock there and line 115
        makes the bonus task unavailable to a robot that never crosses before time expires.
        The mission used to stop at the bins, so it forfeited the bonus (40 points, and our
        다족 category is the only one eligible) and produced no completion time to break
        ties on -- while banking every placement it had already made, which line 125 also
        guarantees. So this is upside with no risk to the task-2 score.
        """
        if ct.FINISH_ARC is None or self.finished:
            return
        if self.off_platform:
            self.log.ev("off the platform — no way to the finish line from here")
            return
        budget = min(self.steps_left(), 3000)
        if budget < 300:
            self.log.ev("no budget left to cross the finish line")
            return

        # GO AROUND THE BINS, not through them. The finish line sits BEYOND the bin row,
        # and the row spans the middle of the course: with the bins at lateral -0.9 / 0.0
        # / +0.9 and each 0.26 m across, they block -1.03..-0.77, -0.13..+0.13 and
        # +0.77..+1.03. Driving straight at the line put the green bin dead ahead and the
        # robot pushed against it for 4 re-aims and gave up 1.7 m short -- measured, it
        # never crossed and forfeited the bonus task it exists to unlock.
        #
        # The widest clear lane is OUTSIDE the row: 1.03 to 1.87 (the belt stanchion posts
        # start at 1.87), so 1.45 is its centre with 0.42 m of margin either side. Pick the
        # side the robot is already on to avoid crossing the row to reach the lane.
        _, lat_now = self.nav.arc_lat()
        # ...but not down the middle of that lane. 1.45 is exactly _edge_lat() and only
        # 0.20 m inside the foot-guard strike line, so every disturbance on the way
        # tripped it and the strike recovery walked back in to 1.2 only for the dash to
        # steer straight out again (hs4e cycled that four times). 1.15 still clears the
        # bin row's outer edge at 1.03 — the geometry above is what matters — with half
        # a metre of margin to the guard.
        # ...and 1.15 is still too far in, because it centres the ROBOT, not a point.
        # The bin's outer edge is at 1.03 and the body is ~0.35 m across, so at lane
        # 1.15 the inboard side sits at 0.98 — inside the bin. Two runs ended exactly
        # there: hs22v5 logged "finish line NOT crossed (arc 17.9 lat +1.14)" after
        # 2,649 steps of no forward progress with 23,000 steps of clock still unspent,
        # and hs21e the same. 1.32 clears the bin by 0.11 m on the body's edge and
        # still leaves 0.13 m to the 1.45 guard line.
        lane = float(os.environ.get("MISSION_FINISH_LANE", "1.32"))
        lane = lane if lat_now >= 0.0 else -lane
        if self.nav.honest:
            # the line is 1.2 m past the bin row (a course rule, the sighted bins give
            # the row); the judge's own crossing check ends the run
            _fin = self._bins_arc() + ct.FINISH_LOCAL
            past = min(_fin + 0.8, self._course_end() - 0.4)
        else:
            _fin = ct.FINISH_ARC
            past = min(ct.FINISH_ARC + 0.8, ct.COURSE_X0 + ct.COURSE_LEN - 0.4)
        self.log.ev(f"heading for the finish line at arc {_fin:.1f} via the "
                    f"lat {lane:+.2f} lane ({budget} steps available)")
        # ease into the lane BEFORE reaching the row, then run past it
        approach = self._bins_arc() - 1.2
        if self.nav.arc_lat()[0] < approach:
            self.goto_arc(approach, lat_target=lane)
        self.goto_arc(past, lat_target=lane)
        if not self.finished:
            a_now, l_now = self.nav.arc_lat()
            self.log.ev(f"finish line NOT crossed (arc {a_now:.1f} lat {l_now:+.2f})")

    def report(self):
        # Score each PHYSICAL OBJECT once, best outcome first. An object that needed a
        # retry leaves two Attempt rows; counting rows double-charged it against the
        # 2-per-category cap and pushed a genuine `correct` out of the score entirely
        # (a run worth 25 was reported as 10).
        rank = {"correct": 3, "wrong": 2, "area": 1}
        best = {}
        for a in self.log.attempts:
            if not a.binned or a.binned == "none":
                continue
            # Key on the PHYSICAL article, not the robot's own label. With honest
            # identity a name is a track id, and a tracker that splits one article into
            # two tracks would otherwise be scored twice — the robot may decide by
            # track, but the scorer counts articles. _bound is the telemetry binding
            # (nearest truth of the track's class, pinned at first use).
            _key = getattr(self, "_bound", {}).get(a.name, a.name)
            cur = best.get(_key)
            if cur is None or rank[a.binned] > rank[cur[1]]:
                best[_key] = (a.cls, a.binned)
        per_cls, total = {}, 0
        for cls, binned in sorted(best.values(), key=lambda r: -rank[r[1]]):
            if per_cls.get(cls, 0) >= MAX_SCORED_PER_CLASS:
                continue
            per_cls[cls] = per_cls.get(cls, 0) + 1
            total += SCORE.get(binned, 0)
        rep = {
            "profile": self.p.name,
            "steps": self.step_i,
            "attempts": [vars(a) for a in self.log.attempts],
            "carried": [n for n, _ in self.carried],
            "score_task2": total,
            "score_breakdown": per_cls,
            "terrain_strikes": self.terrain_strikes,
            "terrain_ok": self.terrain_ok,
            "events": self.log.events,
        }
        # Task 1 and the penalty term. Until now report() emitted the task-2 total ALONE
        # and called it the score, which is the gross number: the guide ranks on
        # task1 + task2 - penalties (line 392: -10 per out-of-bounds, unlimited), and
        # breaks ties on completion time (line 429), which only exists once the robot has
        # crossed the finish line (line 125).
        oob_penalty = 10 * self.oob_count
        rep["score_task1"] = self.task1
        rep["task1_max"] = ct.TASK1_MAX
        rep["checkpoints"] = [ct.CHECKPOINTS[i][2] for i in sorted(self.cp_done)]
        rep["oob_count"] = self.oob_count
        # what the venue would actually have charged (feet, not body centre)
        rep["oob_feet_count"] = self.oob_feet_count
        rep["penalty_feet"] = 10 * self.oob_feet_count
        rep["penalty"] = oob_penalty
        rep["finished"] = self.finished
        rep["finish_min"] = (round(self.finish_step * 0.02 / 60, 2)
                             if self.finish_step else None)
        rep["score_total"] = self.task1 + total - oob_penalty
        rep["venue"] = ct._VENUE
        rep["nav_honest"] = os.environ.get("MISSION_NAV_HONEST") == "1"
        self._torque_audit_write()
        # The report is the ONLY structured record of a run — steps, per-class breakdown
        # and the full event list — and it is written once, at the very end. A 2h45m run
        # died here with FileNotFoundError because its output directory (empty all along,
        # since MISSION_NO_VIDEO=1 writes no mp4s) had been swept by a disk cleanup. Never
        # lose a finished run to a missing directory, and never let this raise: the score
        # is already computed by the time we get here.
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / f"report_{self.p.name}.json").write_text(
                json.dumps(rep, indent=2, default=str))
        except Exception as e:
            alt = RW / "course_mission" / "logs" / f"report_{self.p.name}_rescue.json"
            print(f"[MISSION] could not write the report to {self.out_dir} ({e}) "
                  f"— falling back to {alt}", flush=True)
            try:
                alt.parent.mkdir(parents=True, exist_ok=True)
                alt.write_text(json.dumps(rep, indent=2, default=str))
            except Exception as e2:
                print(f"[MISSION] rescue write failed too: {e2}", flush=True)
        print(f"[MISSION] === score (과제2) = {total} / 160 ===", flush=True)
        return rep
