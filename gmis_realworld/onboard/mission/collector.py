"""The Task 2 collector: pick a target, walk to it, grasp it, deliver it.

Plugs into `mission/fsm.py` through `attach_collector()`, which asks for exactly three
things -- `want(budget)`, `step()`, `carrying()` -- so Task 1 stays finished and frozen
while this moves.

The shape is the simulator's, because the simulator's is measured.  What is different is
what it runs ON: object identity comes from `perceive/objtrack`, not a simulator registry;
grasp success comes from finger width, current and lift, not from the object's true pose;
the bins are where the robot last SAW them, carried on raw odometry.

Carrying is ONE object.  The shipping profile carries in the gripper, so every successful
grasp is a trip to the bins.
"""
from __future__ import annotations

import enum

import numpy as np

from course_mission.catalogue import BIN_OF, OBJ_DIMS, grasp_hint, grasp_obb
from onboard.mission.deliver import Delivery
from onboard.mission.fsm import Phase
from onboard.mission.grasp import GraspEpisode
from onboard.mission.ledger import Ledger
from onboard.oio.iface import ArmCommand
from onboard.perceive.objtrack import ObjectTracker, distance_key


class Sub(enum.Enum):
    IDLE = "idle"
    APPROACH = "approach"
    SETTLE = "settle"
    CROUCH = "crouch"
    RAMP = "ramp"
    GRASP = "grasp"
    LIFT = "lift"
    TO_BIN = "to_bin"
    RELEASE = "release"
    APRON = "apron"


class Collector:
    def __init__(self, cfg, mission_cfg, sup, nav, odom, percep, arm_bus, arm_ctl,
                 grasp_ep: GraspEpisode, tracker: ObjectTracker, delivery: Delivery,
                 ledger: Ledger) -> None:
        self.cfg = cfg
        self.sel = cfg["select"]
        self.app = cfg["approach"]
        self.bud = cfg["budget"]
        self.sup, self.nav, self.odom, self.percep = sup, nav, odom, percep
        self.bus, self.arm = arm_bus, arm_ctl
        self.ep, self.trk, self.deliver, self.ledger = grasp_ep, tracker, delivery, ledger

        self.skip = set(self.sel["skip_classes"])
        self.max_per_cls = int(self.sel["max_scored_per_class"])
        self.attempts_for = dict(self.bud["attempts"])
        self.aim_bias = np.asarray(self.app["aim_bias"], float)

        self.sub = Sub.IDLE
        self.target = None
        self.step_i = 0
        self._t_sub = 0
        self._held_cls: str | None = None
        self.binned_ok: dict[str, int] = {}
        self.picked: dict[str, int] = {}
        self._approach_est: np.ndarray | None = None

    # ---- the FSM's three questions ------------------------------------------------------
    def carrying(self) -> int:
        return int(self._held_cls is not None)

    def want(self, budget) -> bool:
        """Is there something worth walking to, and time to do it?

        This INGESTS as well as answers, and it has to.  The FSM only calls `step()` once
        `want()` has said yes, so ingesting solely in `step()` deadlocks: no tracks means no
        target means `want()` is False means `step()` is never called means no tracks.  The
        first version did exactly that and drove the whole course seeing nothing.

        `want()` is called every DRIVE tick, which is also the right cadence for taking
        detections off the mailbox.
        """
        self.step_i += 1
        self._ingest()
        if self._held_cls is not None or self.sub is not Sub.IDLE:
            return False
        if not budget.may_collect():
            return False
        return self._pick_target() is not None

    def step(self) -> Phase | None:
        # `want()` already advanced the clock and drained the mailboxes on DRIVE ticks; in
        # COLLECT/DELIVER the FSM calls step() instead, so do it here.
        if self.sub is not Sub.IDLE:
            self.step_i += 1
            self._ingest()
        handler = getattr(self, f"_do_{self.sub.value}")
        return handler()

    # ---- perception in -------------------------------------------------------------------
    def _ingest(self) -> None:
        m = self.percep.objects.get()
        if m is not None:
            self.trk.observe_all(m.value or [], self.step_i)
        b = self.percep.bins.get()
        if b is not None and hasattr(self.deliver.mem, "observe"):
            self.deliver.mem.observe(b.value, self._odo(), self.step_i)

    def _odo(self) -> np.ndarray:
        return np.array([self.odom.xy[0], self.odom.xy[1], getattr(self.nav, "heading", 0.0)])

    # ---- target selection ---------------------------------------------------------------
    def _quota_full(self, cls: str) -> bool:
        return self.binned_ok.get(cls, 0) >= self.max_per_cls

    def _pick_target(self):
        cands = []
        for t in self.trk.tracks.values():
            if not self.trk.engageable(t):
                continue
            if BIN_OF.get(t.cls) is None or t.cls in self.skip:
                continue
            if self._quota_full(t.cls):
                continue
            if t.attempts >= self.attempts_for.get(t.cls, 2):
                continue
            d = float(np.linalg.norm(t.p[:2]))
            if not (float(self.sel["range_min_m"]) < d < float(self.sel["range_max_m"])):
                continue
            cands.append(t)
        if not cands:
            return None
        return min(cands, key=lambda t: distance_key(t, (0.0, 0.0),
                                                     dict(self.sel["class_rank"]), self.picked))

    # ---- sub-states ------------------------------------------------------------------------
    def _enter(self, sub: Sub) -> None:
        self.sub = sub
        self._t_sub = self.step_i

    @property
    def _elapsed(self) -> int:
        return self.step_i - self._t_sub

    def _do_idle(self):
        t = self._pick_target()
        if t is None:
            return None
        self.target = t
        self.ledger.start(t, self.step_i)
        self._approach_est = t.p.copy()
        self._enter(Sub.APPROACH)
        return Phase.COLLECT

    def _do_approach(self):
        t = self.target
        aim = self.trk.tracks[t.tid].p[:2] - self.aim_bias
        err = aim - np.zeros(2)                       # target in the body frame already
        ex, ey = float(err[0]) - float(self.app["aim_x"]), float(err[1])
        if (np.hypot(ex, ey) < float(self.app["radius_m"])
                and abs(ey) < float(self.app["lateral_m"])):
            self.sup.set_command(np.zeros(3))
            self._enter(Sub.SETTLE)
            return Phase.COLLECT
        if self._elapsed > int(self.app["timeout_steps"]):
            self._fail("approach_timeout", "never reached the aim point")
            return Phase.DRIVE
        self.sup.set_command(np.array([np.clip(0.9 * ex, -0.35, 0.55),
                                       np.clip(0.9 * ey, -0.25, 0.25), 0.0]))
        return Phase.COLLECT

    def _do_settle(self):
        self.sup.set_command(np.zeros(3))
        if self._elapsed > 45:
            self._enter(Sub.CROUCH)
        return Phase.COLLECT

    def _do_crouch(self):
        self.sup.set_command(np.zeros(3))
        c = self.cfg["crouch"]
        total = int(c["blend_steps"]) + int(c["descend_steps"]) + int(c["stability_steps"])
        if self._elapsed < total:
            return Phase.COLLECT
        # The re-observation, and the number the venue will read.  Two independent looks that
        # disagree are the only evidence available that the target is the wrong article.
        fresh = self.trk.tracks[self.target.tid].p
        if self._approach_est is not None and self.ledger.current is not None:
            self.ledger.current.reobs_delta_m = round(
                float(np.linalg.norm(fresh[:2] - self._approach_est[:2])), 4)
        cls = self.target.cls
        width, _depth = grasp_hint(cls, OBJ_DIMS[cls])
        self.ep.reset(fresh, width, grasp_obb(cls, OBJ_DIMS[cls]))
        if self.ep.drift() > float(self.cfg["grasp"]["bank_drift_abort_m"]):
            self._fail("out_of_bank", f"{self.ep.drift():.2f} m outside the pre-grasp bank")
            return Phase.DRIVE
        self._enter(Sub.RAMP)
        return Phase.COLLECT

    def _do_ramp(self):
        self.bus.write(ArmCommand(q=self.ep.start_pose(),
                                  gripper_stroke=self.ep.open_stroke))
        if self._elapsed > int(self.cfg["grasp"]["ramp_steps"]):
            self._enter(Sub.GRASP)
        return Phase.COLLECT

    def _do_grasp(self):
        st = self.bus.read()
        done = self.ep.step(st, base_quat=self._base_quat(),
                            base_ang_vel_b=self._ang_vel(), base_lin_acc_w=self._acc())
        if not done:
            return Phase.COLLECT
        if self.ledger.current is not None:
            self.ep.fill(self.ledger.current)
            self.ledger.current.gw_end_m = st.gripper_stroke
            self.ledger.current.effort_end = st.gripper_current
        if self.ep.held:
            self._held_cls = self.target.cls
            self.picked[self.target.cls] = self.picked.get(self.target.cls, 0) + 1
            self.trk.forget(self.target)
            self._enter(Sub.TO_BIN)
            return Phase.DELIVER
        self.target.attempts += 1
        stage = ("arrived_empty" if self.ep.hold.loaded_n == 0 and self.ep.best_reach < 0.12
                 else "slipped" if self.ep.hold.loaded_n else "approach_timeout")
        self._fail(stage, self.ep.why())
        return Phase.DRIVE

    def _do_lift(self):
        return Phase.DELIVER

    def _do_to_bin(self):
        colour = self.deliver.bin_for(self._held_cls) or "blue"
        cmd, arrived, why = self.deliver.approach_command(
            colour, self._odo(), self.step_i,
            elapsed_frac=min(1.0, self._elapsed / 3000.0))
        if arrived:
            self._enter(Sub.RELEASE)
            return Phase.DELIVER
        if why == "no fresh sighting" and self._elapsed > 4000:
            self._enter(Sub.APRON)
            return Phase.DELIVER
        self.sup.set_command(cmd)
        return Phase.DELIVER

    def _do_release(self):
        colour = self.deliver.bin_for(self._held_cls) or "blue"
        self.sup.set_command(np.zeros(3))
        st = self.bus.read()
        tcp_b, _ = self.arm.fk(st.q)
        a = self.ledger.current
        if a is not None:
            a.bin_colour = colour
            a.bin_obs_age = self.deliver.mem.age(colour, self.step_i)
            a.bin_moved_m = round(self.deliver.mem.moved(colour, self._odo()), 3)
        shift, refused = self.deliver.recentre(colour, tcp_b, self._odo(), self.step_i)
        if a is not None:
            a.recentre_m = round(float(np.linalg.norm(shift)), 4)
            a.recentre_refused = refused
        if self._elapsed < int(self.cfg["deliver"]["open_steps"]):
            self.bus.write(ArmCommand(q=st.q, gripper_stroke=self.ep.open_stroke))
            return Phase.DELIVER
        believed = self.deliver.believed(colour, tcp_b, self._odo(), self.step_i)
        if a is not None:
            a.released_believed = believed
        if believed:
            self.binned_ok[self._held_cls] = self.binned_ok.get(self._held_cls, 0) + 1
        self.ledger.close("released_believed" if believed else "bin_unreachable")
        self._held_cls = None
        self._enter(Sub.IDLE)
        return Phase.DRIVE

    def _do_apron(self):
        self.sup.set_command(np.zeros(3))
        st = self.bus.read()
        self.bus.write(ArmCommand(q=st.q, gripper_stroke=self.ep.open_stroke))
        if self._elapsed > 200:
            if self.ledger.current is not None:
                self.ledger.current.released_believed = False
            self.ledger.close("apron", "bin unreachable; placed in the 25 % zone")
            self._held_cls = None
            self._enter(Sub.IDLE)
            return Phase.DRIVE
        return Phase.DELIVER

    # ---- helpers ----------------------------------------------------------------------------
    def _fail(self, stage: str, note: str) -> None:
        if self.target is not None:
            self.trk.blacklist_spot(self.target.p)
        self.ledger.close(stage, note)
        self.target = None
        self._enter(Sub.IDLE)

    def _base_quat(self) -> np.ndarray:
        return np.asarray(getattr(self.sup.bus.read(), "quat", [1.0, 0, 0, 0]), float)

    def _ang_vel(self) -> np.ndarray:
        return np.asarray(self.sup.bus.read().gyro, float)

    def _acc(self) -> np.ndarray:
        return np.asarray(self.sup.bus.read().accel, float)

    def report(self) -> dict:
        return {"sub": self.sub.value, "carrying": self.carrying(),
                "binned_ok": self.binned_ok, "picked": self.picked,
                "tracker": self.trk.report(), "ledger": self.ledger.summary()}
