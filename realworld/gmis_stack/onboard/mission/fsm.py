"""The hiking mission: drive the corridor, count checkpoints, respect the clock.

Scope, stated plainly.  This runs Task 1 -- the 80 points that come from walking the course
-- and it owns the run-level concerns that Task 2 also needs: the wall clock, the corridor
estimate, the out-of-bounds guard, recovery, and the finish rule.  The object-collection
decision logic (which target, which bin, when to give up on one) is 8,900 lines in
`course_mission/runner.py` and is not reimplemented here; `attach_collector()` is where it
plugs in, and until it does this FSM walks the course and delivers nothing.

That split is deliberate.  Task 1 is worth 80 points and is a prerequisite for the other
160: a robot that cannot cross the course cannot reach an object, so this is the part that
has to be right first and the part that must keep working when everything else degrades.

Three rules shape the states more than any algorithm does:

  * "Each out-of-bounds violation shall result in a 10-point deduction", judged on where the
    FEET touch, not the body centre.  So the guard is a first-class state, not a clamp.
  * "After crossing the finish line, the robot may not return to the course to collect,
    transport, or place any target items."  Finishing is irreversible and therefore last.
  * "The robot's autonomous return to an operational state shall not be counted as a reset."
    Recovery is worth 10 points every time it works, so a stuck robot tries to free itself
    before anything else.
"""
from __future__ import annotations

import enum
import time

import numpy as np

from onboard.control.states import State as LegState_


class Phase(enum.Enum):
    WAIT = "wait"            # standing, whistle not blown
    DRIVE = "drive"          # following the corridor
    RECOVER_NAV = "recover_nav"   # corridor estimate lost; search rather than guess
    EDGE = "edge"            # too close to the boundary; steer in hard
    STUCK = "stuck"          # no progress; back out
    COLLECT = "collect"      # handed to the collector, if one is attached
    DELIVER = "deliver"      # carrying to the bins
    FINISH = "finish"        # crossing the line; irreversible
    DONE = "done"


class MissionFSM:
    def __init__(self, sup, nav, odom, budget, mission_cfg) -> None:
        self.sup = sup
        self.nav = nav
        self.odom = odom
        self.budget = budget
        self.cfg = mission_cfg

        loco = mission_cfg["locomotion"]
        self.v_nom = float(loco["nominal_vx"])
        self.v_stair = float(loco["stair_vx"])
        course = mission_cfg["course"]
        self.foot_guard = float(course["foot_guard_half_width"])
        self.stance_half = float(course["stance_half_width"])

        self.phase = Phase.WAIT
        self.t_phase = time.monotonic()
        self.collector = None

        self.oob_events = 0
        self._oob_active = False
        self.stuck_events = 0
        self._progress_ref = 0.0
        self._progress_t = time.monotonic()
        self.transitions: list[tuple[float, str, str, str]] = []

    # ---- wiring --------------------------------------------------------------------------
    def attach_collector(self, collector) -> None:
        """A collector exposes `want(budget) -> bool`, `step() -> Phase|None` and
        `carrying() -> int`.  Keeping it behind an interface means Task 1 can be finished,
        tested and frozen while Task 2 is still moving."""
        self.collector = collector

    def _to(self, phase: Phase, why: str = "") -> None:
        if phase is self.phase:
            return
        self.transitions.append((self.budget.elapsed(), self.phase.value, phase.value, why))
        self.phase = phase
        self.t_phase = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.t_phase

    # ---- the tick --------------------------------------------------------------------------
    def step(self) -> Phase:
        # The supervisor is the authority on whether the legs may move at all.  If it has
        # damped or faulted, the mission has no business issuing a velocity.
        if self.sup.state in (LegState_.DAMP, LegState_.FAULT):
            self.sup.set_command(np.zeros(3))
            return self.phase
        if self.sup.state is LegState_.RECOVER:
            return self.phase          # self-righting owns the robot until it is done

        if not self.budget.started:
            self.sup.set_command(np.zeros(3))
            return self.phase
        if self.budget.expired():
            self._to(Phase.DONE, "30 minutes elapsed")
            self.sup.set_command(np.zeros(3))
            return self.phase

        {
            Phase.WAIT: self._wait,
            Phase.DRIVE: self._drive,
            Phase.RECOVER_NAV: self._recover_nav,
            Phase.EDGE: self._edge,
            Phase.STUCK: self._stuck,
            Phase.COLLECT: self._collect,
            Phase.DELIVER: self._deliver,
            Phase.FINISH: self._finish,
            Phase.DONE: self._done,
        }[self.phase]()
        return self.phase

    # ---- phases ------------------------------------------------------------------------------
    def _wait(self) -> None:
        self.sup.set_command(np.zeros(3))
        if self.budget.started:
            self._progress_ref = self.odom.distance
            self._progress_t = time.monotonic()
            self._to(Phase.DRIVE, "whistle")

    def _drive(self) -> None:
        if self._check_edge() or self._check_stuck():
            return
        if self.nav.lost:
            self._to(Phase.RECOVER_NAV, "corridor estimate lost")
            return
        if self.collector is not None and self.budget.may_collect() \
                and self.collector.want(self.budget):
            self._to(Phase.COLLECT, "target in reach")
            return
        if not self.budget.may_collect() and self._carrying():
            self._to(Phase.DELIVER, "reserve reached")
            return
        self.sup.set_command(self.nav.steer(self._speed()))

    def _speed(self) -> float:
        """Slow down where the estimate is poor.

        The corridor estimate's own sigma is the honest speed governor: on the stairs the
        belt detector degrades, sigma grows, and driving slower there both reduces the
        distance travelled per bad fix and gives the filter more fixes per metre.
        """
        sigma = float(self.nav.fuser.lat_sigma)
        if sigma > 0.6:
            return self.v_stair
        return self.v_nom if sigma < 0.3 else 0.5 * (self.v_nom + self.v_stair)

    def _check_edge(self) -> bool:
        """Out of bounds is judged on the feet.  The body-centre number that the simulator
        scored with is optimistic by a stance half-width, and each violation is 10 points
        with no limit on how many you can collect."""
        lat = abs(self.nav.lat) + self.stance_half
        if lat > self.foot_guard:
            if not self._oob_active:
                self.oob_events += 1
                self._oob_active = True
            self._to(Phase.EDGE, f"feet at {lat:.2f} m of {self.foot_guard:.2f}")
            return True
        self._oob_active = False
        return False

    def _edge(self) -> None:
        # Steer in hard and slowly; do not stop, because standing still next to the boundary
        # accumulates nothing but clock.
        side = -np.sign(self.nav.lat) if abs(self.nav.lat) > 1e-3 else 1.0
        self.sup.set_command(np.array([0.25, 0.0, 0.6 * side]))
        if abs(self.nav.lat) + self.stance_half < self.foot_guard - 0.15:
            self._to(Phase.DRIVE, "back inside")
        elif self.elapsed > 8.0:
            self._to(Phase.RECOVER_NAV, "edge recovery timed out")

    def _check_stuck(self) -> bool:
        if time.monotonic() - self._progress_t < 6.0:
            return False
        moved = self.odom.distance - self._progress_ref
        self._progress_ref = self.odom.distance
        self._progress_t = time.monotonic()
        if moved < 0.3:
            self.stuck_events += 1
            self._to(Phase.STUCK, f"{moved:.2f} m in 6 s")
            return True
        return False

    def _stuck(self) -> None:
        """Back out, then turn.  The one thing not to do is push harder: the recorded
        failure here is a robot wedged against a belt post for eight minutes, and more
        forward command is what kept it there."""
        if self.elapsed < 2.0:
            self.sup.set_command(np.array([-0.25, 0.0, 0.0]))
        elif self.elapsed < 4.0:
            self.sup.set_command(np.array([0.0, 0.0, 0.5]))
        else:
            self._progress_ref = self.odom.distance
            self._progress_t = time.monotonic()
            self._to(Phase.DRIVE, "backed out")

    def _recover_nav(self) -> None:
        """No usable corridor estimate.  Creep forward with the last known heading and let
        the filter re-acquire; do not turn, because turning without a reference is how a
        lost robot ends up facing back down the course."""
        self.sup.set_command(np.array([0.2, 0.0, 0.0]))
        if not self.nav.lost:
            self._to(Phase.DRIVE, "corridor re-acquired")
        elif self.elapsed > 20.0:
            # Twenty seconds blind is 4 m. Keep going rather than stopping: the clock does
            # not stop and a stationary robot scores nothing.
            self._to(Phase.DRIVE, "proceeding without a fix")

    def _collect(self) -> None:
        nxt = self.collector.step() if self.collector is not None else None
        if nxt is not None:
            self._to(nxt, "collector")
        elif not self.budget.may_collect():
            self._to(Phase.DELIVER, "reserve reached")

    def _deliver(self) -> None:
        nxt = self.collector.step() if self.collector is not None else None
        if nxt is not None:
            self._to(nxt, "collector")
        elif not self._carrying():
            self._to(Phase.DRIVE, "hands empty")

    def _finish(self) -> None:
        # Crossing ends collection permanently, so this is only ever entered once and only
        # when there is nothing left worth doing.
        self.sup.set_command(np.array([self.v_nom, 0.0, 0.0]))
        if self.elapsed > 6.0:
            self._to(Phase.DONE, "finish line crossed")

    def _done(self) -> None:
        self.sup.set_command(np.zeros(3))

    def _carrying(self) -> int:
        return 0 if self.collector is None else int(self.collector.carrying())

    # ---- reporting -----------------------------------------------------------------------------
    def report(self) -> dict:
        return {
            "phase": self.phase.value,
            "oob_events": self.oob_events,
            "stuck_events": self.stuck_events,
            "distance_m": round(self.odom.distance, 2),
            "transitions": [{"t": round(t, 2), "from": a, "to": b, "why": w}
                            for t, a, b, w in self.transitions],
            "nav": self.nav.report(),
            "budget": self.budget.report(),
        }
