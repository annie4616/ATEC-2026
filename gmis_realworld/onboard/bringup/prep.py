"""The five-minute preparation sequence.

"Each team will have 5 minutes of preparation time ... used for robot initialization,
including system startup, program launch, sensor checks, motor activation, communication
verification, and standing/in-place motion tests.  Without prior permission from the Lead
Judge, the robot must remain within the starting area and must not begin any competition
tasks during the preparation period."

So this is a scored procedure with a hard deadline, not a convenience script.  It brings
everything up in dependency order, times each stage, and ends in a state that can wait
indefinitely for the whistle:

    link -> sensors -> motors -> stand -> zero the estimator -> self-test -> READY

Two things it deliberately does not do.  It does not start the mission -- that is
`startgate`, and separating them is what makes a false start structurally impossible rather
than merely unlikely.  And it does not proceed past a FAIL: five minutes is enough to fix a
cable, and it is not enough to debug a run that started broken.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from onboard.bringup.selftest import R, SelfTest, run as run_selftest
from onboard.control.states import State


@dataclass
class Stage:
    name: str
    seconds: float
    ok: bool
    detail: str = ""


@dataclass
class PrepResult:
    stages: list[Stage] = field(default_factory=list)
    selftest: SelfTest | None = None
    total_s: float = 0.0

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.stages) and (self.selftest is None or self.selftest.ok)

    def render(self, budget_s: float = 300.0) -> str:
        lines = [f"  {'OK ' if s.ok else 'FAIL'}  {s.name:<22s} {s.seconds:6.2f} s  {s.detail}"
                 for s in self.stages]
        lines.append(f"  ---- prep {self.total_s:.1f} s of {budget_s:.0f} s budget"
                     f"  ({budget_s - self.total_s:.0f} s spare)")
        if self.selftest is not None:
            lines.append(self.selftest.render())
        return "\n".join(lines)


class Prep:
    def __init__(self, supervisor, legs, arm=None, perception=None, attitude=None,
                 odom=None, estop=None, budget_s: float = 300.0) -> None:
        self.sup = supervisor
        self.legs = legs
        self.arm = arm
        self.perception = perception
        self.attitude = attitude
        self.odom = odom
        self.estop = estop
        self.budget_s = float(budget_s)

    def run(self, tick, stand_timeout_s: float = 20.0) -> PrepResult:
        """`tick` advances the control loop by one step -- the caller owns the cadence, so
        prep runs inside the same 50 Hz loop that will run the mission rather than in a
        parallel world with different timing."""
        res = PrepResult()
        t_all = time.monotonic()

        with _stage(res, "communication") as s:
            ok = self.legs.wait_for_state(5.0)
            s.ok = ok
            s.detail = "LowState streaming" if ok else "no LowState -- check NIC and msg_type"
        if not res.ok:
            return self._finish(res, t_all)

        with _stage(res, "sensors") as s:
            # Perception is already RUNNING by the time prep starts -- see run.py.  Spawning
            # it here would cost a second of process startup inside the control loop, which
            # is a second of stale motor commands while the robot is standing up.  Prep's
            # job is to confirm it is producing, not to bring it up.
            if self.perception is None:
                s.ok, s.detail = True, "no perception configured"
            else:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline and self.perception.belt.peek() is None \
                        and self.perception.rail.peek() is None:
                    tick()
                got = (self.perception.belt.peek() is not None
                       or self.perception.rail.peek() is not None
                       or self.perception.objects.peek() is not None)
                # A camera that produces nothing is a degraded run, not a dead one: the
                # robot can still walk the course blind on proprioception.
                s.ok = True
                s.detail = "frames flowing" if got else "NO detections yet -- degraded"

        with _stage(res, "motors") as s:
            self.sup.arm()
            s.ok = self.sup.state in (State.RAMP, State.STAND)
            s.detail = self.sup.state.value

        with _stage(res, "stand") as s:
            # Count distinct LowState ticks while standing: two seconds of ramp at 50 Hz is
            # a hundred samples, free, and it measures the link under the load it will
            # actually see rather than in an artificial burst.
            deadline = time.monotonic() + stand_timeout_s
            t_rate, seen = time.monotonic(), set()
            while time.monotonic() < deadline and self.sup.state is not State.STAND:
                tick()
                seen.add(self.legs.read().tick)
                if self.sup.state in (State.DAMP, State.FAULT):
                    break
            span = max(time.monotonic() - t_rate, 1e-6)
            self._lowstate_hz = len(seen) / span if seen else 0.0
            s.ok = self.sup.state is State.STAND
            s.detail = f"{self.sup.state.value}, LowState {self._lowstate_hz:.0f} Hz"

        with _stage(res, "arm") as s:
            if self.arm is None:
                s.ok, s.detail = True, "no arm configured"
            else:
                ok = self.arm.enable()
                # The arm is worth 160 of the 240 points but the hiking 80 does not need it.
                # A dead arm must not stop the run.
                s.ok = True
                s.detail = "enabled" if ok else "NOT enabled -- run will be legs-only"

        with _stage(res, "estimator origin") as s:
            if self.attitude is not None:
                self.attitude.reset(0.0)
            if self.odom is not None:
                self.odom.reset()
            s.ok = True
            s.detail = "heading and odometry zeroed at the start line"

        res.selftest = run_selftest(self.legs, arm=self.arm, perception=self.perception,
                                    estop=self.estop,
                                    lowstate_hz=getattr(self, "_lowstate_hz", None))
        return self._finish(res, t_all)

    def _finish(self, res: PrepResult, t0: float) -> PrepResult:
        res.total_s = time.monotonic() - t0
        if res.total_s > self.budget_s:
            res.stages.append(Stage("BUDGET", res.total_s, False,
                                    f"prep exceeded the {self.budget_s:.0f} s window"))
        return res


class _stage:
    def __init__(self, res: PrepResult, name: str) -> None:
        self.res, self.name = res, name

    def __enter__(self) -> Stage:
        self.t0 = time.monotonic()
        self.s = Stage(self.name, 0.0, False)
        return self.s

    def __exit__(self, *exc) -> bool:
        self.s.seconds = time.monotonic() - self.t0
        if exc[0] is not None:
            self.s.ok = False
            self.s.detail = f"{exc[0].__name__}: {exc[1]}"
        self.res.stages.append(self.s)
        return exc[0] is not None      # a stage failing is a result, not a crash
