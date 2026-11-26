"""The start gate: nothing moves until the whistle.

"Any robot movement before the whistle shall invalidate the current attempt. Three
accumulated false starts shall result in disqualification."  And separately: "the robot must
operate fully autonomously and shall not use teleoperation", with the definition "except for
the initial startup procedure, the robot must complete the entire task workflow
autonomously".

Those two together are the whole design.  The robot has to be powered, standing, sensing and
one signal away from running for up to five minutes, and that signal has to be an *initial
startup* action rather than a task command.  So:

  * the gate latches ONCE and is then irreversible for the run;
  * arming is two-stage (hold, then confirm) so a knocked controller cannot start the run;
  * after the latch nothing reads a stick again -- `LegSupervisor.allow_run` is set and the
    remote's only remaining role is abort.

`ManualTrigger` is the default: a human presses the arm combination on the controller when
the whistle blows, which is a startup action, in the open, and identical in kind to pressing
"go" on a laptop.  `TimerTrigger` exists for rehearsal, where there is no whistle.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field


class GateState(enum.Enum):
    IDLE = "idle"          # powered, not yet ready
    READY = "ready"        # standing, self-test passed, waiting for the whistle
    ARMING = "arming"      # confirmation combination is being held
    STARTED = "started"    # latched; the run has begun


@dataclass
class StartGate:
    # Both buttons held together for hold_s.  A single button is one bump away from a
    # false start, and a false start is the most expensive cheap mistake available.
    combo: tuple[str, ...] = ("L2", "start")
    hold_s: float = 1.0
    state: GateState = GateState.IDLE
    t_started: float | None = None
    _held_since: float | None = None
    log: list[tuple[float, str]] = field(default_factory=list)

    def ready(self) -> None:
        """Self-test passed and the robot is standing: from here the whistle can start us."""
        if self.state is GateState.IDLE:
            self.state = GateState.READY
            self.log.append((time.monotonic(), "ready"))

    def update(self, remote) -> bool:
        """-> True on the tick the run latches.  Idempotent afterwards."""
        if self.state is GateState.STARTED:
            return False
        if self.state is GateState.IDLE:
            return False

        pressed = remote is not None and remote.ok and all(remote.pressed(b) for b in self.combo)
        now = time.monotonic()
        if not pressed:
            if self.state is GateState.ARMING:
                self.state = GateState.READY
                self.log.append((now, "arming aborted"))
            self._held_since = None
            return False

        if self._held_since is None:
            self._held_since = now
            self.state = GateState.ARMING
            self.log.append((now, f"arming: {'+'.join(self.combo)}"))
            return False

        if now - self._held_since >= self.hold_s:
            self.state = GateState.STARTED
            self.t_started = now
            self.log.append((now, "STARTED"))
            return True
        return False

    @property
    def started(self) -> bool:
        return self.state is GateState.STARTED

    def elapsed(self) -> float:
        return 0.0 if self.t_started is None else time.monotonic() - self.t_started


@dataclass
class TimerTrigger:
    """Rehearsal only: start after a fixed delay.  Never use at the venue -- it cannot see
    the whistle, and a start one second early is a voided attempt."""
    delay_s: float = 3.0
    _t0: float | None = None
    state: GateState = GateState.IDLE
    t_started: float | None = None
    log: list[tuple[float, str]] = field(default_factory=list)

    def ready(self) -> None:
        if self.state is GateState.IDLE:
            self.state = GateState.READY
            self._t0 = time.monotonic()
            self.log.append((self._t0, f"ready; auto-start in {self.delay_s}s"))

    def update(self, remote=None) -> bool:
        if self.state is not GateState.READY or self._t0 is None:
            return False
        if time.monotonic() - self._t0 >= self.delay_s:
            self.state = GateState.STARTED
            self.t_started = time.monotonic()
            self.log.append((self.t_started, "STARTED (timer)"))
            return True
        return False

    @property
    def started(self) -> bool:
        return self.state is GateState.STARTED

    def elapsed(self) -> float:
        return 0.0 if self.t_started is None else time.monotonic() - self.t_started
