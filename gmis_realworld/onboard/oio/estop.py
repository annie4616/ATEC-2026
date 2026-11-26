"""Emergency stop: hardware line, software latch, and the fault register.

The rules require "a reliable hardware or software E-Stop capable of immediately disabling
the robot's power", make the E-Stop link the one cable allowed to leave the robot, and
count an activation as a reset: -10 points, three per run.  So this is a scored component,
not hygiene, and it has two jobs that pull in opposite directions:

  * trip on anything genuinely dangerous, because the alternative is damage and a stopped run
  * do NOT trip on anything recoverable, because every trip costs 10 points and one of three
    lives, and a robot that can right itself scores zero penalty

Hence the split: `Fault` records what happened and how bad it is, and only ABORT-class
faults latch.  A latched fault never clears itself -- clearing is a human action during an
assisted reset, which is exactly the accounting the rules describe.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from pathlib import Path


class Severity(enum.IntEnum):
    INFO = 0
    WARN = 1        # logged, run continues
    DEGRADE = 2     # capability lost (no camera, no arm); mission adapts and keeps going
    ABORT = 3       # damping now, latch, wait for a human


@dataclass
class Fault:
    name: str
    severity: Severity
    detail: str = ""
    t: float = field(default_factory=time.monotonic)
    repeats: int = 0
    t_last: float = 0.0

    def __str__(self) -> str:
        rep = f" (x{self.repeats + 1}, {self.t_last - self.t:.1f}s)" if self.repeats else ""
        return f"[{self.severity.name}] {self.name}: {self.detail}{rep}"


class EStop:
    """Software latch plus an optional hardware line.

    The hardware line is read as a sysfs/gpio file so that whatever the safety hardware
    ends up being -- a GPIO, a relay exposed by a USB board, a file written by a watchdog
    daemon -- it needs no code change here.  Absent file means "no hardware E-Stop wired",
    which is reported by selftest rather than assumed to be fine.
    """

    def __init__(self, gpio_path: str | None = None, active_low: bool = True) -> None:
        self.path = Path(gpio_path) if gpio_path else None
        self.active_low = active_low
        self.latched: Fault | None = None
        self.history: list[Fault] = []
        self._hw_available = self.path is not None and self.path.exists()

    @property
    def hardware_present(self) -> bool:
        return self._hw_available

    def hardware_tripped(self) -> bool:
        if not self._hw_available:
            return False
        try:
            raw = self.path.read_text().strip()
        except OSError:
            # The line was there and now is not.  Losing the ability to read the E-Stop is
            # itself an abort condition -- we can no longer prove the robot is safe.
            self.trip(Fault("estop_line_lost", Severity.ABORT, str(self.path)))
            self._hw_available = False
            return True
        val = raw not in ("0", "", "false", "False")
        return (not val) if self.active_low else val

    def _append(self, fault: Fault) -> None:
        """Collapse repeats.

        A condition like 'no LowState' is true on every one of the 50 ticks per second that
        follow it, and a fault log with 1500 identical lines is a log nobody reads.  Keep
        the first occurrence, count the rest, and keep the last timestamp so the report can
        say how long it persisted.
        """
        if self.history:
            last = self.history[-1]
            if last.name == fault.name and last.detail == fault.detail:
                last.repeats += 1
                last.t_last = fault.t
                return
        self.history.append(fault)

    def trip(self, fault: Fault) -> None:
        self._append(fault)
        if fault.severity >= Severity.ABORT and self.latched is None:
            self.latched = fault

    def record(self, fault: Fault) -> None:
        """Non-latching: WARN/DEGRADE go in the log so the post-run report can explain a
        score without anyone having to remember what happened at minute 14."""
        self._append(fault)

    def clear(self, who: str = "operator") -> None:
        """Only ever called by a human action during an assisted reset."""
        if self.latched is not None:
            self.history.append(Fault("estop_cleared", Severity.INFO, f"by {who}"))
        self.latched = None

    def active(self) -> bool:
        return self.latched is not None or self.hardware_tripped()

    def summary(self) -> dict:
        return {
            "latched": str(self.latched) if self.latched else None,
            "hardware_present": self._hw_available,
            "counts": {s.name: sum(1 + f.repeats for f in self.history if f.severity is s)
                       for s in Severity},
            "faults": [{"t": round(f.t, 3), "name": f.name, "sev": f.severity.name,
                        "detail": f.detail, "repeats": f.repeats,
                        "held_s": round(f.t_last - f.t, 2) if f.repeats else 0.0}
                       for f in self.history],
        }
