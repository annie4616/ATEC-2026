"""The 30-minute wall clock.

"The timer starts at the beginning of the round and will not be stopped, paused, or
extended."  Not for a fall, not for a reset, not for our own network.  So every deadline in
the stack reads `time.monotonic()` and nothing counts control steps -- the simulator's
step-count budget silently becomes wrong the moment a real tick takes 21 ms instead of 20.

The reserve is the part that actually earns points: items in the gripper at the finish line
score zero, so collection has to stop early enough to reach the bins.  And crossing the
finish line permanently ends collection, so finishing is always the last act.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Budget:
    total_s: float = 1800.0
    reserve_for_delivery_s: float = 240.0
    t0: float | None = None
    marks: list[tuple[float, str]] = field(default_factory=list)

    def start(self) -> None:
        self.t0 = time.monotonic()
        self.marks.append((0.0, "start"))

    @property
    def started(self) -> bool:
        return self.t0 is not None

    def elapsed(self) -> float:
        return 0.0 if self.t0 is None else time.monotonic() - self.t0

    def remaining(self) -> float:
        return float("inf") if self.t0 is None else max(0.0, self.total_s - self.elapsed())

    def expired(self) -> bool:
        return self.started and self.remaining() <= 0.0

    def may_collect(self) -> bool:
        """New object attempts only while there is still time to deliver what we take."""
        return self.remaining() > self.reserve_for_delivery_s

    def sane(self) -> str | None:
        """-> a complaint if this budget can never collect anything.

        A reserve at or above the total makes `may_collect()` False from the first tick, so
        the robot walks the whole course and never tries to pick anything up -- with no
        error anywhere.  That is a plausible way to lose a run to a config typo, and it cost
        an afternoon of 'the collector sees nothing' during development.
        """
        if self.reserve_for_delivery_s >= self.total_s:
            return (f"delivery reserve {self.reserve_for_delivery_s:.0f} s >= total budget "
                    f"{self.total_s:.0f} s: nothing will ever be collected")
        return None

    def affordable(self, cost_s: float) -> bool:
        """Is a specific attempt worth starting?  Used to refuse a grasp that cannot finish
        and still leave delivery time -- a half-finished attempt scores the same as none."""
        return (self.remaining() - cost_s) > self.reserve_for_delivery_s

    def mark(self, label: str) -> None:
        self.marks.append((round(self.elapsed(), 2), label))

    def report(self) -> dict:
        return {"elapsed_s": round(self.elapsed(), 2),
                "remaining_s": round(self.remaining(), 2),
                "marks": self.marks}
