"""Drift-free control cadence, plus the two OS knobs that make it hold on an N150.

The reference Python deploy loop is `do_work(); time.sleep(control_dt)`, so its real period
is `control_dt + compute_time` and it runs permanently slow.  For a policy whose observation
includes `q̇ * 0.05` and whose action history is indexed in control steps, "50 Hz" quietly
becoming 46 Hz is a change to the dynamics the policy was trained on.  The C++ sibling gets
this right with an absolute deadline, and that is what this is.

The N150 has four E-cores and no SMT.  Perception can and will saturate them, so the control
thread asks for SCHED_FIFO and its own core.  Both requests are best-effort: without
CAP_SYS_NICE they fail, and a loop that runs at normal priority is much better than a loop
that refuses to start.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LoopStats:
    """Jitter is a safety property here: a late tick is a late motor command."""
    period: float
    n: int = 0
    late: int = 0
    overrun: int = 0
    _dt: list[float] = field(default_factory=list)
    _work: list[float] = field(default_factory=list)

    def add(self, dt: float, work: float) -> None:
        self.n += 1
        self._dt.append(dt)
        self._work.append(work)
        if dt > self.period * 1.5:
            self.late += 1
        if work > self.period:
            self.overrun += 1
        if len(self._dt) > 200_000:          # 66 min at 50 Hz; bound the memory
            del self._dt[:100_000]
            del self._work[:100_000]

    def summary(self) -> dict:
        if not self._dt:
            return {"n": 0}
        dt = np.asarray(self._dt) * 1e3
        wk = np.asarray(self._work) * 1e3
        return {
            "n": self.n, "late": self.late, "overrun": self.overrun,
            "hz": 1e3 / float(np.median(dt)),
            "dt_ms_p50": float(np.percentile(dt, 50)),
            "dt_ms_p99": float(np.percentile(dt, 99)),
            "dt_ms_max": float(dt.max()),
            "jitter_ms_p99": float(np.percentile(np.abs(dt - self.period * 1e3), 99)),
            "work_ms_p50": float(np.percentile(wk, 50)),
            "work_ms_p99": float(np.percentile(wk, 99)),
        }

    def __str__(self) -> str:
        s = self.summary()
        if not s.get("n"):
            return "loop: no samples"
        return (f"loop: n={s['n']} {s['hz']:.2f} Hz  dt p50/p99/max "
                f"{s['dt_ms_p50']:.2f}/{s['dt_ms_p99']:.2f}/{s['dt_ms_max']:.2f} ms  "
                f"work p50/p99 {s['work_ms_p50']:.2f}/{s['work_ms_p99']:.2f} ms  "
                f"late={s['late']} overrun={s['overrun']}")


class RateLoop:
    """Absolute-deadline pacer.

        loop = RateLoop(50.0)
        for _ in loop:
            step()

    A tick that overruns does not push every later tick back: the deadline is advanced by
    whole periods until it is in the future, so the loop drops a slot instead of drifting.
    """

    def __init__(self, hz: float, name: str = "loop") -> None:
        self.period = 1.0 / float(hz)
        self.name = name
        self.stats = LoopStats(self.period)
        self._next = 0.0
        self._last = 0.0
        self.running = True

    def start(self) -> None:
        self._next = time.monotonic() + self.period
        self._last = time.monotonic()

    def stop(self) -> None:
        self.running = False

    def __iter__(self):
        if self._next == 0.0:
            self.start()
        while self.running:
            yield self.tick()

    def tick(self) -> float:
        """Sleep to the next deadline; return the actual elapsed period."""
        now = time.monotonic()
        work = now - self._last
        delay = self._next - now
        if delay > 0:
            time.sleep(delay)
        else:
            # Behind.  Skip whole periods rather than trying to catch up by running fast --
            # a burst of back-to-back motor commands is worse than one missed slot.
            skipped = int(-delay // self.period) + 1
            self._next += skipped * self.period
        t = time.monotonic()
        dt = t - self._last
        self._last = t
        self._next += self.period
        self.stats.add(dt, work)
        return dt


def set_realtime(priority: int = 80, core: int | None = None) -> dict:
    """Best-effort SCHED_FIFO + core affinity.  Returns what actually took effect, so
    bench/selftest can report "asked for RT, did not get it" instead of assuming."""
    got = {"sched_fifo": False, "priority": None, "core": None, "errors": []}
    try:
        param = os.sched_param(priority)
        os.sched_setscheduler(0, os.SCHED_FIFO, param)
        got["sched_fifo"] = True
        got["priority"] = priority
    except (PermissionError, OSError, AttributeError) as e:
        got["errors"].append(f"sched_setscheduler: {e}")
    if core is not None:
        try:
            os.sched_setaffinity(0, {core})
            got["core"] = core
        except (OSError, AttributeError) as e:
            got["errors"].append(f"sched_setaffinity: {e}")
    return got


def pin_perception(cores: set[int] | None = None) -> dict:
    """Keep the perception process off the control core.  Default: everything but core 0."""
    got = {"cores": None, "errors": []}
    try:
        # NOT `os.sched_getaffinity(0) - {0}`: this runs in the spawned child, which inherits
        # the parent's mask.  Once the parent had pinned itself to core 0 the child's
        # `avail - {0}` was empty and the `or avail` fallback handed it straight back to
        # core 0 -- the one function whose job is to keep perception OFF the control core
        # was pinning it ON.  Derive from the machine instead of from what we inherited.
        avail = set(range(os.cpu_count() or 1))
        target = cores if cores is not None else (avail - {0} or avail)
        os.sched_setaffinity(0, target)
        got["cores"] = sorted(target)
    except (OSError, AttributeError) as e:
        got["errors"].append(str(e))
    return got
