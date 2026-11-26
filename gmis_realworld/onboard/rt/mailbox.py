"""Age-stamped hand-off between the perception process and the control loop.

Perception costs 190-550 ms per object-detection frame on an N150; the control loop owes
the motors a command every 20 ms.  They cannot share a thread, and once they do not, the
control loop is always reading something that was true a little while ago.

The design rule is that the consumer must never be able to forget that.  So a mailbox slot
carries the time and the LowState tick it was computed against, `age()` is the natural way
to read it, and `get()` returns None past a deadline rather than the last known value.  The
alternative -- a plain shared variable -- is how a robot ends up steering confidently on a
detection from four seconds and three metres ago.

Single producer, single consumer, one slot: the consumer wants the NEWEST result, never a
backlog, so a queue would be the wrong shape.  A lock is enough at these rates and is
easier to reason about than a lock-free swap.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class Stamped:
    t: float
    tick: int
    seq: int
    value: Any

    def age(self, now: float | None = None) -> float:
        return (time.monotonic() if now is None else now) - self.t


class Mailbox:
    """Latest-value slot with an age.  `max_age_s` is the default staleness deadline."""

    def __init__(self, name: str, max_age_s: float = 1.5) -> None:
        self.name = name
        self.max_age = float(max_age_s)
        self._lock = threading.Lock()
        self._slot: Stamped | None = None
        self._seq = 0
        self.n_put = 0
        self.n_dropped = 0          # overwritten before anyone read it
        self.n_stale = 0
        self._read_seq = -1

    def put(self, value: Any, t: float | None = None, tick: int = 0) -> None:
        with self._lock:
            if self._slot is not None and self._slot.seq != self._read_seq:
                self.n_dropped += 1
            self._seq += 1
            self._slot = Stamped(t=time.monotonic() if t is None else t, tick=tick,
                                 seq=self._seq, value=value)
            self.n_put += 1

    def get(self, max_age_s: float | None = None) -> Stamped | None:
        """Newest value if it is fresh enough, else None.  Refusing to answer is a valid
        answer and the caller has to handle it -- that is the point."""
        limit = self.max_age if max_age_s is None else float(max_age_s)
        with self._lock:
            s = self._slot
            if s is None:
                return None
            self._read_seq = s.seq
            if s.age() > limit:
                self.n_stale += 1
                return None
            return s

    def peek(self) -> Stamped | None:
        """The slot regardless of age -- for logging and for code that wants to say 'the
        last fix was 4.2 s ago' rather than 'there is no fix'."""
        with self._lock:
            return self._slot

    def report(self) -> dict:
        with self._lock:
            s = self._slot
            return {"name": self.name, "put": self.n_put, "dropped": self.n_dropped,
                    "stale_reads": self.n_stale,
                    "age_s": None if s is None else round(s.age(), 3)}
