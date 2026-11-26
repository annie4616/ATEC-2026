"""Joint-limit, rate and velocity clamps on outgoing targets.

Neither reference deploy stack has this layer.  unitree_rl_gym computes
`target = default + action * scale` and puts it straight on the wire -- no joint clamp, no
rate clamp, no sanity check.  In simulation that is harmless because IsaacLab silently
saturates; on hardware the driver's response to an out-of-rate or out-of-range target is a
FAULT, and a faulted joint mid-course is a fall, a manual assist and -10 points.

Everything here is a clamp with a counter.  The counter matters as much as the clamp: a
limiter that is quietly active every step is telling you the policy and the robot disagree,
and you want to read that in the log rather than infer it from a limp gait.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClampReport:
    """How often each guard fired.  Dumped at the end of a run and by selftest."""
    limit_hits: np.ndarray
    rate_hits: np.ndarray
    n: int = 0
    worst_rate: float = 0.0
    worst_limit_excess: float = 0.0
    nonfinite: int = 0

    @staticmethod
    def zeros(n_joint: int) -> "ClampReport":
        return ClampReport(limit_hits=np.zeros(n_joint, int), rate_hits=np.zeros(n_joint, int))

    def summary(self, names: list[str] | None = None) -> dict:
        d = {"steps": self.n, "nonfinite": self.nonfinite,
             "worst_rate_rad_per_step": round(float(self.worst_rate), 4),
             "worst_limit_excess_rad": round(float(self.worst_limit_excess), 4),
             "limit_hits": int(self.limit_hits.sum()), "rate_hits": int(self.rate_hits.sum())}
        if names is not None:
            d["per_joint"] = {n: {"limit": int(self.limit_hits[i]), "rate": int(self.rate_hits[i])}
                              for i, n in enumerate(names) if
                              self.limit_hits[i] or self.rate_hits[i]}
        return d


class TargetLimiter:
    """Clamp a joint-position target against soft limits and a per-step rate cap.

    Order is deliberate: sanitise -> rate-clamp against the LAST COMMAND -> position-clamp.
    Rate-clamping against the last *command* rather than the last *measurement* is what
    keeps the limiter from fighting a joint that is being physically held back: if we
    rate-limited against measurement, a blocked leg would drag the target down with it and
    the policy would never build the effort to free itself.
    """

    def __init__(self, lower: np.ndarray, upper: np.ndarray, max_delta, soft: float = 1.0,
                 names: list[str] | None = None) -> None:
        lower = np.asarray(lower, float)
        upper = np.asarray(upper, float)
        if lower.shape != upper.shape:
            raise ValueError("limit arrays must have the same shape")
        mid = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower) * float(soft)
        self.lo = mid - half
        self.hi = mid + half
        # Scalar or per-joint.  Per-joint is the useful case: a quadruped's calf and hip
        # have different velocity ratings, so one number is wrong for one of them.
        self.max_delta = np.broadcast_to(np.asarray(max_delta, float), lower.shape).copy()
        self.names = names
        self.n = lower.size
        self.report = ClampReport.zeros(self.n)
        self._prev: np.ndarray | None = None

    def reset(self, q: np.ndarray | None = None) -> None:
        """Re-anchor the rate clamp.  Call on every state change, otherwise the first tick
        after a transition is rate-limited against a target from another regime."""
        self._prev = None if q is None else np.asarray(q, float).copy()

    def __call__(self, target: np.ndarray) -> np.ndarray:
        t = np.asarray(target, float).copy()

        # A NaN reaching the motors is a fall.  It should be impossible, so if it happens
        # we hold the last good command and count it rather than propagate it.
        bad = ~np.isfinite(t)
        if bad.any():
            self.report.nonfinite += int(bad.sum())
            t[bad] = self._prev[bad] if self._prev is not None else np.clip(0.0, self.lo, self.hi)[bad]

        if self._prev is not None:
            d = t - self._prev
            over = np.abs(d) > self.max_delta
            if over.any():
                self.report.rate_hits += over
                self.report.worst_rate = max(self.report.worst_rate, float(np.abs(d).max()))
            t = self._prev + np.clip(d, -self.max_delta, self.max_delta)

        excess = np.maximum(self.lo - t, t - self.hi)
        hit = excess > 0
        if hit.any():
            self.report.limit_hits += hit
            self.report.worst_limit_excess = max(self.report.worst_limit_excess,
                                                 float(excess.max()))
        t = np.clip(t, self.lo, self.hi)

        self._prev = t
        self.report.n += 1
        return t


class ScalarLimiter:
    """The same guard for a single axis -- the gripper stroke."""

    def __init__(self, lo: float, hi: float, max_delta: float) -> None:
        self.lo, self.hi, self.max_delta = float(lo), float(hi), float(max_delta)
        self._prev: float | None = None
        self.rate_hits = 0
        self.limit_hits = 0

    def reset(self, v: float | None = None) -> None:
        self._prev = None if v is None else float(v)

    def __call__(self, v: float) -> float:
        x = float(v)
        if not np.isfinite(x):
            x = self._prev if self._prev is not None else self.lo
        if self._prev is not None:
            d = x - self._prev
            if abs(d) > self.max_delta:
                self.rate_hits += 1
            x = self._prev + float(np.clip(d, -self.max_delta, self.max_delta))
        if x < self.lo or x > self.hi:
            self.limit_hits += 1
        x = float(np.clip(x, self.lo, self.hi))
        self._prev = x
        return x


def velocity_watchdog(dq: np.ndarray, limit: np.ndarray, margin: float = 1.1) -> np.ndarray:
    """Which joints are running past their rating.

    This never modifies a command -- by the time a joint is over-speed the cause is upstream.
    It exists so the supervisor can turn "the arm is being driven past spec" into a logged
    fault instead of a driver-side fault we only learn about from a stopped robot.
    """
    return np.abs(np.asarray(dq, float)) > (np.asarray(limit, float) * float(margin))
