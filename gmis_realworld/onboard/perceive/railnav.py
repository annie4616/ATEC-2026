"""Corridor from the LiDAR: lateral offset and heading relative to the red guide strips.

Ported from `deploy/walk_course.py::_detect_rail`, which was already written to be
deployment-equivalent: it uses only the sensor's own frame, estimates the ground height from
the cloud itself, and never touches a world pose or a course map.  That makes the port a
change of input plumbing rather than a rewrite, and it means the behaviour has already been
watched in simulation.

Why it earns its place next to the camera belt detector rather than replacing it:

  * The camera detector fits red PIXELS.  It measured 52 % fix rate, 262 mm / 5.4 deg median
    -- and a p90 tail of 1.8 m / 24.8 deg, because at the stairs the belt's projection onto
    a tilted ground plane stops being a straight line in the image.
  * The LiDAR sees RANGE.  It does not care what colour the strip is, does not care about
    lighting, and does not need the ground-plane assumption that breaks on the stairs.
  * But it cannot tell a red guide strip from any other object in the same height band,
    which is why the linearity gate below exists and why the camera stays in the fusion.

Both feed the same EKF (`estimate/navstate.py`), which is the only place they are combined.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RailFix:
    lat: float          # signed distance to the near rail, metres, + when it is on the left
    heading: float      # yaw to ADD to become parallel to it; 0 = aligned
    n_left: int
    n_right: int
    halfwidth: float | None = None   # only when both rails are seen
    residual: float = 0.0

    # Intercepts of each fitted rail in the robot frame (+left), when that side was seen.
    b_left: float | None = None
    b_right: float | None = None

    @property
    def both(self) -> bool:
        return self.b_left is not None and self.b_right is not None

    @property
    def offset(self) -> float | None:
        """Lateral offset from the CORRIDOR CENTRELINE, +left -- navfuse's `w`.

        `lat` is the distance to the nearest rail, which is what you steer on when only one
        rail is visible; it is not the same quantity and mixing them puts the robot half a
        corridor width off.  The centreline needs both rails, so this is None when only one
        was found and the caller must fall back to a rail-following gain.
        """
        if not self.both:
            return None
        return -0.5 * (self.b_left + self.b_right)


class RailDetector:
    def __init__(self, sensors_cfg) -> None:
        r = (sensors_cfg.get("lidar") or {}).get("rail") or {}
        self.z_lo = float(r.get("z_lo", 0.20))
        self.z_hi = float(r.get("z_hi", 0.55))
        self.forward = float(r.get("forward_m", 3.0))
        self.max_resid = float(r.get("max_residual_m", 0.20))
        self.min_points = int(r.get("min_points", 4))
        self.ground_q = float(r.get("ground_quantile", 0.10))

    def detect(self, points: np.ndarray) -> RailFix | None:
        """Cloud in the sensor's own frame (x forward, y left, z up) -> RailFix or None."""
        P = np.asarray(points, dtype=np.float64)
        if P.ndim != 2 or P.shape[0] < 20:
            return None
        fin = np.isfinite(P).all(axis=1)
        if int(fin.sum()) < 20:
            return None
        P = P[fin]
        rx, ry, rz = P[:, 0], P[:, 1], P[:, 2]

        # Ground height from the cloud, not from a state estimate.  The 10th percentile is
        # robust to the strips themselves and to a few stray returns, and it is the reason
        # this works on a slope: the band is measured against the local ground, not a global
        # z that the slope has already invalidated.
        ground = float(np.quantile(rz, self.ground_q))
        dz = rz - ground
        m = (dz > self.z_lo) & (dz < self.z_hi) & (np.abs(rx) < self.forward)
        if int(m.sum()) < self.min_points:
            return None

        xm, ym = rx[m], ry[m]
        left = self._assign(xm, ym)
        n_left, n_right = int(left.sum()), int((~left).sum())

        fL, rL = self._fit(xm[left], ym[left])
        fR, rR = self._fit(xm[~left], ym[~left])
        cands = [(f, r) for f, r in ((fL, rL), (fR, rR)) if f is not None]
        if not cands:
            return None

        # The NEAR rail is the one we steer against: smaller |intercept|.
        (a, b), resid = min(cands, key=lambda c: abs(c[0][1]))
        half = abs(fL[1] - fR[1]) / 2.0 if (fL is not None and fR is not None) else None
        return RailFix(lat=float(b), heading=float(np.arctan(a)),
                       n_left=n_left, n_right=n_right, halfwidth=half, residual=float(resid),
                       b_left=None if fL is None else float(fL[1]),
                       b_right=None if fR is None else float(fR[1]))

    def _assign(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Split the returns into the left and the right rail.

        The obvious split -- sign(y) -- is what the original does, and it is correct only
        while both rails stay on their own side of the camera axis.  They do not.  With the
        robot 0.6 m off centre and 0.2 rad askew, the near rail crosses y = 0 inside the
        3 m window, so its far half is labelled as the other rail; both fits are then
        contaminated and the corridor centre lands roughly half a corridor away.  That
        happens precisely when the robot is already off course and the fix matters most.

        So: seed with sign(y), fit, then reassign each point to whichever fitted line it is
        actually closer to and let the caller refit.  One iteration is enough because the
        seed is only wrong for the tail of one rail, and it degrades to the original when
        a side is too sparse to fit.
        """
        left = y > 0
        if int(left.sum()) < 3 or int((~left).sum()) < 3:
            return left
        fL, _ = self._fit(x[left], y[left], gate=False)
        fR, _ = self._fit(x[~left], y[~left], gate=False)
        if fL is None or fR is None:
            return left
        dL = np.abs(y - (fL[0] * x + fL[1]))
        dR = np.abs(y - (fR[0] * x + fR[1]))
        new = dL < dR
        # If reassignment collapses one side, the two fits were the same line -- keep the
        # seed rather than reporting a corridor of width zero.
        if int(new.sum()) < 3 or int((~new).sum()) < 3:
            return left
        return new

    def _fit(self, x: np.ndarray, y: np.ndarray, gate: bool = True):
        """Least-squares y = a x + b, rejected if it is not actually a line.

        The linearity gate is the whole trick.  A real guide strip is a thin ribbon and fits
        a line with a small residual; a stair riser caught by the same height band is a
        broad slab and does not.  Without this the detector locks onto the steps exactly
        where the camera detector is also failing, which is the worst possible time to have
        two sensors agree on the wrong answer.
        """
        if x.size < 3:
            return None, 0.0
        xc = x - x.mean()
        den = float((xc * xc).sum())
        if den < 1e-6:
            return None, 0.0
        a = float((xc * (y - y.mean())).sum() / den)
        b = float(y.mean() - a * x.mean())
        resid = float(np.std(y - (a * x + b)))
        if gate and x.size >= 5 and resid > self.max_resid:
            return None, resid
        return (a, b), resid
