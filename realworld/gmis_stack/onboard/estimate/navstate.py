"""Where the robot is in the corridor, without a map.

The simulator answers "where am I" by projecting the true world pose onto a surveyed
S-curve.  At the venue that curve does not exist and neither does the true pose, so the
question has to be replaced with one that on-board sensors can actually answer:

    how far off the corridor centreline am I, and how far from parallel?

That is `navfuse.BeltOdomFuser` -- an EKF over [lateral, heading error, course curvature],
already written, already self-tested, already measured against the alternative (98 mm median
/ 342 mm worst versus a plain EMA's 194 mm / 846 mm).  Curvature is a state rather than a
constant because without it every bend reads as a steering error and the correction fights
the course.

This module is the part that was missing: it feeds that filter.  Two independent corridor
sensors, one filter.

  camera  `perception.detect_belt` -- red pixels intersected with the belt plane.  52 % fix
          rate, 262 mm / 5.4 deg median, but a p90 tail of 1.8 m / 24.8 deg because at the
          stairs the belt stops projecting to a straight line on the assumed ground plane.
  LiDAR   `railnav.RailDetector` -- range, so it does not care about colour, lighting, or
          the ground-plane assumption.  It cannot tell a guide strip from any other object
          in the same height band, which the linearity gate handles.

They fail in different places, which is the entire argument for running both.  The EKF's
normalised-innovation gate then does the arbitration, and it does it on the only question
that generalises: is this fix consistent with where I believe I am, given how sure I am?
"""
from __future__ import annotations

import numpy as np

from course_mission.navfuse import BeltOdomFuser
from onboard.estimate.attitude import Attitude
from onboard.estimate.odom import LegOdometry


class NavState:
    def __init__(self, mission_cfg, sensors_cfg) -> None:
        loco = mission_cfg["locomotion"]
        self.fuser = BeltOdomFuser()
        self.lat_gain = float(loco["lat_gain"])
        self.head_gain = float(loco["head_gain"])
        self.lat_cap = float(loco["lat_cap"])
        self.yaw_cap = float(loco["yaw_cap"])
        self.max_age = float(sensors_cfg["perception"].get("max_age_s", 1.5))

        self.n_cam_fix = 0
        self.n_lidar_fix = 0
        self.n_rejected = 0
        self._last_fix_t = 0.0
        self._t: float | None = None

    # ---- prediction -------------------------------------------------------------------
    def predict(self, t: float, odo: LegOdometry, att: Attitude) -> None:
        """Propagate on proprioception alone.  Never on a map, never on a commanded
        velocity -- the command is what we asked for, and between fixes the difference
        between asked and achieved is exactly the error we are trying to carry."""
        if self._t is None:
            self._t = t
            return
        dt = t - self._t
        self._t = t
        if not (1e-4 < dt < 0.2):
            return
        vx, vy = float(odo.vel_body[0]), float(odo.vel_body[1])
        self.fuser.predict(vx, vy, att.yaw_rate, dt)

    # ---- corrections ------------------------------------------------------------------
    def update_camera(self, belt: dict | None, t: float) -> bool:
        """`detect_belt` output.  Its own `lat` is already the offset from the corridor
        centre, which is the filter's `w` directly."""
        if not belt:
            return False
        ok = self.fuser.update(lat=float(belt["lat"]), heading=float(belt["heading"]),
                               halfwidth=belt.get("halfwidth"))
        self.n_cam_fix += int(ok)
        self.n_rejected += int(not ok)
        if ok:
            self._last_fix_t = t
        return ok

    def update_rail(self, fix, t: float) -> bool:
        """A RailFix from the LiDAR.

        Only the two-rail case supplies a centreline offset; with one rail visible the
        heading is still worth having on its own, and feeding a near-rail distance in as if
        it were a centre offset would bias the filter by half a corridor width -- which is
        the kind of error that walks the robot into the strip it is trying to follow.
        """
        if fix is None:
            return False
        off = fix.offset
        ok = self.fuser.update(lat=None if off is None else float(off),
                               heading=float(fix.heading),
                               halfwidth=fix.halfwidth)
        self.n_lidar_fix += int(ok)
        self.n_rejected += int(not ok)
        if ok:
            self._last_fix_t = t
        return ok

    # ---- consumption ------------------------------------------------------------------
    def steer(self, v_nom: float) -> np.ndarray:
        """(vx, vy, wz) for the walking policy."""
        vx, vy, wz = self.fuser.steer(v_nom=v_nom, lat_gain=self.lat_gain,
                                      head_gain=self.head_gain, lat_cap=self.lat_cap,
                                      yaw_cap=self.yaw_cap)
        return np.array([vx, vy, wz])

    @property
    def lat(self) -> float:
        return float(self.fuser.lat)

    @property
    def heading(self) -> float:
        return float(self.fuser.heading)

    @property
    def lost(self) -> bool:
        return bool(self.fuser.lost())

    def stale(self, t: float) -> bool:
        return (t - self._last_fix_t) > self.max_age

    def report(self) -> dict:
        return {"lat": round(self.lat, 3), "heading": round(self.heading, 4),
                "curvature": round(float(self.fuser.curvature), 4),
                "lat_sigma": round(float(self.fuser.lat_sigma), 3),
                "fixes_camera": self.n_cam_fix, "fixes_lidar": self.n_lidar_fix,
                "rejected": self.n_rejected, "lost": self.lost}
