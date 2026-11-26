"""Attitude and heading from the IMU.

The walking policy needs only the gravity direction and the body angular velocity, both of
which the IMU hands over directly.  Everything else here exists for NAVIGATION, which needs
a heading that survives the seconds between belt fixes.

The honest position: yaw from a strapdown IMU drifts and there is no magnetometer we trust
in a steel-framed outdoor venue.  So yaw is treated as a RELATIVE quantity -- integrated
from the gyro, used only for the change since the last fix, and never as an absolute course
direction.  Absolute heading comes from the belt/rail fits through the EKF.  That division
is why `navstate` can survive a 3-second detector dropout and still not accumulate a
heading error that steers the robot off the course.
"""
from __future__ import annotations

import numpy as np

from onboard.oio.iface import LegState


def projected_gravity(quat) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    return np.array([2.0 * (-z * x + w * y),
                     -2.0 * (z * y + w * x),
                     1.0 - 2.0 * (w * w + z * z)])


def roll_pitch(quat) -> tuple[float, float]:
    """From the gravity vector, not Euler decomposition: no gimbal case, no wrap, and it is
    the same quantity the policy consumes."""
    g = projected_gravity(quat)
    return float(np.arctan2(-g[1], -g[2])), float(np.arctan2(g[0], np.hypot(g[1], g[2])))


def wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


class Attitude:
    """Roll/pitch straight from the IMU; yaw integrated from the gyro's z component.

    Gyro yaw rather than the IMU's own quaternion yaw because the on-board fusion may or may
    not be doing something magnetometer-flavoured that we cannot see or reset.  A quantity
    we integrate ourselves is a quantity whose drift we know the shape of.
    """

    def __init__(self) -> None:
        self.yaw = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.gravity = np.array([0.0, 0.0, -1.0])
        self._t: float | None = None
        self.yaw_rate = 0.0

    def reset(self, yaw: float = 0.0) -> None:
        """Zero the heading.  Called once during bring-up, so 'straight ahead at the start
        line' is the origin -- which is the only heading reference the rules give us."""
        self.yaw = float(yaw)
        self._t = None

    def update(self, st: LegState) -> None:
        if not st.ok:
            return
        self.gravity = projected_gravity(st.quat)
        self.roll, self.pitch = roll_pitch(st.quat)
        self.yaw_rate = float(st.gyro[2])
        if self._t is not None:
            dt = min(max(st.t - self._t, 0.0), 0.1)     # clamp: a stall must not jump yaw
            # Body yaw rate -> world yaw rate.  On a slope the body's z axis is not vertical
            # and gyro-z alone under-reads the turn; the cos(roll)cos(pitch) factor is the
            # first-order correction and matters on the 10 deg grass slope.
            k = max(np.cos(self.roll) * np.cos(self.pitch), 0.3)
            self.yaw = wrap_pi(self.yaw + self.yaw_rate * dt / k)
        self._t = st.t

    @property
    def upright(self) -> bool:
        return self.gravity[2] < -0.5
