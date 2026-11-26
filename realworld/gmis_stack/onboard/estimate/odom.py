"""Leg odometry: body velocity and travelled distance from the feet.

The mission's budget, its section progress and its memory of where an object was all need
distance travelled.  There is no wheel encoder, no GPS, and the venue forbids any off-board
positioning system, so the only source is the legs: while a foot is on the ground it is
stationary in the world, therefore the body moved by minus the change in that foot's
position relative to the body.

The error model matters more than the estimator.  Leg odometry on a quadruped drifts a few
percent on good ground and much worse on the pebble section, where the feet slip by design.
So this reports a covariance that grows with the disagreement between feet, and
`navstate.py` uses that to decide how much to trust it between belt fixes.  An odometry that
lies confidently is worse than one that admits it is guessing.
"""
from __future__ import annotations

import numpy as np

from onboard.estimate.attitude import Attitude
from onboard.oio.iface import LegState

# From a2_piperx.urdf.  Both links are 0.275 m, which is what makes the height formula in
# control/states.py and the FK here consistent.
L_THIGH = 0.275
L_CALF = 0.275
HIP_X = 0.25944
HIP_Y = 0.075113
THIGH_Y = 0.12779
# The foot's collision sphere, not a guess.  With 0.025 the standing height came out
# 0.406 m against the configured 0.43, and body height is what sets `ground_z` for the
# detector -- a 2.4 cm error there shifts every unprojected range and every grasp target,
# and it would have looked like a calibration problem rather than an arithmetic one.
FOOT_R = 0.032

# Sign of the hip's y offset per leg, in policy order FR, FL, RR, RL.
_SY = np.array([-1.0, +1.0, -1.0, +1.0])
_SX = np.array([+1.0, +1.0, -1.0, -1.0])


def foot_positions(q: np.ndarray) -> np.ndarray:
    """(12,) joint angles -> (4, 3) foot positions in the body frame.

    Hip abducts about +x, thigh and calf rotate about +y, so the leg lives in a sagittal
    plane that the hip angle then rolls sideways.
    """
    q = np.asarray(q, dtype=np.float64).reshape(4, 3)
    a, t, c = q[:, 0], q[:, 1], q[:, 2]
    # In-plane foot position relative to the thigh joint.
    px = -(L_THIGH * np.sin(t) + L_CALF * np.sin(t + c))
    pz = -(L_THIGH * np.cos(t) + L_CALF * np.cos(t + c)) - FOOT_R
    py = _SY * THIGH_Y
    ca, sa = np.cos(a), np.sin(a)
    return np.stack([
        _SX * HIP_X + px,
        _SY * HIP_Y + ca * py - sa * pz,
        sa * py + ca * pz,
    ], axis=1)


class LegOdometry:
    """Contact-averaged body velocity, integrated into a 2-D pose."""

    def __init__(self, contact_threshold: float = 20.0) -> None:
        self.contact_threshold = float(contact_threshold)
        self.xy = np.zeros(2)
        self.distance = 0.0
        self.vel_body = np.zeros(3)
        self.height = 0.43
        self.sigma = 0.0              # 1-sigma disagreement between stance feet, m/s
        self.n_contact = 0
        self._prev_feet: np.ndarray | None = None
        self._prev_contact: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self.xy[:] = 0.0
        self.distance = 0.0
        self._prev_feet = None
        self._t = None

    def update(self, st: LegState, att: Attitude) -> None:
        if not st.ok:
            # A stale state must not be integrated -- the feet have not stopped moving just
            # because we stopped hearing about them.
            self._prev_feet = None
            self._t = None
            return

        feet = foot_positions(st.q)
        contact = np.asarray(st.foot_force, float) > self.contact_threshold
        # Body height: mean drop of the stance feet, or of all four when airborne.
        use = contact if contact.any() else np.ones(4, bool)
        # Measured, never nominal.  For the record the nominal standing height from exact
        # URDF kinematics is 0.413 m, not the 0.43 quoted in a2.py's docstring: that figure
        # drops the hip abduction, which lifts each foot about 1.3 cm.  It matters only as a
        # sanity bound here because this uses the live joint angles.
        self.height = float(-np.mean(feet[use, 2]))

        if self._prev_feet is not None and self._t is not None:
            dt = st.t - self._t
            if 1e-4 < dt < 0.1:
                # A foot counts only if it was down for the WHOLE interval; a foot that
                # touched down mid-interval moved through the air for part of it and its
                # displacement is not the body's.
                both = contact & self._prev_contact
                if both.any():
                    v = -(feet[both] - self._prev_feet[both]) / dt
                    self.vel_body = v.mean(axis=0)
                    # Spread across stance feet is the honest confidence signal: on the
                    # pebbles the feet disagree because they are slipping, and that is
                    # exactly when the estimate should be down-weighted.
                    self.sigma = float(np.linalg.norm(v.std(axis=0))) if both.sum() > 1 else 0.15
                    c, s = np.cos(att.yaw), np.sin(att.yaw)
                    vx, vy = self.vel_body[0], self.vel_body[1]
                    self.xy += dt * np.array([c * vx - s * vy, s * vx + c * vy])
                    self.distance += float(np.hypot(vx, vy) * dt)
                else:
                    self.sigma = 0.5          # flight phase: no information
        self._prev_feet = feet
        self._prev_contact = contact
        self.n_contact = int(contact.sum())
        self._t = st.t

    def report(self) -> dict:
        return {"xy": [round(float(v), 3) for v in self.xy],
                "distance_m": round(self.distance, 3),
                "vel_body": [round(float(v), 3) for v in self.vel_body],
                "height_m": round(self.height, 3),
                "sigma": round(self.sigma, 3), "n_contact": self.n_contact}
