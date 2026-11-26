"""Belt + leg-odometry fusion: the map-free replacement for arc_of's lateral half.

WHY THIS EXISTS
---------------
Every steering decision, guard and budget in the mission consumes (arc, lat) from
`mission.arc_of`, which projects the TRUE world pose onto the sim's surveyed S-curve.
At the venue that curve does not exist. `perception.detect_belt` supplies the two
quantities that matter -- lateral offset and heading error relative to the red boundary
-- but measured on its own it is not drivable: 52 % fix rate, 262 mm / 5.4 deg median but
a p90 tail of 1.8 m / 24.8 deg, and steering directly on an EMA of it left the course at
the stairs in 400 steps (README "실기 항법").

The missing piece is not a better detector, it is TIME: something has to carry the state
between fixes and decide which fixes to believe. That is this module.

MODEL
-----
State x = [w, psi, c]

    w    lateral offset from the belt centreline, metres, +left
    psi  heading error, radians: the tangent minus the robot's yaw, so psi > 0 means
         "turn left to line up"
    c    course curvature, 1/m -- NOT observed, estimated. This is what makes the filter
         survive a bend: without it every curve looks like a steering error and the
         correction fights the course.

Propagation over dt with body velocity (vx, vy) and yaw rate wz, all from proprioception
(leg odometry + IMU), never from a map:

    w   <- w   + (-vx sin psi + vy cos psi) dt
    psi <- psi + (c vx - wz) dt
    c   <- c                                  (random walk)

Measurements are detect_belt's `lat` and `heading`, each a direct observation of w and
psi. They are accepted through a NORMALISED INNOVATION gate rather than the plausibility
gates the detector used to carry: those were tuned on absolute values and killed good
fixes on curves (114 -> 45 samples, README). An innovation gate asks the only question
that generalises -- "is this fix consistent with where I believe I am, given how sure I
am?" -- so it widens by itself after a dropout and tightens when fixes are flowing.

Pure numpy, no Isaac, no torch: it runs on the N150 and `python -m course_mission.navfuse`
self-tests it against a synthetic S-curve with the measured noise and dropout statistics.
"""

from __future__ import annotations

import math

import numpy as np

# Measurement noise, from the validate_venue_nav.py run recorded in the README:
# lat median |err| 262 mm, heading median 5.4 deg. Median |err| of a zero-mean Gaussian is
# 0.6745 sigma, so sigma_lat = 0.262/0.6745 = 0.39 m and sigma_psi = 5.4/0.6745 = 8.0 deg.
SIG_LAT = 0.39
SIG_PSI = math.radians(8.0)
# Process noise per second. The velocity estimate a real quadruped's leg odometry gives is
# good to a few cm/s on firm ground and much worse on pebbles; the gyro is good but its
# bias walks. c is a genuine unknown that changes as the course bends.
Q_W = 0.12 ** 2          # (m/s)^2   lateral velocity error
Q_PSI = math.radians(6.0) ** 2   # (rad/s)^2 yaw-rate error
# Curvature drifts SLOWLY: a course bends over tens of metres, and at 0.5 m/s that is
# tens of seconds. The first version used 0.02 here and the state ran away to -0.477 1/m
# (a 2.1 m turning radius) inside one sim run -- see C_MAX below.
Q_C = 0.004 ** 2         # (1/m/s)^2 curvature random walk
# Innovation gate in sigmas. 3.0 rejects ~0.3 % of good fixes and most of the 1.8 m tail.
GATE = 3.0
# A fix this far outside the gate several times running means the FILTER is wrong, not the
# fix; after this many consecutive rejections the filter resets to the measurement.
MAX_REJECT = 8

# ---- PHYSICAL BOUNDS. These are not tuning knobs, they are the difference between a
# filter and a runaway. Measured failure, sim run navval_fuse (2026-08-24): curvature
# integrated to -0.477 1/m, i.e. the filter believed the course was turning inside a 2.1 m
# radius. It then steered a constant turn, drove off the course, lost sight of the belt,
# and with no fixes left to correct it the error reached 54 m. Everything below exists to
# make that trajectory unreachable.
C_MAX = 0.10             # 1/m. The sim S-curve peaks at 0.029 (17.5 m min radius); even a
#                          much tighter venue course does not bend inside 10 m.
W_MAX = 3.0              # m. The course is +-1.98 m wide. A belief outside this is lost,
#                          not offset.
PSI_MAX = 1.2            # rad. Believing you are more than ~70 deg off the course tangent
#                          while still walking forward is a lost filter, not a heading.
# NO COVARIANCE CAP. The first version clamped P[0,0] alone, which is not a covariance
# operation: it leaves P[0,1] and P[0,2] untouched and destroys positive semi-definiteness.
# An indefinite P then drives P[0,0] NEGATIVE through the update, the Kalman gain
# K[0] = P00/S flips sign, and every accepted fix pushes the estimate AWAY from the
# measurement -- the exact positive feedback that put the sim run 24.7 m off the course,
# introduced by the "fix" that was supposed to stop divergence. Staleness is bounded by
# LOST_SIGMA + the reset below instead, which are PSD-safe.
LOST_SIGMA = 1.2         # m. Beyond this the estimate is not usable; say so, do not fake it.
FIX_STALE = 150          # steps (3 s). Past this, stop propagating curvature: an unobserved
#                          state integrating into the prediction is what ran away.


class BeltOdomFuser:
    """Fuses detect_belt fixes with leg odometry into a continuous (lat, heading).

    Usage per control step:

        fuser.predict(vx, vy, wz, dt)          # every step, from proprioception
        if belt is not None:                   # ~2 Hz, when the detector returns
            fuser.update(belt["lat"], belt["heading"])
        lat, psi = fuser.lat, fuser.heading    # always available, never None
    """

    def __init__(self, lat0: float = 0.0, psi0: float = 0.0,
                 p_lat: float = 0.5, p_psi: float = math.radians(15.0)):
        self.x = np.array([lat0, psi0, 0.0], float)
        self.P = np.diag([p_lat ** 2, p_psi ** 2, 0.05 ** 2])
        self.n_pred = 0
        self.n_acc = 0
        self.n_rej = 0
        self._streak = 0
        self.since_fix = 10 ** 9       # steps since the last ACCEPTED fix
        self.out_of_range = False      # belief left the physical course: not usable
        self.halfwidth = None          # robust running estimate, for the OOB margin

    # -- propagation ---------------------------------------------------------------
    def predict(self, vx: float, vy: float, wz: float, dt: float) -> None:
        w, psi, c = self.x
        # Curvature is only observable THROUGH the heading measurement. With no fixes
        # arriving it is a free variable multiplying vx, so it must not keep bending the
        # prediction: freeze it once the fixes go stale.
        c_eff = c if self.since_fix < FIX_STALE else 0.0
        sp, cp = math.sin(psi), math.cos(psi)
        self.x = np.array([
            w + (-vx * sp + vy * cp) * dt,
            psi + (c_eff * vx - wz) * dt,
            c,
        ], float)
        self._clamp()
        # F = d(f)/dx
        # F is d(f)/dx of the propagation ACTUALLY used. When c is frozen the curvature no
        # longer feeds psi, so its column must be zero too -- otherwise the filter keeps
        # crediting c with an influence it is not exerting and the covariance lies.
        F = np.array([
            [1.0, (-vx * cp - vy * sp) * dt, 0.0],
            [0.0, 1.0, (vx * dt if self.since_fix < FIX_STALE else 0.0)],
            [0.0, 0.0, 1.0],
        ], float)
        Q = np.diag([Q_W, Q_PSI, Q_C]) * dt
        self.P = 0.5 * (F @ self.P @ F.T + Q + (F @ self.P @ F.T + Q).T)   # keep symmetric
        self.n_pred += 1
        self.since_fix += 1

    def _clamp(self) -> None:
        """Bound the state to what the course physically allows.

        Curvature and heading are genuinely bounded quantities and clamping them is
        honest. The LATERAL offset is different: a belief of 3 m does not mean the robot
        is at 3 m, it means the filter has lost the plot -- so clamping w silently turns a
        detectable failure into a plausible-looking number, and the innovation is then
        computed against the clamped value so the gate rejects the very fixes that would
        have corrected it. Flag it instead; `lost()` reports it and the caller stops
        steering on it.
        """
        self.x[1] = float(np.clip((self.x[1] + math.pi) % (2 * math.pi) - math.pi,
                                  -PSI_MAX, PSI_MAX))
        self.x[2] = float(np.clip(self.x[2], -C_MAX, C_MAX))
        if abs(self.x[0]) > W_MAX:
            self.x[0] = float(np.clip(self.x[0], -W_MAX, W_MAX))
            self.out_of_range = True

    # -- correction ----------------------------------------------------------------
    def update(self, lat: float | None = None, heading: float | None = None,
               halfwidth: float | None = None) -> bool:
        """Fold in one belt fix. Returns True if it was accepted.

        Each component is gated on its own: on a curve the far-line fit can spoil the
        heading while the near line still pins the offset, and throwing away the whole
        fix for one bad half is how the old absolute gates lost their samples.
        """
        if halfwidth is not None and halfwidth > 0.2:
            self.halfwidth = (halfwidth if self.halfwidth is None
                              else 0.9 * self.halfwidth + 0.1 * halfwidth)
        acc = False
        rej = False
        for idx, z, sig in ((0, lat, SIG_LAT), (1, heading, SIG_PSI)):
            if z is None or not np.isfinite(z):
                continue
            H = np.zeros(3)
            H[idx] = 1.0
            nu = float(z) - self.x[idx]
            if idx == 1:                       # heading is an angle
                nu = (nu + math.pi) % (2 * math.pi) - math.pi
            S = float(H @ self.P @ H + sig ** 2)
            if not np.isfinite(S) or S <= 0.0:
                # S is a variance and cannot be <= 0 for a well-formed P. If it is, the
                # covariance has gone bad and the previous code raised
                # "math domain error" out of math.sqrt mid-run; re-seed rather than crash.
                self._reseed(lat, heading)
                return False
            if abs(nu) > GATE * math.sqrt(S):
                rej = True
                continue
            K = (self.P @ H) / S
            self.x = self.x + K * nu
            self.x[1] = (self.x[1] + math.pi) % (2 * math.pi) - math.pi
            # Joseph form: P = (I-KH) P (I-KH)^T + K R K^T. Algebraically equal to
            # (I-KH)P for the optimal gain but positive-semidefinite by construction, so
            # repeated updates cannot walk the covariance into the indefinite region the
            # short form reached here.
            IKH = np.eye(3) - np.outer(K, H)
            self.P = IKH @ self.P @ IKH.T + np.outer(K, K) * (sig ** 2)
            self.P = 0.5 * (self.P + self.P.T)
            self._clamp()
            acc = True
        if acc:
            self.n_acc += 1
            self.since_fix = 0
            self.out_of_range = False
        if rej:
            # Counted even when the OTHER component was accepted. The previous `elif`
            # meant a fix whose heading was folded in reset the streak, so the re-seed --
            # the filter's only escape from a gate lockout -- could never fire while any
            # half of a fix kept being accepted.
            self.n_rej += 1
            self._streak += 1
            if self._streak >= MAX_REJECT:
                # The filter, not the detector, has lost the plot -- most likely the walk
                # slipped or the course bent harder than the curvature state can follow.
                # Re-seed on the measurement and re-open the covariance.
                self._reseed(lat, heading)
        return acc

    def _reseed(self, lat, heading) -> None:
        """Throw the belief away and restart from the measurement."""
        if lat is not None and np.isfinite(lat):
            self.x[0] = float(lat)
        if heading is not None and np.isfinite(heading):
            self.x[1] = float(heading)
        self.x[2] = 0.0
        self.P = np.diag([0.5 ** 2, math.radians(15.0) ** 2, 0.05 ** 2])
        self._clamp()
        self._streak = 0
        self.since_fix = 0
        self.out_of_range = False
        self.n_reseed = getattr(self, "n_reseed", 0) + 1

    # -- outputs -------------------------------------------------------------------
    @property
    def lat(self) -> float:
        return float(self.x[0])

    @property
    def heading(self) -> float:
        return float(self.x[1])

    @property
    def curvature(self) -> float:
        return float(self.x[2])

    @property
    def lat_sigma(self) -> float:
        return float(math.sqrt(max(self.P[0, 0], 0.0)))

    @property
    def trusted(self) -> bool:
        """Tight enough to steer aggressively on.

        Uses the SAME staleness threshold as lost()/fresh. They disagreed before (250 vs
        150), so between 150 and 250 steps the filter reported lost() and trusted() at the
        same time and the caller could not act on either.
        """
        return self.lat_sigma < 0.45 and self.since_fix < FIX_STALE and not self.out_of_range

    def steer(self, v_nom: float = 0.5, lat_gain: float = 0.5, head_gain: float = 1.2,
              lat_cap: float = 0.30, yaw_cap: float = 0.7):
        """(vx, vy, wz) that drives forward while closing on the belt centreline.

        THE TURN IS THE DANGEROUS PART. In the first sim run this returned the full yaw
        command whatever the state of the estimate, so when the filter drifted the robot
        committed to a steady turn, drove off the course, lost sight of the belt, and had
        no way back -- the fixes that would have corrected it were exactly what the turn
        threw away. Speed was throttled on confidence but steering was not, which is
        backwards: driving straight while unsure is recoverable, turning while unsure is
        not. So yaw is scaled by confidence and decays to zero once the fixes go stale.
        """
        lat = self.lat
        psi = self.heading
        conf = 1.0 if self.trusted else 0.55
        # 1.0 while fixes are flowing, fading to 0 as the last one ages out
        fresh = max(0.0, 1.0 - self.since_fix / float(FIX_STALE))
        vx = v_nom * conf
        # BOTH corrections fade with the fix age, not just yaw. The first version decayed
        # wz only, so on a dead estimate the robot still crabbed sideways at up to
        # 0.30 m/s forever -- a slower way to leave the course, not a safer one.
        vy = float(np.clip(-lat_gain * lat, -lat_cap, lat_cap)) * conf * fresh
        wz = float(np.clip(head_gain * (psi - 0.8 * np.clip(lat, -0.8, 0.8)),
                           -yaw_cap, yaw_cap)) * conf * fresh
        return vx, vy, wz

    def lost(self) -> bool:
        """No usable belief: the caller should hold a straight heading and re-acquire."""
        return (self.since_fix >= FIX_STALE or self.lat_sigma >= LOST_SIGMA
                or self.out_of_range)


# ======================================================================================
# self-test: synthetic S-curve with the MEASURED detector statistics, no Isaac needed
# ======================================================================================
def _selftest(n: int = 6000, seed: int = 0) -> dict:
    """Drive a synthetic course with the measured fix rate, noise and heavy tail.

    Ground truth: a reverse-S of amplitude 7.5 m over 36 m (course_sim/terrain.py:432),
    walked at 0.5 m/s. The robot steers on the FILTER's output, so this measures the
    closed loop -- which is the thing that failed on the real harness -- not just the
    filter's error against a recorded track.
    """
    rng = np.random.default_rng(seed)
    dt = 0.02
    L, amp = 36.0, 7.5

    def tangent(s):                       # dtheta/ds of the sim's half-cosine S
        # y(s) = amp/2 * (1 - cos(pi s / L)) -> y' = amp*pi/(2L) sin(pi s/L)
        yp = amp * math.pi / (2 * L) * math.sin(math.pi * s / L)
        return math.atan(yp)

    s = 0.0                                # arc travelled
    w_true = 0.0                           # true lateral offset
    yaw = tangent(0.0)                     # robot yaw in world
    f = BeltOdomFuser()
    err, off = [], []
    fixes = acc = 0
    for i in range(n):
        th = tangent(s)
        psi_true = (th - yaw + math.pi) % (2 * math.pi) - math.pi
        # --- the robot acts on the FILTER, never on truth
        vx, vy, wz = f.steer()
        # --- true kinematics
        w_true += (-vx * math.sin(psi_true) + vy * math.cos(psi_true)) * dt
        yaw += wz * dt
        s += vx * dt
        # --- proprioception: velocity and gyro, with realistic error
        f.predict(vx * (1 + rng.normal(0, 0.06)) + rng.normal(0, 0.02),
                  vy + rng.normal(0, 0.02),
                  wz + rng.normal(0, math.radians(3.0)), dt)
        # --- detector at 5 Hz with a 52 % fix rate and a heavy tail
        if i % 10 == 0:
            if rng.random() < 0.52:
                fixes += 1
                heavy = rng.random() < 0.10          # the p90 tail
                zl = w_true + rng.normal(0, SIG_LAT * (4.0 if heavy else 1.0))
                zp = psi_true + rng.normal(0, SIG_PSI * (4.0 if heavy else 1.0))
                acc += bool(f.update(zl, zp, halfwidth=1.98))
        err.append(abs(f.lat - w_true))
        off.append(abs(w_true))
        if s > L:
            break
    return {
        "steps": i + 1,
        "fixes": fixes,
        "accepted": acc,
        "est_err_median": float(np.median(err)),
        "est_err_p90": float(np.percentile(err, 90)),
        "true_off_median": float(np.median(off)),
        "true_off_max": float(np.max(off)),
        "arc": s,
    }


if __name__ == "__main__":
    for sd in range(5):
        r = _selftest(seed=sd)
        print(f"seed {sd}: arc {r['arc']:5.1f} m  fixes {r['fixes']:4d} "
              f"accepted {r['accepted']:4d}  est err median {r['est_err_median']*1000:4.0f} mm "
              f"p90 {r['est_err_p90']*1000:5.0f} mm  |  TRUE offset median "
              f"{r['true_off_median']*1000:4.0f} mm max {r['true_off_max']*1000:5.0f} mm")
