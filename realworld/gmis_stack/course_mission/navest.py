"""NavEst v2 — the robot-side pose service for map-free ("실전형") missions.

WHAT COUNTS AS PRIOR-FREE. With MISSION_NAV_HONEST=1 every ROBOT decision runs in the
"S-frame": a straightened course frame the robot builds for itself from

    * leg odometry + IMU (3 % velocity scale noise, gyro bias + noise, outlier clamp),
    * CORRIDOR fixes: the belt stanchions (black posts) and near belt webbing seen by the
      head RGB-D camera, fitted as two parallel lines a rules-level 4 m apart; the fit's
      slope corrects the heading, its intercepts the lateral offset. The line direction is
      resolved against the gyro-continued tangent, so a fix taken while the robot faces
      across the course can never flip the frame (the v1 vanishing-point fixes had no
      direction and did exactly that after every deposit spin),
    * bins discovered BY SIGHT (perception.detect_bins) with the S-frame camera pose, and
      re-sighted later as a drift landmark.

The sim's surveyed curve (course_sim.terrain arc_of/curve_theta/curve_map), its section
table and bin_targets() are never consulted by robot code. The judge-side scoring in
runner.py (checkpoints, OOB, landing classification) keeps the truth — that is the
venue's job, not the robot's. What the robot MAY know, because the event rules say so:
it is placed on the start line facing along the course, the course is 4 m wide between
the belts, and the nominal length is a budget horizon (never a position).

S-frame conventions. x = estimated arc s (start line = the sim's start arc, a labelling
convention so logs stay comparable), y = estimated lateral w (+ = left of the course
direction), z = the sim's root height (a stand-in for the leg-kinematic height the real
robot has), yaw_S = heading relative to the course tangent = -psi_dr. Every position the
robot stores (detections, memory, bins, blacklists) lives in S; the truth-side helpers
(`S_of_true`, `true_of_S`) exist only for sim bookkeeping and calibration logs.

With MISSION_NAV_HONEST unset, every method passes the ORACLE through, byte-identical to
the pre-NavEst behaviour, so one code path serves both modes.
"""

from __future__ import annotations

import os
from collections import deque

import numpy as np

W_HALF = 2.0                 # rules: course width 4 m, belts at +-2 m
COURSE_LEN_NOMINAL = 36.0    # rules: 6 sections x 6 m — a horizon for budgets only
FIX_EVERY = int(os.environ.get("NAVEST_FIX_EVERY", "20"))
BINS_EVERY = 100
TRACK_STEPS = int(os.environ.get("NAVEST_TRACK", "500"))      # tracker window (10 s ~ 5 m)
STAIR_PITCH_ON, STAIR_PITCH_OFF = 0.20, 0.12                  # rad, EMA of |pitch|
INCLINE_PITCH = float(os.environ.get("NAVEST_INCLINE_PITCH", "0.10"))   # slope gate
# stance-foot grade (front-rear foot height over their spacing, leg kinematics + IMU
# attitude): the 9 deg ramp reads ~0.16, flat ground < 0.03 — hs1 spent 3,000 steps
# on three slope stations because the body pitch EMA there sat at 0.06-0.10
INCLINE_GRADE = float(os.environ.get("NAVEST_INCLINE_GRADE", "0.08"))
FOOT_R = float(os.environ.get("NAVEST_FOOT_R", "0.02"))   # foot-body centre above ground
# bin box (guide: top 0.40 x 0.26, bottom 0.32 x 0.195): mean half extents, used to
# move the detector's FACE point to the bin's centre
BIN_HALF_ALONG, BIN_HALF_ACROSS = 0.18, 0.115
# A body-relative sighting is carried forward by RAW ODOMETRY (never corrected), so it
# decays with TIME as well as distance: the yaw channel integrates a per-run gyro bias,
# which at 0.65 m of lever arm is worth several centimetres a minute even standing
# still. Measured over the 20 scored drops of hs4a-hs4e: every drop whose sighting was
# younger than ~2,600 steps landed in the bin (10/10) and none of the older ones did
# (0/7) — the drift ramps about 7 mm per 100 steps. The robot stands at a bin for
# 1,500-3,000 steps per unload while it fishes the next object out of the basket, so
# the SECOND drop of a stop is always the one that goes wrong. Keep the step expiry and
# add a distance one, because walking spends the same budget faster.
BIN_OBS_MAX_AGE = int(os.environ.get("NAVEST_BIN_OBS_AGE", "2500"))    # steps
# The distance gate must not undo the sighting the approach is FOR: the landmark window
# is 2.4-3.5 m out, the drop stand-off is 0.55 m, and goto_bin backs off ~1.1 m to
# strafe between bins — a legitimate walk-in spends about 4 m, which a 3 m gate cut
# (hs11d's second trip: every drop logged "1000000000 steps since the bin was last
# SEEN", i.e. the body sighting had been dropped, and the S-frame fallback then asked
# for +134..+175 mm corrections that had to be refused). 6 m keeps the approach's own
# sighting and still drops one carried across a pick trip; the AGE gate above is what
# handles the drift that accumulates while the robot stands at the bins.
BIN_OBS_MAX_MOVE = float(os.environ.get("NAVEST_BIN_OBS_MOVE", "6.0"))  # m of odometry
# a sighting beyond this range is a heading, not a position: the S frame is the LOCAL
# tangent, and the course curves (measured curv up to 0.08) — at 7 m the centreline
# itself sits 1.2 m off the tangent line, at 3 m 0.2 m
BIN_NEAR_RANGE = float(os.environ.get("NAVEST_BIN_NEAR_RANGE", "3.5"))


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot2(a: float, xy: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    xy = np.asarray(xy, np.float64)
    return np.stack([c * xy[..., 0] - s * xy[..., 1], s * xy[..., 0] + c * xy[..., 1]], -1)


class NavEst:
    def __init__(self, runner):
        self.r = runner
        self.honest = os.environ.get("MISSION_NAV_HONEST") == "1"
        self._init = False
        # ---- S-frame state
        self.ox = 0.0                 # est arc s
        self.oy = 0.0                 # est lateral w
        self.yaw_g = 0.0              # gyro-integrated heading (relative to t0)
        self.tangent_yaw = 0.0        # course tangent heading in the gyro frame
        self.curv = 0.0               # local curvature estimate (rad/m), feed-forward
        self.gyro_bias = np.random.normal(0.0, np.radians(0.1))   # rad/s, AHRS-grade
        self.v_prev = (0.0, 0.0)
        self.prev_yaw_true = None
        self.dt = None
        self.s0 = 0.0
        # ---- raw (uncorrected) odometry chain for the corridor tracker
        self.odo = np.zeros(3)        # x, y, yaw in the t0 body frame
        self._posts = deque()         # (step, odo(3), xy_b(2), weight, range)
        self._belt = deque()          # same, dense webbing points
        # ---- fix bookkeeping
        self.since_fix = 10 ** 9      # steps since the last ACCEPTED corridor fix
        self.fixes_belt = 0           # accepted corridor fixes (name kept for the runner)
        self.fixes_bins = 0
        self.fit_tries = 0
        self.fit_ok = 0
        self.last_fit = None          # dict of the last accepted fit (for logs)
        self._reloc = False
        self._pose_key = None
        # ---- bins in S: colour -> np.array([s, w]); centre likewise
        self.bins = {}
        self.bins_centre = None
        self._bins_w = {}             # accumulated range weights per colour
        self._bin_obs = {}            # colour -> (step, odo(3), rel_body(2), range)
        # ---- IMU terrain identity (no map): |pitch| EMA hysteresis + roughness
        self._pitch_ema = 0.0
        self._rough = 0.0
        self._rough_walk = 0.0
        self._prev_rp = None
        self._stairs = False

    # ================================================================== feeding
    def step(self):
        """Once per control step, from proprioception + the head camera. No-op unless
        honest."""
        if not self.honest:
            return
        r = self.r
        d = r.robot.data
        if not self._init:
            _ue = r.env.unwrapped
            self.dt = float(getattr(_ue, "step_dt", None)
                            or _ue.physics_dt * getattr(_ue.cfg, "decimation", 4))
            from course_mission.mission import arc_of
            # the robot is placed ON THE START LINE facing along the course — the
            # event's own setup rule. The arc label of that line is the sim's start
            # arc so est and true logs share an origin; nothing about the course
            # shape or its lateral placement enters here.
            self.s0 = float(arc_of(*r.base_pose_true()[:2])[0])
            self.ox, self.oy = self.s0, 0.0
            self.prev_yaw_true = r.base_pose_true()[3]
            self._init = True
            r.log.ev(f"[NAVEST] honest navigation ON (v2 S-frame) — anchored at the "
                     f"start line (s {self.s0:+.1f})")
        # ---- AHRS-delta yaw: true per-step delta + bias + noise
        yt = r.base_pose_true()[3]
        dyaw = _wrap(yt - self.prev_yaw_true)
        self.prev_yaw_true = yt
        dyaw_meas = dyaw + (self.gyro_bias + np.random.normal(0, 0.003)) * self.dt
        self.yaw_g += dyaw_meas
        # ---- body velocity with scale noise, outlier clamp
        v = d.root_lin_vel_b[0].cpu().numpy()
        vx = float(v[0]) * (1.0 + np.random.normal(0, 0.03))
        vy = float(v[1]) * (1.0 + np.random.normal(0, 0.03))
        if abs(vx) > 2.5 or abs(vy) > 2.5:
            vx, vy = self.v_prev
        self.v_prev = (vx, vy)
        # ---- raw odometry chain (never corrected; the tracker re-expresses its
        # stored observations through it)
        self.odo[2] += dyaw_meas
        self.odo[:2] += _rot2(self.odo[2], np.array([vx, vy])) * self.dt
        # ---- S-frame dead reckoning at ANY heading: yaw_S = yaw_g - tangent_yaw
        ys = self.yaw_S()
        dS = _rot2(ys, np.array([vx, vy])) * self.dt
        self.ox += float(dS[0])
        self.oy += float(dS[1])
        # the tangent turns with the course; feed the last fit's curvature forward
        # while the fixes are fresh enough to have measured it
        if self.since_fix < 600 and abs(self.curv) > 1e-4:
            self.tangent_yaw += float(self.curv) * float(dS[0])
        self.since_fix += 1
        self._imu_terrain()
        self._pose_key = None
        # ---- corridor fixes (posts + near belt) and bins fixes on their cadences
        if r.step_i % FIX_EVERY == 0:
            try:
                self._corridor_observe()
                self._corridor_fix()
            except Exception as e:          # a detector fault must never stop the walk
                if os.environ.get("NAVEST_FIXLOG") == "1":
                    r.log.ev(f"  [NAVFIX] fault: {e!r}")
        if r.step_i % BINS_EVERY == 50:
            try:
                self._bins_fix()
            except Exception as e:
                if os.environ.get("NAVEST_FIXLOG") == "1":
                    r.log.ev(f"  [NAVFIX] bins fault: {e!r}")
        self._publish_extrinsics()
        if r.step_i % int(os.environ.get("NAVEST_LOG", "2000")) == 0:
            self._calib_log()

    # ------------------------------------------------------------- IMU terrain
    def _imu_terrain(self):
        from isaaclab.utils import math as math_utils
        d = self.r.robot.data
        rpy = math_utils.euler_xyz_from_quat(d.root_quat_w)
        roll, pitch = float(rpy[0][0]), float(rpy[1][0])
        roll, pitch = _wrap(roll), _wrap(pitch)
        self._pitch_ema += 0.05 * (abs(pitch) - self._pitch_ema)
        if self._prev_rp is not None:
            j = abs(roll - self._prev_rp[0]) + abs(pitch - self._prev_rp[1])
            self._rough += 0.02 * (j - self._rough)
            # roughness OF THE GROUND is only observable while walking on it: latch
            # the walking-time jitter so a stationary grasp cycle (minutes) does not
            # decay it before the deposit asks "is this rough ground?"
            if abs(self.v_prev[0]) > 0.15 or abs(self.v_prev[1]) > 0.15:
                self._rough_walk += 0.02 * (j - self._rough_walk)
        self._prev_rp = (roll, pitch)
        if not self._stairs and self._pitch_ema > STAIR_PITCH_ON:
            self._stairs = True
        elif self._stairs and self._pitch_ema < STAIR_PITCH_OFF:
            self._stairs = False

    def on_stairs(self) -> bool:
        """IMU-only stand-in for section_of(ax) == 'stairs': a sustained body pitch."""
        if not self.honest:
            from course_mission.mission import section_of, arc_of
            return section_of(arc_of(*self.r.base_pose_true()[:2])[0]) == "stairs"
        return self._stairs

    def rough(self) -> float:
        """Walking-time attitude jitter EMA (rad/step): the honest stand-in for
        'pebble/mixed'."""
        return float(self._rough_walk)

    def inclined(self) -> bool:
        """IMU stand-in for 'on a slope or the stairs': a sustained body pitch well
        above the flat-ground trot jitter (~0.03-0.05 rad); the 9 deg ramp reads
        ~0.15. Used to skip survey spins on ground the rules put no objects on."""
        if not self.honest:
            from course_mission.mission import section_of, arc_of
            return section_of(arc_of(*self.r.base_pose_true()[:2])[0]) in ("slope", "stairs")
        return (self._stairs or self._pitch_ema > INCLINE_PITCH
                or abs(self.foot_grade()) > INCLINE_GRADE)

    def foot_grade(self) -> float:
        """Ground grade under the robot from the STANCE feet: the lowest front foot
        vs the lowest rear foot (leg kinematics + IMU attitude — the same signal
        ground_z() rests on), over their horizontal spacing. + = climbing. Zero when
        the foot bodies are unknown. The min per pair rejects the swing foot of a
        trot, so it reads while walking, not only at a standstill."""
        r = self.r
        fi = getattr(r, "_foot_idx", None)
        if not fi or len(fi) < 4:
            return 0.0
        try:
            names = [r.robot.body_names[i] for i in fi]
        except Exception:
            return 0.0
        pw = r.robot.data.body_pos_w[0, fi].detach().cpu().numpy().astype(np.float64)
        front = [k for k, n in enumerate(names) if n.startswith("F")]
        rear = [k for k, n in enumerate(names) if n.startswith("R")]
        if not front or not rear:
            return 0.0
        pf = pw[front][np.argmin(pw[front][:, 2])]
        pr = pw[rear][np.argmin(pw[rear][:, 2])]
        dxy = float(np.hypot(pf[0] - pr[0], pf[1] - pr[1]))
        if dxy < 0.2:
            return 0.0
        return float((pf[2] - pr[2]) / dxy)

    # --------------------------------------------------------- corridor sensing
    def _cam_arrays(self):
        cd = self.r.percep.cam.data
        rgb = cd.output["rgb"][0, ..., :3].cpu().numpy()
        depth = cd.output["distance_to_image_plane"][0].cpu().numpy()
        K = cd.intrinsic_matrices[0].cpu().numpy()
        return rgb, depth, K

    def _corridor_observe(self):
        """Run the post / belt-point detectors in the gravity-aligned body frame and
        push the hits into the odometry-referenced tracker."""
        from course_mission.perception import (cam_pose_body, detect_belt_points,
                                               detect_posts)
        r = self.r
        m = r.percep.mount
        if m is None:
            return
        rgb, depth, K = self._cam_arrays()
        Xt, Yt, Zt, yaw_t = r.base_pose_true()
        cam_pb, cam_Rb = cam_pose_body(m.pos_w, m.R_ros, (Xt, Yt, Zt), yaw_t)
        odo = self.odo.copy()
        st = r.step_i
        posts = detect_posts(rgb, depth, K, cam_pb, cam_Rb)
        for p in posts:
            rng = float(p["range"])
            self._posts.append((st, odo, np.asarray(p["xy"], np.float64),
                                1.0 / (1.0 + (rng / 3.0) ** 2), rng))
        pts = detect_belt_points(rgb, depth, K, cam_pb, cam_Rb)
        if pts is not None and len(pts):
            xy = pts[:, :2]
            rng = np.hypot(xy[:, 0], xy[:, 1])
            keep = rng < 6.0
            xy, rng = xy[keep], rng[keep]
            if len(xy) > 24:
                sel = np.linspace(0, len(xy) - 1, 24).astype(int)
                xy, rng = xy[sel], rng[sel]
            for q, g in zip(xy, rng):
                self._belt.append((st, odo, q.astype(np.float64),
                                   0.12 / (1.0 + (g / 3.0) ** 2), float(g)))
        while self._posts and st - self._posts[0][0] > TRACK_STEPS:
            self._posts.popleft()
        while self._belt and st - self._belt[0][0] > TRACK_STEPS:
            self._belt.popleft()

    def _tracked_S(self, store):
        """Stored (odo pose, body xy) -> S coordinates through the CURRENT estimate."""
        if not store:
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
        odo = np.array([o for _, o, _, _, _ in store], np.float64)      # (N,3)
        xy = np.array([q for _, _, q, _, _ in store], np.float64)       # (N,2)
        wts = np.array([w for _, _, _, w, _ in store], np.float64)
        rng = np.array([g for _, _, _, _, g in store], np.float64)
        c, s = np.cos(odo[:, 2]), np.sin(odo[:, 2])
        p_o = np.stack([c * xy[:, 0] - s * xy[:, 1],
                        s * xy[:, 0] + c * xy[:, 1]], 1) + odo[:, :2]
        p_b = _rot2(-self.odo[2], p_o - self.odo[:2])
        pts = _rot2(self.yaw_S(), p_b) + np.array([self.ox, self.oy])
        return pts, wts, rng

    def _corridor_fix(self):
        """MAP fit of the corridor seen over the tracking window: both belt lines share
        one centreline  w = a + b*ds + c*ds^2  (a rules-level 2W apart), and the fit's
        a / b correct the lateral offset / heading, its c the local curvature.

        Why MAP and not plain least squares: from a short baseline (only the 3-6 m the
        camera sees ahead) heading and curvature are not separable — a bend ahead
        looks exactly like a heading error. Priors settle that: the heading is gyro-
        continued so it is tight (sigma ~2 deg, growing with time/distance since the
        last fix), the curvature is loosely tied to the last measurement, and the
        lateral offset is loose. As the robot walks, the tracker keeps points BEHIND it
        (TRACK_STEPS of odometry-referenced history), the baseline grows to ~10 m and
        the data take over from the priors.

        Direction ambiguity: a line has no direction, but the fit is done in the
        S-frame whose x axis IS the gyro-continued course direction, so the slope
        residual lives in (-pi/2, pi/2) around the current tangent — the frame cannot
        flip unless the gyro were 90 deg wrong, which the fixes keep it far from."""
        P, Wt, Rg = self._tracked_S(self._posts)
        B, Wb, Rb = self._tracked_S(self._belt)
        n_posts = len(P)
        if n_posts == 0 and len(B) < 8:
            return
        self.fit_tries += 1
        pts = np.concatenate([P, B]) if len(B) else P
        wts = np.concatenate([Wt, Wb]) if len(B) else Wt
        is_post = np.concatenate([np.ones(n_posts, bool), np.zeros(len(B), bool)])
        ds = pts[:, 0] - self.ox
        wv = pts[:, 1] - self.oy
        ok = (ds > -7.0) & (ds < 7.0) & (np.abs(wv) < 2.0 * W_HALF + 1.0)
        if int(ok.sum()) < 2:
            return
        wts, ds, wv, is_post = wts[ok], ds[ok], wv[ok], is_post[ok]
        n_posts = int(is_post.sum())
        spread = float(np.ptp(ds)) if len(ds) > 1 else 0.0
        belt_spread = float(np.ptp(ds[~is_post])) if int((~is_post).sum()) > 1 else 0.0
        if not (n_posts >= 2 or (n_posts >= 1 and belt_spread >= 1.0)
                or belt_spread >= 2.0):
            return
        if spread < 1.0:
            return
        sw = np.sqrt(wts)
        k_prior = float(self.curv)
        # ---- priors (in metres of a weight-1 point; sigma_y = 0.05 m)
        dt = float(self.dt or 0.02)
        t_since = min(self.since_fix, 10 ** 6) * dt
        fp = getattr(self, "_fix_pose", None)
        m_since = float(np.hypot(self.ox - fp[0], self.oy - fp[1])) if fp is not None \
            else 5.0
        if self._reloc:
            s_a, s_b, s_c = 2.0, 0.5, 0.02
        else:
            s_a = 0.2 + 0.1 * m_since
            s_b = 0.03 + 0.002 * t_since + 0.03 * m_since
            s_c = 0.008 + 0.003 * m_since
        s_y = 0.05
        prior_A = np.diag([s_y / s_a, s_y / s_b, s_y / s_c])
        prior_y = np.array([0.0, 0.0, (s_y / s_c) * 0.5 * k_prior])

        def _solve(side, use):
            # model: w_i - side_i*W = a + b*ds_i + c*ds_i^2*(1 + side_i*W*2c); a_L - a_R
            # = 2W is the rules-level belt spacing, which is what lets ONE visible
            # belt fix lat. The side factor is the offset-curve curvature (the outer
            # belt of a bend is flatter than the inner one); without it the fit is
            # biased ~0.06 rad at the S-curve peak.
            k = k_prior
            n = int(use.sum())
            for _ in range(2):
                k_used = k
                q2 = ds[use] ** 2 * (1.0 + side[use] * W_HALF * k)
                A = np.stack([np.ones(n), ds[use], q2], 1) * sw[use, None]
                y = (wv[use] - side[use] * W_HALF) * sw[use]
                A = np.concatenate([A, prior_A])
                y = np.concatenate([y, prior_y])
                x, *_ = np.linalg.lstsq(A, y, rcond=None)
                k = float(np.clip(2.0 * x[2], -0.1, 0.1))
            pred_all = x[0] + x[1] * ds + x[2] * ds ** 2 * (1.0 + side * W_HALF * k_used)
            return x, pred_all

        # SIDE ASSIGNMENT IS THE ONLY DISCRETE CHOICE. Seed it from several heading
        # hypotheses around the estimate, iterate assignment <-> least squares, keep
        # the hypothesis with the smallest residual; ties go to the one nearest the
        # current estimate. A wrong estimate can rotate far posts across the
        # centreline, which is what a one-shot assignment got wrong.
        deltas = np.arange(-1.2, 1.21, 0.3) if self._reloc else (-0.6, -0.3, 0.0, 0.3, 0.6)
        e_gate, w_gate = (1.2, 2.5) if self._reloc else (0.6, 1.0)
        best = None
        rejected = []
        use0 = np.ones(len(ds), bool)
        w_all = float(wts.sum())
        for dl in deltas:
            # the hypothesis: the robot's heading error is dl more than estimated, so
            # the centreline runs at angle dl through the robot -> sides by that line
            side = np.sign(_rot2(-dl, np.stack([ds, wv], 1))[:, 1])
            side[side == 0] = 1.0
            try:
                for _ in range(4):
                    x, pred = _solve(side, use0)
                    ns = np.sign(wv - pred)
                    ns[ns == 0] = 1.0
                    if np.array_equal(ns, side):
                        break
                    side = ns
                resid = wv - side * W_HALF - pred
                use = np.abs(resid) < 0.4
                if int(use.sum()) >= 2 and int(use.sum()) < len(ds):
                    x, pred = _solve(side, use)
                    resid = wv - side * W_HALF - pred
                else:
                    use = use0
            except np.linalg.LinAlgError:
                continue
            rms = float(np.sqrt(np.average(resid[use] ** 2, weights=wts[use])))
            frac = float(wts[use].sum()) / max(w_all, 1e-9)
            eps_h = float(np.arctan(x[1]))
            # the model's intercept is RELATIVE to the estimate (wv = w_S - oy), so
            # a = -(w_true) when the sides are right; the estimate's error is a + oy
            wres_h = float(x[0]) + float(self.oy)
            if abs(eps_h) > e_gate or abs(wres_h) > w_gate or rms > 0.25 or frac < 0.6:
                rejected.append((dl, eps_h, wres_h, rms, frac))
                continue
            score = rms - 0.5 * frac + 0.02 * abs(eps_h) + 0.02 * abs(wres_h)
            if best is None or score < best[0]:
                best = (score, x, side.copy(), use.copy(), rms, eps_h, wres_h)
        if best is None:
            if os.environ.get("NAVEST_FIXLOG") == "1" and rejected:
                dl, e_, w_, rm, fr = min(rejected, key=lambda t: t[3])
                self.r.log.ev(f"  [NAVFIX] rejected eps {e_:+.2f} w_res {w_:+.2f} "
                              f"rms {rm:.2f} inl {fr:.2f} posts {n_posts} pts {len(ds)}")
            return
        _, x, side, use, rms, eps, w_res = best
        c = float(x[2])
        n_posts = int((is_post & use).sum())
        n_use = int(use.sum())
        sides = int(len(set(side[use].tolist())))
        if n_posts < 2 and sides < 2 and n_use < 8:
            return
        ds_u = ds[use]
        long_base = float(np.ptp(ds_u)) >= 5.0 and float(ds_u.min()) < -1.0
        if self._reloc:
            self.tangent_yaw += eps
            self.oy -= w_res
        else:
            self.tangent_yaw += float(np.clip(eps, -0.3, 0.3))
            self.oy -= float(np.clip(w_res, -0.6, 0.6))
        self.curv = float(np.clip(2.0 * c, -0.08, 0.08))
        self.fixes_belt += 1
        self.fit_ok += 1
        self.since_fix = 0
        self._fix_pose = (float(self.ox), float(self.oy))
        self._pose_key = None
        self.last_fit = {"eps": eps, "w_res": w_res, "rms": rms, "posts": n_posts,
                         "pts": n_use, "sides": sides, "quad": long_base,
                         "curv": self.curv, "span": float(np.ptp(ds_u))}
        if os.environ.get("NAVEST_FIXLOG") == "1":
            from course_mission.mission import arc_of
            axt, wt = arc_of(*self.r.base_pose_true()[:2])
            self.r.log.ev(f"  [NAVFIX] ok eps {eps:+.3f} w_res {w_res:+.2f} rms {rms:.2f} "
                          f"posts {n_posts} pts {n_use} sides {sides} span "
                          f"{np.ptp(ds_u):.1f} curv {self.curv:+.3f} -> est s "
                          f"{self.ox:+.1f} w {self.oy:+.2f} (true {axt:+.1f} {wt:+.2f}) "
                          f"psi {self.psi_dr():+.2f}")

    def _bins_fix(self):
        """Bins by sight, positioned with the S-frame camera pose (no truth). A
        re-sighting of an already-known row is a drift landmark: the apparent motion
        of a fixed row is odometry error.

        Two things learned from hs1 (first drop 0.31 m short, second re-centred the
        load 96 mm AWAY from the bin): (1) the detector's point is the bin's visible
        FACE — add the box's half depth along the viewing direction to get the centre;
        (2) the memorised S position is an average dominated by the first far
        sightings, and the S pose itself walks 0.15 m under one-sided belt fits while
        the robot stands at the bin. So every sighting is ALSO kept body-relative in
        the raw odometry chain (`_bin_obs`), and bin_pos/bin_base dead-reckon from the
        latest one while it is fresh — a metre of odometry is a few cm, the S detour
        was 0.3."""
        from course_mission.perception import detect_bins, BIN_FRONT_TO_CENTRE
        rgb, depth, K = self._cam_arrays()
        cam_p, cam_R = self.cam_pose_S()
        hit = detect_bins(rgb, K, cam_p, cam_R, self.ground_z(), depth=depth)
        if hit is None:
            return
        new, rng = {}, {}
        _front = hit.get("front", {})
        # The bins stand across the course at THEIR arc, and that tangent is not the
        # robot's S tangent (kappa*d: ~6 deg two metres before the row on the S
        # course, 2 cm of lateral per bin if the half depth is added along S-x). With
        # two front points the row's own direction gives the face normal directly.
        n_row = np.array([1.0, 0.0])
        if len(_front) >= 2:
            _cs = sorted(_front)                   # blue..red = the widest pair
            fa = np.asarray(_front[_cs[0]], np.float64)[:2]
            fb = np.asarray(_front[_cs[-1]], np.float64)[:2]
            d = fb - fa
            dn = float(np.hypot(*d))
            if dn > 0.5:
                d = d / dn
                n = np.array([-d[1], d[0]])
                if n[0] < 0.0:
                    n = -n
                # within ~37 deg of S-x, else distrust. The measured normal stays good
                # out to 25 deg (bins probe b8n, dpsi -20: y err 1.4 cm) while the
                # [1, 0] fallback under the same yaw error is off 4-8 cm, so the gate
                # only needs to catch a broken pair, not a heading estimate that is
                # 20 deg out.
                if n[0] > 0.8:
                    n_row = n
        for col, pw in hit["colors"].items():
            p = np.asarray(pw, np.float64)[:2]
            u = p - np.asarray(cam_p, np.float64)[:2]
            un = float(np.hypot(*u))
            if col in _front:
                # depth point cloud: the bin's FRONT rim and centre line in S (the
                # bins face the course, so the centre is half a bin further along s).
                # The ray-direction correction below over-shot side bins ~0.05 m
                # outward per sighting (bins probe 0905), which is where hs4a's
                # blue-bin drops 0.11-0.24 m past the bin came from.
                f = np.asarray(_front[col], np.float64)
                p = f[:2] + BIN_FRONT_TO_CENTRE * n_row
            elif un > 1e-3:
                # face -> centre: the box's support distance in the viewing direction
                # (tapered bin: mean half extents 0.18 along the course, 0.115 across)
                u = u / un
                p = p + u * (BIN_HALF_ALONG * abs(u[0]) + BIN_HALF_ACROSS * abs(u[1]))
            new[col] = p
            rng[col] = float(hit.get("ranges", {}).get(col, un))
        # A FAR sighting is a tangent-line projection: the S frame is only the local
        # tangent, and a 7 m ray across the curving course lands 1.0-1.2 m off the
        # centreline (hs3b: "blue bin at w +1.8" seen from the stairs, true w +0.9 —
        # the robot walked to the deck edge chasing it, and the re-sighting was then
        # read as 0.44 m of lateral "drift"). Never treat a far view as a landmark.
        _far = max(rng.values()) > BIN_NEAR_RANGE
        if _far:
            # keep it only where nothing nearer is remembered; no drift correction
            for c, p in new.items():
                wn = 1.0 / (1.0 + (rng[c] / 2.0) ** 2)
                if c not in self.bins or self._bins_w.get(c, 0.0) <= wn:
                    self.bins[c] = p
                    self._bins_w[c] = wn
        elif self.bins_centre is not None and len(self.bins) >= 2:
            # only a NEAR memory is a landmark: comparing the first close view with a
            # far-derived memory would read the 1 m tangent bias as odometry drift
            _w_near = 1.0 / (1.0 + (BIN_NEAR_RANGE / 2.0) ** 2)
            common = [c for c in new if c in self.bins and self._bins_w.get(c, 0.0) >= _w_near]
            if common:
                dlt = np.mean([new[c] - self.bins[c] for c in common], axis=0)
                if float(np.hypot(*dlt)) < 1.5:
                    corr = np.clip(0.3 * dlt, -0.3, 0.3)
                    self.ox -= float(corr[0])
                    self.oy -= float(corr[1])
                    self._pose_key = None
                    new = {c: p - corr for c, p in new.items()}
                    for c, p in new.items():
                        # range-weighted: a 2 m view outweighs an 8 m one 17:1
                        wn = 1.0 / (1.0 + (rng[c] / 2.0) ** 2)
                        wo = self._bins_w.get(c, 0.0)
                        if wo < _w_near:          # far-derived memory: replace it
                            wo = 0.0
                        self.bins[c] = ((self.bins[c] * wo + p * wn) / (wo + wn)
                                        if c in self.bins and wo > 0.0 else p)
                        self._bins_w[c] = wo + wn
                else:
                    self.bins = new            # not the same row we remembered
                    self._bins_w = {c: 1.0 / (1.0 + (rng[c] / 2.0) ** 2) for c in new}
            else:
                self.bins.update(new)
                for c in new:
                    self._bins_w[c] = 1.0 / (1.0 + (rng[c] / 2.0) ** 2)
        else:
            self.bins.update(new)
            for c in new:
                self._bins_w[c] = 1.0 / (1.0 + (rng[c] / 2.0) ** 2)
        # body-relative memory in the raw odometry chain (yaw-only body frame): the
        # S point of THIS sighting minus the S pose of this instant is exact, and the
        # odometry carries it to the drop without the corridor fits in between
        for c, p in new.items():
            rel = _rot2(-self.yaw_S(), p - np.array([self.ox, self.oy]))
            self._bin_obs[c] = (self.r.step_i, self.odo.copy(), rel, rng[c])
        self.bins_centre = np.mean(list(self.bins.values()), axis=0)
        self.fixes_bins += 1
        if os.environ.get("NAVEST_FIXLOG") == "1":
            self.r.log.ev(f"  [NAVFIX] bins {sorted(new)} rng "
                          f"{'/'.join(f'{rng[c]:.1f}' for c in sorted(new))} -> centre s "
                          f"{self.bins_centre[0]:+.1f} w {self.bins_centre[1]:+.2f}")

    def bin_obs_age(self, colour: str) -> int:
        """Steps since the bin was last SEEN (not dead-reckoned). Huge when never."""
        ob = self._bin_obs.get(colour)
        return 10 ** 9 if ob is None else int(self.r.step_i - ob[0])

    def bin_rel_body(self, colour: str, max_age: int = BIN_OBS_MAX_AGE):
        """The bin's centre in the CURRENT yaw-only body frame, dead-reckoned through
        the raw odometry chain from the latest direct sighting; None when there is no
        sighting fresher than `max_age` steps."""
        ob = self._bin_obs.get(colour)
        if ob is None:
            return None
        st, odo0, rel, _ = ob
        if self.r.step_i - st > max_age:
            return None
        if float(np.hypot(*(self.odo[:2] - odo0[:2]))) > BIN_OBS_MAX_MOVE:
            return None      # walked away from it: odometry has carried it too far
        # sighting body frame -> odo frame -> current body frame
        p_odo = odo0[:2] + _rot2(odo0[2], rel)
        return _rot2(-self.odo[2], p_odo - self.odo[:2])

    # ================================================================ S-frame pose
    def yaw_S(self) -> float:
        return _wrap(self.yaw_g - self.tangent_yaw)

    def _true_root(self):
        from isaaclab.utils import math as math_utils
        d = self.r.robot.data
        p = d.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        R = math_utils.matrix_from_quat(d.root_quat_w)[0].detach().cpu().numpy() \
            .astype(np.float64)
        yaw = float(math_utils.euler_xyz_from_quat(d.root_quat_w)[2][0])
        return p, R, yaw

    def _frames(self):
        """(p_root_S, Rz(yaw_S - yaw_true), R_true, p_root_true) cached per step."""
        key = (self.r.step_i, self.fixes_belt, self.fixes_bins, self._pose_key)
        if getattr(self, "_fr_key", None) == key:
            return self._fr
        p_t, R_t, yaw_t = self._true_root()
        Rd = _rz(self.yaw_S() - yaw_t)
        p_S = np.array([self.ox, self.oy, p_t[2]])
        self._fr = (p_S, Rd, R_t, p_t)
        self._fr_key = key
        return self._fr

    def pose(self):
        """(s, w, z, yaw_S) — the honest base pose for robot decisions."""
        p_S, _, _, _ = self._frames()
        return float(p_S[0]), float(p_S[1]), float(p_S[2]), self.yaw_S()

    def S_of_true(self, p_w) -> np.ndarray:
        """Sim-bookkeeping only: express a TRUE world point in the robot's S-frame
        (rigid, through the current estimate)."""
        p_S, Rd, _, p_t = self._frames()
        p_w = np.asarray(p_w, np.float64)
        if p_w.shape[-1] == 2:
            p_w = np.array([p_w[0], p_w[1], p_t[2]])
        return p_S + Rd @ (p_w - p_t)

    def true_of_S(self, p) -> np.ndarray:
        p_S, Rd, _, p_t = self._frames()
        p = np.asarray(p, np.float64)
        if p.shape[-1] == 2:
            p = np.array([p[0], p[1], p_S[2]])
        return p_t + Rd.T @ (p - p_S)

    def to_base(self, p) -> np.ndarray:
        """S-frame point -> full body frame (roll/pitch included, like the oracle's
        to_base_frame)."""
        p_S, Rd, R_t, _ = self._frames()
        p = np.asarray(p, np.float64)
        if p.shape[-1] == 2:
            p = np.array([p[0], p[1], p_S[2]])
        return R_t.T @ (Rd.T @ (p - p_S))

    def of_base(self, p_b) -> np.ndarray:
        p_S, Rd, R_t, _ = self._frames()
        p_b = np.asarray(p_b, np.float64)
        if p_b.shape[-1] == 2:
            p_b = np.array([p_b[0], p_b[1], 0.0])
        return p_S + Rd @ (R_t @ p_b)

    def cam_pose_S(self):
        """Head camera extrinsics in S: mount offset + IMU attitude + the estimate."""
        m = self.r.percep.mount
        p_S, Rd, _, p_t = self._frames()
        cam_p = p_S + Rd @ (np.asarray(m.pos_w, np.float64) - p_t)
        cam_R = Rd @ np.asarray(m.R_ros, np.float64)
        return cam_p, cam_R

    def _publish_extrinsics(self):
        m = self.r.percep.mount
        if m is None:
            return
        try:
            cam_p, cam_R = self.cam_pose_S()
            m.pos_est, m.R_est, m.use_est = cam_p, cam_R, True
        except Exception:
            m.use_est = False

    # ================================================================== queries
    def arc_lat(self):
        """(arc, lat) for robot decisions. Oracle passthrough unless honest."""
        if not self.honest:
            from course_mission.mission import arc_of
            return arc_of(*self.r.base_pose_true()[:2])
        return float(self.ox), float(self.oy)

    def lat(self) -> float:
        return self.arc_lat()[1]

    def psi_dr(self) -> float:
        """Dead-reckoned heading error (tangent - yaw), valid at any angle."""
        return _wrap(self.tangent_yaw - self.yaw_g)

    def psi(self, dir_sign: float = 1.0) -> float:
        """Heading error to the course direction (0 = aligned with dir_sign travel)."""
        if not self.honest:
            import course_sim.terrain as ct
            from course_mission.mission import arc_of
            X, Y, _, yaw = self.r.base_pose_true()
            ax, _ = arc_of(X, Y)
            th = float(ct.curve_theta(ax))
            if dir_sign < 0:
                th += np.pi
            return _wrap(th - yaw)
        p = self.psi_dr()
        if dir_sign < 0:
            p = _wrap(p + np.pi)
        return p

    def s_start(self) -> float:
        if not self.honest:
            import course_sim.terrain as ct
            return float(ct.COURSE_X0)
        return float(self.s0)

    def bins_arc(self):
        """Arc of the bin row: sighted centre (honest) or the surveyed centre; None
        while honest and never sighted."""
        if not self.honest:
            import course_sim.terrain as ct
            return float(ct.SECTION_CENTERS["bins"])
        if self.bins_centre is not None:
            return float(self.bins_centre[0])
        return None

    def bin_pos(self, colour: str):
        """Bin position in (arc, lat), or None if never sighted (honest)."""
        if not self.honest:
            from course_mission.runner import bin_targets
            bx, blat = bin_targets()[colour]
            return float(bx), float(blat)
        # fresh direct sighting: place it relative to the CURRENT S pose so the
        # runner's stand-off band (tgt - ax) is the true body-relative distance
        rel = self.bin_rel_body(colour)
        if rel is not None:
            p = np.array([self.ox, self.oy]) + _rot2(self.yaw_S(), rel)
            return float(p[0]), float(p[1])
        if colour in self.bins:
            return float(self.bins[colour][0]), float(self.bins[colour][1])
        if self.bins_centre is not None:
            return float(self.bins_centre[0]), float(self.bins_centre[1])
        return None

    # ---------------------------------------------------------- object anchoring
    def object_anchor(self, rel_b):
        """Start tracking the CURRENT approach target as a drift landmark (rel_b: the
        target's base-frame xy). Subsequent object_fix() calls read the target's
        apparent motion in S as odometry drift and cancel it."""
        if not self.honest:
            return
        self._obj_anchor = self._est_of_rel(rel_b)

    def object_fix(self, rel_b):
        if not self.honest or getattr(self, "_obj_anchor", None) is None:
            return
        p_new = self._est_of_rel(rel_b)
        dlt = p_new - self._obj_anchor
        n = float(np.hypot(*dlt))
        if n > 1.0:      # not the same object / bad detection — re-anchor instead
            self._obj_anchor = p_new
            return
        corr = np.clip(0.3 * dlt, -0.3, 0.3)
        self.ox -= float(corr[0])
        self.oy -= float(corr[1])
        self._pose_key = None
        self.since_fix = min(self.since_fix, 200)   # a landmark fix is a (partial) fix

    def _est_of_rel(self, rel_b):
        """Gravity-aligned body xy -> S (rotation by yaw_S, the same one step() uses)."""
        v = _rot2(self.yaw_S(), np.asarray(rel_b, np.float64)[:2])
        return np.array([self.ox + v[0], self.oy + v[1]])

    def reacquire(self, r, max_steps: int = 600):
        """Rotate slowly in place until the corridor is re-fitted (honest recovery).

        After a wedge or long occlusion the tracker is empty and the tangent may be
        stale; a slow scan brings posts back into view. Because the fit resolves its
        direction against the gyro-continued tangent, spinning cannot flip the frame."""
        if not self.honest:
            return
        import torch
        r.log.ev("  [NAVEST] reacquiring the corridor — scanning in place")
        self._reloc = True
        self._posts.clear()
        self._belt.clear()
        cmd = torch.zeros(1, 3, device=r.device)
        cmd[0, 0] = 0.02
        cmd[0, 2] = 0.35
        start_fix = self.fixes_belt
        good = 0
        for _ in range(max_steps):
            r._write_arm(r.travel_q, open_=False)
            r._step(r.legs.step(cmd))
            if self.fixes_belt > start_fix:
                start_fix = self.fixes_belt
                lf = self.last_fit or {}
                if lf.get("posts", 0) >= 2 or lf.get("pts", 0) >= 20:
                    good += 1
                if good >= 2:
                    r.log.ev(f"  [NAVEST] corridor reacquired (est s {self.ox:+.1f} "
                             f"w {self.oy:+.2f} psi {self.psi_dr():+.2f})")
                    self._reloc = False
                    return
        self._reloc = False
        r.log.ev("  [NAVEST] reacquire budget spent without a solid fit")

    def bin_base(self, colour: str):
        """Base-frame position of the bin's rim centre — the drop-centring target."""
        import course_sim.terrain as ct
        r = self.r
        if not self.honest:
            from course_mission.runner import bin_targets
            bx, blat = bin_targets()[colour]
            BX, BY = ct.curve_map(bx, blat)
            return r.to_base_frame_true(np.array([float(BX), float(BY), ct.BIN_HEIGHT]))
        body = self.bin_rel_body(colour)
        if body is None:
            bp = self.bin_pos(colour)
            if bp is None:
                return None
            body = _rot2(-self.yaw_S(), np.array([bp[0] - self.ox, bp[1] - self.oy]))
        bz = float(r.robot.data.root_pos_w[0, 2].cpu())
        return np.array([body[0], body[1], (self.ground_z() + ct.BIN_HEIGHT) - bz])

    def ground_z(self) -> float:
        """Self-estimated local ground height: the lowest foot (leg kinematics — what
        the joint encoders give) minus the foot radius; a fixed stance height below
        the base when no foot bodies are known."""
        r = self.r
        bz = float(r.robot.data.root_pos_w[0, 2].cpu())
        if not self.honest:
            from course_mission.runner import ground_z_arc
            from course_mission.mission import arc_of
            return ground_z_arc(arc_of(*r.base_pose_true()[:2])[0])
        fi = getattr(r, "_foot_idx", None)
        if fi:
            fz = r.robot.data.body_pos_w[0, fi, 2].detach().cpu().numpy()
            return float(np.min(fz)) - FOOT_R
        h = 0.24 if getattr(r.legs, "mode", "walk") == "crouch" else 0.42
        return bz - h

    # ------------------------------------------------------------ calibration
    def _calib_log(self):
        """Truth enters ONLY this print."""
        from course_mission.mission import arc_of, section_of
        import course_sim.terrain as ct
        r = self.r
        X, Y, _, yaw_t = r.base_pose_true()
        axt, wt = arc_of(X, Y)
        psi_t = _wrap(float(ct.curve_theta(axt)) - yaw_t)
        try:
            from course_mission.runner import ground_z_world
            gz_err = self.ground_z() - ground_z_world(X, Y)
        except Exception:
            gz_err = float("nan")
        r.log.ev(f"[NAVEST] step {r.step_i}: est s {self.ox:+.1f} (true {axt:+.1f}) "
                 f"w {self.oy:+.2f} (true {wt:+.2f}) psi {self.psi_dr():+.2f} "
                 f"(true {psi_t:+.2f}) fixes {self.fixes_belt}/{self.fit_tries} "
                 f"bins {self.fixes_bins} since {self.since_fix} curv {self.curv:+.3f} "
                 f"stairs {int(self._stairs)} pitch {self._pitch_ema:.2f} "
                 f"grade {self.foot_grade():+.3f} "
                 f"rough {self._rough:.4f}/{self._rough_walk:.4f} gz_err {gz_err:+.3f} [{section_of(axt)}]")
