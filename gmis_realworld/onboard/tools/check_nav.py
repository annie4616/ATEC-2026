#!/usr/bin/env python3
"""Closed-loop corridor following, with no robot and no simulator.

The pieces were each verified alone -- the rail detector recovers offset to a few mm, the
EKF self-tests against a synthetic S-curve, the steering law came from the sim.  None of
that says the loop is stable.  Detection lag, the filter's own dynamics and the steering
gains interact, and the failure they produce is the one already seen once in this project:
an estimate that looks fine in isolation and a robot that leaves the course at the stairs.

So: close the loop.  The mock robot's true lateral offset drives the synthetic LiDAR, the
detector and filter see only what the sensor would give, the steering output is fed back as
the velocity command, and the truth is scored against the estimate.  Dropouts and a heavy
noise tail are injected because the real detector has a 52 % fix rate, not 100 %.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onboard import cfg as C  # noqa: E402
from onboard.estimate.navstate import NavState  # noqa: E402
from onboard.perceive.railnav import RailDetector  # noqa: E402


def run(steps: int, dt: float, fix_rate: float, tail: float, curve: float,
        lat0: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    det = RailDetector(C.load("sensors"))
    nav = NavState(C.load("mission_hk"), C.load("sensors"))
    half = 1.9775

    lat, head = float(lat0), 0.0      # truth: offset from centreline (+left), heading error
    v_nom = 0.7
    err, oob = [], 0
    t = 0.0

    for i in range(steps):
        # ---- truth propagation ---------------------------------------------------------
        # Course curvature turns the corridor under the robot; the heading error grows with
        # it unless the controller keeps up, which is the thing being tested.
        cmd = nav.steer(v_nom)
        vx, vy, wz = float(cmd[0]), float(cmd[1]), float(cmd[2])
        head += (curve * vx - wz) * dt
        lat += (-vx * np.sin(head) + vy * np.cos(head)) * dt
        t += dt

        # ---- proprioception (what the filter propagates on) ------------------------------
        class _Odo:
            vel_body = np.array([vx, vy, 0.0])

        class _Att:
            yaw_rate = wz
        nav.predict(t, _Odo(), _Att())

        # ---- sensor ---------------------------------------------------------------------
        if rng.random() < fix_rate:
            pts = _synth_cloud(rng, lat, head, half)
            fix = det.detect(pts)
            # Perturb the FIX, not the points.  Per-point noise averages out across the
            # hundreds of returns in a fit -- the first version of this test did that and
            # reported a 0.2 mm estimate error, which would have been a very reassuring
            # lie.  Real detector error is a per-FRAME bias: a pitch or ground-plane error
            # displaces the whole fit at once.  These magnitudes are the measured ones for
            # detect_belt (262 mm / 5.4 deg median, with a p90 tail at 1.8 m / 24.8 deg).
            if fix is not None:
                heavy = rng.random() < tail
                sl = 1.2 if heavy else 0.26
                sh = np.deg2rad(24.8 if heavy else 5.4)
                fix.b_left += rng.normal(0.0, sl)
                fix.b_right += rng.normal(0.0, sl)
                fix.heading += rng.normal(0.0, sh)
            nav.update_rail(fix, t)

        err.append(abs(nav.lat - lat))
        if abs(lat) > half:
            oob += 1

    e = np.asarray(err)
    return {"est_err_median_mm": float(np.median(e) * 1e3),
            "est_err_p90_mm": float(np.percentile(e, 90) * 1e3),
            "est_err_max_mm": float(e.max() * 1e3),
            "true_lat_final_mm": float(lat * 1e3),
            "oob_steps": oob,
            "fixes": nav.n_lidar_fix, "rejected": nav.n_rejected}


def _synth_cloud(rng, lat, head, half, n=300):
    """Two rails as the LiDAR would see them, in the sensor's own frame."""
    noise = 0.012
    x = rng.uniform(0.3, 3.0, n)
    out = []
    for side in (+1.0, -1.0):
        y = side * half - lat + x * np.tan(head) + rng.normal(0, noise, n)
        z = rng.uniform(0.02, 0.40, n)
        out.append(np.stack([x, y, z], 1))
    out.append(np.stack([rng.uniform(0.3, 3.0, n), rng.uniform(-1.5, 1.5, n),
                         rng.normal(0, 0.01, n)], 1))
    return np.concatenate(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)     # 60 s at 50 Hz
    ap.add_argument("--fix-rate", type=float, default=0.52)  # the measured detector rate
    ap.add_argument("--tail", type=float, default=0.10)      # fraction of fixes in the heavy tail
    ap.add_argument("--curve", type=float, default=0.029)    # sim S-curve peak, 1/m
    ap.add_argument("--lat0", type=float, default=0.8)       # start well off centre
    a = ap.parse_args()

    print(f"closed-loop corridor following: fix_rate={a.fix_rate} tail={a.tail} "
          f"curvature={a.curve}/m  start offset={a.lat0} m")
    worst_true, fails = 0.0, 0
    for seed in range(8):
        r = run(a.steps, 0.02, a.fix_rate, a.tail, a.curve, a.lat0, seed)
        worst_true = max(worst_true, abs(r["true_lat_final_mm"]))
        bad = r["oob_steps"] > 0
        fails += bad
        print(f"  seed {seed}: est err med {r['est_err_median_mm']:6.1f} mm  "
              f"p90 {r['est_err_p90_mm']:6.1f}  max {r['est_err_max_mm']:7.1f}  |  "
              f"final true offset {r['true_lat_final_mm']:+7.1f} mm  "
              f"fixes {r['fixes']}/{r['fixes'] + r['rejected']}  "
              f"{'OUT OF BOUNDS' if bad else 'in bounds'}")
    ok = fails == 0 and worst_true < 500.0
    print(f"\nworst final true offset {worst_true:.0f} mm over 8 seeds, "
          f"{fails} runs left the corridor")
    print("NAV OK" if ok else "NAV FAILED")
    raise SystemExit(0 if ok else 1)
