#!/usr/bin/env python3
"""Timing and depth-substitution study on recorded frames.

Two questions, both answerable today with no hardware:

1. **Does the perception budget close on an N150?**  Measured here on a desktop core and
   reported with the 4-6x E-core penalty applied, because that is the number that decides
   whether object detection runs at 5 Hz or 1 Hz.

2. **What does it cost if the built-in camera has no depth channel?**  This is real-hardware
   confirmation #1 and the largest open question in the perception plan.  Rather than wait
   for the venue to find out, run the detector twice over the same recorded frames -- once
   with the true depth, once with `DepthSynth` standing in -- and compare the detections
   directly.  If the answer is "barely any difference", the risk evaporates.

Usage:
    python onboard/tools/bench.py --frames <dir-of-npz>
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import onboard  # noqa: E402,F401  (sets threads and MISSION_REAL_NOISE before perception)
from onboard import cfg as C  # noqa: E402
from onboard.perceive.depthsynth import DepthSynth  # noqa: E402

N150_PENALTY = (4.0, 6.0)


def _t(fn, *a, **k):
    t0 = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t0


def summarise(name: str, times: list[float]) -> None:
    a = np.asarray(times) * 1e3
    p50, p90 = float(np.percentile(a, 50)), float(np.percentile(a, 90))
    lo, hi = p50 * N150_PENALTY[0], p50 * N150_PENALTY[1]
    print(f"  {name:16s} p50 {p50:7.1f} ms  p90 {p90:7.1f}  max {a.max():7.1f}   "
          f"-> N150 {lo:6.0f}-{hi:6.0f} ms  ({1000/hi:4.1f}-{1000/lo:4.1f} Hz)")


def match(a: list[dict], b: list[dict], tol: float = 0.25) -> tuple[int, int, int]:
    """Greedy nearest match between two detection lists on class + position."""
    used, hit = set(), 0
    for da in a:
        best, bj = tol, None
        for j, db in enumerate(b):
            if j in used or db.get("cls") != da.get("cls"):
                continue
            d = float(np.linalg.norm(np.asarray(da["pos_w"])[:2] - np.asarray(db["pos_w"])[:2]))
            if d < best:
                best, bj = d, j
        if bj is not None:
            used.add(bj)
            hit += 1
    return hit, len(a) - hit, len(b) - hit      # matched, only-in-a, only-in-b


def lidar_like(depth: np.ndarray, lines: int, per_line: int, dilate: int,
               rng, sigma: float = 0.02) -> np.ndarray:
    """Carve a sparse scan out of the true depth: what a real 3D LiDAR would return,
    already projected into this pinhole.  Rows are scan planes, columns azimuth samples.

    Using the recorded depth as the source is the point -- it isolates the ONE variable we
    care about (sparsity) from everything else that differs between a depth camera and a
    LiDAR, so the retention numbers answer 'how much does sparsity cost' rather than
    'how good is my LiDAR model'.
    """
    from onboard.perceive.depthsynth import _dilate_nanmin
    out = np.full_like(depth, np.nan)
    rows = np.linspace(4, depth.shape[0] - 5, lines).astype(int)
    cols = np.linspace(0, depth.shape[1] - 1, per_line).astype(int)
    out[np.ix_(rows, cols)] = depth[np.ix_(rows, cols)] + rng.normal(0, sigma, (lines, per_line))
    return _dilate_nanmin(out.astype(np.float32), dilate)


def depth_study(files: list[str], K: np.ndarray, ds, seed: int = 0) -> None:
    """Reproduce the depth-substitution table in the README.

    This is the measurement that decides how much of Task 2 survives if the built-in camera
    turns out to have no depth channel, and it is worth being able to re-run rather than
    trusting a number in a document.
    """
    from course_mission.perception import Perception
    rng = np.random.default_rng(seed)

    def retained(make_depth, label: str) -> None:
        n_true = n_match = n_synth = 0
        for f in files:
            d = np.load(f, allow_pickle=True)
            dep = np.asarray(d["depth"], np.float32)
            cp = np.asarray(d["cam_pos"], np.float64)
            cR = np.asarray(d["cam_rot"], np.float64)
            gz = float(d["ground_z"])
            A = Perception.detect_arrays(d["rgb"], dep, K, cp, cR, gz, add_noise=False)
            B = Perception.detect_arrays(d["rgb"], make_depth(dep, cp, cR, gz), K, cp, cR,
                                         gz, add_noise=False)
            n_true += len(A)
            n_synth += len(B)
            m, _, _ = match(A, B)
            n_match += m
        pct = 100.0 * n_match / max(n_true, 1)
        print(f"  {label:34s} true {n_true:3d}  synth {n_synth:3d}  retained {pct:5.1f} %")

    print("\n-- depth substitution study (README table) --")
    retained(lambda dep, cp, cR, gz: dep, "true depth (control)")
    retained(lambda dep, cp, cR, gz: ds.from_ground(cp, cR, gz), "ground plane only")
    for lines, per_line, dil in ((16, 900, 3), (32, 1200, 2), (64, 1800, 2), (128, 2400, 1)):
        retained(lambda dep, cp, cR, gz, l=lines, p=per_line, k=dil:
                 lidar_like(dep, l, p, k, rng), f"LiDAR {lines} lines")

    def filled(dep, cp, cR, gz):
        z = lidar_like(dep, 16, 900, 3, rng)
        return np.where(np.isnan(z), ds.from_ground(cp, cR, gz), z)
    retained(filled, "LiDAR 16 + ground-plane fill")
    print("  (the last row is WORSE than either input alone: detect_arrays discriminates on\n"
          "   height above ground, and a plane-filled pixel is by construction at height 0)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--depth-study", action="store_true",
                    help="reproduce the README's depth-substitution table")
    a = ap.parse_args()

    from course_mission.perception import Perception, detect_belt, detect_bins

    files = sorted(glob.glob(str(Path(a.frames) / "*.npz")))[: a.limit]
    if not files:
        print(f"no .npz frames under {a.frames}", file=sys.stderr)
        return 2
    print(f"== {len(files)} recorded frames from {a.frames} ==")

    sensors = C.load("sensors")
    d0 = np.load(files[0], allow_pickle=True)
    K = np.asarray(d0["K"], np.float64)
    h, w = d0["rgb"].shape[:2]
    ds = DepthSynth(sensors, K, w, h)
    print(f"   frame {w}x{h}  fx={K[0,0]:.1f} cx={K[0,2]:.1f}  "
          f"(virtual pinhole: fx={sensors['camera']['virtual_pinhole']['fx']}, "
          f"cx={sensors['camera']['virtual_pinhole']['cx']})")

    tb, tn, to, tg = [], [], [], []
    n_true, n_synth, n_match, n_lost, n_extra = 0, 0, 0, 0, 0
    depth_err = []

    for f in files:
        d = np.load(f, allow_pickle=True)
        rgb = d["rgb"]
        depth = np.asarray(d["depth"], np.float32)
        cam_p = np.asarray(d["cam_pos"], np.float64)
        cam_R = np.asarray(d["cam_rot"], np.float64)
        gz = float(d["ground_z"])

        _, dt = _t(detect_belt, rgb, K, cam_p, cam_R, gz, add_noise=False)
        tb.append(dt)
        _, dt = _t(detect_bins, rgb, K, cam_p, cam_R, gz, add_noise=False)
        tn.append(dt)

        synth, dt = _t(ds.from_ground, cam_p, cam_R, gz)
        tg.append(dt)

        # How wrong is the substitute where both are valid?  Only over pixels the real
        # sensor saw AND the plane model covers -- elsewhere they are answering different
        # questions, and averaging that in would flatter or damn the model for no reason.
        both = np.isfinite(synth) & np.isfinite(depth) & (depth > 0.1)
        if both.any():
            depth_err.append(float(np.median(np.abs(synth[both] - depth[both]))))

        det_true, dt = _t(Perception.detect_arrays, rgb, depth, K, cam_p, cam_R, gz,
                          add_noise=False)
        to.append(dt)
        det_synth = Perception.detect_arrays(rgb, synth, K, cam_p, cam_R, gz, add_noise=False)

        m, lost, extra = match(det_true, det_synth)
        n_true += len(det_true)
        n_synth += len(det_synth)
        n_match += m
        n_lost += lost
        n_extra += extra

    print("\n-- timing (this core; N150 projection at 4-6x) --")
    summarise("detect_belt", tb)
    summarise("detect_bins", tn)
    summarise("detect_arrays", to)
    summarise("depth from plane", tg)

    print("\n-- depth substitution: ground plane instead of the depth channel --")
    if depth_err:
        e = np.asarray(depth_err)
        print(f"  median |synth - true| over valid pixels: {np.median(e)*100:.1f} cm "
              f"(p90 {np.percentile(e,90)*100:.1f} cm)")
    print(f"  detections with true depth : {n_true}")
    print(f"  detections with synth depth: {n_synth}")
    print(f"  agreeing (same class, <25 cm): {n_match}"
          f"   lost: {n_lost}   spurious: {n_extra}")
    if n_true:
        print(f"  retained {100.0 * n_match / n_true:.1f} % of the depth-camera detections")

    if a.depth_study:
        depth_study(files, K, ds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
