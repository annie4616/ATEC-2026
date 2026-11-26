"""Effective detector rate on the N100, per _blobs backend, from recorded frames.

bench_blobs.py timed the blob step in isolation.  This times the WHOLE detector frame the
mission actually calls -- detect_arrays + detect_belt + detect_bins, i.e. label_map + every
_blobs call together -- once with BLOBS_BACKEND=py and once with =fast, over real recorded
frames, and converts each to the effective detect_hz the mission would see on the mini-PC's
N100 (desktop time x the 4-6x E-core penalty).

Those two rates are exactly what the before/after mission videos should run at:
  before = MISSION_DETECT_HZ=<py rate>   BLOBS_BACKEND=py
  after  = MISSION_DETECT_HZ=<fast rate> BLOBS_BACKEND=fast

Usage:
    python course_mission/scripts/bench_detect_hz.py --frames course_mission/dataset_bench
"""
from __future__ import annotations

import argparse
import glob
import importlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

N100_PENALTY = (4.0, 6.0)


def _time_backend(files, backend: str, warmup: int = 3):
    import os
    os.environ["BLOBS_BACKEND"] = backend
    # perception reads _BLOBS_BACKEND at import; reload so the switch takes effect.
    import course_mission.perception as P
    importlib.reload(P)

    frames = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        frames.append((d["rgb"], np.asarray(d["depth"], np.float32),
                       np.asarray(d["K"], np.float64),
                       np.asarray(d["cam_pos"], np.float64),
                       np.asarray(d["cam_rot"], np.float64), float(d["ground_z"])))

    def one(fr):
        rgb, depth, K, cp, cR, gz = fr
        P.Perception.detect_arrays(rgb, depth, K, cp, cR, gz, add_noise=False)
        P.detect_belt(rgb, K, cp, cR, gz, add_noise=False)
        try:
            P.detect_bins(rgb, K, cp, cR, gz, add_noise=False, depth=depth)
        except Exception:
            pass

    for _ in range(warmup):
        one(frames[0])
    per = []
    for fr in frames:
        t0 = time.perf_counter()
        one(fr)
        per.append(time.perf_counter() - t0)
    return np.asarray(per) * 1e3   # ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()

    files = sorted(glob.glob(str(Path(a.frames) / "*.npz")))[: a.limit]
    if not files:
        print(f"no .npz under {a.frames}", file=sys.stderr)
        return 2
    print(f"== whole-detector timing over {len(files)} frames ==\n")

    rows = {}
    for backend in ("py", "fast"):
        ms = _time_backend(files, backend)
        p50, p90 = float(np.percentile(ms, 50)), float(np.percentile(ms, 90))
        # effective rate: the detector cannot start the next frame until this one finishes,
        # so sustained rate = 1 / frame_time.  Use p90 (the venue must survive the slow ones).
        n100_lo, n100_hi = p90 * N100_PENALTY[0], p90 * N100_PENALTY[1]
        hz_lo, hz_hi = 1000.0 / n100_hi, 1000.0 / n100_lo
        rows[backend] = (p50, p90, n100_lo, n100_hi, hz_lo, hz_hi)
        print(f"  BLOBS_BACKEND={backend:4s}  desktop p50 {p50:6.1f} ms  p90 {p90:6.1f} ms")
        print(f"       -> N100 frame {n100_lo:6.0f}-{n100_hi:6.0f} ms   "
              f"effective {hz_lo:4.1f}-{hz_hi:4.1f} Hz\n")

    # recommendation for the videos: conservative end (slowest) of each backend's range
    py_hz = rows["py"][4]
    fast_hz = rows["fast"][4]
    print("-- recommended before/after mission video settings --")
    print(f"  before : BLOBS_BACKEND=py   MISSION_DETECT_HZ={py_hz:.1f}")
    print(f"  after  : BLOBS_BACKEND=fast MISSION_DETECT_HZ={fast_hz:.1f}")
    speedup = rows["py"][1] / max(rows["fast"][1], 1e-9)
    print(f"  (whole-detector speedup: {speedup:.1f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
