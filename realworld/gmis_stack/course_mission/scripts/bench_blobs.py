"""Equivalence + speed A/B for the two `_blobs` backends.

`_blobs` is the connected-components step every object detector calls (detect_arrays,
detect_belt, detect_bins).  The pure-Python `_blobs_py` held the GIL and dominated the
perception frame on the mini-PC's N100 E-cores; `_blobs_fast` is a vectorised + scipy
replacement.  Correctness FIRST: for a random battery of masks shaped like the real ones
(object blobs, frame-wide speckle, empty frames) the two backends must return the same
blobs -- same pixel sets, same order, under every (min_pix, max_blobs, min_cell) the call
sites use.  Then the speed, with the N100 penalty applied.

No simulator, no GPU: run it anywhere.
    python course_mission/scripts/bench_blobs.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from course_mission.perception import _blobs_fast, _blobs_py  # noqa: E402

N100_PENALTY = (4.0, 6.0)
H, W = 360, 640
# (min_pix, max_blobs, min_cell) exactly as the five call sites pass them.
CALL_SITES = [
    (30, 16, 0),      # detect_height mask
    (30, 16, 0),      # detect_arrays object mask (MIN_PIX)
    (10, 16, 0),      # detect_belt red (min_cell=0)
    (10, 16, 0),      # detect_belt red (2nd)
    (250, 6, 3),      # detect_bins (min_pix=250, max_blobs=6, min_cell=3)
]


def _blobset(blobs):
    """A hashable, order-sensitive signature: list of frozenset of (y,x)."""
    return [frozenset(zip(map(int, ys), map(int, xs))) for ys, xs in blobs]


def make_mask(rng: np.random.Generator, kind: str) -> np.ndarray:
    m = np.zeros((H, W), bool)
    if kind == "empty":
        return m
    if kind in ("objects", "objects+speckle"):
        for _ in range(rng.integers(1, 6)):
            cy, cx = rng.integers(20, H - 20), rng.integers(20, W - 20)
            ry, rx = rng.integers(6, 40), rng.integers(6, 40)
            y0, y1 = max(0, cy - ry), min(H, cy + ry)
            x0, x1 = max(0, cx - rx), min(W, cx + rx)
            m[y0:y1, x0:x1] = True
    if kind in ("speckle", "objects+speckle"):
        n = rng.integers(200, 3000)
        m[rng.integers(0, H, n), rng.integers(0, W, n)] = True
    if kind == "dense":
        m[rng.integers(0, H, 40000), rng.integers(0, W, 40000)] = True
    return m


def equivalence(seed: int = 0) -> bool:
    rng = np.random.default_rng(seed)
    kinds = ["empty", "objects", "speckle", "objects+speckle", "dense"]
    ok = True
    checked = 0
    for trial in range(400):
        kind = kinds[trial % len(kinds)]
        mask = make_mask(rng, kind)
        for min_pix, max_blobs, min_cell in CALL_SITES:
            a = _blobs_py(mask, min_pix, max_blobs, min_cell)
            b = _blobs_fast(mask, min_pix, max_blobs, min_cell)
            checked += 1
            if _blobset(a) != _blobset(b):
                ok = False
                print(f"  MISMATCH  kind={kind} args=({min_pix},{max_blobs},{min_cell}) "
                      f"py={len(a)} blobs fast={len(b)} blobs")
                # show the first differing blob sizes
                sa = sorted(len(x[0]) for x in a)
                sb = sorted(len(x[0]) for x in b)
                print(f"            py sizes={sa}\n            fast sizes={sb}")
                if checked > 6:
                    return ok
    print(f"  equivalence: {'PASS' if ok else 'FAIL'}  ({checked} comparisons)")
    return ok


def speed(seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    masks = [make_mask(rng, k) for k in
             ["objects+speckle"] * 40 + ["objects"] * 20 + ["dense"] * 10]
    print("\n-- speed (this core; N100 projection at 4-6x) --")
    for name, fn in (("_blobs_py", _blobs_py), ("_blobs_fast", _blobs_fast)):
        ts = []
        for _ in range(3):                       # warm + measure
            t0 = time.perf_counter()
            for mk in masks:
                fn(mk, 30, 16, 0)
            ts.append((time.perf_counter() - t0) / len(masks))
        per = min(ts) * 1e3
        lo, hi = per * N100_PENALTY[0], per * N100_PENALTY[1]
        print(f"  {name:12s} {per:6.2f} ms/mask   -> N100 {lo:6.1f}-{hi:6.1f} ms")


if __name__ == "__main__":
    print("== _blobs backend A/B ==")
    passed = equivalence()
    speed()
    raise SystemExit(0 if passed else 1)
