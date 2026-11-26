#!/usr/bin/env python3
"""Score the object tracker offline, against recorded frames with ground truth.

The number this exists to produce: **what fraction of the tracks the mission would walk to
are pointing at the wrong article.**  In the simulator's honest runs that was 32 % -- the
grasp-point error distribution is bimodal, 68 % under 100 mm and 32 % over a metre, with
nothing in between, which is the signature of aiming at a different object rather than of
aiming imprecisely at the right one.

That is the single largest, cheapest defect in Task 2, and it is measurable here on a
desktop with no robot and no simulator: the recorded frames carry `objs`/`classes`, so every
track can be compared against the nearest true article of its class.

    python onboard/tools/check_collector.py --frames <dir-of-npz>

Reports the mis-aim rate, the error distribution, and how the `min_views` gate trades
coverage against ghosts.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import onboard  # noqa: E402,F401  threads + noise off before perception
from onboard import cfg as C  # noqa: E402
from onboard.perceive.objtrack import ObjectTracker  # noqa: E402


def nearest_true(p, cls, names, classes, objs):
    """Distance from a track to the nearest TRUE article of the same class."""
    best = None
    for i, c in enumerate(classes):
        if str(c) != cls:
            continue
        d = float(np.linalg.norm(np.asarray(objs[i], float)[:2] - np.asarray(p, float)[:2]))
        if best is None or d < best[0]:
            best = (d, str(names[i]))
    return best


def run(files: list[str], min_views: int, verbose: bool) -> dict:
    from course_mission.perception import Perception

    cfg = C.load("collector")
    cfg["track"] = dict(cfg["track"]) | {"min_views": min_views}
    trk = ObjectTracker(cfg)

    names = classes = objs = None
    n_det = 0
    for step, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        names, classes, objs = d["names"], d["classes"], d["objs"]
        dets = Perception.detect_arrays(
            d["rgb"], np.asarray(d["depth"], np.float32), np.asarray(d["K"], np.float64),
            np.asarray(d["cam_pos"], np.float64), np.asarray(d["cam_rot"], np.float64),
            float(d["ground_z"]), add_noise=False)
        n_det += len(dets)
        trk.observe_all(dets, step)

    # Apply the SELECTION gates too.  "The tracks the mission would walk to" is not the same
    # set as "the tracks that exist": distractors (bottle, purplebag) are never grasped, the
    # skip list removes the bag, and the corridor/range gates drop what the robot would not
    # reach.  Scoring without them measures the detector, not the decision.
    from course_mission.catalogue import BIN_OF
    sel = cfg["select"]
    skip = set(sel["skip_classes"])
    corridor = float(sel["corridor_lat_m"])

    rows = []
    for t in trk.tracks.values():
        if not trk.engageable(t):
            continue
        if BIN_OF.get(t.cls) is None or t.cls in skip:
            continue                      # distractor or explicitly skipped
        # NOTE: the corridor gate is deliberately NOT applied here.  In the mission `pos_w`
        # is an S-frame point whose y IS the lateral offset; in these recordings the camera
        # extrinsics are world, so y is a world coordinate that sweeps +-7.5 m along the
        # S-curve.  Applying the gate offline rejects almost everything for a reason that
        # has nothing to do with the tracker.
        _ = corridor
        hit = nearest_true(t.p, t.cls, names, classes, objs)
        rows.append({"tid": t.tid, "cls": t.cls, "views": t.n_views,
                     "best_r": t.best_r, "spread": t.spread,
                     "err": 9.9 if hit is None else hit[0],
                     "match": None if hit is None else hit[1]})
    if verbose:
        for r in sorted(rows, key=lambda r: -r["err"]):
            print(f"    {r['tid']} {r['cls']:9s} views={r['views']:2d} "
                  f"best_r={r['best_r']:.2f} spread={r['spread']*1000:5.0f}mm "
                  f"err={r['err']*1000:7.0f}mm -> {r['match']}")
    errs = np.array([r["err"] for r in rows]) if rows else np.zeros(0)
    return {"tracker": trk.report(), "rows": rows, "errs": errs, "n_det": n_det}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    files = sorted(glob.glob(str(Path(a.frames) / "*.npz")))[: a.limit]
    if not files:
        print(f"no .npz frames under {a.frames}", file=sys.stderr)
        return 2
    print(f"== object tracker on {len(files)} recorded frames ==\n")

    base = run(files, min_views=int(C.load("collector")["track"]["min_views"]), verbose=a.verbose)
    e, rep = base["errs"], base["tracker"]
    print(f"  detections fed      : {base['n_det']}")
    print(f"  tracks created      : {rep['tracks']}  (engageable {rep['engageable']})")
    print(f"  associations / new  : {rep['associations']} / {rep['new_tracks']}")
    print(f"  live by class       : {rep['by_class']}")

    if not len(e):
        print("\n  no engageable tracks -- nothing to score")
        return 1

    # TWO different defects hide in one distance, and they want opposite fixes.
    #
    #   * No article of this class anywhere near  -> the CLASS is wrong (or the blob is not
    #     an article at all).  The detector's measured precision is 45-59 %, so this is
    #     expected and it is a classification problem.
    #   * An article of this class nearby, but not where we think  -> LOCALISATION.
    #
    # Reporting one number for both is how "the tracker is bad" gets concluded from a
    # detector precision figure.  Split at 1 m: two articles of the same class are never
    # that close on this course.
    phantom = e > 1.0
    loc = e[~phantom]
    print(f"\n  -- what the mission would walk to --")
    print(f"  n = {len(e)}")
    print(f"  no article of that class within 1 m : {int(phantom.sum()):3d} "
          f"({100*float(phantom.mean()):5.1f} %)   <- CLASSIFICATION")
    if len(loc):
        print(f"  the rest, localisation error        : n={len(loc)}  "
              f"median {np.median(loc)*1000:5.0f} mm  p90 {np.percentile(loc, 90)*1000:5.0f} mm")
        print(f"      of those, < 50 mm (grasp-viable): {int((loc < 0.05).sum())} "
              f"({100*float((loc < 0.05).mean()):.0f} %)")
    print(f"\n  This pass scores tracks as the SEARCH leaves them.  The mission commits only")
    print(f"  after `stop_and_look` -- back off to 1.25 m, stand still, take six confirming")
    print(f"  views -- which is why its measured figure is better than this one.  The gap")
    print(f"  between the two IS the value of that refinement, and it is worth knowing.")
    print(f"\n  Reference points from the simulator's honest runs:")
    print(f"    grasp attempts with aim error > 1 m ......... 32 %")
    print(f"    grasp attempts with aim error < 50 mm ....... 16 %")
    print(f"    detector precision / recall (photo world) ... 59 % / 76 %")

    print(f"\n  -- min_views trade-off --")
    for mv in (1, 2, 3):
        r = run(files, min_views=mv, verbose=False)
        ee = r["errs"]
        if not len(ee):
            print(f"    min_views={mv}: no engageable tracks")
            continue
        ph = ee > 1.0
        lo = ee[~ph]
        print(f"    min_views={mv}: {len(ee):3d} engageable, "
              f"{100*float(ph.mean()):5.1f} % wrong-class, "
              f"localisation median "
              f"{(np.median(lo)*1000 if len(lo) else float('nan')):5.0f} mm")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
