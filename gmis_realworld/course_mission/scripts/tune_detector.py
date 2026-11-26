"""Score the detector against the recorded dataset — no GPU, no Isaac, ~1 s per pass.

Reports the two numbers that decide whether the mission can run autonomously:
  precision — of everything reported, how much was a real object of that class nearby
  recall    — of the objects genuinely in view, how many were reported

and, because the smoke run showed the classes were right while the positions were not,
it separates the failures: wrong class vs right class with a bad position, resolved by
range. Run with --patch to try a variant without touching mission.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_S = Path(__file__).resolve()
RW = _S.parents[2]
for _p in (str(RW / "source" / "atec_rl_lab"), str(RW / "eunha_rl"), str(RW / "a2pro_rl"), str(RW)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from course_mission.perception import Perception  # noqa: E402

HIT = 0.45          # a detection counts as correct within this radius (mission tolerance)
CONE = 0.55         # half-angle of the "genuinely in view" test, radians
                    # true half-FOV = atan(cx/fx) = atan(320/488.666) = 0.579 rad;
                    # 0.45 (25.8 deg) under-counted what is genuinely visible
VIS_RANGE = 2.2     # = Perception.FAR_CUT: beyond it nothing can ever be reported


def visible_objects(f):
    """Ground-truth objects inside the camera cone, in the robot's frame."""
    q = f["root_quat"]
    w, x, y, z = q
    Rz = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                   [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    out = []
    for nm, cl, p in zip(f["names"], f["classes"], f["objs"]):
        rel = Rz.T @ (p - f["root_pos"])
        if rel[0] > 0.5 and np.linalg.norm(rel[:2]) < VIS_RANGE and \
                abs(np.arctan2(rel[1], rel[0])) < CONE:
            out.append((str(nm), str(cl), p))
    return out


def score(files, method="colour", add_noise=True, seeds=3):
    tp = fp_class = fp_pos = 0
    errs, by_range = [], {}
    found, seen = set(), set()
    per_cls = {}
    for fi, fn in enumerate(files):
        f = np.load(fn, allow_pickle=True)
        fn_det = (Perception.detect_height if method == "height"
                  else Perception.detect_arrays)
        rot = f["cam_rot"] if "cam_rot" in f else f["cam_quat_ros"]
        vis = visible_objects(f)
        for nm, _cl, _p in vis:
            seen.add(nm)
        # Sensor noise is RANDOM, so one draw per frame is one sample of a distribution —
        # and it is the distribution the robot actually runs in (deploy.REAL_NOISE
        # defaults to 1 and the mission log prints "sensor noise=ON"). This tool used to
        # score add_noise=False, so every threshold in perception.py was tuned against a
        # detector that has never been deployed. Seed per (seed, frame) so an A/B is
        # comparable rather than a reroll of the noise.
        dets = []
        for sd in range(max(1, seeds)):
            np.random.seed(1000 * sd + fi)
            dets += list(fn_det(f["rgb"], f["depth"], f["K"], f["cam_pos"], rot,
                                float(f["ground_z"]), add_noise=add_noise))
        for d in dets:
            same = [(nm, p) for nm, cl, p in zip(f["names"], f["classes"], f["objs"])
                    if str(cl) == d["cls"]]
            if not same:
                fp_class += 1
                continue
            e = min(float(np.linalg.norm(p[:2] - d["pos_w"][:2])) for _nm, p in same)
            nm = min(same, key=lambda t: np.linalg.norm(t[1][:2] - d["pos_w"][:2]))[0]
            errs.append((float(d["dist"]), e))
            b = min(int(d["dist"] / 0.5), 5)
            by_range.setdefault(b, []).append(e)
            if e < HIT:
                tp += 1
                found.add(str(nm))
                per_cls.setdefault(d["cls"], [0, 0])[0] += 1
            else:
                # right class exists in the world, but this report points somewhere else
                fp_pos += 1
                per_cls.setdefault(d["cls"], [0, 0])[1] += 1
    tot = tp + fp_class + fp_pos
    print(f"frames={len(files)} x {max(1, seeds)} noise draw(s)  noise={add_noise}  detections={tot}")
    for c in sorted(per_cls):
        good, bad = per_cls[c]
        print(f"    {c:11s} reports={good + bad:4d}  good={good:4d} "
              f"({100 * good / max(good + bad, 1):3.0f}%)")
    print(f"  precision      {tp}/{tot} ({100 * tp / max(tot, 1):.0f}%)")
    print(f"  recall         {len(found & seen)}/{len(seen)} "
          f"({100 * len(found & seen) / max(len(seen), 1):.0f}%)")
    print(f"  wrong class    {fp_class}")
    print(f"  bad position   {fp_pos}   <- right class, position off by >{HIT} m")
    if errs:
        a = np.array([e for _r, e in errs])
        print(f"  position error median {np.median(a):.2f} m  p90 {np.percentile(a, 90):.2f} m")
    for b in sorted(by_range):
        v = np.array(by_range[b])
        lo = b * 0.5
        print(f"  range {lo:.1f}-{lo + 0.5:.1f} m: n={len(v):4d} "
              f"within-{HIT:.2f}={100 * (v < HIT).mean():3.0f}%  median={np.median(v):.2f} m")
    return tp, tot, len(found & seen), len(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(RW / "course_mission" / "dataset"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--method", default="both", choices=["colour", "height", "both"])
    ap.add_argument("--set", action="append", default=[],
                    help="Perception attribute override, e.g. --set MAX_SPAN=0.22")
    ap.add_argument("--noise", type=int, default=1,
                    help="1 = score as deployed (deploy.REAL_NOISE default), 0 = clean")
    ap.add_argument("--seeds", type=int, default=3,
                    help="noise draws per frame; >1 because one draw is one sample")
    args = ap.parse_args()
    files = sorted(Path(args.data).glob("f*.npz"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"no frames in {args.data} — run scripts/collect_frames.py first")
        return
    for kv in args.set:
        k, v = kv.split("=", 1)
        setattr(Perception, k, eval(v))
        print(f"[override] Perception.{k} = {getattr(Perception, k)}")
    for m in (["colour", "height"] if args.method == "both" else [args.method]):
        print(f"===== {m} detector =====")
        score(files, m, add_noise=bool(args.noise), seeds=args.seeds)
        print()


if __name__ == "__main__":
    main()
