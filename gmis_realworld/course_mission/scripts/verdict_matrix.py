"""Fit the PORTABLE grasp verdict against the sim-truth one.

`held()` (runner.py) decides success from the object's true pose, velocity and rest
height — none of which a real robot has. The two signals it does have are finger WIDTH
and finger EFFORT, both now logged per attempt as `[VERDICT]` lines. This reads those
lines out of any number of run logs and answers one question:

    can width+effort reproduce held()'s verdict, and at what thresholds?

Usage:
    python course_mission/scripts/verdict_matrix.py course_mission/logs/deploy_probe_*.log
"""
import re
import sys
from itertools import product

LINE = re.compile(
    r"\[VERDICT\]\s+(\S+)\s+truth=(HELD|MISS)\s+"
    r"gw_shut=([-\d.]+)mm\s+gw_min=([-\d.]+)mm\s+"
    r"effort_peak=([-\d.]+)(?:\s+gw_end=([-\d.]+)mm\s+effort_end=([-\d.]+))?"
    r"\s+lift=([-+\d.]+)")
# the first probe wave logged "effort=" before the end-of-attempt fields were added
LINE_OLD = re.compile(
    r"\[VERDICT\]\s+(\S+)\s+truth=(HELD|MISS)\s+gw_shut=([-\d.]+)mm\s+"
    r"gw_min=([-\d.]+)mm\s+effort=([-\d.]+)\s+lift=([-+\d.]+)")


def parse(paths):
    rows = []
    for p in paths:
        tag = p.split("deploy_probe_")[-1].split(".log")[0] if "deploy_probe_" in p else p
        try:
            txt = open(p, errors="ignore").read()
        except OSError:
            continue
        for m in LINE.finditer(txt):
            name, truth, gw_shut, gw_min, ep, gw_end, ee, lift = m.groups()
            rows.append(dict(tag=tag, name=name, held=truth == "HELD",
                             gw_shut=float(gw_shut), gw_min=float(gw_min),
                             effort_peak=float(ep),
                             gw_end=None if gw_end is None else float(gw_end),
                             effort_end=None if ee is None else float(ee),
                             lift=float(lift)))
        if not any(r["tag"] == tag for r in rows):
            for m in LINE_OLD.finditer(txt):
                name, truth, gw_shut, gw_min, ep, lift = m.groups()
                rows.append(dict(tag=tag, name=name, held=truth == "HELD",
                                 gw_shut=float(gw_shut), gw_min=float(gw_min),
                                 effort_peak=float(ep), gw_end=None, effort_end=None,
                                 lift=float(lift)))
    return rows


def score(rows, rule):
    tp = fp = tn = fn = 0
    for r in rows:
        pred = rule(r)
        if pred and r["held"]:
            tp += 1
        elif pred and not r["held"]:
            fp += 1
        elif not pred and r["held"]:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def main(paths):
    rows = parse(paths)
    if not rows:
        print("no [VERDICT] lines found")
        return 1
    n_held = sum(r["held"] for r in rows)
    print(f"{len(rows)} attempts: {n_held} HELD / {len(rows) - n_held} MISS")
    print()
    print(f"{'tag':10s} {'obj':9s} {'truth':5s} {'gw_shut':>8s} {'eff_peak':>9s} "
          f"{'gw_end':>7s} {'eff_end':>8s} {'lift':>7s}")
    for r in sorted(rows, key=lambda r: (r["tag"], r["name"])):
        gwe = "      -" if r["gw_end"] is None else "%7.1f" % r["gw_end"]
        efe = "       -" if r["effort_end"] is None else "%8.2f" % r["effort_end"]
        truth = "HELD" if r["held"] else "MISS"
        print("%-10s %-9s %-5s %8.1f %9.2f %s %s %+7.3f"
              % (r["tag"], r["name"], truth, r["gw_shut"], r["effort_peak"],
                 gwe, efe, r["lift"]))
    print()

    have_end = [r for r in rows if r["effort_end"] is not None]
    print("--- single-signal rules ---")
    for label, rule in (
        ("effort_peak > 20", lambda r: r["effort_peak"] > 20),
        ("effort_peak > 40", lambda r: r["effort_peak"] > 40),
        ("gw_shut > 2 mm", lambda r: r["gw_shut"] > 2.0),
    ):
        tp, fp, tn, fn = score(rows, rule)
        print(f"  {label:22s} TP {tp:2d}  FP {fp:2d}  TN {tn:2d}  FN {fn:2d}")

    if have_end:
        print(f"\n--- end-of-attempt rules ({len(have_end)} attempts have them) ---")
        best = None
        for et, wt in product([1, 5, 10, 20, 30, 40], [0.0, 0.5, 1.0, 2.0, 3.0]):
            rule = (lambda r, et=et, wt=wt:
                    r["effort_end"] > et and r["gw_end"] > wt)
            tp, fp, tn, fn = score(have_end, rule)
            acc = (tp + tn) / len(have_end)
            if best is None or acc > best[0]:
                best = (acc, et, wt, tp, fp, tn, fn)
        acc, et, wt, tp, fp, tn, fn = best
        print(f"  best: effort_end > {et} AND gw_end > {wt} mm  -> "
              f"TP {tp} FP {fp} TN {tn} FN {fn}  ({acc*100:.0f}% agreement with held())")
    else:
        print("\n(no end-of-attempt fields yet — rerun the probes after the instrumentation "
              "change to fit the deployable rule)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["course_mission/logs/deploy_probe_base.log"]))
