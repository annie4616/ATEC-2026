"""Parse a Task D collect_demos diag log and classify each attempt's outcome.

For every attempt it extracts:
  - the last push-controller state reached (BACK_UP/ALIGN_Y/PUSH_Y/ALIGN_X/PUSH_X/CROSS)
  - outcome: SUCCESS, ABORT (robot fell), or END (env termination + active terms)
  - the final box (x,y,z) and robot (x,y,z) before the attempt ended

Then it groups failures by (last_state, outcome) so we can see WHERE and WHY
attempts fail — the data needed to decide "fixable by oracle logic" vs "needs RL".

Usage: python scripts/act/classify_failures.py <diag_log_path>
"""

import re
import sys
from collections import defaultdict

STEP_RE = re.compile(
    r"\[step\s+(\d+)\]\s+root=\(([-+0-9.]+),([-+0-9.]+),([-+0-9.]+)\)"
    r"(?:\s+box=\(([-+0-9.]+),([-+0-9.]+),([-+0-9.]+)\))?"
    r"(?:\s+state=(\w+))?"
)
DEMO_RE = re.compile(r"\[INFO\] Demo (\d+)/\d+\s+\(attempt (\d+)\)")
TRAJ_RE = re.compile(r"\[INFO\] traj_(\d+):")
ABORT_RE = re.compile(r"\[ABORT step (\d+)\] robot fallen \(z=([-+0-9.]+)\)")
END_RE = re.compile(
    r"\[END step (\d+)\].*root_x=([-+0-9.]+) root_z=([-+0-9.]+) active_terms=\[([^\]]*)\]"
)
WARN_SKIP = re.compile(r"\[WARN\] Did not reach goal")


def main(path):
    attempts = []          # list of dicts
    cur = None

    def flush():
        if cur is not None:
            attempts.append(cur)

    with open(path) as fh:
        for line in fh:
            m = DEMO_RE.search(line)
            if m:
                flush()
                cur = {"attempt": int(m.group(2)), "last_state": None,
                       "last_step": None, "root": None, "box": None,
                       "outcome": None, "detail": ""}
                continue
            if cur is None:
                continue

            m = STEP_RE.search(line)
            if m:
                cur["last_step"] = int(m.group(1))
                cur["root"] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
                if m.group(5) is not None:
                    cur["box"] = (float(m.group(5)), float(m.group(6)), float(m.group(7)))
                if m.group(8):
                    cur["last_state"] = m.group(8)
                continue

            m = ABORT_RE.search(line)
            if m:
                cur["outcome"] = "ABORT_FELL"
                cur["detail"] = f"z={m.group(2)} @step{m.group(1)}"
                continue

            m = END_RE.search(line)
            if m:
                cur["outcome"] = "END_TERM"
                cur["detail"] = f"terms=[{m.group(4)}] root_x={m.group(2)} root_z={m.group(3)}"
                continue

            if TRAJ_RE.search(line):
                cur["outcome"] = "SUCCESS"
                continue
            if WARN_SKIP.search(line):
                if cur["outcome"] is None:
                    cur["outcome"] = "SKIP_NOGOAL"
                continue
    flush()

    # --- per-attempt table ---
    print(f"\n{'att':>3} {'outcome':<11} {'last_state':<10} {'step':>5} "
          f"{'box(x,y,z)':<22} {'robot(x,y,z)':<22} detail")
    print("-" * 110)
    for a in attempts:
        box = ("(%.2f,%.2f,%.2f)" % a["box"]) if a["box"] else "-"
        root = ("(%.2f,%.2f,%.2f)" % a["root"]) if a["root"] else "-"
        print(f"{a['attempt']:>3} {str(a['outcome']):<11} {str(a['last_state']):<10} "
              f"{str(a['last_step']):>5} {box:<22} {root:<22} {a['detail']}")

    # --- classification summary ---
    groups = defaultdict(list)
    for a in attempts:
        groups[(a["outcome"], a["last_state"])].append(a)
    print("\n=== classification: (outcome, last_state) -> count ===")
    for (outc, st), lst in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        n = len(lst)
        # representative box/robot ranges
        boxes = [x["box"] for x in lst if x["box"]]
        roots = [x["root"] for x in lst if x["root"]]
        bx = f"bx[{min(b[0] for b in boxes):.2f},{max(b[0] for b in boxes):.2f}]" if boxes else ""
        by = f"by[{min(b[1] for b in boxes):.2f},{max(b[1] for b in boxes):.2f}]" if boxes else ""
        bz = f"bz[{min(b[2] for b in boxes):.2f},{max(b[2] for b in boxes):.2f}]" if boxes else ""
        ry = f"ry[{min(r[1] for r in roots):.2f},{max(r[1] for r in roots):.2f}]" if roots else ""
        print(f"  {str(outc):<11} {str(st):<10} x{n:<3} {bx} {by} {bz} {ry}")

    succ = sum(1 for a in attempts if a["outcome"] == "SUCCESS")
    print(f"\ntotal attempts={len(attempts)}  success={succ}  "
          f"fail={len(attempts)-succ}  success_rate={succ/max(1,len(attempts)):.0%}")


if __name__ == "__main__":
    main(sys.argv[1])
