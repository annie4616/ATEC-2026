"""Render a grasp attempt with the numbers that decide it printed on the frame.

Every claim about why a grasp fails in this project has so far come from a single summary
line written after the attempt was already over. This shows the quantities AS THEY MOVE, so
the cause can be seen rather than argued:

  finger width   the gap between the fingertips. 100 mm is wide open, 0 is shut. If it
                 never leaves ~100 while the command says CLOSE, the fingers are not
                 moving; if it goes to ~0, they shut on nothing.
  command        what the policy is asking of the gripper this step.
  reach          TCP to object centre.
  |a_rot|        the policy's rotation command, the quantity whose median separated every
                 outcome in the log record.

Run:  python course_mission/scripts/make_grasp_overlay.py --trace_dir <dir> --out <mp4>
"""
import argparse
import glob
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--trace_dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=float, default=12.0)
a = ap.parse_args()

W, H = 640, 360
PLOT_H = 150
BAR_H = 44


def panel(npz):
    d = np.load(npz, allow_pickle=True)
    frames, tr = d["frames"], d["trace"]
    name, cls, sec = (str(x) for x in d["meta"])
    # Frames now start at the APPROACH while the trace only covers the grasp loop, so the
    # two are different lengths on purpose: the tail of the frames is the grasp, and for a
    # skipped grasp there is no tail at all. Pad the trace at the FRONT so the plot lines up
    # with the frames it belongs to, and mark where the grasp actually began.
    # The trace now spans the WHOLE attempt (approach -> crouch -> grasp) while frames are
    # sampled every few steps, so resample the trace onto the frames instead of padding.
    if len(tr):
        idx = np.linspace(0, len(tr) - 1, num=len(frames)).round().astype(int)
        tr = tr[idx]
    if len(frames) == 0:
        # An EMPTY trace is the finding, not an error: the grasp policy never ran a single
        # step. run_grasp bails at runner.py:991 when the object sits more than 0.40 m from
        # the pregrasp bank anchor, so nothing downstream of that — finger width, squeeze,
        # friction, rotation cap — was ever exercised.
        card = np.full((BAR_H + H + PLOT_H, W, 3), 18, np.uint8)
        for k, (txt, col, sz) in enumerate([
                (f"{cls}  ({sec})", (255, 255, 255), 0.7),
                ("GRASP POLICY NEVER RAN", (80, 80, 255), 0.8),
                ("0 steps, 0 frames", (200, 200, 200), 0.6),
                ("object too far from the pregrasp bank", (200, 200, 200), 0.55),
                ("(runner.py:991  drift > 0.40 m -> skip)", (160, 160, 160), 0.5)]):
            cv2.putText(card, txt, (24, 90 + 46 * k), cv2.FONT_HERSHEY_SIMPLEX, sz, col,
                        2 if k == 1 else 1, cv2.LINE_AA)
        return [card], f"{cls}/{sec}"
    n = len(frames)
    if len(tr) < n:                       # skipped grasp: no trace at all
        tr = np.full((n, 9), np.nan, np.float32)
    tr = tr[:n]
    step, reach, width, closed, arot = (tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3], tr[:, 4])
    tilt = tr[:, 7] if tr.shape[1] > 7 else np.full(len(tr), np.nan)
    ing = tr[:, 9] if tr.shape[1] > 9 else np.zeros(len(tr))
    footd = tr[:, 8] if tr.shape[1] > 8 else np.full(len(tr), np.nan)
    out = []
    for i in range(n):
        im = cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR), (W, H))
        bar = np.full((BAR_H, W, 3), 20, np.uint8)
        cv2.putText(bar, f"{cls}  ({sec})", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        pre = ing[i] < 0.5
        cmd = "approach / crouch" if pre else ("CLOSE" if closed[i] > 0.5 else "open")
        col = (150, 150, 150) if pre else ((60, 220, 60) if closed[i] > 0.5
                                          else (200, 200, 200))
        cv2.putText(bar, cmd if pre else f"command {cmd}", (8, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        f = (lambda v, u: "  --  " if not np.isfinite(v) else f"{v*u:6.1f}")
        cv2.putText(bar, f"finger {f(width[i],1000)} mm", (200, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, f"reach {f(reach[i],1000)} mm", (380, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 120), 1, cv2.LINE_AA)
        cv2.putText(bar, f"|a_rot| {f(arot[i],1)}", (520, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 160, 255), 1, cv2.LINE_AA)
        # object tilt: 0 = upright, 90 = flat on its side. Red once it is past 45.
        tv = tilt[i]
        tc = ((80, 80, 255) if np.isfinite(tv) and tv > 45 else (140, 255, 180))
        cv2.putText(bar, f"tilt {('--' if not np.isfinite(tv) else f'{tv:5.1f}')} deg",
                    (200, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tc, 1, cv2.LINE_AA)
        fd = footd[i]
        cv2.putText(bar, f"nearest foot {('--' if not np.isfinite(fd) else f'{fd*1000:4.0f}')} mm",
                    (380, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 140), 1,
                    cv2.LINE_AA)

        pl = np.full((PLOT_H, W, 3), 16, np.uint8)
        # finger width 0..110 mm, and the close command as a filled band
        for k in range(1, n):
            x0, x1 = int(W * (k - 1) / n), int(W * k / n)
            if np.isfinite(closed[k]) and closed[k] > 0.5:
                cv2.rectangle(pl, (x0, 0), (x1, PLOT_H), (28, 60, 28), -1)
        def yy(v, lo, hi):
            return int(PLOT_H - 6 - (PLOT_H - 14) * np.clip((v - lo) / (hi - lo), 0, 1))
        for k in range(1, n):
            if not (np.isfinite(width[k]) and np.isfinite(width[k - 1])):
                continue
            x0, x1 = int(W * (k - 1) / n), int(W * k / n)
            cv2.line(pl, (x0, yy(width[k - 1], 0, 0.110)), (x1, yy(width[k], 0, 0.110)),
                     (120, 200, 255), 2, cv2.LINE_AA)
            cv2.line(pl, (x0, yy(reach[k - 1], 0, 0.40)), (x1, yy(reach[k], 0, 0.40)),
                     (255, 200, 120), 1, cv2.LINE_AA)
            if np.isfinite(tilt[k]) and np.isfinite(tilt[k - 1]):
                cv2.line(pl, (x0, yy(tilt[k - 1], 0, 95)), (x1, yy(tilt[k], 0, 95)),
                         (80, 80, 255), 2, cv2.LINE_AA)
        cur = int(W * i / n)
        cv2.line(pl, (cur, 0), (cur, PLOT_H), (255, 255, 255), 1)
        cv2.putText(pl, "finger width", (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (120, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(pl, "reach", (120, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 200, 120), 1, cv2.LINE_AA)
        cv2.putText(pl, "tilt", (168, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (80, 80, 255), 1, cv2.LINE_AA)
        cv2.putText(pl, "green band = commanded CLOSE", (210, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 220, 60), 1, cv2.LINE_AA)
        out.append(np.vstack([bar, im, pl]))
    return out, f"{cls}/{sec}"


files = sorted(glob.glob(os.path.join(a.trace_dir, "trace_*.npz")))
if not files:
    raise SystemExit(f"no traces in {a.trace_dir}")
panels = [panel(f) for f in files]
n = max(len(p[0]) for p in panels)
h, w = panels[0][0][0].shape[:2]
vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w * len(panels), h))
for i in range(n):
    row = [p[0][min(i, len(p[0]) - 1)] for p in panels]
    vw.write(np.hstack(row))
vw.release()
print(f"[OVERLAY] {len(panels)} panels x {n} frames -> {a.out}")
for f, (fr, lab) in zip(files, panels):
    d = np.load(f, allow_pickle=True)["trace"]
    if d.ndim != 2 or len(d) == 0:
        print(f"  {lab:<16s} GRASP POLICY NEVER RAN (0 steps)")
        continue
    cl = d[:, 3] > 0.5
    print(f"  {lab:<16s} steps={len(d)}  commanded-close {int(cl.sum())}  "
          f"finger min {d[:,2].min()*1000:.1f} mm  "
          f"finger min WHILE closed {(d[cl,2].min()*1000 if cl.any() else float('nan')):.1f} mm  "
          f"reach min {d[:,1].min()*1000:.0f} mm")
