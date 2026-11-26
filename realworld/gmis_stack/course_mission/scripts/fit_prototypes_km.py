"""Fit the detector's colour prototypes as CLUSTERS over pixels, not medians over frames.

Why this exists alongside fit_prototypes.py: a flat vertex colour IS one point in (H, S),
so one prototype per surface described it exactly. A photograph is not. Real grass spans
roughly H 60-110 and S 0.2-0.7 in one frame, and its bright saturated tail lands nearer the
watermelon-skin prototype than the grass prototype -- measured, that alone produced 306
rind reports of which 22 were real. Collapsing a texture to its median throws away exactly
the part of the distribution that causes false positives.

So: sample the actual pixels, k-means them in the SAME metric label_map uses (hue scaled
by H_SCALE on the circle, saturation by S_SCALE), and emit several prototypes per surface.
label_map already loops over BG_PROTO.values(), so extra background entries need no code
change; extra object entries need an alias (Perception.PROTO_ALIAS).

Run:  ~/.conda/envs/isaaclab/bin/python course_mission/scripts/fit_prototypes_km.py \
          --data course_mission/dataset_v4 --bg_k 4 --obj_k 2
"""

import argparse
import glob
from collections import defaultdict

import cv2
import numpy as np

H_SCALE, S_SCALE = 20.0, 0.15          # must match Perception


def to_metric(h, s):
    """(hue, sat) -> the plane label_map measures distance in: hue on a circle of
    circumference 360/H_SCALE, saturation scaled by S_SCALE."""
    a = np.deg2rad(h)
    return np.stack([np.cos(a) * 180.0 / (np.pi * H_SCALE),
                     np.sin(a) * 180.0 / (np.pi * H_SCALE), s / S_SCALE], axis=-1)


def from_metric(c):
    h = np.rad2deg(np.arctan2(c[1], c[0])) % 360.0
    return float(h), float(c[2] * S_SCALE)


def kmeans(pts, k, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    c = pts[rng.choice(len(pts), k, replace=False)]
    for _ in range(iters):
        d = ((pts[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        new = np.stack([pts[lab == j].mean(0) if (lab == j).any() else c[j]
                        for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return c, np.bincount(lab, minlength=k)


ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--bg_k", type=int, default=4, help="prototypes per terrain section")
ap.add_argument("--obj_k", type=int, default=2, help="prototypes per object class")
ap.add_argument("--patch", type=int, default=5)
ap.add_argument("--min_px", type=int, default=15)
ap.add_argument("--max_px", type=int, default=200000, help="pixels kept per surface")
args = ap.parse_args()

obj_px = defaultdict(list)
bg_px = defaultdict(list)
rng = np.random.default_rng(0)

for fn in sorted(glob.glob(f"{args.data}/f*.npz")):
    f = np.load(fn, allow_pickle=True)
    rgb = np.asarray(f["rgb"]).astype(np.uint8)
    depth = np.asarray(f["depth"], float)
    K = np.asarray(f["K"], float)
    cam = np.asarray(f["cam_pos"], float)
    R = np.asarray(f["cam_rot"], float)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S = hsv[..., 0] * 2.0, hsv[..., 1] / 255.0
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    sec = str(f["section"])
    band = (slice(250, 340), slice(120, 520))
    bg_px[sec].append(np.stack([H[band].ravel(), S[band].ravel()], 1))
    # Sky = pixels with NO geometry behind them. Taking the top rows of the image instead
    # sampled the green recycling bins and the far end of the course, which produced a
    # "sky" prototype at (143, 0.72) -- saturated green, 32 degrees from the watermelon
    # skin it is supposed to be distinguished from.
    sky_m = ~np.isfinite(depth)
    if sky_m.sum() > 200:
        bg_px["sky"].append(np.stack([H[sky_m], S[sky_m]], 1))

    for cl, p in zip(f["classes"], np.asarray(f["objs"], float)):
        cl = str(cl)
        d = R.T @ (p - cam)
        if d[2] < 0.4 or d[2] > 2.2:
            continue
        u = fx * d[0] / d[2] + cx
        v = fy * d[1] / d[2] + cy
        if not (args.patch < u < rgb.shape[1] - args.patch
                and args.patch < v < rgb.shape[0] - args.patch):
            continue
        u, v = int(u), int(v)
        sl = (slice(v - args.patch, v + args.patch + 1),
              slice(u - args.patch, u + args.patch + 1))
        dz = depth[sl]
        m = np.isfinite(dz) & (np.abs(dz - d[2]) < 0.20)
        if m.sum() < args.min_px:
            continue
        obj_px[cl].append(np.stack([H[sl][m], S[sl][m]], 1))


def fit(name, chunks, k):
    a = np.concatenate(chunks, 0)
    if len(a) > args.max_px:
        a = a[rng.choice(len(a), args.max_px, replace=False)]
    k = min(k, max(1, len(a) // 50))
    c, n = kmeans(to_metric(a[:, 0], a[:, 1]), k)
    out = [from_metric(c[j]) for j in np.argsort(-n)]
    print(f"  {name:<12s} n={len(a):7d}  " +
          "  ".join(f"({h:5.1f},{s:.2f})x{100*cnt/len(a):.0f}%"
                    for (h, s), cnt in zip(out, sorted(n, reverse=True))))
    return out


print("=== object clusters ===")
obj_fit = {}
for cl in sorted(obj_px):
    obj_fit[cl] = fit(cl, obj_px[cl], args.obj_k)

print("\n=== background clusters ===")
bg_fit = {}
for sec in sorted(bg_px):
    if sec in ("stairs", "slope"):            # no objects are placed there
        continue
    # "bins" IS fitted. Excluding it was a mistake: the recycling bins are permanent
    # scenery the camera stares at from 0.4 m every time it unloads, and the green one
    # measures (140, 0.73) -- 24 degrees from the photo-fitted watermelon skin (116, 0.77)
    # and at the same saturation. With no prototype of its own it went to the nearest
    # object class: 171 of 179 rind reports in dataset_v5 were the green bin at 0.41 m.
    bg_fit[sec] = fit(sec, bg_px[sec], args.bg_k)

print("\n=== paste into course_mission/perception.py ===")
print("OBJ_PROTO = {")
alias = {}
MIN_OBJ_SAT = 0.20
for cl in ["can", "rind", "redbag", "purplebag", "bottle"]:
    if cl not in obj_fit:
        continue
    kept = 0
    for j, (h, s) in enumerate(obj_fit[cl]):
        # Drop near-achromatic OBJECT clusters. Below S ~0.2 the hue is numerically
        # meaningless, so such a prototype does not describe the object -- it claims a
        # whole empty wedge of the hue circle at low saturation and every desaturated
        # terrain pixel in it. The bottle's clear body fits at (104, 0.06), which would
        # have made "bottle" the nearest label for washed-out ground anywhere near that
        # hue. Achromatic objects are handled by the LB_SAT/LB_VAL gate instead, which
        # tests value as well. BACKGROUND clusters keep theirs: absorbing those pixels is
        # precisely their job.
        if kept and s < MIN_OBJ_SAT:
            print(f'    # {cl}_{j} ({h:.1f}, {s:.2f}) dropped: S < {MIN_OBJ_SAT}')
            continue
        key = cl if kept == 0 else f"{cl}_{j}"
        if kept:
            alias[key] = cl
        kept += 1
        print(f'    "{key}": ({h:.1f}, {s:.2f}),')
print("}")
print(f"PROTO_ALIAS = {alias}")
print("BG_PROTO = {")
for sec, protos in bg_fit.items():
    for j, (h, s) in enumerate(protos):
        print(f'    "{sec}_{j}": ({h:.1f}, {s:.2f}),')
print("}")
