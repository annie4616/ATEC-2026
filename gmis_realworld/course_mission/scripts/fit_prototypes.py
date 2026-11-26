"""Re-fit the detector's colour prototypes from recorded frames. Isaac-free.

WHY: perception.OBJ_PROTO / BG_PROTO were fitted against flat vertex-coloured terrain and a
uniformly green rind cuboid. Both changed — the ground now carries photo textures from the
real venue surfaces (course_sim/textures) and the rind is a two-sided shell whose concave
face is RED — so every prototype has to be re-measured or the classifier is aiming at
colours that no longer exist.

Object pixels are sampled by PROJECTING the labelled ground-truth positions into the image
(the frames carry names/classes/objs/K/cam_pos/cam_rot), not by colour thresholds, which
would beg the question. Background pixels come from the lower image band per section.

Run:  ~/.conda/envs/isaaclab/bin/python course_mission/scripts/fit_prototypes.py \
          --data course_mission/dataset_v2
"""

import argparse
import glob
from collections import defaultdict

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--patch", type=int, default=5, help="half-size of the object sample patch")
ap.add_argument("--min_px", type=int, default=15)
args = ap.parse_args()

obj_px = defaultdict(list)     # class -> list of (H, S)
bg_px = defaultdict(list)      # section -> list of (H, S)

for fn in sorted(glob.glob(f"{args.data}/f*.npz")):
    f = np.load(fn, allow_pickle=True)
    rgb = np.asarray(f["rgb"]).astype(np.uint8)
    depth = np.asarray(f["depth"], float)
    K = np.asarray(f["K"], float)
    cam = np.asarray(f["cam_pos"], float)
    R = np.asarray(f["cam_rot"], float)          # already the ROS-convention matrix
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = hsv[..., 0] * 2.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # --- background: the strip just in front of the robot, away from image edges
    sec = str(f["section"])
    band = (slice(250, 340), slice(120, 520))
    bg_px[sec].append((float(np.median(H[band])), float(np.median(S[band])),
                       float(np.median(V[band]))))
    # sky: the top rows above the horizon. It is a BACKGROUND the classifier must own --
    # the old "sky_pebble" prototype covered it, and dropping it would leave every sky
    # pixel to be claimed by whichever object prototype is nearest.
    sky = (slice(0, 40), slice(80, 560))
    bg_px["sky"].append((float(np.median(H[sky])), float(np.median(S[sky])),
                         float(np.median(V[sky]))))

    # --- objects: project the known positions
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
        obj_px[cl].append((float(np.median(H[sl][m])), float(np.median(S[sl][m])),
                           float(np.median(V[sl][m]))))

def summarise(name, samples):
    a = np.asarray(samples)
    # circular median for hue
    ang = np.deg2rad(a[:, 0])
    hm = float(np.rad2deg(np.arctan2(np.median(np.sin(ang)), np.median(np.cos(ang)))) % 360)
    return name, hm, float(np.median(a[:, 1])), float(np.median(a[:, 2])), len(samples)

obj_fit = {}
print(f"=== objects (n = sightings within 2.2 m) ===")
for cl in sorted(obj_px):
    if cl == "rind":
        # TWO faces: split on hue, they are not one cluster and one prototype cannot
        # describe both. skin ~110 deg, flesh ~345 deg.
        a = np.asarray(obj_px[cl])
        red = (a[:, 0] < 40) | (a[:, 0] > 320)
        for tag, sel in (("rind(skin)", ~red), ("rind_flesh", red)):
            if sel.sum() >= 3:
                nm, h, s, vv, n = summarise(tag, a[sel])
                obj_fit["rind" if tag == "rind(skin)" else "rind_flesh"] = (h, s)
                print(f"  {nm:<12s} H {h:5.1f}  S {s:.3f}  V {vv:.3f}   n={n}")
            else:
                print(f"  {tag:<12s} too few samples ({int(sel.sum())})")
        continue
    nm, h, s, vv, n = summarise(cl, obj_px[cl])
    obj_fit[cl] = (h, s)
    print(f"  {nm:<12s} H {h:5.1f}  S {s:.3f}  V {vv:.3f}   n={n}")

print(f"\n=== background by section ===")
bg_fit = {}
for sec in sorted(bg_px):
    nm, h, s, vv, n = summarise(sec, bg_px[sec])
    bg_fit[sec] = (h, s)
    print(f"  {nm:<12s} H {h:5.1f}  S {s:.3f}  V {vv:.3f}   n={n}")

# ------------------------------------------------------------------ paste-ready block
# perception.label_map returns indices into list(OBJ_PROTO), and detect_arrays uses the
# KEY as the class name, so a second rind prototype must be aliased back to "rind" there
# rather than added as a new class. RIND_ALIAS in perception.py does that.
print("\n=== paste into course_mission/perception.py ===")
print("OBJ_PROTO = {")
for cl in ["can", "rind", "rind_flesh", "redbag", "purplebag", "bottle", "lunchbox"]:
    if cl not in obj_fit:
        print(f"    # {cl!r}: NOT MEASURED in this dataset -- left as-is")
        continue
    h, s = obj_fit[cl]
    print(f"    {cl!r}: ({h:.1f}, {s:.2f}),")
print("}")
print("BG_PROTO = {")
for sec, key in (("mud", "mud"), ("grass", "grass"), ("pebble", "pebble"),
                 ("mixed", "mixed")):
    if sec in bg_fit:
        h, s = bg_fit[sec]
        print(f"    {key!r}: ({h:.1f}, {s:.2f}),")
print("}")
