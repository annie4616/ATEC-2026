"""Contact sheet of the CLOSEST camera view of each prop class in a recorded dataset.

Ground truth positions are projected into the frame rather than searched for by colour,
so this shows what the prop actually looks like on screen without assuming anything about
the detector -- which is the point, since the detector is what gets re-fitted afterwards.
"""
import argparse, glob, os
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="course_mission/dataset_v3")
ap.add_argument("--out", default="/tmp/prop_closeups.png")
ap.add_argument("--crop", type=int, default=150, help="half-size of the crop in px")
a = ap.parse_args()

best = {}                                   # class -> (distance, crop, section)
for fn in sorted(glob.glob(os.path.join(a.data, "f*.npz"))):
    f = np.load(fn, allow_pickle=True)
    K, cam, R = f["K"], f["cam_pos"], f["cam_rot"]
    rgb = np.asarray(f["rgb"]).astype(np.uint8)
    h, w = rgb.shape[:2]
    for cls, p in zip(f["classes"], f["objs"]):
        d = R.T @ (np.asarray(p, np.float64) - cam)
        if d[2] <= 0.25:
            continue
        u = K[0, 0] * d[0] / d[2] + K[0, 2]
        v = K[1, 1] * d[1] / d[2] + K[1, 2]
        r = a.crop
        if not (0 <= u < w and 0 <= v < h):
            continue
        dist = float(np.linalg.norm(d))
        if cls in best and best[cls][0] <= dist:
            continue
        # clamp instead of skipping: requiring a full margin dropped can and rind entirely,
        # because the robot only ever passed them at the edge of frame
        u0 = int(min(max(u - r, 0), w - 2 * r)); v0 = int(min(max(v - r, 0), h - 2 * r))
        best[cls] = (dist, rgb[v0:v0 + 2 * r, u0:u0 + 2 * r].copy(), str(f["section"]),
                     (u - u0, v - v0))

tiles = []
for cls in ["can", "bottle", "lunchbox", "rind", "redbag", "purplebag"]:
    if cls not in best:
        t = np.full((260, 260, 3), 25, np.uint8)
        cv2.putText(t, cls + ": none", (8, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 1, cv2.LINE_AA)
        tiles.append(t)
        continue
    dist, crop, sec, (ox, oy) = best[cls]
    im = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), (260, 240),
                    interpolation=cv2.INTER_NEAREST)
    # Mark WHICH object the tile is about: clamping the crop to the frame moves the target
    # off centre, and without this the can tile reads as a picture of a paper bag.
    mx, my = int(ox * 260 / crop.shape[1]), int(oy * 240 / crop.shape[0])
    for col, th in ((( 0, 0, 0), 3), ((0, 255, 255), 1)):
        cv2.line(im, (mx - 14, my), (mx - 5, my), col, th, cv2.LINE_AA)
        cv2.line(im, (mx + 5, my), (mx + 14, my), col, th, cv2.LINE_AA)
        cv2.line(im, (mx, my - 14), (mx, my - 5), col, th, cv2.LINE_AA)
        cv2.line(im, (mx, my + 5), (mx, my + 14), col, th, cv2.LINE_AA)
    lab = f"{cls} {dist:.1f}m {sec}"
    bar = np.full((20, 260, 3), 25, np.uint8)
    cv2.putText(bar, lab, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                cv2.LINE_AA)
    tiles.append(np.vstack([bar, im]))
    print(f"  {cls:<10s} {dist:.2f} m  {sec}")
rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
cv2.imwrite(a.out, np.vstack(rows))
print("wrote", a.out)
