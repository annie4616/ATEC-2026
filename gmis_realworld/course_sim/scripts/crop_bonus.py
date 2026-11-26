"""Cut the 보너스 과제 (보물찾기) textures out of the organiser photos.

These two need hand-picked boxes rather than the automatic segmentation in crop_props.py:
  - the chest photo's biggest clean region contains the recessed handle and a lid seam,
    and a tiling wood texture would repeat both across the whole box. The handle is a
    separate part in the guide anyway (황동 손잡이 162 x 96 x 20 mm), so it is modelled as
    geometry and the wood must be plain.
  - the gold bar is one matte surface; the automatic crop straddled its top edge, baking a
    hard corner into what should be a flat face.
Boxes were read off a coordinate grid overlaid on each photo.
"""
import os
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "image")
DST = os.path.join(HERE, "..", "..", "a2pro_rl", "assets", "textures")

# (source, output, x0, y0, x1, y1, tile-seamlessly)
JOBS = [
    # chest: clean staves below the handle, inside the brass corner hardware
    ("treasure_chest.png", "chest_wood", 100, 185, 340, 292, True),
    # chest lid: the top slab, above the seam
    ("treasure_chest.png", "chest_lid", 80, 103, 360, 140, True),
]

# The gold bar sits on a plain blue backdrop, so its box is FOUND rather than typed: mask
# out the blue, then take the largest inscribed rectangle. Hand-picked boxes kept clipping
# either the ingot's edge or the backdrop, because its faces are rotated quads and no
# axis-aligned guess stays inside one.
GOLD = ("gold_bar.png", "gold")


def tileable(im, blend=0.20):
    im = im.astype(np.float32)
    h, w = im.shape[:2]
    bx, by = max(1, int(w * blend)), max(1, int(h * blend))
    rx = np.linspace(0.0, 1.0, bx, dtype=np.float32)[None, :, None]
    im[:, :bx] = im[:, :bx] * rx + im[:, -bx:][:, ::-1] * (1 - rx)
    ry = np.linspace(0.0, 1.0, by, dtype=np.float32)[:, None, None]
    im[:by] = im[:by] * ry + im[-by:][::-1] * (1 - ry)
    return np.clip(im, 0, 255).astype(np.uint8)


def largest_rect(mask, max_aspect=3.0):
    """Largest all-foreground axis-aligned rectangle (histogram/stack, O(h*w))."""
    h, w = mask.shape
    heights = np.zeros(w, np.int32)
    best = (0, 0, 0, 0, 0)
    for y in range(h):
        heights = np.where(mask[y], heights + 1, 0)
        stack = []
        for x in range(w + 1):
            cur = heights[x] if x < w else 0
            start = x
            while stack and stack[-1][1] >= cur:
                sx, sh = stack.pop()
                bw = x - sx
                if sh and bw and max(bw / sh, sh / bw) <= max_aspect and sh * bw > best[0]:
                    best = (sh * bw, y - sh + 1, sx, y, x - 1)
                start = sx
            if cur:
                stack.append((start, cur))
    return best[1:]


def crop_gold():
    src, out = GOLD
    bgr = cv2.imread(os.path.join(SRC, src))
    if bgr is None:
        print(f"  {out:<12s} MISSING {src}")
        return
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(np.float32) * 2.0
    # backdrop is blue (H ~200); gold is warm (H ~45). Also drop the dark contact shadow.
    m = ((hue < 90) | (hue > 330)) & (hsv[..., 2] > 60)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        m = (lbl == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
    m = cv2.erode(m.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    y0, x0, y1, x1 = largest_rect(m)
    sub = tileable(bgr[y0:y1 + 1, x0:x1 + 1])
    cv2.imwrite(os.path.join(DST, out + ".png"), sub)
    hsv2 = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    a = np.deg2rad(hsv2[..., 0].astype(np.float32) * 2)
    hm = float(np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360)
    print(f"  {out:<12s} {sub.shape[1]:3d}x{sub.shape[0]:3d}  H {hm:5.1f}  "
          f"S {hsv2[...,1].mean()/255:.3f}  V {hsv2[...,2].mean()/255:.3f}  "
          f"box x{x0}..{x1} y{y0}..{y1}")


if __name__ == "__main__":
    os.makedirs(DST, exist_ok=True)
    print("[BONUS] course_sim/image -> a2pro_rl/assets/textures")
    for src, out, x0, y0, x1, y1, tile in JOBS:
        bgr = cv2.imread(os.path.join(SRC, src))
        if bgr is None:
            print(f"  {out:<12s} MISSING {src}")
            continue
        sub = bgr[y0:y1, x0:x1]
        if tile:
            sub = tileable(sub)
        k = 512 / max(sub.shape[:2])
        if k < 1:
            sub = cv2.resize(sub, (int(sub.shape[1] * k), int(sub.shape[0] * k)),
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(DST, out + ".png"), sub)
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        a = np.deg2rad(hsv[..., 0].astype(np.float32) * 2)
        hm = float(np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360)
        seam = float(np.abs(sub[:, 0].astype(int) - sub[:, -1].astype(int)).mean())
        print(f"  {out:<12s} {sub.shape[1]:3d}x{sub.shape[0]:3d}  H {hm:5.1f}  "
              f"S {hsv[...,1].mean()/255:.3f}  V {hsv[...,2].mean()/255:.3f}  seam {seam:4.1f}")
    crop_gold()
