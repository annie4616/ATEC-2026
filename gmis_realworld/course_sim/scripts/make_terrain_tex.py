"""Turn a venue ground photo into a tileable 512x512 terrain texture.

The section meshes are UV-mapped from world XY (terrain._apply_photo_texture), so the
image repeats every `tile_m` metres in both axes and any seam is visible as a grid across
the whole course. Edges are therefore cross-faded against the mirrored image, which is
seamless by construction and keeps the grain direction the photo already has.
"""
import argparse, os
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "image")
DST = os.path.join(HERE, "..", "textures")


def make_tileable(bgr, size=512, blend=0.18):
    h, w = bgr.shape[:2]
    s = min(h, w)                                  # centre square: no aspect distortion
    im = bgr[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    b = int(size * blend)
    ramp = np.linspace(0.0, 1.0, b, dtype=np.float32)[:, None, None]
    # left edge <- mirrored right edge, then the same for the top
    im[:, :b] = im[:, :b] * ramp.transpose(1, 0, 2) + \
        im[:, -b:][:, ::-1] * (1 - ramp.transpose(1, 0, 2))
    im[:b] = im[:b] * ramp + im[-b:][::-1] * (1 - ramp)
    return np.clip(im, 0, 255).astype(np.uint8)


def report(name, im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    a = np.deg2rad(hsv[..., 0].astype(np.float32) * 2)
    hm = float(np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360)
    # seam check: how different are the wrapped-neighbour columns/rows
    seam = float(np.abs(im[:, 0].astype(int) - im[:, -1].astype(int)).mean() +
                 np.abs(im[0].astype(int) - im[-1].astype(int)).mean()) / 2
    print(f"  {name:<16s} H {hm:5.1f}  S {hsv[...,1].mean()/255:.3f}  "
          f"V {hsv[...,2].mean()/255:.3f}   seam {seam:4.1f}/255")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated stems, default all")
    a = ap.parse_args()
    jobs = ["mud", "grass", "grass_rock", "rocks"]
    if a.only:
        jobs = [j for j in a.only.split(",") if j]
    os.makedirs(DST, exist_ok=True)
    print("[TEX] course_sim/image/*.jpg -> course_sim/textures/*.png")
    for stem in jobs:
        src = os.path.join(SRC, stem + ".jpg")
        bgr = cv2.imread(src)
        if bgr is None:
            print(f"  {stem:<16s} MISSING {src}")
            continue
        out = make_tileable(bgr)
        cv2.imwrite(os.path.join(DST, stem + ".png"), out)
        report(stem, out)
