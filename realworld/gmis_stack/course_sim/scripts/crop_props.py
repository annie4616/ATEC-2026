"""Cut each prop out of its desk photo and write a texture map for the USD builders.

The photos are all the same setup: one prop on a light wooden desk, so the background is
segmented by colour distance from the border median rather than by any learned model.
Output is written to a2pro_rl/assets/textures/ as PNG, already oriented the way the USD
builder wants to sample it (long axis horizontal for cylinder wraps, upright for faces).
"""
import argparse, os
import cv2
import numpy as np

SRC = os.path.join(os.path.dirname(__file__), "..", "image")
DST = os.path.join(os.path.dirname(__file__), "..", "..",
                   "a2pro_rl", "assets", "textures")


def _trace_silhouette(fg):
    """Solid outline of the prop, traced row by row from its widest row outward.

    A plain row fill (leftmost to rightmost foreground pixel) is wrong here: GrabCut hands
    back the prop AND a lobe of lit desk beside it, and at the bottle's narrow neck the
    fill bridges straight across to that lobe. Inside the silhouette the bottle's
    clear-plastic recolour then turned the desk into pale blue speckle -- and pale plus
    achromatic is exactly the lunchbox trigger in perception.label_map. Following the run
    that touches the row before it stays on the prop and drops anything not connected to
    its body.
    """
    sil = np.zeros_like(fg)
    rows = np.where(fg.any(axis=1))[0]
    if not len(rows):
        return sil

    def runs(y):
        x = np.where(fg[y])[0]
        if not len(x):
            return []
        cuts = np.where(np.diff(x) > 1)[0]
        return [(g[0], g[-1]) for g in np.split(x, cuts + 1)]

    widths = {y: (max((r[1] - r[0] for r in runs(y)), default=0)) for y in rows}
    y_seed = max(widths, key=widths.get)
    seed = max(runs(y_seed), key=lambda r: r[1] - r[0])
    for direction in (-1, 1):
        lo, hi = seed
        y = y_seed
        while True:
            sil[y, lo:hi + 1] = True
            y += direction
            if y < 0 or y >= fg.shape[0] or not fg[y].any():
                break
            cand = [r for r in runs(y) if r[1] >= lo and r[0] <= hi]
            if not cand:
                break
            lo, hi = min(c[0] for c in cand), max(c[1] for c in cand)
    return sil


def prop_mask(bgr, rect_frac=0.72, return_silhouette=False):
    """Boolean mask of the prop, by GrabCut seeded with a central rectangle.

    An earlier colour-distance-from-the-border version failed on every photo here: the
    props are large and often touch the frame, so the border median was not always desk,
    and a fixed percentile threshold picked the wood GRAIN as the foreground blob. GrabCut
    only needs "the prop is roughly in the middle", which is true of all 18 photos.
    """
    H, W = bgr.shape[:2]
    # GrabCut is O(pixels) with a big constant; on these ~3000 px phone photos a full-res
    # solve takes minutes each. The mask only needs to be accurate to a few px at output
    # scale (512), so solve at 480 and upsample.
    k = 480.0 / max(H, W)
    small = cv2.resize(bgr, (int(W * k), int(H * k)), interpolation=cv2.INTER_AREA) if k < 1 else bgr
    h, w = small.shape[:2]
    m = np.zeros((h, w), np.uint8)
    rw, rh = int(w * rect_frac), int(h * rect_frac)
    rect = ((w - rw) // 2, (h - rh) // 2, rw, rh)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(small, m, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.resize(fg, (W, H), interpolation=cv2.INTER_NEAREST)
    rw, rh = int(W * rect_frac), int(H * rect_frac)   # full-res rect, for the desk model
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return np.ones((h, w), bool)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fg = (lbl == keep)
    sil = _trace_silhouette(fg)

    # GrabCut's "probable foreground" leaks desk wherever the prop is pale (the lunchbox,
    # the bottle's clear body, the rind's shadowed rim), and one leaked corner is enough to
    # put wood grain inside the inscribed rectangle. The contaminant is always the SAME
    # single surface, so model it explicitly from pixels the rect never claimed and
    # subtract it. This is only a refinement of a mask that is already mostly right -- the
    # earlier version that used this test ALONE is what mistook the grain for the prop.
    outside = np.zeros((H, W), bool)
    outside[:(H - rh) // 2] = True
    outside[(H + rh) // 2:] = True
    outside[:, :(W - rw) // 2] = True
    outside[:, (W + rw) // 2:] = True
    outside &= ~fg
    if outside.sum() > 2000:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        desk = np.median(lab[outside], axis=0)
        spread = np.median(np.abs(lab[outside] - desk), axis=0) + 1.5
        d = np.abs(lab - desk) / spread
        fg &= (d.max(axis=2) > 4.0)      # 4 median-absolute-deviations from the desk
        fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN,
                              np.ones((7, 7), np.uint8)).astype(bool)
    return (fg, sil) if return_silhouette else fg


def largest_rect(mask, max_aspect=3.0, work=600):
    """Largest axis-aligned all-foreground rectangle, capped at `max_aspect` w:h or h:w.

    A bounding box is the wrong crop for a texture: every one of these photos leaves desk
    in the corners of the prop's bbox, and painting it over with the prop median left a
    visible wood-shaped smear. An inscribed rectangle is pure prop surface by construction.

    The aspect cap matters as much as the area: the unconstrained maximum was repeatedly a
    1-2 px-tall full-width sliver (the rind skin came out 512x51), which carries the right
    colour but stretches to mush across a face. Solved on a downscaled copy -- the O(h*w)
    scan over a 12 MP phone photo took minutes per image in Python.
    """
    H, W = mask.shape
    k = min(1.0, work / max(H, W))
    m = cv2.resize(mask.astype(np.uint8), (max(1, int(W * k)), max(1, int(H * k))),
                   interpolation=cv2.INTER_NEAREST).astype(bool)
    h, w = m.shape
    heights = np.zeros(w, np.int32)
    best = (0, 0, 0, 0, 0)                       # area, y0, x0, y1, x1
    for y in range(h):
        heights = np.where(m[y], heights + 1, 0)
        stack = []                               # (start_x, height), increasing height
        for x in range(w + 1):
            cur = heights[x] if x < w else 0
            start = x
            while stack and stack[-1][1] >= cur:
                sx, sh = stack.pop()
                bw = x - sx
                if sh and bw and max(bw / sh, sh / bw) <= max_aspect:
                    area = sh * bw
                    if area > best[0]:
                        best = (area, y - sh + 1, sx, y, x - 1)
                start = sx
            if cur:
                stack.append((start, cur))
    if best[0] == 0:
        return 0, 0, H - 1, W - 1
    y0, x0, y1, x1 = best[1:]
    return (int(y0 / k), int(x0 / k), min(H - 1, int(y1 / k)), min(W - 1, int(x1 / k)))


def crop(name, out, rotate=0, mode="rect", pad=0.0):
    """mode="rect": largest inscribed rectangle (pure surface, for tiled textures).
       mode="bbox": full prop bounding box with the background flooded to the prop median
                    (for face textures where the silhouette itself carries information)."""
    p = next((q for q in (os.path.join(SRC, name + e)
                          for e in (".jpg", ".png"))
              if os.path.exists(q)), os.path.join(SRC, name + ".jpg"))
    bgr = cv2.imread(p)
    if bgr is None:
        raise SystemExit(f"missing {p}")
    if rotate:
        bgr = cv2.rotate(bgr, {90: cv2.ROTATE_90_CLOCKWISE,
                               180: cv2.ROTATE_180,
                               270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotate])
    m = prop_mask(bgr)
    if m.sum() < 500:
        raise SystemExit(f"{name}: mask too small ({int(m.sum())} px)")
    if mode == "rect":
        # erode first: GrabCut's boundary is a few px soft, and a rectangle that touches it
        # picks up a rim of desk-blended pixels along one edge
        e = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        y0, x0, y1, x1 = largest_rect(e if e.sum() > 500 else m)
        sub = bgr[y0:y1 + 1, x0:x1 + 1].copy()
    else:
        ys, xs = np.where(m)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        py, px = int((y1 - y0) * pad), int((x1 - x0) * pad)
        y0 = max(0, y0 - py); y1 = min(bgr.shape[0] - 1, y1 + py)
        x0 = max(0, x0 - px); x1 = min(bgr.shape[1] - 1, x1 + px)
        sub = bgr[y0:y1 + 1, x0:x1 + 1].copy()
        sm = m[y0:y1 + 1, x0:x1 + 1].astype(np.uint8)
        # Keep only the prop's own component. The desk subtraction punches the clear part
        # of the bottle out of the mask and leaves its specular highlights behind as
        # islands, which flooding then framed as bright speckle across the body.
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(sm, 8)
        if n > 1:
            sm = (lbl == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
            sm = cv2.morphologyEx(sm.astype(np.uint8), cv2.MORPH_CLOSE,
                                  np.ones((31, 31), np.uint8)).astype(bool)
        else:
            sm = sm.astype(bool)
        sub[~sm] = np.median(sub[sm], axis=0)
    k = 512 / max(sub.shape[:2])
    if k < 1:
        sub = cv2.resize(sub, (max(1, int(sub.shape[1] * k)), max(1, int(sub.shape[0] * k))),
                         interpolation=cv2.INTER_AREA)
    os.makedirs(DST, exist_ok=True)
    cv2.imwrite(os.path.join(DST, out + ".png"), sub)
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    ang = np.deg2rad(hsv[..., 0].astype(np.float32) * 2.0)
    hmed = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360)
    print(f"  {out:<16s} {mode:4s} {sub.shape[1]:3d}x{sub.shape[0]:3d}  "
          f"H {hmed:5.1f}  S {hsv[...,1].mean()/255:.3f}  V {hsv[...,2].mean()/255:.3f}")


def bottle_texture(out="bottle_side"):
    """The bottle is CLEAR PET with a yellow label band, and the generic bbox crop got it
    badly wrong: the desk subtraction removes the see-through part of the body (it looks
    like desk, because it IS desk seen through plastic) and the median flood then painted
    the whole bottle label-yellow. Measured on the resulting frames that put the bottle at
    H 50 / S 0.66 against the can's H 48 / S 0.84 -- a 2-degree hue gap between a target
    worth 20 points and a distractor worth 0, which is an artefact of this crop and not
    something the real prop does.

    So composite it: the label band exactly as photographed, and the clear body rendered
    as pale desaturated plastic that keeps the photo's own specular structure.
    """
    bgr = cv2.imread(os.path.join(SRC, "bottle_front.jpg"))
    fg, sil = prop_mask(bgr, return_silhouette=True)

    # GrabCut hands back the bottle WITH the lit desk beside it fused into one component,
    # so the traced silhouette ran a long way to the left of the prop and the composite
    # painted that desk as "clear plastic". The opaque parts that survive the desk test --
    # cap, label band -- do mark the bottle's true column range (the label wraps its widest
    # point), so use them to clip. Then drop the leftover specks of desk highlight that
    # also survived, which is what the red mask flecks were.
    n, lbl, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    if n > 1:
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        x0c, wc = st[big, cv2.CC_STAT_LEFT], st[big, cv2.CC_STAT_WIDTH]
        pad = int(0.06 * wc)
        band = np.zeros_like(sil)
        band[:, max(0, x0c - pad):x0c + wc + pad] = True
        sil &= band
        keep = np.zeros_like(fg)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= 0.02 * st[big, cv2.CC_STAT_AREA]:
                keep |= (lbl == i)
        fg = keep & band

    ys, xs = np.where(sil)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    sub = bgr[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    label = fg[y0:y1 + 1, x0:x1 + 1]                 # survived the desk test -> opaque
    body = sil[y0:y1 + 1, x0:x1 + 1] & ~label        # see-through
    hsv = cv2.cvtColor(sub.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    # clear PET: keep relative brightness (highlights, the neck's caustics), drop the
    # colour it borrowed from whatever was behind it, and lift it the way glass reads
    # Blur the value channel before reusing it. The see-through region is a photograph of
    # WOOD GRAIN behind plastic; keeping its per-pixel detail and merely desaturating it
    # left a speckled patch that reads as pale-achromatic -- the lunchbox trigger in
    # perception.label_map. Blurring keeps the broad light falloff and the neck's caustics
    # while erasing the grain, which is what clear PET actually looks like.
    v = cv2.GaussianBlur(hsv[..., 2], (0, 0), 21)
    hsv[..., 1] = np.where(body, np.minimum(hsv[..., 1], 0.06 * 255), hsv[..., 1])
    hsv[..., 0] = np.where(body, 105.0 / 2.0, hsv[..., 0])          # faint cool cast
    hsv[..., 2] = np.where(body, np.clip(0.55 * v + 0.42 * 255, 0, 255), hsv[..., 2])
    outimg = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    outside = ~sil[y0:y1 + 1, x0:x1 + 1]
    if outside.any() and body.any():
        outimg[outside] = np.median(outimg[body], axis=0)
    k = 512 / max(outimg.shape[:2])
    if k < 1:
        outimg = cv2.resize(outimg, (int(outimg.shape[1] * k), int(outimg.shape[0] * k)),
                            interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(DST, out + ".png"), outimg)
    hsv2 = cv2.cvtColor(outimg, cv2.COLOR_BGR2HSV)
    ang = np.deg2rad(hsv2[..., 0].astype(np.float32) * 2.0)
    hm = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360)
    print(f"  {out:<16s} pet  {outimg.shape[1]:3d}x{outimg.shape[0]:3d}  "
          f"H {hm:5.1f}  S {hsv2[...,1].mean()/255:.3f}  V {hsv2[...,2].mean()/255:.3f}  "
          f"(label {100*label.sum()/max(1,sil[y0:y1+1,x0:x1+1].sum()):.0f}% of silhouette)")


JOBS = [
    # (photo stem, texture name, mode)
    #
    # mode="bbox" wherever the prop's silhouette is already close to a rectangle AND its
    # printed artwork is the point: the inscribed rectangle dodges the desk by landing on
    # a blank part of the surface, which threw away the ATEC 2026 logo on the can and on
    # both bags. A bbox of a standing can/bag is >90 % prop, so flooding the few corner
    # pixels with the prop median costs nothing visible.
    # mode="rect" for the rest, where a bbox would drag in a wedge of desk.
    ("can_front",              "can_side",       "bbox"),
    ("can_top",                "can_cap",        "rect"),
    ("bottle_top",             "bottle_cap",     "rect"),
    ("lunchbox_top",           "lunchbox_top",   "bbox"),
    ("lunchbox_side",          "lunchbox_side",  "rect"),
    ("red_paper_bag_front",    "redbag_face",    "bbox"),
    ("red_paper_bag_side",     "redbag_side",    "rect"),
    ("red_paper_bag_top",      "redbag_top",     "rect"),
    ("purple_paper_bag_front", "purplebag_face", "bbox"),
    ("purple_paper_bag_top",   "purplebag_top",  "rect"),
    # watermelon: the two faces are separate materials in build_rind_usd.py.
    #   watermelon_back = striped green outer skin, watermelon_top = red flesh
    ("watermelon_back",        "rind_skin",      "rect"),
    ("watermelon_top",         "rind_flesh",     "rect"),
    # 보너스 과제 (보물찾기). The chest photo is a front-on studio shot, so a pure
    # inscribed rectangle of its wood is what a box face wants; the gold bar is a
    # single matte metal surface and needs no silhouette either.
    ("treasure_chest",         "chest_wood",     "rect"),
    ("gold_bar",               "gold",           "rect"),
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated texture names, default all")
    a = ap.parse_args()
    want = set(a.only.split(",")) if a.only else None
    print("[CROP] course_sim/image/*.jpg -> a2pro_rl/assets/textures/")
    for src, out, mode in JOBS:
        if want is None or out in want:
            crop(src, out, mode=mode)
    if want is None or "bottle_side" in want:
        bottle_texture()
