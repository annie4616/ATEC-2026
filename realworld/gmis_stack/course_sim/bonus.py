"""보너스 과제: 보물찾기 (treasure hunt) — geometry and scoring.

Guide 2026-08-20, lines 300-340. Worth 40 points and open ONLY to the 다족 / 휠-레그
category, which is ours — the humanoid teams cannot enter it (line 308). It runs on a
SEPARATE course (line 310) after the core task is complete, and the core task is only
complete once the robot has crossed the finish line (line 115), so course_mission's
cross_finish() is its precondition.

The course is two zones divided by a boundary line (line 307):
  보물 상자 구역   the chest sits at a fixed position and orientation, the bar inside it
  배치 구역        concentric squares, inner 40 x 40 cm and outer 75 x 75 cm (line 337)

Scoring is on the item's FINAL RESTING position (line 314), and the run ends the first time
it comes to rest in the placement zone -- no second attempt from there (line 313):
  40  fully inside the inner square
  20  between the inner and outer boundaries
   0  outside the outer boundary
Overlapping zones take the LOWER score (line 320), so "fully inside" means the item's own
extent counts, not just its centre.
"""

import numpy as np
import trimesh

# ---- guide dimensions [m] -------------------------------------------------------------
INNER = 0.40
OUTER = 0.75
LINE_W = 0.03                 # painted line width
LINE_T = 0.004                # paint thickness, so it never trips the robot
CHEST_TO_PLACE = 3.0          # zone separation; the guide fixes neither, this is a working
                              # value and the one number here that is NOT from the guide
BONUS_SCORE = (40, 20, 0)

WHITE = np.array([245, 245, 245, 255], np.uint8)
LINE_COLOR = WHITE


def _box(extents, translate, color):
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(translate)
    m.visual.vertex_colors = np.tile(color, (m.vertices.shape[0], 1))
    return m


def _square(side, cx, cy, z, dashed=False, dash=0.10, gap=0.07):
    """Four painted edges of an axis-aligned square, solid or dashed."""
    out = []
    h = side / 2
    for axis in (0, 1):
        for sgn in (+1.0, -1.0):
            if not dashed:
                ext = ((side + LINE_W, LINE_W, LINE_T) if axis == 0
                       else (LINE_W, side + LINE_W, LINE_T))
                off = ((cx, cy + sgn * h) if axis == 0 else (cx + sgn * h, cy))
                out.append(_box(ext, (off[0], off[1], z), LINE_COLOR))
                continue
            n = max(int(round((side + LINE_W) / (dash + gap))), 1)
            step = (side + LINE_W) / n
            for k in range(n):
                t = -h - LINE_W / 2 + (k + 0.5) * step
                ext = (dash, LINE_W, LINE_T) if axis == 0 else (LINE_W, dash, LINE_T)
                off = ((cx + t, cy + sgn * h) if axis == 0 else (cx + sgn * h, cy + t))
                out.append(_box(ext, (off[0], off[1], z), LINE_COLOR))
    return out


def placement_zone(cx=0.0, cy=0.0, z=0.0):
    """The concentric scoring squares. Matches course_sim/image/placement_area.png:
    a solid inner square inside a dashed outer one."""
    return _square(INNER, cx, cy, z + LINE_T / 2, dashed=False) + \
        _square(OUTER, cx, cy, z + LINE_T / 2, dashed=True)


def divider_line(x, half_width, z=0.0):
    """The 경계선 dividing the chest zone from the placement zone (guide 307). An item
    dropped on the chest side may be re-picked; one that stops in the placement zone
    ends the task."""
    return [_box((LINE_W, 2 * half_width, LINE_T), (x, 0.0, z + LINE_T / 2), LINE_COLOR)]


def score(item_xy, item_half_extent, centre=(0.0, 0.0)):
    """Points for an item at rest. `item_half_extent` is half its largest horizontal
    dimension, because the guide scores on the item being COMPLETELY inside a zone and
    applies the LOWER score where zones overlap (lines 316-320)."""
    dx = abs(float(item_xy[0]) - centre[0])
    dy = abs(float(item_xy[1]) - centre[1])
    reach = max(dx, dy) + item_half_extent
    if reach <= INNER / 2:
        return BONUS_SCORE[0]
    if reach <= OUTER / 2:
        return BONUS_SCORE[1]
    return BONUS_SCORE[2]
