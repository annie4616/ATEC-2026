"""Competition course terrain (test env): mud -> pebble -> grass -> stairs -> slope ->
mixed -> bin zone, laid out linearly along +x via BetterTerrainGenerator's
terrain_sequence (task_a pattern). Section visuals use trimesh vertex colors
(task_b pattern), so the importer's visual_material must stay None.

Constraints honored: slopes < 10 deg, stair step height 0.15 m (<= 0.17), pebbles as a
1-2 cm irregular heightfield, red 0.4 m boundary strips along both edges of every
section, three tapered trash bins (top 0.40x0.26, bottom length 0.32, red/green/blue)
in the final zone.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class _Absent:
    """Stand-in for a simulator symbol on a machine that has no simulator.

    THE COURSE GEOMETRY IS NOT A SIMULATOR THING. curve_map, SECTION_CENTERS, BIN_*,
    CHECKPOINTS, BOUNDARY_Y and friends are plain numbers that the mission reads on every
    step and that the robot needs just as much as the sim does — but they lived in a
    module whose first ten lines import trimesh, isaaclab and the task registry, so the
    whole file was unreachable without Isaac Sim installed. Isaac Sim does not install on
    a GPU-less N150.

    Rather than split the file (and risk the constants drifting between two copies), the
    simulator-only names degrade to this: any attribute access returns another _Absent and
    any call returns None. The mesh builders and the terrain-importer config then evaluate
    to None instead of raising, and everything numeric above them is untouched. In the
    simulator the import succeeds and NOTHING here is used.
    """

    def __init__(self, name="isaac"):
        self._name = name

    def __getattr__(self, k):
        return _Absent(f"{self._name}.{k}")

    def __call__(self, *a, **kw):
        return None


class _AbsentCfg:
    """Permissive stand-in base for the simulator's config dataclasses.

    It has to accept arbitrary keyword arguments: the module builds the sub-terrain cfg
    with `_mk_cfg(s_course)(proportion=1.0)`, and a bare `object` base rejects that with
    "takes no arguments" — which would put the geometry back out of reach on the robot.
    """

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


try:
    import trimesh
    from isaaclab.terrains import SubTerrainBaseCfg, TerrainImporterCfg
    import isaaclab.sim as sim_utils
    from isaaclab.utils import configclass

    from atec_rl_lab.tasks.task_base import (
        BetterTerrainGenerator,
        BetterTerrainGeneratorCfg,
        BetterTerrainImporter,
    )
    SIM_AVAILABLE = True
except ModuleNotFoundError as _e:
    # Hardware build: no simulator packages at all. If SOME of them are importable the
    # install is broken rather than absent, and degrading silently would hide it — so
    # print which one went missing instead of swallowing it without trace.
    print(f"[course_sim.terrain] simulator packages unavailable ({_e.name}); "
          f"geometry only, COURSE_TERRAIN_CFG will be None")
    SIM_AVAILABLE = False
    trimesh = _Absent("trimesh")
    sim_utils = _Absent("isaaclab.sim")
    SubTerrainBaseCfg = _AbsentCfg
    BetterTerrainImporter = _AbsentCfg
    BetterTerrainGenerator = None
    TerrainImporterCfg = _Absent("TerrainImporterCfg")
    BetterTerrainGeneratorCfg = _Absent("BetterTerrainGeneratorCfg")

    def configclass(c):        # the decorator is a no-op without isaaclab
        return c

# Which venue's course to build. Read EARLY: the strip and bin geometry below depend
# on it, and Python executes this module top to bottom.
_VENUE = os.environ.get("COURSE_VENUE", "hk").lower()

SECTION_LEN = 6.0   # cell size along x [m]
COURSE_WIDTH = 4.0  # cell size along y [m]
# Red boundary strip. The 2026-08-20 guide (line 296) gives it as 높이 40cm, 폭 4.5cm --
# a STANDING strip 0.40 m tall and 0.045 m wide. This file previously read the "40 cm" of
# the older guide as the strip's WIDTH and laid it flat (0.40 x 0.014 m), which is the
# wrong axis: the venue marker is knee-high, not a floor stripe.
#
# The judged boundary is NOT the strip. Guide line 131: the scored line is drawn on the
# ground UNDER the strip, and the test is where the SUPPORTING SURFACE (the feet) touches
# down -- body, arm, gripper and sensor overhang past the line explicitly do not count.
# BOUNDARY_Y is that line; the strip stands centred on it.
# It is a RETRACTABLE BELT STANCHION, not a wall: black post on a black round base with a
# red webbing belt strung between posts. So the guide's "높이 40cm, 폭 4.5cm" describes the
# BELT -- it hangs 40 cm above the ground and the webbing is 4.5 cm wide. An earlier version
# read those two numbers as the dimensions of a solid red panel and built a 40 cm red wall
# down both edges of every section, which is wrong twice over: the barrier is mostly black,
# and everything below 37.5 cm is open air apart from the posts.
BELT_Z = 0.40       # belt centre height above the LOCAL ground [m]
BELT_W = 0.045      # belt webbing width [m]
BELT_T = 0.006      # webbing thickness
POST_SPACING = 2.0  # between stanchions [m]
POST_R, POST_H = 0.022, 0.44
BASE_R, BASE_H = 0.110, 0.020
STRIP_W = 0.045     # footprint of the barrier line, for BOUNDARY_Y below
# legacy keeps the OLD geometry on purpose: it exists to reproduce the four 120/160 runs,
# and a course whose barriers and bins have moved is not the course they were scored on.
STRIP_FLAT = (os.environ.get("COURSE_STRIP_FLAT", "1" if _VENUE == "legacy" else "0") == "1")
BOUNDARY_Y = COURSE_WIDTH / 2 - STRIP_W / 2               # judged line, +-1.9775 m

RED = np.array([220, 30, 30, 255], np.uint8)
BLACK = np.array([28, 28, 30, 255], np.uint8)
MUD = np.array([101, 67, 33, 255], np.uint8)
PEBBLE = np.array([150, 150, 150, 255], np.uint8)
GRASS = np.array([80, 160, 60, 255], np.uint8)
STAIR = np.array([215, 215, 215, 255], np.uint8)
# REAL bin colours, measured from the actual competition bins (2026-08-26,
# course_sim/image/trashcan_{red,green,blue}.jpg — dominant-hue median of the bin body,
# ~130k pixels each). The old values were guesses and all three were too bright and too
# saturated: red (220,30,30)->(137,37,37), green (30,170,60) hue 133 sat 0.82 ->
# (48,121,70) hue 137.5 sat 0.60, blue (30,80,220) hue 224 -> (36,68,113) hue 217.5.
# detect_bins' hue windows (perception.BIN_COLORS_NAV: red 330..25, green 90..150,
# blue 200..250) contain all three measured hues, and its sat/val floors (>=0.3/0.1)
# clear the measured 0.60-0.73 / 0.44-0.54 — checked before this edit, so navigation
# keeps working. The perception BACKGROUND prototypes for the green bin were refreshed
# in the same change (perception.py) because the rind-vs-bin confusion is hue-adjacent.
BIN_COLORS = {
    "red": np.array([137, 37, 37, 255], np.uint8),
    "green": np.array([48, 121, 70, 255], np.uint8),
    "blue": np.array([36, 68, 113, 255], np.uint8),
}


def _colored_box(extents, translate, color):
    m = trimesh.creation.box(
        extents=extents, transform=trimesh.transformations.translation_matrix(translate)
    )
    m.visual.vertex_colors = np.tile(color, (m.vertices.shape[0], 1))
    return m


def _colored_cyl(radius, height, translate, color, sections=16):
    m = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    m.apply_translation(translate)
    m.visual.vertex_colors = np.tile(color, (m.vertices.shape[0], 1))
    return m


def _boundary_strips(cfg, z_at=None, seg=0.5):
    """The course boundary: retractable belt stanchions down both edges.

    z_at(x) gives the local surface height at course-x, so posts stand on the terrain and
    the belt follows it (stairs and the slope would otherwise leave them floating or buried).

    Tagged in two groups because they behave differently. The POSTS are rigid and collide.
    The BELT does not: it is webbing, and a 50 kg quadruped walking into it pulls the belt
    out or tips the post rather than being stopped. Giving it a collider would let the
    simulation catch the robot at the boundary and under-report exactly the excursions the
    venue charges 10 points for (guide 392).

    COURSE_STRIP_FLAT=1 restores the old 0.40 x 0.014 m painted floor marking, which is what
    every 120/160 run was scored on.
    """
    out = []
    n = max(int(round(cfg.size[0] / seg)), 1)
    step = cfg.size[0] / n
    if STRIP_FLAT:
        y = cfg.size[1] / 2 - 0.40 / 2
        for k in range(n):
            x = -cfg.size[0] / 2 + (k + 0.5) * step
            z = (z_at(x) if z_at else 0.0) + 0.006
            for sgn in (+1.0, -1.0):
                m = _colored_box((step + 0.02, 0.40, 0.014), (x, sgn * y, z), RED)
                m.metadata["boundary_strip"] = True
                out.append(m)
        return out

    y = cfg.size[1] / 2 - STRIP_W / 2
    # belt: short segments so it follows the S-warp and the local surface
    for k in range(n):
        x = -cfg.size[0] / 2 + (k + 0.5) * step
        z0 = (z_at(x) if z_at else 0.0)
        for sgn in (+1.0, -1.0):
            m = _colored_box((step + 0.02, BELT_T, BELT_W), (x, sgn * y, z0 + BELT_Z), RED)
            m.metadata["belt"] = True
            out.append(m)
    # stanchions
    npost = max(int(round(cfg.size[0] / POST_SPACING)), 1)
    for k in range(npost):
        x = -cfg.size[0] / 2 + (k + 0.5) * cfg.size[0] / npost
        z0 = (z_at(x) if z_at else 0.0)
        for sgn in (+1.0, -1.0):
            for m in (_colored_cyl(BASE_R, BASE_H, (x, sgn * y, z0 + BASE_H / 2), BLACK),
                      _colored_cyl(POST_R, POST_H, (x, sgn * y, z0 + BASE_H + POST_H / 2),
                                   BLACK)):
                m.metadata["boundary_strip"] = True
                out.append(m)
    return out


def _heightfield_mesh(cfg, z_fn, color_fn, res=0.10):
    """Surface grid mesh z = z_fn(x, y) with per-vertex colors, centered at (0, 0)."""
    nx = int(cfg.size[0] / res) + 1
    ny = int(cfg.size[1] / res) + 1
    xs = np.linspace(-cfg.size[0] / 2, cfg.size[0] / 2, nx)
    ys = np.linspace(-cfg.size[1] / 2, cfg.size[1] / 2, ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    gz = z_fn(gx, gy)
    verts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            faces += [[a, a + ny, a + 1], [a + 1, a + ny, a + ny + 1]]
    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=False)
    mesh.visual.vertex_colors = color_fn(verts)
    return mesh


def _flat_section(cfg, color, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    z_fn = (lambda x, y: rng.uniform(0.0, noise, x.shape)) if noise > 0 else (lambda x, y: np.zeros_like(x))
    mesh = _heightfield_mesh(cfg, z_fn, lambda v: np.tile(color, (len(v), 1)))
    return [mesh] + _boundary_strips(cfg, z_at=lambda x: noise), np.array([0.0, 0.0, 0.05])


def mud_section(difficulty, cfg):
    return _flat_section(cfg, MUD, noise=0.008, seed=1)


def pebble_section(difficulty, cfg):
    # Guide line 245: 자갈 지름 약 1~2cm (natural, irregular). Surface relief of a packed
    # 1-2 cm gravel bed: 0.4-2.2 cm at 3 cm cells (~pebble spacing). The previous build
    # was 0.5-5.0 cm at 5 cm cells — 2.5x rougher than the spec — with a comment that
    # claimed 1-2 cm; runner.ground_z_arc's pebble mean and the boundary-post z here are
    # coupled to this range and move with it.
    rng = np.random.default_rng(2)

    def colors(v):
        shade = rng.uniform(0.75, 1.05, len(v))[:, None]
        c = np.clip(PEBBLE[None, :3] * shade, 0, 255).astype(np.uint8)
        return np.concatenate([c, np.full((len(v), 1), 255, np.uint8)], axis=1)

    mesh = _heightfield_mesh(cfg, lambda x, y: rng.uniform(0.004, 0.022, x.shape), colors, res=0.03)
    return [mesh] + _boundary_strips(cfg, z_at=lambda x: 0.022), np.array([0.0, 0.0, 0.05])


def grass_section(difficulty, cfg):
    return _flat_section(cfg, GRASS, noise=0.005, seed=3)


def stairs_section(difficulty, cfg):
    """Linear staircase across the course width: 4 steps up (0.15 m each, <=0.17 ok),
    plateau, 4 steps down. Step depth 0.35 m."""
    meshes = [
        _colored_box((cfg.size[0], cfg.size[1], 0.02), (0.0, 0.0, -0.01), STAIR)
    ]
    step_h, step_d, n = 0.15, 0.35, 4
    x0 = -cfg.size[0] / 2 + 0.8
    for i in range(n):  # up
        h = step_h * (i + 1)
        meshes.append(_colored_box((step_d, cfg.size[1], h), (x0 + step_d * (i + 0.5), 0.0, h / 2), STAIR))
    plat_x0 = x0 + n * step_d
    plat_len = cfg.size[0] - 2 * 0.8 - 2 * n * step_d
    meshes.append(_colored_box((plat_len, cfg.size[1], step_h * n),
                               (plat_x0 + plat_len / 2, 0.0, step_h * n / 2), STAIR))
    for i in range(n):  # down
        h = step_h * (n - i)
        meshes.append(_colored_box((step_d, cfg.size[1], h),
                                   (plat_x0 + plat_len + step_d * (i + 0.5), 0.0, h / 2), STAIR))

    def z_at(x):  # local stair surface height at course-x
        if x < x0:
            return 0.02
        if x < plat_x0:
            return step_h * min(int((x - x0) / step_d) + 1, n)
        if x < plat_x0 + plat_len:
            return step_h * n
        if x < plat_x0 + plat_len + n * step_d:
            return step_h * (n - min(int((x - plat_x0 - plat_len) / step_d), n - 1))
        return 0.02

    return meshes + _boundary_strips(cfg, z_at=z_at, seg=0.35), np.array([0.0, 0.0, 0.05])


def slope_section(difficulty, cfg):
    """Gentle ramp up-down, max grade 9 deg (< 10 deg constraint)."""
    grade = np.tan(np.deg2rad(9.0))
    half = cfg.size[0] / 2
    ramp = 2.2
    peak = grade * ramp  # ~0.35 m

    def z_fn(x, y):
        up = np.clip((x + half) - 0.8, 0.0, ramp) * grade
        down = np.clip((half - x) - 0.8, 0.0, ramp) * grade
        return np.minimum(np.minimum(up, down), peak)

    def colors(v):
        return np.tile(GRASS, (len(v), 1))

    mesh = _heightfield_mesh(cfg, z_fn, colors, res=0.05)
    zx = lambda x: float(z_fn(np.array([x]), np.array([0.0]))[0])
    return [mesh] + _boundary_strips(cfg, z_at=zx, seg=0.35), np.array([0.0, 0.0, 0.05])


def mixed_section(difficulty, cfg):
    """Grass + pebble mix: 1-2 cm noise with blended green/gray patches."""
    rng = np.random.default_rng(4)

    def colors(v):
        t = (np.sin(v[:, 0] * 2.1) * np.sin(v[:, 1] * 1.7) + 1) / 2
        c = (GRASS[None, :3] * t[:, None] + PEBBLE[None, :3] * (1 - t[:, None])).astype(np.uint8)
        return np.concatenate([c, np.full((len(v), 1), 255, np.uint8)], axis=1)

    mesh = _heightfield_mesh(cfg, lambda x, y: rng.uniform(0.005, 0.035, x.shape), colors, res=0.05)
    return [mesh] + _boundary_strips(cfg, z_at=lambda x: 0.035), np.array([0.0, 0.0, 0.05])


# Bin geometry, guide line 293: 개구부 40cm x 26cm, 바닥 32cm x 19.5cm, 높이 47cm.
# The height was 0.55 here; exported now because course_mission/runner.py's
# "did it land IN the bin" test needs the rim height and had it written out as a literal.
BIN_TOP = (0.40, 0.26)
BIN_BOT_L = 0.32
BIN_BOT_W = 0.195       # guide gives the bottom explicitly; deriving it from the top's
                        # aspect ratio (0.26*0.32/0.40) lands 1.3 cm wide at 0.208
BIN_HEIGHT = 0.55 if _VENUE == "legacy" else 0.47
# Cross-course centres, keyed by colour so nothing has to remember an order. The
# organiser photo of the bin row (course_sim/image, and the arrangement the team was
# shown) is BLUE - GREEN - RED reading left to right as the robot approaches, and +lateral
# is screen-left: an earlier version put red at +0.9 and rendered the row mirrored.
#
# Getting this wrong is silent and expensive: the robot would drive to the position it
# believes is blue, drop a can into the red bin, and score 10 instead of 20 with nothing
# in any log to say why. course_mission/mission.bin_targets() and env_cfg.bin_world_xy()
# both used to restate the mapping independently; they now read this.
BIN_LATERAL_OF = {"blue": 0.9, "green": 0.0, "red": -0.9}
BIN_LATERAL = tuple(BIN_LATERAL_OF.values())

# 일반 배치 구역, guide line 295: a rectangle 26 cm outside the three bins' OUTER edges.
# Worth 25 % (5 points, guide 377) and it is a real rectangle, not "anywhere near the end
# of the course" -- which is what the code used to award it for.
PLACEMENT_PAD = 0.26
PLACEMENT_HALF_ALONG = BIN_TOP[0] / 2 + PLACEMENT_PAD                       # 0.46 m
PLACEMENT_HALF_CROSS = (max(abs(v) for v in BIN_LATERAL_OF.values())
                        + BIN_TOP[1] / 2 + PLACEMENT_PAD)          # 1.29 m


def _tapered_bin(cx, cy, color, top=BIN_TOP, bot_l=BIN_BOT_L, height=BIN_HEIGHT,
                 wall_t=0.014):
    """Open-top tapered bin: four THIN-BOX walls plus a thick bottom plate that overlaps
    the wall feet.

    The first version used zero-thickness trapezoid surfaces whose bottom edge met a 2 cm
    plate edge-to-edge, with no overlap. Objects dropped into the bin slipped through that
    seam and reappeared underneath it (confirmed on video, and in the landing logs: an
    object released at z=0.26 inside the bin was found at z=0.01 outside it). Solid walls
    and an oversized plate close the leak.
    """
    bot_w = BIN_BOT_W
    tl, tw = top[0] / 2, top[1] / 2
    bl, bw = bot_l / 2, bot_w / 2
    z0 = 0.04                                   # plate top: the interior floor
    H = height - z0
    meshes = []

    def _wall(along_x, sgn):
        """One tilted side wall as a solid box (tapers are approximated by the mean width)."""
        d = (tl - bl) if along_x else (tw - bw)
        L = float(np.hypot(d, H))
        ang = float(np.arctan2(d, H))           # lean-out angle from vertical
        if along_x:
            extents = (wall_t, (bw + tw) + 2 * wall_t, L)
            offs = (sgn * (bl + tl) / 2, 0.0)
            axis, sign = [0.0, 1.0, 0.0], sgn
        else:
            extents = ((bl + tl) + 2 * wall_t, wall_t, L)
            offs = (0.0, sgn * (bw + tw) / 2)
            axis, sign = [1.0, 0.0, 0.0], -sgn
        m = trimesh.creation.box(extents=extents)
        T = trimesh.transformations.rotation_matrix(sign * ang, axis)
        T[0, 3], T[1, 3], T[2, 3] = cx + offs[0], cy + offs[1], z0 + H / 2
        m.apply_transform(T)
        m.visual.vertex_colors = np.tile(color, (m.vertices.shape[0], 1))
        return m

    for sgn in (+1.0, -1.0):
        meshes.append(_wall(True, sgn))
        meshes.append(_wall(False, sgn))
    # bottom plate: thicker than any per-substep fall distance and wider than the wall
    # feet, so there is real overlap instead of a seam
    meshes.append(_colored_box((bot_l + 0.05, bot_w + 0.05, 0.08), (cx, cy, 0.0), color))
    return meshes


def bins_section(difficulty, cfg):
    """Designated end zone: light-gray pad + red/green/blue tapered bins."""
    meshes = [
        _heightfield_mesh(cfg, lambda x, y: np.zeros_like(x),
                          lambda v: np.tile(np.array([190, 190, 190, 255], np.uint8), (len(v), 1)))
    ]
    for name, cy in BIN_LATERAL_OF.items():
        meshes += _tapered_bin(0.0, cy, BIN_COLORS[name])
    # 결승선. Guide 123: after placing what it collected in the designated area the robot
    # keeps going and completes the core task by CROSSING the red finish line; guide 115
    # makes the bonus task unavailable if it never does. Painted across the full width,
    # 1.2 m past the bins, flat so it is never an obstacle.
    if FINISH_LOCAL is not None:
        meshes.append(_colored_box((0.10, cfg.size[1] - 2 * STRIP_W, 0.012),
                                   (FINISH_LOCAL, 0.0, 0.006), RED))
    return meshes + _boundary_strips(cfg), np.array([0.0, 0.0, 0.05])


def _mk_cfg(fn):
    @configclass
    class _Cfg(SubTerrainBaseCfg):
        function = fn
    return _Cfg


# ---------------------------------------------------------------- venue course layouts
# COURSE_VENUE=hk (default) builds the HONG KONG course, which is what the team competes
# on (실전 예선 – 홍콩, 2026-09-12). The 2026-08-20 guide scores it in three sections
# (lines 368-371), NOT the seven this file used to build:
#
#   잔디 경사 및 진흙 구간 (혼합 지형)   2 scoring points, +20 per half   -> 40
#   자갈 구간                            section checkpoint, +20          -> 20
#   계단 구간                            scored at each half, +10         -> 20
#                                                                    total 80
#
# So the combined grass-slope + mud section is modelled as two cells (slope, mud) whose
# cell boundaries ARE its two scoring points, preceded by a flat grass run-in that carries
# the start zone. Shanghai's layout is different again (stairs twice, no dedicated grass
# slope); it is not built here because we do not compete there.
#
# COURSE_VENUE=legacy restores the original seven-section course. Every 120/160 result was
# scored on that one, so it stays runnable. (The pre-HK snapshot dir lives only in the
# original dev tree: real_world_preliminary/.../course_mission/snapshots/pre_hk_0821_*.)
COURSE_SEQUENCE_HK = ["grass", "slope", "mud", "pebble", "stairs", "bins"]
COURSE_SEQUENCE_LEGACY = ["mud", "pebble", "grass", "stairs", "slope", "mixed", "bins"]
COURSE_SEQUENCE = COURSE_SEQUENCE_LEGACY if _VENUE == "legacy" else COURSE_SEQUENCE_HK

# per-section ground friction (static; dynamic = 0.9x). Mud is slick, pebble loose-but-
# grippy, stairs/bin pad hard concrete-like.
SECTION_FRICTION = {
    "mud": 0.55, "pebble": 0.90, "grass": 0.80, "stairs": 1.00,
    "slope": 0.80, "mixed": 0.85, "bins": 1.00,
}
# filled by s_course(): per-section merged mesh in FINAL (corner-convention) coords
SECTION_MESH_GROUPS: dict[str, trimesh.Trimesh] = {}
# The barrier, kept out of the ground mesh so the photo texture never lands on it.
# Two groups because they differ physically: posts collide, webbing does not.
STRIP_MESH_GROUPS: dict[str, trimesh.Trimesh] = {}
BELT_MESH_GROUPS: dict[str, trimesh.Trimesh] = {}
COURSE_LEN = len(COURSE_SEQUENCE) * SECTION_LEN
COURSE_X0 = -COURSE_LEN / 2
SECTION_CENTERS = {
    name: COURSE_X0 + SECTION_LEN * (i + 0.5) for i, name in enumerate(COURSE_SEQUENCE)
}
# ---- task 1 (하이킹) checkpoints -----------------------------------------------------
# Guide line 246: an IR detector sits at the END of each sub-section, and passing it
# completes that sub-section. The Hong Kong point table is guide lines 368-371; each
# sub-section is scored ONCE (line 239: "각 하위 구간은 1회만 채점됩니다"), which matters
# because the mission shuttles back and forth along the course many times per run.
#
# (arc position [m], points, label). Positions are section boundaries, so they move with
# SECTION_LEN and the sequence rather than being pinned to today's 36 m course.
def _end_of(sec):
    return SECTION_CENTERS[sec] + SECTION_LEN / 2


def _checkpoints():
    if _VENUE == "legacy":
        return []                       # the seven-section course has no scored table
    st = SECTION_CENTERS["stairs"]
    return [
        (_end_of("slope"), 20, "잔디 경사 (혼합 지형 1/2)"),
        (_end_of("mud"), 20, "진흙 (혼합 지형 2/2)"),
        (_end_of("pebble"), 20, "자갈"),
        (st, 10, "계단 1/2"),
        (_end_of("stairs"), 10, "계단 2/2"),
    ]


CHECKPOINTS = _checkpoints()
TASK1_MAX = sum(p for _, p, _ in CHECKPOINTS)

# Finish line, 1.2 m beyond the bins inside the final cell (local x within that cell).
FINISH_LOCAL = 1.2 if _VENUE != "legacy" else None
FINISH_ARC = (SECTION_CENTERS["bins"] + FINISH_LOCAL) if FINISH_LOCAL is not None else None

_SECTION_FNS = {
    "mud": mud_section, "pebble": pebble_section, "grass": grass_section,
    "stairs": stairs_section, "slope": slope_section, "mixed": mixed_section,
    "bins": bins_section,
}

# ---- reverse-S centerline (course map shape): start upper-left -> end lower-right ----
S_AMP = 7.5

# The S is ONE half-cosine over COURSE_LEN, so shortening the course sharpens it without
# anyone asking: the minimum radius of curvature is COURSE_LEN**2 / (S_AMP * pi**2), and
# curve_map offsets each vertex by its lateral coordinate along the local normal. Once that
# radius drops below the course HALF-WIDTH the inner edge inverts through itself -- the mesh
# folds, the collision geometry self-intersects, and nothing raises: it imports and renders.
# 42 m gives R = 23.8 m, 36 m gives 17.5 m, and the failure point is near 12 m.
_S_RADIUS = COURSE_LEN ** 2 / (S_AMP * np.pi ** 2) if S_AMP else float("inf")
if _S_RADIUS <= COURSE_WIDTH / 2 * 1.5:
    raise ValueError(
        f"S-curve too sharp for a {COURSE_LEN} m course: min radius {_S_RADIUS:.2f} m vs "
        f"half-width {COURSE_WIDTH / 2:.2f} m — the warped mesh will fold. "
        f"Reduce S_AMP (currently {S_AMP}) or lengthen the course.")


def curve_y(x):
    return S_AMP * np.cos(np.pi * (np.asarray(x, float) - COURSE_X0) / COURSE_LEN)


def curve_theta(x):
    return np.arctan(-S_AMP * np.pi / COURSE_LEN
                     * np.sin(np.pi * (np.asarray(x, float) - COURSE_X0) / COURSE_LEN))


def curve_map(x, w):
    """(course arc-x, lateral offset) -> world (X, Y) on the S-course."""
    th = curve_theta(x)
    return np.asarray(x, float) - w * np.sin(th), curve_y(x) + w * np.cos(th)


def s_course(difficulty, cfg):
    """The whole S-shaped course as ONE sub-terrain: build each straight section
    locally, then warp its vertices onto the reverse-S centerline."""
    import types as _types

    sub = _types.SimpleNamespace(size=(SECTION_LEN, COURSE_WIDTH))
    meshes = []
    SECTION_MESH_GROUPS.clear()
    STRIP_MESH_GROUPS.clear()
    BELT_MESH_GROUPS.clear()
    for i, name in enumerate(COURSE_SEQUENCE):
        mlist, _ = _SECTION_FNS[name](difficulty, sub)
        cx = COURSE_X0 + SECTION_LEN * (i + 0.5)
        for m in mlist:
            v = m.vertices.copy()
            X, Y = curve_map(cx + v[:, 0], v[:, 1])
            m.vertices = np.stack([X, Y, v[:, 2]], axis=1)  # world-intended coords
        # registry: WORLD coords (imported directly by CourseTerrainImporter, which
        # bypasses the generator's net -size/2 translation)
        _strips = [m for m in mlist if m.metadata.get("boundary_strip")]
        _belt = [m for m in mlist if m.metadata.get("belt")]
        _ground = [m for m in mlist
                   if not (m.metadata.get("boundary_strip") or m.metadata.get("belt"))]
        SECTION_MESH_GROUPS[name] = trimesh.util.concatenate([m.copy() for m in _ground])
        if _strips:
            STRIP_MESH_GROUPS[name] = trimesh.util.concatenate([m.copy() for m in _strips])
        if _belt:
            BELT_MESH_GROUPS[name] = trimesh.util.concatenate([m.copy() for m in _belt])
        for m in mlist:
            # generator path expects CORNER coords [0, size] (net -size/2 applied later)
            m.vertices = m.vertices + np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])
            meshes.append(m)
    return meshes, np.array([0.0, 0.0, 0.05])


# Photo textures for the ground, from course_sim/image/*.jpg (the real venue surfaces).
# OFF BY DEFAULT: the detector's BACKGROUND prototypes in course_mission/perception.py are
# fitted to the flat vertex colours these replace — measured, the grass photo sits at
# S 0.50 against the grass prototype's S 0.20 — so switching this on without re-fitting
# would wreck detection and with it the 120/160 result. Enable with COURSE_TERRAIN_TEX=1,
# re-record frames (scripts/collect_frames.py), then re-fit the prototypes.
_TEX_DIR = Path(__file__).resolve().parent / "textures"
SECTION_TEXTURE = {
    "mud": ("mud.png", 0.60),           # (file, metres per tile)
    "grass": ("grass.png", 0.50),
    "pebble": ("rocks.png", 0.45),      # crushed gravel, 1-2 cm stones -> small tile
    "slope": ("grass.png", 0.50),       # 잔디 경사 — same turf as the flat grass
    "mixed": ("grass_rock.png", 0.80),  # grass growing through pavers
}


def _apply_photo_texture(prim_path, mesh, tex_file, tile_m):
    """UV-map a tiling photo onto one section's mesh and bind it as the visual material.

    create_prim_from_mesh writes points/faces and a displayColor primvar but NO uvs, so the
    st primvar is added here. A bound material takes precedence over displayColor.
    """
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()
    mesh_prim = stage.GetPrimAtPath(f"{prim_path}/mesh")
    if not mesh_prim or not mesh_prim.IsValid():
        print(f"[COURSE] texture skipped, no mesh prim at {prim_path}", flush=True)
        return False
    # World XY -> uv, so tile size is in METRES and stays consistent across sections.
    st = (np.asarray(mesh.vertices)[:, :2] / float(tile_m))[np.asarray(mesh.faces).ravel()]
    pv = UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    pv.Set([Gf.Vec2f(float(a), float(b)) for a, b in st])

    mp = f"{prim_path}/photoMaterial"
    mat = UsdShade.Material.Define(stage, mp)
    reader = UsdShade.Shader.Define(stage, mp + "/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, mp + "/Tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(_TEX_DIR / tex_file))
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result")
    sh = UsdShade.Shader.Define(stage, mp + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.95)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(mat)
    return True


class CourseTerrainImporter(BetterTerrainImporter):
    """Imports each course section as its OWN prim with a per-section physics material
    (real friction differences), instead of one merged mesh with one material.
    The generator still runs normally (env origins etc.); only the import differs.
    NOTE the sections in SECTION_MESH_GROUPS are in final world coords already — the
    merged-mesh translation net is zero for this 1x1 grid, so direct import matches."""

    def import_mesh(self, name, mesh):
        from isaaclab.terrains.utils import create_prim_from_mesh

        if name != "terrain" or not SECTION_MESH_GROUPS:
            return super().import_mesh(name, mesh)
        for sec, m in SECTION_MESH_GROUPS.items():
            prim_path = self.cfg.prim_path + f"/terrain_{sec}"
            if prim_path in self.terrain_prim_paths:
                continue
            self.terrain_prim_paths.append(prim_path)
            f = SECTION_FRICTION[sec]
            create_prim_from_mesh(
                prim_path, m,
                visual_material=None,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=f, dynamic_friction=round(0.9 * f, 3),
                    restitution=0.0,
                ),
            )
            if (os.environ.get("COURSE_TERRAIN_TEX") == "1"
                    or os.environ.get("MISSION_VENUE_PHOTO") == "1") \
                    and sec in SECTION_TEXTURE:
                _f, _t = SECTION_TEXTURE[sec]
                if _apply_photo_texture(prim_path, m, _f, _t):
                    print(f"[COURSE] {sec}: photo texture {_f} @ {_t} m/tile", flush=True)
        for sec, m in STRIP_MESH_GROUPS.items():
            prim_path = self.cfg.prim_path + f"/strip_{sec}"
            if prim_path in self.terrain_prim_paths:
                continue
            self.terrain_prim_paths.append(prim_path)
            create_prim_from_mesh(
                prim_path, m,
                visual_material=None,          # keeps the RED vertex colours
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=0.9, dynamic_friction=0.8, restitution=0.0,
                ),
            )
        for sec, m in BELT_MESH_GROUPS.items():
            prim_path = self.cfg.prim_path + f"/belt_{sec}"
            if prim_path in self.terrain_prim_paths:
                continue
            self.terrain_prim_paths.append(prim_path)
            create_prim_from_mesh(prim_path, m, visual_material=None)
            # VISUAL ONLY. create_prim_from_mesh always enables the collider, so it is
            # switched off again here: webbing does not stop a quadruped, and a belt that
            # did would hide the boundary excursions the venue charges 10 points for.
            sim_utils.define_collision_properties(
                prim_path + "/mesh",
                sim_utils.CollisionPropertiesCfg(collision_enabled=False))
        print(f"[COURSE] boundary: "
              f"{'flat painted marking' if STRIP_FLAT else f'belt stanchions, webbing at {BELT_Z} m'}"
              f" — {len(STRIP_MESH_GROUPS)} post prims, {len(BELT_MESH_GROUPS)} belt prims",
              flush=True)
        print(f"[COURSE] per-section terrain import: "
              f"{ {k: SECTION_FRICTION[k] for k in SECTION_MESH_GROUPS} }", flush=True)


COURSE_TERRAIN_CFG = TerrainImporterCfg(
    class_type=CourseTerrainImporter,
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=BetterTerrainGeneratorCfg(
        class_type=BetterTerrainGenerator,
        seed=0,
        size=(COURSE_LEN + 6.0, 2 * (S_AMP + COURSE_WIDTH / 2 + 2.0)),
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={"s_course": _mk_cfg(s_course)(proportion=1.0)},
    ),
    max_init_terrain_level=0,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    ),
    visual_material=None,  # REQUIRED: vertex colors carry the section look
    debug_vis=False,
)
