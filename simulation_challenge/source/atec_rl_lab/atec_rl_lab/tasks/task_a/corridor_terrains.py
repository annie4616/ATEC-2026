"""Corridor-symmetric terrains for the walled Task A course.

The stock IsaacLab pyramid slope/stairs put their apex at the TILE CENTER, so a narrow
corridor running down a y-band crosses the pyramid's flank at an angle -- the corridor
floor tilts sideways and the robot slides off. These terrains instead vary height ONLY
along x (the corridor's travel direction) and are CONSTANT along y (the corridor's width),
so inside the corridor the floor is left-right symmetric and the robot climbs the ramp /
stairs head-on, exactly like a real ramp or staircase spanning a hallway.

height array shape is (width_pixels, length_pixels) = (x, y), matching IsaacLab's HF convention.
"""

import numpy as np
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh


@height_field_to_mesh
def transverse_ramp_terrain(difficulty: float, cfg: "TransverseRampCfg") -> np.ndarray:
    """A ridge that rises along +x to a peak at the tile center then falls back -- constant
    across y. The robot walks up one face and down the other, straight along the corridor."""
    slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    wpx = int(cfg.size[0] / cfg.horizontal_scale)   # x pixels (travel dir)
    lpx = int(cfg.size[1] / cfg.horizontal_scale)   # y pixels (corridor width)
    # triangular profile along x: 0 at ends, peak at center. peak height = slope * (size_x/2)
    hmax = slope * cfg.size[0] / 2.0 / cfg.vertical_scale
    x = np.arange(wpx)
    xn = x / max(1, wpx - 1)                          # 0..1
    prof = 1.0 - np.abs(xn - 0.5) * 2.0              # 0 at ends, 1 at center
    # optional flat platform at the peak
    if cfg.platform_width > 0.0:
        pw = cfg.platform_width / cfg.size[0]         # as fraction of x
        prof = np.clip(prof / max(1e-6, 1.0 - pw), 0.0, 1.0)
    col = (hmax * prof).astype(np.int16)             # (wpx,)
    hf = np.repeat(col.reshape(wpx, 1), lpx, axis=1)  # constant across y
    return hf


@height_field_to_mesh
def transverse_stairs_terrain(difficulty: float, cfg: "TransverseStairsCfg") -> np.ndarray:
    """Stairs whose steps run across the corridor (each step spans the full y width): the
    robot ascends step-by-step along +x to the center, then descends -- constant across y."""
    step_h = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0])
    wpx = int(cfg.size[0] / cfg.horizontal_scale)
    lpx = int(cfg.size[1] / cfg.horizontal_scale)
    step_w_px = max(1, int(cfg.step_width / cfg.horizontal_scale))
    step_h_u = step_h / cfg.vertical_scale            # step height in vertical units
    x = np.arange(wpx)
    center = wpx // 2
    # number of steps from each end up to the center
    dist = np.abs(x - center)                         # 0 at center, grows to ends
    n_from_end = (center - dist) // step_w_px          # 0 at ends, max at center
    col = (n_from_end * step_h_u).astype(np.int16)
    hf = np.repeat(col.reshape(wpx, 1), lpx, axis=1)
    return hf


@configclass
class TransverseRampCfg(HfTerrainBaseCfg):
    """A y-constant ridge ramp along the corridor (peak at tile center)."""
    function = transverse_ramp_terrain
    slope_range: tuple[float, float] = (0.3, 0.3)
    platform_width: float = 2.0     # flat cap at the peak (m along x)


@configclass
class TransverseStairsCfg(HfTerrainBaseCfg):
    """y-constant stairs along the corridor (ascend to center, descend after)."""
    function = transverse_stairs_terrain
    step_height_range: tuple[float, float] = (0.10, 0.10)
    step_width: float = 0.35        # tread depth along x (m)
