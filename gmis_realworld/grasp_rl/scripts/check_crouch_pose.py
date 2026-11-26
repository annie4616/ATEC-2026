"""Gate check for a freshly extracted crouch pose (guards against the depth-0.26-style
degenerate attractor: twisted collapse that settles far below the commanded height).

Usage: check_crouch_pose.py <depth_tag> [poses_json]
Exits non-zero (and prints GATE_FAIL) when the settled height is off-target or a hip is
outside its clamped range; prints GATE_OK otherwise.
"""

import json
import sys
from pathlib import Path

POSES = Path(sys.argv[2]) if len(sys.argv) > 2 else (
    Path(__file__).resolve().parents[1] / "banks" / "crouch_poses.json"
)
HEIGHT_TOL = 0.03
HIP_MAX = 0.95

# B2 joint limits; a settled pose pinned AT a limit is the twisted-collapse signature
# (right height, contorted kneel) — reject anything outside the interior margin
import os
if os.environ.get("CROUCH_LIMITS_PRESET") == "a2":
    LIMITS = {"_hip_": (-1.01, 1.01), "_thigh_": (-2.34, 3.15), "_calf_": (-2.77, -0.54)}
else:
    LIMITS = {"_hip_": (-0.87, 0.87), "_thigh_": (-0.94, 4.69), "_calf_": (-2.82, -0.43)}
MARGIN = 0.02

tag = sys.argv[1]
poses = json.loads(POSES.read_text())["poses"]
if tag not in poses:
    sys.exit(f"GATE_FAIL crouch {tag}: no entry in {POSES}")
entry = poses[tag]
z = entry["base_height_mean"]
if abs(z - float(tag)) > HEIGHT_TOL:
    sys.exit(f"GATE_FAIL crouch {tag}: settled z={z:.3f} (target {float(tag):.2f} +/- {HEIGHT_TOL})")
bad = {}
for k, v in entry["joint_pos"].items():
    for pat, (lo, hi) in LIMITS.items():
        if pat in k and not (lo + MARGIN <= v <= hi - MARGIN):
            bad[k] = round(v, 3)
if bad:
    sys.exit(f"GATE_FAIL crouch {tag}: joints at/over limits (twisted pose) {bad}")
hips = {k: round(v, 2) for k, v in entry["joint_pos"].items() if "_hip_" in k}
print(f"GATE_OK crouch {tag}: z={z:.3f} hips={hips}")
