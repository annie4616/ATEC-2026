"""v3/v4 basket: the v1 basket (same footprint, walls, mount and mass) plus a LOW LIP
across the floor — v3: face at base-frame x -0.15, 30 mm high; v4: face -0.16, 20 mm.

Why (exp/grasp-boost, 0903): the arm's in-basket reach ends at base x -0.15 while the
v1 interior runs back to -0.214, and the stair climbs to the bins slide the load
against the rear wall — svJ's rind was deposited at x -0.111 and sat at -0.184 when
the robot reached the green bin: six basket-pick tries, 3,000 steps, scored none.
Cans released in the aft slot (x -0.13) also cost 3-4 tries because their +0.07 hover
point is out of reach. The lip stops a sliding can (r 33 mm) or rind (70 x 25 mm) at
x >= -0.117, inside the window, and at 30 mm it sits 10 cm under the deposit swing
(release TCP z 0.278, object underside 0.228) and clear of the q_pick entry.

v4 (0903, svN): the aft release slot lands a can at x -0.125 +- 0.005 (r 33 mm), which
overlaps a lip face at -0.15 by 8 mm — svN's first can came to rest ON the 30 mm lip
(rel z 0.153 vs 0.131 on the floor), one nudge from tipping behind it. Face at -0.16
clears the landing by ~6 mm, still stops a sliding can at x -0.127 / rind at -0.125
(window x >= -0.15), and 20 mm holds a can up to ~60 deg of pitch (roll-over needs
tan(pitch) > sqrt(2rh-h^2)/(r-h) = 3.4) while being harder to perch on.

Pass to a mission with
  BASKET_USD=eunha_rl/assets/a2_piperx_basket_v4.usd  BASKET_LIP_X=-0.16
(runner.pick_from_basket skips the wrist-roll tries for objects resting on the lip).
Build with `--v 4`, convert with `convert_basket_v3.py --v 4`. v1/v2 assets stay untouched.
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PRESETS = {3: (-0.15, 0.03), 4: (-0.16, 0.02), 5: (-0.16, 0.02), 6: (-0.16, 0.02)}
ap = argparse.ArgumentParser()
ap.add_argument("--v", type=int, default=3, choices=sorted(PRESETS))
ap.add_argument("--tilt", type=float, default=14.0,
                help="v5 only: floor tilt in degrees, rising toward the OUTBOARD wall")
args = ap.parse_args()

# v5 (0906): the v4 lip fixed the fore-aft slide and left the LATERAL one untouched,
# which is now the biggest single loss on the board. The arm's in-basket window ends
# near base y -0.24: measured over hs16a/hs16b/hs21b, every pick from y -0.18..-0.23
# converted (servo residual 5-9 mm) and every one from y -0.25..-0.34 stalled at
# 44-121 mm and was abandoned — four objects, 80 points, all scored `none` while
# sitting in the basket. And the load MIGRATES outboard on the ride: hs21b deposited
# trash_1 at y -0.276 and picked at -0.336, 60 mm further out, hard against the far
# wall. A rail would only move the perching risk around, so tilt the FLOOR instead:
# a plate rising toward the outboard wall makes every article settle against the near
# wall at y ~ -0.19, which is the middle of the window that always works. No capacity
# lost along x, no edge to balance on.

RW = Path(__file__).resolve().parents[2]
SRC = RW / "a2pro_rl" / "assets" / "a2_piperx.urdf"
OUT = RW / "a2pro_rl" / "assets" / f"a2_piperx_basket_v{args.v}.urdf"

IN_X, IN_Y, DEPTH, T = 0.28, 0.20, 0.15, 0.008
# v6 (0908): the PURCHASED basket — outer 260 x 185 x 160 mm, moulded plastic,
# ROUNDED rim. Interior estimated at 245 x 170 x 154 with 2.5 mm walls (re-measure
# the inside and re-run when the team calipers it). The wedge insert the team glues
# in reproduces the v5 slope+lip, so v6 keeps both. The rim is modelled as 8 mm
# cylinders along the top edges — a rounded rim sheds perched objects instead of
# balancing them, which is the behaviour that matters for deposits.
if args.v >= 6:
    IN_X, IN_Y, DEPTH, T = 0.245, 0.170, 0.154, 0.0025
CX, CY, CZ_BOT = -0.07, -0.27, 0.09  # base_link frame: side-rear saddle mount (v1)
LIP_FRONT_B, LIP_H = PRESETS[args.v]  # base-frame x of the lip's front face, height
LIP_T = 0.01                          # thickness (x)

tree = ET.parse(SRC)
root = tree.getroot()

link = ET.SubElement(root, "link", {"name": "basket"})
inertial = ET.SubElement(link, "inertial")
ET.SubElement(inertial, "origin", {"xyz": "0 0 0.06", "rpy": "0 0 0"})
ET.SubElement(inertial, "mass", {"value": "0.3"})
ET.SubElement(inertial, "inertia", {"ixx": "0.002", "ixy": "0", "ixz": "0",
                                    "iyy": "0.002", "iyz": "0", "izz": "0.003"})


def box(cx, cy, cz, sx, sy, sz, rpy="0 0 0"):
    for tag in ("visual", "collision"):
        el = ET.SubElement(link, tag)
        ET.SubElement(el, "origin", {"xyz": f"{cx} {cy} {cz}", "rpy": rpy})
        geo = ET.SubElement(el, "geometry")
        ET.SubElement(geo, "box", {"size": f"{sx} {sy} {sz}"})


# geometry in the BASKET link frame (origin at basket bottom-center)
hx, hy = IN_X / 2, IN_Y / 2
box(0, 0, T / 2, IN_X + 2 * T, IN_Y + 2 * T, T)                       # bottom
box(+(hx + T / 2), 0, T + DEPTH / 2, T, IN_Y + 2 * T, DEPTH)          # +x wall
box(-(hx + T / 2), 0, T + DEPTH / 2, T, IN_Y + 2 * T, DEPTH)          # -x wall
box(0, +(hy + T / 2), T + DEPTH / 2, IN_X, T, DEPTH)                  # +y wall
box(0, -(hy + T / 2), T + DEPTH / 2, IN_X, T, DEPTH)                  # -y wall
# rear lip: front face at base x LIP_FRONT_B -> link x (LIP_FRONT_B - CX)
lip_cx = (LIP_FRONT_B - CX) - LIP_T / 2
box(lip_cx, 0, T + LIP_H / 2, LIP_T, IN_Y, LIP_H)                      # rear lip
if args.v >= 5:
    # SLOPED FALSE FLOOR (see the v5 note at the top). Link +y is INBOARD (base
    # y -0.17, the side the arm reaches over); link -y is the far wall. A roll of
    # +a about x lifts +y, so roll by -a to lift the FAR side and let everything
    # settle inboard. 14 deg over the 200 mm interior is a 50 mm rise at the far
    # wall — past the ~10 deg where a can stops rolling on this contact material,
    # and shallow enough that the plate never reaches the 150 mm rim.
    _a = -np.radians(args.tilt)
    _pt = 0.006                                  # plate thickness
    _len = IN_Y / np.cos(abs(_a))                # so the plate still spans the box
    _cz = T + _pt / 2 + (IN_Y / 2) * np.sin(abs(_a))
    box(0.0, 0.0, _cz, IN_X, _len, _pt, rpy=f"{_a} 0 0")               # sloped floor
if args.v >= 6:
    # ROUNDED RIM: an 8 mm-radius bar along each top edge. Contact-wise this is what
    # a rolled plastic rim does — nothing balances on it, sweeps slide over it.
    _rr = 0.008
    _rz = T + DEPTH + _rr / 2

    def cyl(cx, cy, length, axis):
        rpy = "0 1.5707963 0" if axis == "x" else "1.5707963 0 0"
        for tag in ("visual", "collision"):
            el = ET.SubElement(link, tag)
            ET.SubElement(el, "origin", {"xyz": f"{cx} {cy} {_rz}", "rpy": rpy})
            geo = ET.SubElement(el, "geometry")
            ET.SubElement(geo, "cylinder", {"radius": f"{_rr}", "length": f"{length}"})

    cyl(0, +(hy + T / 2), IN_X + 2 * T, "x")
    cyl(0, -(hy + T / 2), IN_X + 2 * T, "x")
    cyl(+(hx + T / 2), 0, IN_Y + 2 * T, "y")
    cyl(-(hx + T / 2), 0, IN_Y + 2 * T, "y")

j = ET.SubElement(root, "joint", {"name": "basket_mount", "type": "fixed"})
ET.SubElement(j, "origin", {"xyz": f"{CX} {CY} {CZ_BOT}", "rpy": "0 0 0"})
ET.SubElement(j, "parent", {"link": "base_link"})
ET.SubElement(j, "child", {"link": "basket"})

tree.write(OUT, encoding="unicode", xml_declaration=True)
print(f"written: {OUT}")
print(f"basket interior {IN_X}x{IN_Y}x{DEPTH} m, opening z=+{CZ_BOT + T + DEPTH:.3f} "
      f"center=({CX},{CY}); lip x [{LIP_FRONT_B - LIP_T:.3f},{LIP_FRONT_B:.3f}] "
      f"top z=+{CZ_BOT + T + LIP_H:.3f} (base frame)")
