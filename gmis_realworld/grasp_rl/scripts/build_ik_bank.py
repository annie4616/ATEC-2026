"""Build an IK pre-grasp bank: a canonical detection -> arm-configuration mapping.

WHAT THE OLD BANK IS AND ISN'T. `pregrasp_bank_<depth>.npz` is a SAMPLED reach map:
random arm configurations that happened to put the TCP in the workspace. Training (RSI)
starts every episode at one of them; deployment picks the entry nearest the detection.
That works, but the mapping is arbitrary — two nearby detections can select very
different elbow shapes, and there is no closed-form "given the object HERE, the pre-grasp
is THIS" that a real robot could compute from its own detection.

THIS SCRIPT replaces the sampling with IK on a grid. For every grasp-point cell in the
reachable window it solves the analytic chain (course_mission/armkin.py — verified
against PhysX FK to 1.4e-3 mm) for the TCP hovering a fixed offset above that point,
SEEDED FROM THE NEAREST OLD-BANK ENTRY so the elbow branch and wrist orientation stay
inside the distribution the champion policies were trained on. The output uses the same
npz schema (arm_q / ee_pos_b / crouch_z), so BOTH consumers pick it up unchanged via
GRASP_BANK_DIR:

  * training RSI (graspenv/mdp/events.py reset_pregrasp_rsi)
  * deployment (runner loads ckpts["bank"], picks nearest entry, joint-ramps onto it)

CPU only, no simulator.

    python grasp_rl/scripts/build_ik_bank.py --depth 0.24 \
        --old eunha_rl/s0821/banks/pregrasp_bank_0.24.npz \
        --out eunha_rl/s0821/banks_ik/pregrasp_bank_0.24.npz
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

_RW = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "armkin", _RW / "course_mission" / "armkin.py")
armkin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(armkin)

parser = argparse.ArgumentParser()
parser.add_argument("--depth", type=str, required=True)
parser.add_argument("--old", type=str, required=True, help="existing sampled bank (seed poses)")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--nx", type=int, default=14)
parser.add_argument("--ny", type=int, default=21)
parser.add_argument("--iters", type=int, default=120)
parser.add_argument("--tol", type=float, default=0.004, help="position tolerance [m]")
args = parser.parse_args()

old = np.load(args.old, allow_pickle=True)
old_q = np.asarray(old["arm_q"], float)
old_ee = np.asarray(old["ee_pos_b"], float)
crouch_z = float(old["crouch_z"])

# Target window = the old bank's envelope INTERSECTED with the stage-2 operating region.
# The raw bank spans x -0.19..0.73 — it contains rear-reaching poses that the stage-2 RSI
# filters out (min_ee_x = 0.50, grasp_env_cfg.py:403: objects further back get stomped by
# the now-moving front feet) and that a deployed pre-grasp could never use. Building IK
# cells there would also drag the bank's median (= the runner's grasp_anchor, its
# approach aim point and drift gate) away from where grasps actually happen.
keep = old_ee[:, 0] >= 0.50
lo = old_ee[keep].min(axis=0) + 0.01
hi = old_ee[keep].max(axis=0) - 0.01
z_hover = float(np.median(old_ee[keep, 2]))  # the height the policies expect to start at
print(f"window x [{lo[0]:.3f},{hi[0]:.3f}] y [{lo[1]:.3f},{hi[1]:.3f}] "
      f"z_hover {z_hover:.3f} (old bank {len(old_q)} configs, crouch_z {crouch_z:.3f})")

xs = np.linspace(lo[0], hi[0], args.nx)
ys = np.linspace(lo[1], hi[1], args.ny)
# HEIGHT VARIETY IS PART OF THE TRAINED DISTRIBUTION. The first version solved every cell
# at the single median height and the RSI start states collapsed from a 24 cm z-spread to
# 8 mm — measured in training, that bank alone dropped the warm-started champion's mean
# reward from 3.45 (reference) to 0.71. Span the old bank's interquartile range instead.
z_lo, z_hi = np.percentile(old_ee[keep, 2], 20), np.percentile(old_ee[keep, 2], 80)
zs = np.linspace(z_lo, z_hi, 3)
# The same measurement exposed the second defect: position-only DLS let the WRIST drift
# to 47-61 deg of tool tilt where the trained bank's p90 is 33 deg. Solve the full 6D
# pose — position to the grid target, ORIENTATION to the seed's — and reject any residue.
TILT_MAX = np.radians(35.0)


def _rotvec(Ra, Rb):
    """axis*angle taking Rb to Ra."""
    R = Ra @ Rb.T
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    ang = np.arccos(c)
    if ang < 1e-8:
        return np.zeros(3)
    return ang / (2 * np.sin(ang)) * np.array([R[2, 1] - R[1, 2],
                                               R[0, 2] - R[2, 0],
                                               R[1, 0] - R[0, 1]])


out_q, out_ee, miss = [], [], 0
for x in xs:
    for y in ys:
        for z in zs:
            tgt = np.array([x, y, z])
            near = np.argmin(np.linalg.norm(old_ee - tgt, axis=1))
            seed = old_q[near].copy()
            _, R_seed, _, _ = armkin.fk(seed)
            q = seed.copy()
            ok = False
            for _ in range(args.iters):
                p, R, _, _ = armkin.fk(q)
                e_pos = tgt - p
                e_rot = _rotvec(R_seed, R)
                if np.linalg.norm(e_pos) < args.tol and np.linalg.norm(e_rot) < 0.05:
                    ok = True
                    break
                J = armkin.jacobian(q)
                dq = armkin.dls_step(J, np.concatenate([e_pos, 0.6 * e_rot]), lam=0.05)
                q = np.clip(q + np.clip(dq, -0.15, 0.15),
                            [l for l, _ in armkin.JOINT_LIMITS],
                            [h for _, h in armkin.JOINT_LIMITS])
            if not ok:
                miss += 1
                continue
            margin = min(min(q[i] - armkin.JOINT_LIMITS[i][0],
                             armkin.JOINT_LIMITS[i][1] - q[i]) for i in range(6))
            p, R, _, _ = armkin.fk(q)
            tilt = np.arccos(np.clip(-R[2, 2], -1, 1))
            if margin < 0.03 or tilt > TILT_MAX:
                miss += 1
                continue
            out_q.append(q)
            out_ee.append(p)

out_q = np.asarray(out_q, np.float32)
out_ee = np.asarray(out_ee, np.float32)
print(f"solved {len(out_q)}/{args.nx * args.ny} cells ({miss} rejected)")

# sanity: the IK entries must live inside the old distribution, not invent a new one
tilts = [np.degrees(np.arccos(np.clip(-armkin.fk(q)[1][2, 2], -1, 1))) for q in out_q]
print(f"tool tilt: median {np.median(tilts):.1f} deg  p90 {np.percentile(tilts, 90):.1f} deg "
      f"(old bank p90 ~33)")
print(f"z spread: [{out_ee[:, 2].min():.3f}, {out_ee[:, 2].max():.3f}]")
d_near = [np.linalg.norm(old_q - q, axis=1).min() for q in out_q]
print(f"nearest-old-config joint distance: median {np.median(d_near):.3f} rad, "
      f"max {np.max(d_near):.3f} rad")
res = [np.linalg.norm(armkin.fk(q)[0] - ee) for q, ee in zip(out_q, out_ee)]
print(f"stored-vs-FK residual max {max(res) * 1000:.2f} mm")

outp = Path(args.out)
outp.parent.mkdir(parents=True, exist_ok=True)
if outp.exists():
    raise SystemExit(f"REFUSING to overwrite {outp}")
np.savez(outp, arm_q=out_q, ee_pos_b=out_ee, crouch_z=crouch_z)
print(f"wrote {outp}  ({len(out_q)} configs)")
