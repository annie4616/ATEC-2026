"""PiPER-X forward kinematics and geometric Jacobian, from the URDF, with no physics engine.

WHY
---
`ArmDriver._jacobian()` reads the Jacobian out of PhysX:

    j = self.robot.root_physx_view.get_jacobians()[:, body, :, joints]   (mission.py)

`root_physx_view` exists only inside a running simulator, so every arm behaviour in the
mission — the RL grasp loop, `servo_tcp_pose`, `servo_tcp_to`, the scripted pinches, the
bin release — stops at that line on real hardware. The Jacobian is not sensor data
though: it is a property of the robot MODEL, and the model is `a2_piperx.urdf`. This
module computes it.

CHAIN (from a2_piperx.urdf, all six arm joints revolute about their own local z):

    base_link --arm_mount(fixed, xyz 0.16 0 0.08)--> arm_base
      j1  xyz (0, 0, 0.123)          rpy (0, 0, 0)
      j2  xyz (0, 0, 0)              rpy (1.5708, -0.13586, 3.1416)
      j3  xyz (0.28503, 0, 0)        rpy (0, 0, 2.8377)
      j4  xyz (0.27264, 0, 0)        rpy (0, 0, 0.075129)
      j5  xyz (0.076857, 0.00062395, 0)  rpy (-1.5708, 0, 0)
      j6  xyz (0.035, 0, 0)          rpy (1.5708, 0, 1.5708)
    gripper_base == arm_link6 (fixed, zero offset); TCP is +0.135 m along its local z,
    which is also where arm_joint7/8 (the fingers) are mounted.

The Jacobian is returned in the BASE frame at the TCP, which is exactly what
`mission.ArmDriver._jacobian()` produces after its rotate-and-shift block — so it is a
drop-in for that method, and `scripts/check_armkin.py` compares the two numerically
inside a live sim before anyone trusts it.

numpy only: no torch, no Isaac, runs on the N150.
"""

from __future__ import annotations

import os

import numpy as np

# (translation xyz, rotation rpy) of each joint frame relative to the previous one.
ARM_ORIGINS = (
    ((0.0, 0.0, 0.123), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.5708, -0.13586, 3.1416)),
    ((0.28503, 0.0, 0.0), (0.0, 0.0, 2.8377)),
    ((0.27264, 0.0, 0.0), (0.0, 0.0, 0.075129)),
    ((0.076857, 0.00062395, 0.0), (-1.5708, 0.0, 0.0)),
    ((0.035, 0.0, 0.0), (1.5708, 0.0, 1.5708)),
)
# base_link -> arm_base (urdf joint "arm_mount", fixed)
ARM_MOUNT_XYZ = np.array([0.16, 0.0, 0.08])
# gripper_base -> TCP. Read from the SAME env var mission.py uses (mission.py:81), not a
# hardcoded 0.135: with GRASP_TCP_OFFSET set, a fixed value here would silently compare a
# different tool point in check_armkin.py and still report agreement.
TCP_LOCAL = np.array([0.0, 0.0, float(os.environ.get("GRASP_TCP_OFFSET", "0.135"))])
JOINT_LIMITS = (
    (-2.618, 2.618), (0.0, 3.14), (-2.9671, 0.0),
    (-1.57, 1.57), (-1.57, 1.57), (-3.14, 3.14),
)


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


_FIXED_R = tuple(_rpy(*rpy) for _, rpy in ARM_ORIGINS)
_FIXED_T = tuple(np.asarray(xyz, float) for xyz, _ in ARM_ORIGINS)


def _rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def fk(q6, tcp_local=TCP_LOCAL):
    """Forward kinematics in the BASE (body) frame.

    Returns (p_tcp, R_ee, origins, axes):
        p_tcp   (3,)    TCP position
        R_ee    (3, 3)  gripper_base orientation
        origins (6, 3)  each joint's origin — the Jacobian needs them
        axes    (6, 3)  each joint's rotation axis in base coordinates
    """
    q = np.asarray(q6, float).reshape(6)
    p = ARM_MOUNT_XYZ.astype(float).copy()
    R = np.eye(3)
    origins = np.zeros((6, 3))
    axes = np.zeros((6, 3))
    for i in range(6):
        p = p + R @ _FIXED_T[i]
        R = R @ _FIXED_R[i]
        origins[i] = p
        axes[i] = R[:, 2]              # every arm joint turns about its local z
        R = R @ _rotz(q[i])
    return p + R @ np.asarray(tcp_local, float), R, origins, axes


def jacobian(q6, tcp_local=TCP_LOCAL) -> np.ndarray:
    """6x6 geometric Jacobian at the TCP, expressed in the base frame.

    Rows 0-2 map joint rates to TCP linear velocity, rows 3-5 to angular velocity —
    the same layout the DLS solve in ArmDriver.ik_step consumes.
    """
    p_tcp, _, origins, axes = fk(q6, tcp_local)
    J = np.zeros((6, 6))
    for i in range(6):
        J[:3, i] = np.cross(axes[i], p_tcp - origins[i])
        J[3:, i] = axes[i]
    return J


def jacobian_training_convention(q6, tcp_local=TCP_LOCAL) -> np.ndarray:
    """The Jacobian the GRASP POLICY WAS TRAINED THROUGH — deliberately not the correct one.

    IsaacLab's DifferentialInverseKinematicsAction shifts the base-frame Jacobian to the
    tool point with `J_lin += -[r]x J_ang` using the offset expressed in the END-EFFECTOR's
    frame while J_ang is already in the BASE frame
    (IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py:257-260, and
    mission.py:352-364 is a line-for-line copy of it). The frames do not match, so the
    linear block is wrong by up to 0.26 -- measured against this module in a live sim:
    median 0.1361, max 0.2142, while FK and the angular block agree exactly.

    It is still what matters for deployment. The RL grasp policy was trained inside that
    action term, so its actions are adapted to that plant; feeding it the mathematically
    correct Jacobian is a train/deploy MISMATCH, not a free improvement. Measured:
    MISSION_IK_FRAME_FIX=1 scored 0/5 against a 2-3/6 baseline in the manipulation probe.

    So: use THIS for anything driving the trained policy, and jacobian() for scripted
    Cartesian motion where being correct is what you want.
    """
    p_tcp, R_ee, origins, axes = fk(q6, tcp_local)
    J = np.zeros((6, 6))
    p_gb = p_tcp - R_ee @ np.asarray(tcp_local, float)   # gripper_base origin
    for i in range(6):
        J[:3, i] = np.cross(axes[i], p_gb - origins[i])
        J[3:, i] = axes[i]
    r_local = np.asarray(tcp_local, float)
    skew = np.array([[0.0, -r_local[2], r_local[1]],
                     [r_local[2], 0.0, -r_local[0]],
                     [-r_local[1], r_local[0], 0.0]])
    J[:3] = J[:3] + (-skew) @ J[3:]
    return J


def dls_step(J: np.ndarray, dx: np.ndarray, lam: float = 0.05) -> np.ndarray:
    """Damped least squares joint step: dq = J^T (J J^T + lam^2 I)^-1 dx.

    lam=0.05 is `MISSION_IK_LAMBDA`'s default and is not cosmetic — it damps the
    near-singular configurations where an ordinary pseudo-inverse turns a normal pose
    command into an arm whip (mission.py's own note on the 120-scoring config).
    """
    J = np.asarray(J, float)
    dx = np.asarray(dx, float).reshape(6)
    A = J @ J.T + (lam ** 2) * np.eye(6)
    return J.T @ np.linalg.solve(A, dx)


def _selftest() -> None:
    """Numerical check that jacobian() really is d(pose)/dq of fk(), no simulator needed."""
    rng = np.random.default_rng(0)
    worst_v = worst_w = 0.0
    for _ in range(200):
        q = np.array([rng.uniform(lo, hi) for lo, hi in JOINT_LIMITS])
        J = jacobian(q)
        p0, R0, _, _ = fk(q)
        h = 1e-6
        for i in range(6):
            qp = q.copy(); qp[i] += h
            p1, R1, _, _ = fk(qp)
            worst_v = max(worst_v, np.abs((p1 - p0) / h - J[:3, i]).max())
            # angular: dR R^T is the skew of the angular velocity
            W = (R1 - R0) / h @ R0.T
            w = np.array([W[2, 1] - W[1, 2], W[0, 2] - W[2, 0], W[1, 0] - W[0, 1]]) / 2.0
            worst_w = max(worst_w, np.abs(w - J[3:, i]).max())
    print(f"finite-difference check over 200 random poses:")
    print(f"  linear  max|J - dp/dq|  = {worst_v:.2e}")
    print(f"  angular max|J - dR/dq|  = {worst_w:.2e}")
    q = np.zeros(6)
    p, _, _, _ = fk(q)
    print(f"  TCP at q=0 (base frame) = {np.round(p, 4).tolist()}")
    print(f"  reach at q=0            = {np.linalg.norm(p - ARM_MOUNT_XYZ):.3f} m")
    # DLS round trip: ask for a small Cartesian move, check it lands
    J = jacobian(np.array([0.1, 1.0, -1.0, 0.2, 0.3, 0.0]))
    dx = np.array([0.01, 0.0, -0.005, 0.0, 0.0, 0.0])
    dq = dls_step(J, dx)
    print(f"  DLS residual |J dq - dx| = {np.abs(J @ dq - dx).max():.2e} "
          f"(damped, so nonzero by design)")
    # the training-convention variant should differ from the correct one by exactly the
    # amount the live-sim comparison measured (median 0.1361 / max 0.2142)
    d = []
    for _ in range(200):
        qq = np.array([rng.uniform(lo, hi) for lo, hi in JOINT_LIMITS])
        d.append(np.abs(jacobian_training_convention(qq)[:3] - jacobian(qq)[:3]).max())
    print(f"  training-convention vs correct: median {np.median(d):.4f} max {np.max(d):.4f}"
          f"   (live sim measured 0.1361 / 0.2142)")


if __name__ == "__main__":
    _selftest()
