"""The grasp policy's 42-dimensional observation, and the pre-grasp bank.

This is the one substantial piece the onboard stack did not have.  `onboard/` already
carries the arm controller, the analytic FK and Jacobian, the vendored IK solver, the
gripper conversion and the hold verdict -- but nothing assembled the vector the policy eats.

The layout is a contract with a frozen checkpoint, so it is written out term by term and
checked, not paraphrased:

    [ 0: 3]  ee_pos_b        gripper_base origin in the BASE frame   <- NOT the TCP
    [ 3: 7]  ee_quat_b       gripper_base orientation, (w, x, y, z)
    [ 7:10]  grasp_vec_ee    (target - TCP) expressed in the EE FRAME <- not the base frame
    [10:13]  target_axis_b   world down, in the base frame  = projected gravity
    [13:16]  tool_axis_b     gripper local +z, in the base frame
    [16:17]  gripper_width   q7 - q8, which IS the real opening in metres
    [17:24]  last_action     the previous RAW 7-vector output (no scale, no clip)
    [24:29]  grasp_params    [target_width, 0, 0.5, 0.5, 0.5]
    [29:32]  obb_dims        object extents x dim_err
    [32:36]  base_quat       body orientation, YAW STRIPPED  (see below)
    [36:39]  base_ang_vel    body frame x 0.25
    [39:42]  base_lin_acc    clamped +-50, x 0.1

Four traps, each of which produces a plausible-looking vector that is wrong:

1. `ee_pos_b` is the gripper_base, and the TCP is 0.135 m further along its local z.  Using
   the TCP here shifts the policy's whole sense of where its hand is.
2. `grasp_vec` lives in the EE frame.  Handing it the base-frame residual rotates the
   policy's error signal by the wrist angle.
3. `gripper_width` is already the real opening.  The factor of two between the simulator's
   per-finger width and the real stroke applies to the COMMAND, not to this reading.
4. **Yaw.**  Training reset the robot at yaw ~ 0 on every episode, so the policy has never
   seen a world heading in `base_quat`.  On the course the robot faces anywhere, which makes
   this term out-of-distribution for most of a run.  On hardware there is no world yaw to
   send in the first place -- so building it yaw-free is both the only thing we CAN do and
   the thing that matches training.  `collector.yaml: grasp.noyaw`.
"""
from __future__ import annotations

import numpy as np

from onboard.cfg import onboard_path
from onboard.perceive.camadapter import quat_to_mat, rot_z, yaw_of

OBS_DIM = 42
ACT_DIM = 7


def quat_mul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_conj(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_apply(q, v) -> np.ndarray:
    return quat_to_mat(q) @ np.asarray(v, float)


def quat_apply_inv(q, v) -> np.ndarray:
    return quat_to_mat(q).T @ np.asarray(v, float)


def quat_from_mat(R: np.ndarray) -> np.ndarray:
    from onboard.control.arm import quat_from_mat as _q
    return _q(R)


def strip_yaw(quat) -> np.ndarray:
    """Remove the world heading, keeping roll and pitch.  Trap 4."""
    q = np.asarray(quat, float)
    y = yaw_of(quat_to_mat(q))
    qy = np.array([np.cos(y / 2), 0.0, 0.0, np.sin(y / 2)])
    return quat_mul(quat_conj(qy), q)


class PregraspBank:
    """Arm configurations whose gripper lands in the workspace at the crouch height.

    The bank is not a trajectory library -- it is the definition of *where the object must
    be* for a grasp to be attempted at all.  `anchor` (the median kept position) is the aim
    point the approach drives to, and `drift` past 0.40 m means the approach never converged
    and retrying from the same stance only spends clock.
    """

    def __init__(self, path, min_x: float = 0.50) -> None:
        z = np.load(onboard_path(path), allow_pickle=True)
        ee = np.asarray(z["ee_pos_b"], float)
        keep = ee[:, 0] >= float(min_x)
        self.ee = ee[keep]
        self.q = np.asarray(z["arm_q"], float)[keep]
        self.crouch_z = float(z["crouch_z"]) if "crouch_z" in z else float("nan")
        if not len(self.ee):
            raise ValueError(f"pregrasp bank {path} is empty after the x >= {min_x} filter")
        self.anchor = np.median(self.ee, axis=0)

    def __len__(self) -> int:
        return len(self.ee)

    def drift(self, target_b) -> float:
        return float(np.linalg.norm(np.asarray(target_b, float)[:2] - self.anchor[:2]))

    def pick(self, target_b, mode: str = "far", min_sep: float = 0.15) -> int:
        """A start pose whose fingertips are NOT already on the object.

        Starting on top of the target is the obvious choice and the wrong one: the policy
        was trained from a standoff and its first actions assume one.
        """
        d = np.linalg.norm(self.ee[:, :2] - np.asarray(target_b, float)[:2], axis=1)
        if mode == "over":
            ok = np.nonzero(d < 0.05)[0]
            if len(ok):
                return int(ok[np.argmin(d[ok])])
        far = np.nonzero(d > float(min_sep))[0]
        return int(far[np.argmin(d[far])]) if len(far) else int(np.argmin(d))


class GraspObs:
    """Assembles the 42-vector.  Stateful only in `last_action`."""

    def __init__(self, cfg, piper_cfg) -> None:
        g = cfg["grasp"]
        self.noyaw = bool(g.get("noyaw", True))
        self.tcp_offset = float(piper_cfg["kinematics"]["tcp_offset"])
        self.last_action = np.zeros(ACT_DIM)
        self.dim_err = 1.0
        self.target_width = 0.0
        self.obb = np.zeros(3)

    def reset(self, target_width: float, obb, dim_err: float = 1.0) -> None:
        """New attempt.  `last_action` MUST be zeroed: it is the policy's only memory, and
        carrying the previous object's last action into a new approach is a step of garbage
        at exactly the moment the arm starts moving."""
        self.last_action = np.zeros(ACT_DIM)
        self.target_width = float(target_width)
        self.obb = np.asarray(obb, float)
        self.dim_err = float(dim_err)

    def build(self, ee_pos_b, ee_quat_b, target_b, gripper_stroke,
              base_quat, base_ang_vel_b, base_lin_acc_w) -> np.ndarray:
        ee_pos_b = np.asarray(ee_pos_b, float)
        ee_quat_b = np.asarray(ee_quat_b, float)
        tcp_b = ee_pos_b + quat_apply(ee_quat_b, [0.0, 0.0, self.tcp_offset])
        grasp_vec = quat_apply_inv(ee_quat_b, np.asarray(target_b, float) - tcp_b)

        bq = np.asarray(base_quat, float)
        R_wb = quat_to_mat(bq)
        target_axis_b = R_wb.T @ np.array([0.0, 0.0, -1.0])      # projected gravity
        tool_axis_b = quat_to_mat(ee_quat_b)[:, 2]               # gripper +z, base frame

        acc = np.clip(np.asarray(base_lin_acc_w, float), -50.0, 50.0) * 0.1
        base_state = np.concatenate([
            strip_yaw(bq) if self.noyaw else bq,
            np.asarray(base_ang_vel_b, float) * 0.25,
            acc,
        ])

        obs = np.concatenate([
            ee_pos_b, ee_quat_b, grasp_vec, target_axis_b, tool_axis_b,
            [float(gripper_stroke)], self.last_action,
            [self.target_width, 0.0, 0.5, 0.5, 0.5],
            self.obb * self.dim_err,
            base_state,
        ])
        if obs.shape != (OBS_DIM,):
            raise ValueError(f"grasp obs is {obs.shape}, expected ({OBS_DIM},)")
        return obs.astype(np.float32)


SLICES = {
    "ee_pos_b": slice(0, 3), "ee_quat_b": slice(3, 7), "grasp_vec_ee": slice(7, 10),
    "target_axis_b": slice(10, 13), "tool_axis_b": slice(13, 16),
    "gripper_width": slice(16, 17), "last_action": slice(17, 24),
    "grasp_params": slice(24, 29), "obb_dims": slice(29, 32),
    "base_quat": slice(32, 36), "base_ang_vel": slice(36, 39), "base_lin_acc": slice(39, 42),
}
