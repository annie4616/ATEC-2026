"""Motor index / sign / offset mapping between the SDK and the policy.

This is the highest-risk file in the bridge, and the reason is worth stating plainly: the
policy's joint order is FR, FL, RR, RL x (hip, thigh, calf), which is character-for-
character the Unitree SDK motor order used on Go2 -- so the map is *probably* the identity.
"Probably" is not good enough.  IsaacLab's own DOF order for the same URDF is the
type-grouped one (four hips, then four thighs, then four calves), and a stack that confused
those two would still stand, still pass a smoke test, and then fold the first time it tried
to turn.  A2 Pro is newer than anything either reference repo covers.

So: the map lives in yaml, its default is the Go2 convention, and tools/dump_motor_map.py
replaces that default with a measurement before the first stand.  Sign and offset live here
too, because a joint that reads backwards looks exactly like a joint that is mapped wrong
until you move it by hand and watch which number moves.
"""
from __future__ import annotations

import numpy as np

N_LEG = 12


class LegMap:
    """Bidirectional SDK<->policy conversion for the 12 leg joints.

        q_policy[i] = sign[i] * (q_motor[idx[i]] - offset[i])
        q_motor[idx[i]] = q_policy[i] / sign[i] + offset[i]

    Velocities and torques get the sign but not the offset.
    """

    def __init__(self, cfg) -> None:
        self.names: list[str] = list(cfg["joint_names"])
        self.idx = np.asarray(cfg["leg_joint2motor_idx"], dtype=np.int64)
        self.sign = np.asarray(cfg["joint_sign"], dtype=np.float64)
        self.offset = np.asarray(cfg["joint_offset"], dtype=np.float64)
        self.q_default = np.asarray(cfg["default_angles"], dtype=np.float64)
        self.action_scale = np.asarray(cfg["action_scale"], dtype=np.float64)
        self.limits = np.asarray(cfg["joint_limits"], dtype=np.float64)
        self.effort = np.asarray(cfg["effort_limits"], dtype=np.float64)
        self.vel_limit = np.asarray(cfg["velocity_limits"], dtype=np.float64)

        for name, arr, shape in (
            ("joint_names", self.names, (N_LEG,)),
            ("leg_joint2motor_idx", self.idx, (N_LEG,)),
            ("joint_sign", self.sign, (N_LEG,)),
            ("joint_offset", self.offset, (N_LEG,)),
            ("default_angles", self.q_default, (N_LEG,)),
            ("action_scale", self.action_scale, (N_LEG,)),
            ("joint_limits", self.limits, (N_LEG, 2)),
            ("effort_limits", self.effort, (N_LEG,)),
            ("velocity_limits", self.vel_limit, (N_LEG,)),
        ):
            got = (len(arr),) if isinstance(arr, list) else arr.shape
            if got != shape:
                raise ValueError(f"a2pro.yaml {name}: expected {shape}, got {got}")

        if sorted(self.idx.tolist()) != list(range(N_LEG)):
            raise ValueError(
                f"leg_joint2motor_idx must be a permutation of 0..11, got {self.idx.tolist()}")
        if not np.all(np.isin(self.sign, (-1.0, 1.0))):
            raise ValueError("joint_sign entries must be +1 or -1")

        # The standing pose must be reachable, or the ramp drives into a hard stop and the
        # driver faults during the 5-minute prep window -- with the judges watching.
        lo, hi = self.limits[:, 0], self.limits[:, 1]
        bad = np.where((self.q_default < lo) | (self.q_default > hi))[0]
        if bad.size:
            names = ", ".join(self.names[i] for i in bad)
            raise ValueError(f"default_angles outside joint_limits for: {names}")

    # ---- SDK -> policy ----------------------------------------------------------------
    def q_from_motor(self, q_motor: np.ndarray) -> np.ndarray:
        return self.sign * (np.asarray(q_motor, dtype=np.float64)[self.idx] - self.offset)

    def rate_from_motor(self, v_motor: np.ndarray) -> np.ndarray:
        """Velocity or torque: signed, but not offset."""
        return self.sign * np.asarray(v_motor, dtype=np.float64)[self.idx]

    # ---- policy -> SDK ----------------------------------------------------------------
    def q_to_motor(self, q_policy: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        m = np.zeros(N_LEG) if out is None else out
        m[self.idx] = np.asarray(q_policy, dtype=np.float64) / self.sign + self.offset
        return m

    def scalar_to_motor(self, v: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Gains: index-permuted only.  kp/kd are unsigned magnitudes."""
        m = np.zeros(N_LEG) if out is None else out
        m[self.idx] = np.asarray(v, dtype=np.float64)
        return m

    # ---- the one line the whole bridge exists to compute -------------------------------
    def action_to_target(self, action: np.ndarray) -> np.ndarray:
        """q_target = q_default + action_scale * a.

        The simulator reaches the same place in two hops -- the policy emits `a`, the code
        multiplies by `action_scale / ENV_LEG_SCALE`, and the env's action term applies
        `default + 0.5 * a_env`.  On hardware there is no action term, so those two hops
        collapse into this.  Applying the 0.5 as well is the classic way to halve the gait
        amplitude and spend a day wondering why the robot shuffles.
        """
        return self.q_default + self.action_scale * np.asarray(action, dtype=np.float64)

    def clamp(self, q: np.ndarray, soft: float = 1.0) -> np.ndarray:
        """Clamp to the (optionally shrunk) joint range.  Shrinking matters: a target
        sitting exactly on a hard stop is where real drivers fault rather than saturate."""
        mid = 0.5 * (self.limits[:, 0] + self.limits[:, 1])
        half = 0.5 * (self.limits[:, 1] - self.limits[:, 0]) * soft
        return np.clip(q, mid - half, mid + half)

    def describe(self) -> str:
        rows = [f"{i:2d} {n:9s} motor={self.idx[i]:2d} sign={self.sign[i]:+.0f} "
                f"off={self.offset[i]:+.3f} q0={self.q_default[i]:+.2f} "
                f"scale={self.action_scale[i]:.3f} lim=[{self.limits[i,0]:+.2f},{self.limits[i,1]:+.2f}]"
                for i, n in enumerate(self.names)]
        return "\n".join(rows)
