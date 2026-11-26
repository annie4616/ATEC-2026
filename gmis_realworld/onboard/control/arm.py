"""The arm control step: the grasp policy's action -> six joint targets + a gripper stroke.

This reproduces `mission.ArmDriver` without a simulator.  Two substitutions carry all the
risk, and both have already been measured elsewhere in the project:

**The Jacobian comes from the URDF, not from PhysX.**  `ArmDriver._jacobian()` reads
`root_physx_view.get_jacobians()`, which exists only inside a running simulator, and it is
the single line that stops every arm behaviour on hardware.  `course_mission/armkin.py`
computes it analytically and `scripts/check_armkin.py` has already compared the two against
a live articulation.

**It must be the WRONG Jacobian.**  IsaacLab's `DifferentialInverseKinematicsAction` mixes
an end-effector-frame offset into a base-frame angular block; `armkin.jacobian_training_
convention()` reproduces that quirk deliberately (median divergence from the correct one:
0.136, max 0.214).  The grasp policy was trained through it, so the quirk is part of its
interface -- switching to the correct Jacobian scored 0/5 against a 2-3/6 baseline.  Use
`jacobian()` for scripted Cartesian motion, `jacobian_training_convention()` for anything
the policy drives.  This class takes that as a constructor flag rather than a guess.

The rate limiter is not optional on hardware.  The deployed grasp checkpoint commands pose
deltas of 75-124 mm and 22-32 deg per 20 ms step, against a training contract of 2 cm/step;
in the simulator that saturates harmlessly, and on a real driver it is a fault.
"""
from __future__ import annotations

import numpy as np
import torch

from course_mission import armkin
from onboard.control.limits import ScalarLimiter, TargetLimiter
from onboard.oio.iface import ArmCommand, ArmState
from onboard.vendor.diffik import DifferentialIKController, DifferentialIKControllerCfg


def quat_from_mat(R: np.ndarray) -> np.ndarray:
    """3x3 -> (w, x, y, z), via the largest-diagonal branch for numerical stability."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


class ArmController:
    def __init__(self, piper_cfg, action_scale: float = 0.02) -> None:
        self.cfg = piper_cfg
        k = piper_cfg["kinematics"]
        self.tcp_offset = float(k["tcp_offset"])
        self.lam = float(k["ik_lambda"])
        self.use_training_jac = bool(k.get("use_training_jacobian", True))
        self.action_scale = float(action_scale)

        lim = np.asarray(piper_cfg["joint_limits"], dtype=np.float64)
        L = piper_cfg["limits"]
        self.limiter = TargetLimiter(lim[:, 0], lim[:, 1],
                                     max_delta=float(L["max_target_delta_rad"]),
                                     soft=float(L["soft_limit_factor"]),
                                     names=list(piper_cfg["joint_names"]))
        g = piper_cfg["gripper"]
        self.max_stroke = float(g["max_stroke_m"])
        self.stroke_per_width = float(g["stroke_per_width"])
        self.grip_limiter = ScalarLimiter(0.0, self.max_stroke,
                                          float(g["max_stroke_delta_m"]))
        self.hold_current = float(g["hold_current_a"])

        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True,
                                        ik_method="dls",
                                        ik_params={"lambda_val": self.lam}),
            num_envs=1, device="cpu")
        self._tcp_local = np.array([0.0, 0.0, self.tcp_offset])

    # ---- kinematics ---------------------------------------------------------------------
    def fk(self, q6) -> tuple[np.ndarray, np.ndarray]:
        p, R, _, _ = armkin.fk(np.asarray(q6, dtype=np.float64), self._tcp_local)
        return p, R

    def jacobian(self, q6) -> np.ndarray:
        q = np.asarray(q6, dtype=np.float64)
        return (armkin.jacobian_training_convention(q, self._tcp_local)
                if self.use_training_jac else armkin.jacobian(q, self._tcp_local))

    # ---- policy-driven step ---------------------------------------------------------------
    def ik_step(self, q6: np.ndarray, action6: np.ndarray) -> np.ndarray:
        """One relative-pose IK step from a 6-vector policy action -> joint targets."""
        p, R = self.fk(q6)
        quat = quat_from_mat(R)
        cmd = torch.from_numpy(np.asarray(action6, dtype=np.float64)[:6]
                               * self.action_scale).float().unsqueeze(0)
        tp = torch.from_numpy(p).float().unsqueeze(0)
        tq = torch.from_numpy(quat).float().unsqueeze(0)
        self.ik.set_command(cmd, tp, tq)
        J = torch.from_numpy(self.jacobian(q6)).float().unsqueeze(0)
        jp = torch.from_numpy(np.asarray(q6, dtype=np.float64)).float().unsqueeze(0)
        des = self.ik.compute(tp, tq, J, jp)[0].numpy().astype(np.float64)
        return self.limiter(des)

    def servo_to(self, q6: np.ndarray, dxyz: np.ndarray) -> np.ndarray:
        """Position-only DLS step: the tool orientation is left free.

        A full pose solve with a zero rotation delta silently also demands "do not rotate",
        which leaves a dead zone wherever the target is reachable only with a tilted tool --
        measured as a stalled pick at a 6-7 cm residual in 4 of 4 runs at the basket's
        front-outboard corner.  Using the 3x6 linear block lets the wrist tilt to close it.
        """
        J = self.jacobian(q6)[0:3, :]
        dq = armkin.dls_step(J, np.asarray(dxyz, dtype=np.float64)[:3], lam=self.lam)
        return self.limiter(np.asarray(q6, dtype=np.float64) + dq)

    # ---- gripper -----------------------------------------------------------------------------
    def gripper(self, half_width: float) -> float:
        """Simulator half-width -> rate-limited real stroke.  The factor of two is applied
        here; everything downstream is already in real metres."""
        return self.grip_limiter(float(half_width) * self.stroke_per_width)

    def loaded(self, st: ArmState, commanded_stroke: float, tol: float = 2e-3) -> bool:
        """Are the fingers pressing on something right now?

        Width stopped short of the commanded stroke AND the motor is pulling current.  This
        is necessary but NOT sufficient -- see `HoldDetector` for why.
        """
        return (st.ok
                and st.gripper_stroke > commanded_stroke + tol
                and st.gripper_current > self.hold_current)

    def holding(self, st: ArmState, commanded_stroke: float, tol: float = 2e-3) -> bool:
        """Deprecated single-shot form; kept so existing callers do not silently change
        meaning.  Prefer `HoldDetector`, which adds the lift term that does the separating.
        """
        return self.loaded(st, commanded_stroke, tol)

    def command(self, q_target: np.ndarray, half_width: float) -> ArmCommand:
        return ArmCommand(q=np.asarray(q_target, dtype=np.float64),
                          gripper_stroke=self.gripper(half_width))

    def reset(self, q6: np.ndarray | None = None, stroke: float | None = None) -> None:
        self.limiter.reset(q6)
        self.grip_limiter.reset(stroke)

    def report(self) -> dict:
        return {"joints": self.limiter.report.summary(list(self.cfg["joint_names"])),
                "gripper": {"rate_hits": self.grip_limiter.rate_hits,
                            "limit_hits": self.grip_limiter.limit_hits},
                "training_jacobian": self.use_training_jac}


class HoldDetector:
    """Did the gripper actually pick the object up?

    Replaces the simulator's `held()` oracle, which reads the object's true pose and its
    velocity relative to the hand.  On hardware there are three deployable signals -- finger
    width, finger current, and how far the TCP has risen since the fingers first loaded --
    and the fit over 1,156 recorded attempts (298 held / 858 missed) says exactly how to
    combine them:

        width >= 3 mm AND effort >= 5   recovers 298/298 holds, with 33 false positives
        ... and every one of those 33 false positives had NO LIFT.

    So the lift term is not a refinement, it is the discriminator.  An earlier version of
    this class had only width and current and would have credited all 33.

    Two details that are easy to get wrong and were learned the expensive way:

    * A load is a JAMMED finger, not a finger passing through the band.  The width must be
      in band, the gripper must be COMMANDED SHUT, and the width must have stopped changing.
      Without the "stopped changing" test, every closing sweep reads as a load for the few
      steps it spends crossing the band.
    * Losing the load for one sample is not losing the object.  The reference debounces over
      three consecutive empty reads before forgetting the reference height; a first-miss
      reset makes the lift measurement restart from wherever the hand happens to be.
    """

    def __init__(self, cfg, arm: "ArmController") -> None:
        g = cfg["gripper"]
        h = cfg.get("hold", {}) or {}
        self.arm = arm
        # Bands, in REAL stroke metres (the sim's gw is already the real opening).
        self.gw_lo = float(h.get("gw_lo", 0.003))
        self.gw_hi = float(h.get("gw_hi", 0.0995))
        self.effort = float(h.get("effort", g.get("hold_current_a", 0.6)))
        self.lift = float(h.get("lift", 0.04))
        # Strong evidence: enough loaded steps and enough rise that no artefact explains it.
        self.strong_loaded_n = int(h.get("strong_loaded_n", 15))
        self.strong_lift = float(h.get("strong_lift", 0.12))
        self.still_eps = float(h.get("still_eps", 0.001))
        self.debounce = int(h.get("debounce", 3))
        self.reset()

    def reset(self) -> None:
        self.z_load: float | None = None
        self.loaded_n = 0
        self.sensed_lift = 0.0
        self._load_run = 0
        self._empty_run = 0
        self._gw_prev: float | None = None
        self.strong = False

    def update(self, st: ArmState, closed_cmd: bool, tcp_z: float) -> bool:
        """One control step.  -> True once the object is believed to be held."""
        if not st.ok:
            return False
        gw = float(st.gripper_stroke)
        in_band = self.gw_lo <= gw <= self.gw_hi
        strong_enough = st.gripper_current >= self.effort
        still = self._gw_prev is not None and abs(gw - self._gw_prev) < self.still_eps
        self._gw_prev = gw

        loaded = bool(closed_cmd and in_band and strong_enough and still)
        self._load_run = self._load_run + 1 if loaded else 0
        self._empty_run = 0 if (closed_cmd and in_band and strong_enough) else self._empty_run + 1

        if self._load_run >= 3:
            self.loaded_n += 1
            if self.z_load is None:
                self.z_load = tcp_z
            self.sensed_lift = max(self.sensed_lift, tcp_z - self.z_load)
        elif self._load_run == 0 and self._empty_run >= self.debounce:
            self.z_load = None

        if (self.loaded_n >= self.strong_loaded_n
                and self.sensed_lift >= self.strong_lift
                and gw > 0.008 and loaded):
            self.strong = True
            return True
        return self.z_load is not None and self.sensed_lift >= self.lift

    def report(self) -> dict:
        return {"loaded_n": self.loaded_n, "lift_m": round(self.sensed_lift, 4),
                "strong": self.strong}
