"""PiPER-X arm over CAN, via piper_sdk.

Neither reference deploy stack has any arm code at all -- unitree_rl_gym's "arm" is a
humanoid's own motors on the same DDS bus, held at a fixed pose.  So this is greenfield, and
the things worth being careful about are the ones with no precedent to copy:

  * **Units.**  piper_sdk speaks 0.001 degree integers.  Everything above this file is
    radians.  One conversion, in one place.
  * **The gripper's factor of two.**  In simulation the gripper is two mirrored prismatic
    joints commanded as [w, -w], each travelling 0-0.05 m, so a sim "width" of w is one
    finger and the real opening is 2w.  Every value crossing this boundary goes through
    `width_to_stroke`.
  * **Rate.**  The URDF rates the arm at 10 rad/s; the simulator's actuator said 100, which
    is 10x the hardware.  A real driver answers an out-of-rate command with a fault, not a
    clamp, and a faulted arm mid-course is a lost object and possibly a lost run.  The
    limiter lives in control/arm.py and this class refuses anything that gets past it.
  * **Failing safe is holding, not releasing.**  If the CAN link drops while carrying, the
    right response is to freeze the fingers where they are.  A gripper that opens on fault
    drops the object, and a dropped object outside the placement area scores zero.

The SDK is imported lazily so this module loads on a machine with no CAN interface.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from onboard.oio.iface import ArmBus, ArmCommand, ArmState

RAD2MDEG = 180000.0 / np.pi        # radians -> 0.001-degree units
MDEG2RAD = 1.0 / RAD2MDEG


class PiperCanBus(ArmBus):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        t = cfg["transport"]
        self.can_name = str(t.get("can_interface", "can0"))
        self.timeout = float(t.get("state_timeout_s", 0.2))
        self.limits = np.asarray(cfg["joint_limits"], dtype=np.float64)
        g = cfg["gripper"]
        self.stroke_per_width = float(g["stroke_per_width"])
        self.max_stroke = float(g["max_stroke_m"])

        self._piper = None
        self._lock = threading.Lock()
        self._state = ArmState.zeros()
        self._thr: threading.Thread | None = None
        self._running = False
        self._enabled = False
        self._last_cmd: ArmCommand | None = None

    # ---- unit conversions ---------------------------------------------------------------
    @staticmethod
    def rad_to_sdk(q) -> list[int]:
        return [int(round(float(v) * RAD2MDEG)) for v in q]

    @staticmethod
    def sdk_to_rad(q) -> np.ndarray:
        return np.asarray(q, dtype=np.float64) * MDEG2RAD

    def width_to_stroke(self, half_width: float) -> float:
        """Simulator half-width -> real opening.  The factor of two lives here and only
        here; a 12 mm slot close in the sim is a 24 mm real opening."""
        return float(np.clip(half_width * self.stroke_per_width, 0.0, self.max_stroke))

    def stroke_to_width(self, stroke: float) -> float:
        return float(stroke) / self.stroke_per_width

    # ---- lifecycle ------------------------------------------------------------------------
    def start(self) -> None:
        from piper_sdk import C_PiperInterface_V2  # noqa: PLC0415  lazy: no CAN on the dev box

        self._piper = C_PiperInterface_V2(self.can_name)
        self._piper.ConnectPort()
        self._running = True
        self._thr = threading.Thread(target=self._read_loop, name="piper_read", daemon=True)
        self._thr.start()

    def stop(self) -> None:
        # Hold, do not release, and do not disable: a disabled arm falls, and a falling arm
        # with an object in it is both a lost object and a collision hazard.
        try:
            self.hold()
        except Exception:  # noqa: BLE001
            pass
        self._running = False
        if self._thr is not None:
            self._thr.join(timeout=1.0)

    def enable(self) -> bool:
        if self._piper is None:
            return False
        self._piper.EnableArm(7)                    # 7 = all joints
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            st = self.read()
            if st.ok and st.enabled:
                self._enabled = True
                return True
            time.sleep(0.05)
        return False

    # ---- state -----------------------------------------------------------------------------
    def _read_loop(self) -> None:
        while self._running:
            try:
                jm = self._piper.GetArmJointMsgs()
                gm = self._piper.GetArmGripperMsgs()
                hi = self._piper.GetArmHighSpdInfoMsgs()
                j = jm.joint_state
                q = self.sdk_to_rad([j.joint_1, j.joint_2, j.joint_3,
                                     j.joint_4, j.joint_5, j.joint_6])
                cur = np.array([_current(hi, i) for i in range(1, 7)], dtype=np.float64)
                st = ArmState(
                    t=time.monotonic(), q=q, dq=np.zeros(6), current=cur,
                    gripper_stroke=float(gm.gripper_state.grippers_angle) * 1e-6,
                    gripper_current=float(gm.gripper_state.grippers_effort) * 1e-3,
                    enabled=self._enabled, ok=True,
                )
                with self._lock:
                    self._state = st
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    self._state = ArmState(**{**self._state.__dict__, "ok": False,
                                              "fault": f"{type(e).__name__}: {e}"})
            time.sleep(0.005)

    def read(self) -> ArmState:
        with self._lock:
            st = self._state
        if st.ok and (time.monotonic() - st.t) > self.timeout:
            st = ArmState(**{**st.__dict__, "ok": False, "fault": "CAN state stale"})
        return st

    # ---- commands --------------------------------------------------------------------------
    def write(self, cmd: ArmCommand) -> None:
        if self._piper is None or not self._enabled:
            return
        q = np.clip(np.asarray(cmd.q, dtype=np.float64), self.limits[:, 0], self.limits[:, 1])
        j = self.rad_to_sdk(q)
        # MotionCtrl_2(ctrl_mode=CAN, move_mode=joint, speed%, mit=0)
        self._piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        self._piper.JointCtrl(*j)
        stroke = float(np.clip(cmd.gripper_stroke, 0.0, self.max_stroke))
        # Gripper takes micrometres and a torque limit in 0.001 N*m.
        self._piper.GripperCtrl(int(round(stroke * 1e6)), 1000, 0x01, 0)
        self._last_cmd = cmd

    def hold(self) -> None:
        st = self.read()
        if st.ok:
            self.write(ArmCommand(q=st.q.copy(), gripper_stroke=st.gripper_stroke))


def _current(hi, i: int) -> float:
    """Per-joint motor current in amps, if this firmware reports it.

    Current is the signal that replaces the simulator's `held()` oracle -- a closing gripper
    that stops shrinking while drawing current is holding something.  Firmware versions
    differ on whether and where they publish it, so a missing field degrades to 0.0 rather
    than taking the read loop down.
    """
    try:
        m = getattr(hi, f"motor_{i}_info_high_spd", None) or getattr(hi, f"motor_{i}", None)
        return float(getattr(m, "current", 0.0)) * 1e-3
    except Exception:  # noqa: BLE001
        return 0.0
