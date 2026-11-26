"""Mock and replay backends -- how the stack gets tested before anyone touches the robot.

The A2 Pro is shared equipment we do not get our hands on until check-in day.  If the only
way to exercise the control loop, the state machine, the safety supervisor and the mission
FSM is to have the robot, then all of that code arrives untested on the one day there is no
time to debug it.  So every driver has a stand-in here.

What the mock is honest about:
  * timing, ordering, and the shape of every message
  * the state machine, watchdogs and fault paths, including injected failures
  * joint targets actually reaching the bus, with rate and limit clamps applied

What it is NOT:
  * a gait simulator.  The base follows the *commanded* velocity through a first-order lag.
    It answers "does the stack behave", never "does the policy walk" -- that question
    belongs to Isaac Sim, which we still have.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from onboard.oio.a2_map import LegMap
from onboard.oio.iface import (
    ArmBus, ArmCommand, ArmState, CameraSource, Cloud, Frame, LegBus, LegCommand, LegState,
    LidarSource, RemoteState,
)

FAULTS = ("none", "lowstate_stall", "inverted", "estop", "joint_fault", "arm_fault")


class MockLegBus(LegBus):
    """A2 Pro stand-in: first-order joint tracking, kinematic base, injectable faults."""

    def __init__(self, cfg, fault: str = "none", fault_at_s: float = 8.0,
                 joint_tau_s: float = 0.03, seed: int = 0) -> None:
        if fault not in FAULTS:
            raise ValueError(f"unknown fault {fault!r}; choose from {FAULTS}")
        self.cfg = cfg
        self.map = LegMap(cfg)
        self.fault = fault
        self.fault_at = fault_at_s
        self.joint_tau = joint_tau_s
        self.rng = np.random.default_rng(seed)

        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self._q = self.map.q_default.copy()
        self._dq = np.zeros(12)
        self._cmd = LegCommand.zero_torque()
        self._tick = 0
        self._last = self._t0
        # Base pose in a fixed world frame -- what a perfect estimator would recover, kept
        # only so replay_run.py can score the estimator against something.
        self.base_xy = np.zeros(2)
        self.base_yaw = 0.0
        self.base_z = 0.43
        self._vel_cmd = np.zeros(3)          # (vx, vy, wz) the mission asked for
        self._vel = np.zeros(3)
        self._roll = 0.0
        self._pitch = 0.0
        self._running = False
        self._thr: threading.Thread | None = None
        self._remote = RemoteState(t=0.0, buttons={k: 0 for k in
                                                   ("start", "select", "A", "B", "L2", "R2")},
                                   ok=True)

    # ---- the mission tells us what it asked the legs to do -----------------------------
    def set_velocity_command(self, v: np.ndarray) -> None:
        """The policy's velocity command.  In the mock the base follows it directly; on
        hardware the gait is what turns it into motion."""
        with self._lock:
            self._vel_cmd = np.asarray(v, dtype=np.float64)[:3].copy()

    def start(self) -> None:
        self._running = True
        self._thr = threading.Thread(target=self._loop, name="mock_robot", daemon=True)
        self._thr.start()

    def stop(self) -> None:
        self._running = False
        if self._thr is not None:
            self._thr.join(timeout=1.0)

    def _loop(self) -> None:
        """Integrate at 200 Hz -- the real PD rate -- so joint lag is not an artefact of
        the control period."""
        dt = 0.005
        nxt = time.monotonic()
        while self._running:
            with self._lock:
                self._step(dt)
            nxt += dt
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.monotonic()

    def _step(self, dt: float) -> None:
        el = time.monotonic() - self._t0
        cmd = self._cmd

        # Joint tracking: only where kp > 0.  Under damping (kp = 0) the joints coast and
        # sag, which is what makes DAMP visibly different from HOLD in a replay log.
        driven = cmd.kp > 1e-6
        target = np.where(driven, cmd.q, self._q)
        alpha = dt / max(self.joint_tau, dt)
        newq = self._q + alpha * (target - self._q)
        if np.any(~driven):
            sag = np.where(~driven, -0.6 * dt * np.maximum(cmd.kd, 1e-3) / 8.0, 0.0)
            newq = newq + sag
        newq = self.map.clamp(newq)
        self._dq = (newq - self._q) / dt
        self._q = newq

        if self.fault == "joint_fault" and el > self.fault_at:
            self._q[4] = self.map.limits[4, 1]        # FL_thigh jams at its upper stop
            self._dq[4] = 0.0

        # Base: first-order response to the commanded velocity, and only while the legs are
        # actually driven -- a damped robot does not travel.
        gain = 1.0 if np.any(driven) else 0.0
        self._vel += (dt / 0.25) * (gain * self._vel_cmd - self._vel)
        self.base_yaw += self._vel[2] * dt
        c, s = np.cos(self.base_yaw), np.sin(self.base_yaw)
        self.base_xy += dt * np.array([c * self._vel[0] - s * self._vel[1],
                                       s * self._vel[0] + c * self._vel[1]])
        self.base_z = 0.43 if np.any(driven) else max(0.10, self.base_z - 0.4 * dt)

        if self.fault == "inverted" and el > self.fault_at:
            self._roll += dt * 2.5                    # roll over in ~0.6 s
            self._roll = min(self._roll, np.pi)
        self._tick += 1
        self._last = time.monotonic()

    # ---- LegBus -----------------------------------------------------------------------
    def read(self) -> LegState:
        with self._lock:
            el = time.monotonic() - self._t0
            if self.fault == "lowstate_stall" and el > self.fault_at:
                st = LegState.zeros()
                st.t = self._last          # frozen: age grows, watchdog must catch it
                return st
            r, p, y = self._roll, self._pitch, self.base_yaw
            quat = _quat_from_rpy(r, p, y)
            g = _projected_gravity(quat)
            noise = self.rng.normal(0.0, 0.002, 12)
            return LegState(
                t=time.monotonic(), tick=self._tick,
                q=self._q + noise, dq=self._dq + self.rng.normal(0.0, 0.02, 12),
                tau=np.clip(self._dq * 0.5, -self.map.effort, self.map.effort),
                quat=quat,
                gyro=np.array([0.0, 0.0, self._vel[2]]) + self.rng.normal(0.0, 0.01, 3),
                accel=np.array([0.0, 0.0, -9.81 * g[2]]),
                foot_force=np.full(4, 60.0 if self.base_z > 0.3 else 0.0),
                voltage=25.6, ok=True,
            )

    def write(self, cmd: LegCommand) -> None:
        with self._lock:
            self._cmd = cmd

    def damp(self) -> None:
        self.write(LegCommand.damping(float(self.cfg["safety"]["damp_kd"])))

    def remote(self) -> RemoteState:
        with self._lock:
            el = time.monotonic() - self._t0
            rc = RemoteState(t=time.monotonic(), buttons=dict(self._remote.buttons),
                             ok=True)
            if self.fault == "estop" and el > self.fault_at:
                rc.buttons["select"] = 1        # the abort button
            return rc

    def press(self, name: str, value: int = 1) -> None:
        """Drive a button from a test."""
        with self._lock:
            self._remote.buttons[name] = value

    def truth(self) -> dict:
        with self._lock:
            return {"xy": self.base_xy.copy(), "yaw": self.base_yaw, "z": self.base_z,
                    "vel": self._vel.copy(), "roll": self._roll}


class MockArmBus(ArmBus):
    def __init__(self, cfg, fault: str = "none", fault_at_s: float = 8.0) -> None:
        self.cfg = cfg
        self.fault = fault
        self.fault_at = fault_at_s
        self._t0 = time.monotonic()
        self.limits = np.asarray(cfg["joint_limits"], dtype=np.float64)
        self._q = np.asarray(cfg["home"], dtype=np.float64)
        self._target = self._q.copy()
        self._stroke = float(cfg["gripper"]["max_stroke_m"])
        self._stroke_target = self._stroke
        self._enabled = False
        self._last = time.monotonic()
        # Something is "in" the gripper below this opening: closing further draws current,
        # which is the signal that replaces the simulator's held() oracle.
        self.object_width = None

    def start(self) -> None:
        self._last = time.monotonic()

    def stop(self) -> None:
        self._enabled = False

    def enable(self) -> bool:
        self._enabled = True
        return True

    def hold(self) -> None:
        self._target = self._q.copy()
        self._stroke_target = self._stroke

    def _advance(self) -> None:
        now = time.monotonic()
        dt = min(max(now - self._last, 0.0), 0.1)
        self._last = now
        rate = float(self.cfg["limits"]["velocity_limit"])
        step = rate * dt
        self._q += np.clip(self._target - self._q, -step, step)
        self._q = np.clip(self._q, self.limits[:, 0], self.limits[:, 1])
        s_step = float(self.cfg["gripper"]["vel_guard_mps"]) * dt
        want = self._stroke_target
        if self.object_width is not None:
            want = max(want, self.object_width)     # fingers stop on the object
        self._stroke += float(np.clip(want - self._stroke, -s_step, s_step))

    def read(self) -> ArmState:
        self._advance()
        el = time.monotonic() - self._t0
        if self.fault == "arm_fault" and el > self.fault_at:
            return ArmState(t=time.monotonic(), q=self._q.copy(), dq=np.zeros(6),
                            current=np.zeros(6), gripper_stroke=self._stroke,
                            gripper_current=0.0, enabled=False,
                            fault="CAN timeout", ok=False)
        squeezing = (self.object_width is not None
                     and self._stroke <= self.object_width + 1e-4
                     and self._stroke_target < self._stroke)
        return ArmState(t=time.monotonic(), q=self._q.copy(),
                        dq=np.zeros(6), current=np.full(6, 0.2),
                        gripper_stroke=self._stroke,
                        gripper_current=1.2 if squeezing else 0.1,
                        enabled=self._enabled, ok=True)

    def write(self, cmd: ArmCommand) -> None:
        if not self._enabled:
            return
        self._target = np.clip(np.asarray(cmd.q, dtype=np.float64),
                               self.limits[:, 0], self.limits[:, 1])
        self._stroke_target = float(np.clip(cmd.gripper_stroke, 0.0,
                                            self.cfg["gripper"]["max_stroke_m"]))


class ReplayCamera(CameraSource):
    """Serve recorded frames.

    Accepts a directory of .npz files each holding `rgb` and optionally `depth`, which is
    the format tools/record_bag.py writes and the format the existing offline detector
    harnesses already read.  With no directory it emits a synthetic scene so the perception
    pipeline still has something with the right dtype and geometry to chew on.
    """

    def __init__(self, cfg, path: str | Path | None = None, fps: float = 5.0) -> None:
        self.cfg = cfg
        vp = cfg["camera"]["virtual_pinhole"]
        self.w, self.h = int(vp["width"]), int(vp["height"])
        self.K = np.array([[vp["fx"], 0.0, vp["cx"]],
                           [0.0, vp["fy"], vp["cy"]],
                           [0.0, 0.0, 1.0]])
        self.files = sorted(Path(path).glob("*.npz")) if path else []
        self.period = 1.0 / max(fps, 1e-6)
        self._i = 0
        self._next = 0.0
        self.tick = 0

    def start(self) -> None:
        self._next = time.monotonic()

    def stop(self) -> None: ...

    def grab(self) -> Frame | None:
        now = time.monotonic()
        if now < self._next:
            return None
        self._next = max(now, self._next + self.period)
        if self.files:
            d = np.load(self.files[self._i % len(self.files)], allow_pickle=True)
            rgb = d["rgb"]
            depth = d["depth"] if "depth" in d.files else None
            # Recorded frames carry their own intrinsics; using the configured virtual
            # pinhole instead would unproject them with the wrong focal length.
            if "K" in d.files:
                self.K = np.asarray(d["K"], np.float64)
            self._i += 1
        else:
            rgb, depth = self._synthetic()
        return Frame(t=now, tick=self.tick, rgb=rgb, depth=depth, K=self.K, ok=True)

    def _synthetic(self):
        """A ground plane, two red boundary strips and a bright blob.  Not a scene the
        detector should score well on -- just valid geometry and dtypes so the pipeline,
        the mailbox and the timing are exercised end to end."""
        rgb = np.zeros((self.h, self.w, 3), np.uint8)
        rgb[: self.h // 2] = (150, 170, 200)                 # sky
        rgb[self.h // 2:] = (90, 110, 70)                    # grass
        for x in (60, self.w - 60):
            rgb[self.h // 2: self.h - 30, x - 4: x + 4] = (200, 40, 40)
        cx, cy = self.w // 2, int(self.h * 0.75)
        rgb[cy - 8: cy + 8, cx - 8: cx + 8] = (210, 210, 215)
        depth = np.full((self.h, self.w), 3.0, np.float32)
        ys = np.arange(self.h, dtype=np.float32)[:, None]
        depth[self.h // 2:] = np.clip(
            60.0 / np.maximum(ys[self.h // 2:] - self.h / 2 + 1.0, 1.0), 0.3, 10.0)
        return rgb, depth


class ReplayLidar(LidarSource):
    """A synthetic rail pair at +-1.0 m so railnav can be exercised without hardware."""

    def __init__(self, half_width: float = 1.0, n: int = 400, hz: float = 10.0,
                 seed: int = 0) -> None:
        self.half = half_width
        self.n = n
        self.period = 1.0 / hz
        self._next = 0.0
        self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.lateral = 0.0          # tests move the robot off centre through this
        self.heading = 0.0

    def start(self) -> None:
        self._next = time.monotonic()

    def stop(self) -> None: ...

    def grab(self) -> Cloud | None:
        now = time.monotonic()
        if now < self._next:
            return None
        self._next = max(now, self._next + self.period)
        x = self.rng.uniform(0.3, 3.0, self.n)
        pts = []
        for side in (+1.0, -1.0):
            y = side * self.half - self.lateral + x * np.tan(self.heading)
            z = self.rng.uniform(0.02, 0.40, self.n)          # 40 cm strip
            pts.append(np.stack([x, y + self.rng.normal(0, 0.01, self.n), z], 1))
        ground = np.stack([self.rng.uniform(0.3, 3.0, self.n),
                           self.rng.uniform(-1.0, 1.0, self.n),
                           self.rng.normal(0.0, 0.01, self.n)], 1)
        return Cloud(t=now, tick=self.tick, points=np.concatenate(pts + [ground]), ok=True)


# ---- small maths, kept local so the mock does not depend on the estimator ---------------
def _quat_from_rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


def _projected_gravity(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([2.0 * (-z * x + w * y),
                     -2.0 * (z * y + w * x),
                     1.0 - 2.0 * (w * w + z * z)])
