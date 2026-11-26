"""The hardware boundary.

Everything above this file -- the policies, the estimator, the mission FSM, the safety
supervisor -- talks only to these types.  Everything below is a transport: DDS to the legs,
CAN to the arm, an SDK to the camera, or a mock that replays a recording.

That split is not tidiness.  The robot is a shared unit we do not get to touch until
check-in day, so the only way to have working software by then is for the entire stack
above this line to run to completion on a desktop against oio/replay.py.  If a DDS import
or a /dev/ node leaks upward, that stops being possible.

Conventions, fixed here once:
  * angles in radians, lengths in metres, time in seconds from time.monotonic()
  * leg vectors are ALWAYS in policy order: FR, FL, RR, RL x (hip, thigh, calf)
  * quaternions are (w, x, y, z), matching IsaacLab and the Unitree IMU
  * a driver never raises for a stale link; it reports `ok=False` and lets the supervisor
    decide.  A transport exception during a 30-minute run is a lost run.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

N_LEG = 12
N_ARM = 6


@dataclass
class LegState:
    """One LowState, already remapped into policy order and policy units."""
    t: float                                  # time.monotonic() when received
    tick: int                                 # robot-side counter; identity for frame sync
    q: np.ndarray                             # (12,) joint position
    dq: np.ndarray                            # (12,) joint velocity
    tau: np.ndarray                           # (12,) estimated joint torque
    quat: np.ndarray                          # (4,) IMU orientation, (w, x, y, z)
    gyro: np.ndarray                          # (3,) body angular velocity
    accel: np.ndarray                         # (3,) body linear acceleration
    foot_force: np.ndarray                    # (4,) FR, FL, RR, RL contact force
    temperature: np.ndarray | None = None     # (12,) motor temp, if the SDK reports it
    voltage: float = 0.0
    ok: bool = True                           # False => stale/absent, do not trust the rest

    @staticmethod
    def zeros() -> "LegState":
        return LegState(
            t=0.0, tick=0,
            q=np.zeros(N_LEG), dq=np.zeros(N_LEG), tau=np.zeros(N_LEG),
            quat=np.array([1.0, 0.0, 0.0, 0.0]), gyro=np.zeros(3), accel=np.zeros(3),
            foot_force=np.zeros(4), ok=False,
        )


@dataclass
class LegCommand:
    """One LowCmd.  Position control with driver-side PD; we never send torque.

    kp/kd are per-joint so damping and holding are the same message with different gains
    rather than a separate code path -- one less thing to get wrong when the robot is
    already falling.
    """
    q: np.ndarray                             # (12,) target position
    kp: np.ndarray                            # (12,)
    kd: np.ndarray                            # (12,)
    dq: np.ndarray = field(default_factory=lambda: np.zeros(N_LEG))
    tau: np.ndarray = field(default_factory=lambda: np.zeros(N_LEG))

    @staticmethod
    def damping(kd: float) -> "LegCommand":
        """Zero position gain, pure velocity damping: the robot sinks instead of dropping.
        This is the ONLY safe terminal state, and it is a state, not a packet -- the
        publisher must keep sending it."""
        return LegCommand(q=np.zeros(N_LEG), kp=np.zeros(N_LEG), kd=np.full(N_LEG, kd))

    @staticmethod
    def zero_torque() -> "LegCommand":
        return LegCommand(q=np.zeros(N_LEG), kp=np.zeros(N_LEG), kd=np.zeros(N_LEG))


@dataclass
class ArmState:
    t: float
    q: np.ndarray                             # (6,) rad
    dq: np.ndarray                            # (6,) rad/s
    current: np.ndarray                       # (6,) A -- the grasp-success signal
    gripper_stroke: float                     # m, REAL opening (not the sim half-width)
    gripper_current: float                    # A
    enabled: bool = False
    fault: str | None = None
    ok: bool = True

    @staticmethod
    def zeros() -> "ArmState":
        return ArmState(t=0.0, q=np.zeros(N_ARM), dq=np.zeros(N_ARM),
                        current=np.zeros(N_ARM), gripper_stroke=0.0, gripper_current=0.0,
                        ok=False)


@dataclass
class ArmCommand:
    q: np.ndarray                             # (6,) target position
    gripper_stroke: float                     # m, REAL opening


@dataclass
class Frame:
    """One camera frame, already rectified into the virtual pinhole.

    `tick` is the LowState tick the frame was captured against.  Without it the detector
    unprojects with whatever pose happened to be current when the worker got around to the
    frame, and at 0.7 m/s a 100 ms slip is 7 cm of range error applied to every detection.
    """
    t: float
    tick: int
    rgb: np.ndarray                           # (H, W, 3) uint8
    depth: np.ndarray | None                  # (H, W) float32 metres, z-depth; None if absent
    K: np.ndarray                             # (3, 3)
    ok: bool = True


@dataclass
class Cloud:
    t: float
    tick: int
    points: np.ndarray                        # (N, 3) in the LiDAR's own frame, x fwd/y left/z up
    ok: bool = True


@dataclass
class RemoteState:
    """Buttons and sticks from the wireless remote.

    Used ONLY before the whistle -- for the operator to arm, stand and disarm during the
    5-minute preparation window.  The rules forbid teleoperation once the run starts, so
    the mission loop never reads the sticks; startgate.py latches the run and from then on
    the remote is an emergency input and nothing else.
    """
    t: float
    buttons: dict[str, int] = field(default_factory=dict)
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    ok: bool = False

    def pressed(self, name: str) -> bool:
        return bool(self.buttons.get(name, 0))


class LegBus(abc.ABC):
    """Legs.  Implementations: oio/a2_dds.py (real), oio/replay.py (mock/recorded)."""

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def read(self) -> LegState:
        """Latest state.  Never blocks; sets ok=False rather than waiting."""

    @abc.abstractmethod
    def write(self, cmd: LegCommand) -> None:
        """Latch a command.  The transport is responsible for republishing it at its own
        rate so the motor watchdog stays fed between policy ticks."""

    @abc.abstractmethod
    def remote(self) -> RemoteState: ...

    def wait_for_state(self, timeout_s: float = 5.0) -> bool:
        """Block until the first valid state arrives.  Default polls read()."""
        import time
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.read().ok:
                return True
            time.sleep(0.01)
        return False


class ArmBus(abc.ABC):
    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def read(self) -> ArmState: ...

    @abc.abstractmethod
    def write(self, cmd: ArmCommand) -> None: ...

    @abc.abstractmethod
    def enable(self) -> bool: ...

    @abc.abstractmethod
    def hold(self) -> None:
        """Freeze at the current measured pose.  What we do on a fault, and what the arm
        does for the whole of the walking phase."""


class CameraSource(abc.ABC):
    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def grab(self) -> Frame | None:
        """Newest frame, or None if none is ready.  Never blocks the caller."""


class LidarSource(abc.ABC):
    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def grab(self) -> Cloud | None: ...


class NullLidar(LidarSource):
    """The LiDAR may simply not be reachable through the SDK.  That is a supported outcome,
    not an error: T1 navigation falls back to camera belt detection, which is the path that
    was actually validated in sim."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def grab(self) -> Cloud | None:
        return None
