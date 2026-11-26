"""A2 Pro low-level bridge over CycloneDDS (unitree_sdk2py).

Three things here are deliberate departures from the reference implementations:

1. **The command publisher runs on its own thread at 500 Hz.**  unitree_rl_gym's Python
   deploy publishes one LowCmd per 50 Hz policy tick; its C++ sibling runs a 2 ms
   `low_cmd_write` thread over a double buffer.  The C++ one is right.  A perception hiccup
   or a GC pause on the N150 must not open a hole in the motor stream, and damping must not
   depend on a single packet arriving.

2. **Damping is a state, not a call.**  `create_damping_cmd` in the reference is published
   exactly once on exit.  Here `damp()` latches the command and the writer thread keeps
   sending it until something else latches over it.

3. **LowState freshness is checked continuously.**  Both references check connectivity once
   at startup and never again.  A link that dies at minute 12 of a 30-minute run then looks
   like a robot standing perfectly still while the policy integrates stale observations.

The SDK is imported lazily inside `start()` so this module -- and everything that imports
it -- loads on a machine with no DDS, no CAN and no robot.  That is what lets the whole
stack be exercised against oio/replay.py before we ever touch the hardware.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from onboard.oio.a2_map import LegMap
from onboard.oio.iface import LegBus, LegCommand, LegState, RemoteState
from onboard.oio.remote import decode as decode_remote

# Sentinels from the `go` IDL meaning "ignore this setpoint" on a motor we are not driving.
POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0
MOTOR_MODE_SERVO = 0x0A
MOTOR_MODE_WEAK = 0x01


def _as_float(v) -> float:
    """Battery voltage is a scalar on some IDLs and a one-element sequence on others.
    Telemetry only, so a bad read must degrade to 0.0 rather than break the state callback."""
    try:
        if isinstance(v, (list, tuple)) or hasattr(v, "__len__"):
            return float(v[0]) if len(v) else 0.0
        return float(v)
    except Exception:  # noqa: BLE001
        return 0.0


class A2DdsBus(LegBus):
    def __init__(self, cfg, net_interface: str | None = None) -> None:
        self.cfg = cfg
        self.map = LegMap(cfg)
        t = cfg["transport"]
        self.msg_type = str(t["msg_type"]).lower()
        if self.msg_type not in ("go", "hg"):
            raise ValueError(f"transport.msg_type must be 'go' or 'hg', got {self.msg_type!r}")
        self.domain_id = int(t.get("domain_id", 0))
        self.nic = net_interface or t.get("net_interface")
        if not self.nic:
            raise ValueError(
                "no network interface: pass --nic or set transport.net_interface. "
                "The robot is on 192.168.123.161; this must be the NIC facing it.")
        self.topic_cmd = t.get("lowcmd_topic", "rt/lowcmd")
        self.topic_state = t.get("lowstate_topic", "rt/lowstate")
        self.republish_dt = 1.0 / float(t.get("cmd_republish_hz", 500.0))
        self.state_timeout = float(t.get("state_timeout_s", 0.10))
        self.n_motor = int(t.get("num_motor", 12))

        self._lock = threading.Lock()
        self._state = LegState.zeros()
        self._remote = RemoteState(t=0.0, ok=False)
        self._cmd = LegCommand.zero_torque()
        self._cmd_dirty = True
        self._writer: threading.Thread | None = None
        self._running = False
        self._pub = None
        self._sub = None
        self._low_cmd = None
        self._crc = None
        self._tx = 0
        self._rx = 0

    # ---- lifecycle ---------------------------------------------------------------------
    def start(self) -> None:
        from unitree_sdk2py.core.channel import (  # noqa: PLC0415  (lazy: no SDK on dev box)
            ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
        )
        from unitree_sdk2py.utils.crc import CRC  # noqa: PLC0415

        if self.msg_type == "go":
            from unitree_sdk2py.idl.default import (  # noqa: PLC0415
                unitree_go_msg_dds__LowCmd_ as LowCmdDefault,
            )
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import (  # noqa: PLC0415
                LowCmd_ as LowCmdT, LowState_ as LowStateT,
            )
        else:
            from unitree_sdk2py.idl.default import (  # noqa: PLC0415
                unitree_hg_msg_dds__LowCmd_ as LowCmdDefault,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (  # noqa: PLC0415
                LowCmd_ as LowCmdT, LowState_ as LowStateT,
            )

        ChannelFactoryInitialize(self.domain_id, self.nic)
        self._crc = CRC()

        self._low_cmd = LowCmdDefault()
        self._init_cmd()

        self._pub = ChannelPublisher(self.topic_cmd, LowCmdT)
        self._pub.Init()
        self._sub = ChannelSubscriber(self.topic_state, LowStateT)
        self._sub.Init(self._on_state, 10)

        self._running = True
        self._writer = threading.Thread(target=self._write_loop, name="lowcmd_write",
                                        daemon=True)
        self._writer.start()

    def stop(self) -> None:
        # Leave the robot damped, and keep sending it long enough that the last packet is
        # not the thing standing between the robot and the floor.
        if self._running:
            self.damp()
            time.sleep(0.2)
        self._running = False
        if self._writer is not None:
            self._writer.join(timeout=1.0)
        for ch in (self._sub, self._pub):
            try:
                if ch is not None:
                    ch.Close()
            except Exception:  # noqa: BLE001  shutdown must not raise
                pass

    # ---- LowCmd ------------------------------------------------------------------------
    def _init_cmd(self) -> None:
        """Magic header + per-motor mode.  Without these the firmware ignores the message
        and there is no error anywhere -- the robot simply does not move."""
        c = self._low_cmd
        if self.msg_type == "go":
            c.head[0] = 0xFE
            c.head[1] = 0xEF
            c.level_flag = 0xFF
            c.gpio = 0
        weak = set(self.cfg["transport"].get("weak_motor", []) or [])
        for i in range(len(c.motor_cmd)):
            m = c.motor_cmd[i]
            m.mode = MOTOR_MODE_WEAK if i in weak else MOTOR_MODE_SERVO
            # Every motor we are NOT driving is parked on the sentinels, so an untouched
            # index cannot be interpreted as "hold position 0" by the driver.
            m.q, m.dq, m.kp, m.kd, m.tau = POS_STOP_F, VEL_STOP_F, 0.0, 0.0, 0.0

    def _apply_cmd(self, cmd: LegCommand) -> None:
        c = self._low_cmd
        q = self.map.q_to_motor(cmd.q)
        kp = self.map.scalar_to_motor(cmd.kp)
        kd = self.map.scalar_to_motor(cmd.kd)
        dq = self.map.scalar_to_motor(cmd.dq)
        tau = self.map.scalar_to_motor(cmd.tau)
        for i in self.map.idx.tolist():
            m = c.motor_cmd[i]
            m.q = float(q[i])
            m.dq = float(dq[i])
            m.kp = float(kp[i])
            m.kd = float(kd[i])
            m.tau = float(tau[i])

    def _write_loop(self) -> None:
        """Drift-free cadence: advance an absolute deadline instead of sleeping for a
        duration, so publish jitter does not accumulate into a slow stream."""
        next_t = time.monotonic()
        while self._running:
            with self._lock:
                cmd = self._cmd
                dirty = self._cmd_dirty
                self._cmd_dirty = False
            if dirty:
                self._apply_cmd(cmd)
            try:
                self._low_cmd.crc = self._crc.Crc(self._low_cmd)
                self._pub.Write(self._low_cmd)
                self._tx += 1
            except Exception:  # noqa: BLE001  a dropped write must not kill the thread
                pass
            next_t += self.republish_dt
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()   # we fell behind; resync rather than spin

    # ---- LowState ----------------------------------------------------------------------
    def _on_state(self, msg) -> None:
        now = time.monotonic()
        n = self.n_motor
        q = np.empty(n)
        dq = np.empty(n)
        tau = np.empty(n)
        temp = np.empty(n)
        for i in range(n):
            ms = msg.motor_state[i]
            q[i] = ms.q
            dq[i] = ms.dq
            tau[i] = ms.tau_est
            temp[i] = getattr(ms, "temperature", 0.0)

        imu = msg.imu_state
        st = LegState(
            t=now,
            tick=int(getattr(msg, "tick", 0)),
            q=self.map.q_from_motor(q),
            dq=self.map.rate_from_motor(dq),
            tau=self.map.rate_from_motor(tau),
            quat=np.asarray(imu.quaternion, dtype=np.float64),      # (w, x, y, z)
            gyro=np.asarray(imu.gyroscope, dtype=np.float64),
            accel=np.asarray(imu.accelerometer, dtype=np.float64),
            foot_force=np.asarray(getattr(msg, "foot_force", [0, 0, 0, 0])[:4], dtype=np.float64),
            temperature=self.map.rate_from_motor(temp) * self.map.sign,   # unsigned
            voltage=_as_float(getattr(msg, "power_v", 0.0)),
            ok=True,
        )
        rc = decode_remote(getattr(msg, "wireless_remote", None))
        with self._lock:
            self._state = st
            self._rx += 1
            if rc.ok:
                self._remote = rc

    # ---- LegBus ------------------------------------------------------------------------
    def read(self) -> LegState:
        with self._lock:
            st = self._state
        if st.ok and (time.monotonic() - st.t) > self.state_timeout:
            # Hand back the last values but flag them, so a consumer that ignores `ok`
            # gets plausible numbers rather than zeros -- and a consumer that checks it
            # (the supervisor does) sees the link is gone.
            st = LegState(**{**st.__dict__, "ok": False})
        return st

    def write(self, cmd: LegCommand) -> None:
        with self._lock:
            self._cmd = cmd
            self._cmd_dirty = True

    def damp(self) -> None:
        self.write(LegCommand.damping(float(self.cfg["safety"]["damp_kd"])))

    def remote(self) -> RemoteState:
        with self._lock:
            return self._remote

    def stats(self) -> dict:
        with self._lock:
            return {"tx": self._tx, "rx": self._rx, "state_age_s": time.monotonic() - self._state.t}
