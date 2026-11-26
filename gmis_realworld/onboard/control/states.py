"""The leg control state machine.

    BOOT -> ZERO_TORQUE -> RAMP -> STAND -> RUN -> (DAMP | FAULT)
                                      ^        |
                                      +--RECOVER

Shaped by the reference stacks and by three rules that cost points:

  * "Any robot movement before the whistle shall invalidate the current attempt. Three
    accumulated false starts shall result in disqualification."  So STAND is a real state
    the robot can sit in indefinitely, and nothing may command velocity until the start
    gate latches.
  * "The robot's autonomous return to an operational state shall not be counted as a
    reset."  A fall that we recover from ourselves costs 0; a fall a human fixes costs 10.
    That makes RECOVER a scoring feature, which is why it exists before anything in T2.
  * "An E-Stop activation shall count as a reset."  FAULT is therefore terminal until a
    human clears it, and we would rather sit in DAMP (recoverable) than latch FAULT.

DAMP vs FAULT: DAMP is a posture we can leave; FAULT is latched. Both stream commands
continuously -- unitree_rl_gym publishes its damping command exactly once on exit, so a
single dropped DDS packet leaves the robot holding its last position target instead.
"""
from __future__ import annotations

import enum
import time

import numpy as np

from onboard.control.legs import projected_gravity
from onboard.control.limits import TargetLimiter
from onboard.oio.estop import EStop, Fault, Severity
from onboard.oio.iface import LegBus, LegCommand, LegState


class State(enum.Enum):
    BOOT = "boot"
    ZERO_TORQUE = "zero_torque"
    RAMP = "ramp"
    STAND = "stand"
    RUN = "run"
    RECOVER = "recover"
    DAMP = "damp"
    FAULT = "fault"


class LegSupervisor:
    """Owns the bus.  Nothing else writes a LegCommand.

    The mission asks for a velocity via `set_command()` and calls `step()` at the control
    rate; whether that velocity reaches the legs is this class's decision.
    """

    def __init__(self, bus: LegBus, policy, a2_cfg, estop: EStop | None = None) -> None:
        self.bus = bus
        self.policy = policy
        self.cfg = a2_cfg
        self.estop = estop or EStop()
        self.map = policy.map

        s = a2_cfg["safety"]
        self.grav_z_abort = float(s["grav_z_abort"])
        self.max_roll = float(s["max_roll_rad"])
        self.max_pitch = float(s["max_pitch_rad"])
        self.fall_height = float(s["fall_height_m"])
        self.fall_hold = float(s["fall_hold_s"])
        self.damp_kd = float(s["damp_kd"])
        self.damp_min_s = float(s["damp_min_s"])

        c = a2_cfg["control"]
        self.kp = float(c["kp"])
        self.kd = float(c["kd"])
        self.dt = 1.0 / float(c["rate_hz"])

        lim = a2_cfg["limits"]
        self.stand_ramp_s = float(lim["stand_ramp_s"])
        self.max_delta = self._resolve_max_delta(lim)
        self.limiter = TargetLimiter(
            self.map.limits[:, 0], self.map.limits[:, 1],
            max_delta=self.max_delta,
            soft=float(lim["soft_limit_factor"]), names=self.map.names)

        self.state = State.BOOT
        self.t_state = time.monotonic()
        self.cmd = np.zeros(3)
        self._ramp_from: np.ndarray | None = None
        self._bad_since: float | None = None
        self._recover_tries = 0
        self.max_recover_tries = 3
        self.transitions: list[tuple[float, str, str, str]] = []
        self.allow_run = False          # start gate; see bringup/startgate.py

        # Controller abort.  The rules require an E-Stop and count its use as a reset, so
        # the human watching from the observation zone needs a way to stop the robot that
        # does not involve walking onto the course.  This is that path, and it is the ONLY
        # thing the remote is read for once the run has started.
        self.abort_buttons = tuple(s.get("abort_buttons", ["select"]))
        # A rate limiter that is active every tick is not protecting the robot, it is
        # replacing the policy.  Track the streak so that shows up as a fault rather than
        # as an unexplained limp.
        self._rate_streak = 0
        self.rate_streak_limit = int(s.get("rate_streak_limit", 50))   # 1 s at 50 Hz

    def _resolve_max_delta(self, lim) -> np.ndarray:
        """Per-joint target-rate cap.

        "auto" ties the cap to each joint's own velocity rating rather than one number for
        all twelve, which matters because the calf is rated 14.67 rad/s and the hip/thigh 22:
        a single cap is either loose on the calf or tight on the hips.
        """
        v = lim["max_target_delta_rad"]
        if isinstance(v, str) and v.lower() == "auto":
            margin = float(lim.get("max_target_delta_margin", 1.5))
            return self.map.vel_limit * self.dt * margin
        return np.full(12, float(v))

    # ---- transitions --------------------------------------------------------------------
    def _to(self, state: State, why: str = "") -> None:
        if state is self.state:
            return
        self.transitions.append((time.monotonic(), self.state.value, state.value, why))
        self.state = state
        self.t_state = time.monotonic()
        # Every transition re-anchors the rate limiter.  Without this the first tick after
        # a mode change is rate-clamped against a target from the previous regime, which
        # shows up as an unexplained hitch exactly when the robot is least stable.
        st = self.bus.read()
        self.limiter.reset(st.q if st.ok else None)
        if state in (State.RUN, State.RECOVER):
            self.policy.reset()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.t_state

    # ---- external requests ---------------------------------------------------------------
    def set_command(self, v) -> None:
        m = self.cfg["command"]
        v = np.asarray(v, float)[:3]
        self.cmd = np.array([
            np.clip(v[0], -float(m["max_vx"]), float(m["max_vx"])),
            np.clip(v[1], -float(m["max_vy"]), float(m["max_vy"])),
            np.clip(v[2], -float(m["max_wz"]), float(m["max_wz"])),
        ])

    def request(self, state: State, why: str = "external") -> None:
        """Ask for a transition.  The safe states are sticky on purpose.

        DAMP and FAULT are where the robot ends up when something went wrong, and the one
        way to make them useless is to let ordinary mission logic walk back out of them.
        The mission has plenty of reasons to ask for STAND -- budget expired, run finished,
        waiting for a judge -- and if any of those could lift a damping decision then a
        robot that damped because it was tipping would stand back up into the same fall.

        Leaving DAMP is therefore an explicit re-arm (-> RAMP), which is a deliberate act
        with a ramp behind it.  Leaving FAULT additionally requires estop.clear(), which is
        a human action during an assisted reset -- exactly the accounting the rules use.
        """
        if self.state is State.FAULT:
            return
        if self.state is State.DAMP and state not in (State.RAMP, State.FAULT, State.DAMP):
            return
        self._to(state, why)

    def arm(self) -> None:
        """Operator: leave zero-torque and stand up.  Preparation window only."""
        if self.state in (State.ZERO_TORQUE, State.DAMP, State.BOOT):
            self._to(State.RAMP, "armed")

    def start_run(self) -> None:
        """Whistle.  The one place `allow_run` is set."""
        self.allow_run = True
        if self.state is State.STAND:
            self._to(State.RUN, "start gate")

    # ---- health ---------------------------------------------------------------------------
    def _check(self, st: LegState) -> Fault | None:
        if self.estop.active():
            return self.estop.latched or Fault("estop_hardware", Severity.ABORT, "line tripped")

        rc = self.bus.remote()
        if rc is not None and rc.ok:
            for b in self.abort_buttons:
                if rc.pressed(b):
                    return Fault("estop_remote", Severity.ABORT, f"button {b}")

        if not st.ok:
            return Fault("lowstate_stale", Severity.ABORT,
                         f"no LowState for >{self.cfg['transport']['state_timeout_s']}s")

        g = projected_gravity(st.quat)
        if g[2] > self.grav_z_abort:
            return Fault("orientation", Severity.ABORT, f"projected gravity z={g[2]:+.2f}")

        # Roll/pitch from the gravity vector rather than Euler angles: no gimbal case, no
        # wrap, and it is the same quantity the policy sees.
        roll = float(np.arctan2(-g[1], -g[2]))
        pitch = float(np.arctan2(g[0], np.hypot(g[1], g[2])))
        if abs(roll) > self.max_roll or abs(pitch) > self.max_pitch:
            return Fault("tipping", Severity.ABORT,
                         f"roll={roll:+.2f} pitch={pitch:+.2f} rad")

        over = np.abs(st.dq) > self.map.vel_limit * 1.25
        if over.any():
            names = ", ".join(self.map.names[i] for i in np.where(over)[0])
            return Fault("joint_overspeed", Severity.WARN, names)

        if self._rate_streak >= self.rate_streak_limit:
            n = self._rate_streak
            self._rate_streak = 0
            return Fault("rate_limiter_saturated", Severity.DEGRADE,
                         f"{n} consecutive ticks clamped; policy and robot disagree")
        return None

    def _fallen(self, st: LegState) -> bool:
        """Body low and staying low.  Height from leg kinematics rather than a map: the
        mean vertical foot offset under the hips, which is the only height signal that
        exists on hardware."""
        h = self._body_height(st)
        low = h < self.fall_height
        now = time.monotonic()
        if low:
            self._bad_since = self._bad_since or now
            return (now - self._bad_since) > self.fall_hold
        self._bad_since = None
        return False

    def _body_height(self, st: LegState) -> float:
        """Crude but honest: thigh and calf are both 0.275 m, so the hip-to-foot vertical
        drop of each leg is 0.275*(cos(thigh) + cos(thigh+calf)).  Averaged over the four
        legs this tracks the standing height (~0.43 m) well enough to tell 'standing' from
        'on its belly', which is all the fall detector needs."""
        q = np.asarray(st.q, float)
        thigh = q[1::3]
        calf = q[2::3]
        drop = 0.275 * (np.cos(thigh) + np.cos(thigh + calf))
        return float(np.mean(np.abs(drop)))

    # ---- the tick ---------------------------------------------------------------------------
    def step(self) -> State:
        st = self.bus.read()
        fault = self._check(st)

        if fault is not None and fault.severity >= Severity.ABORT:
            self.estop.trip(fault)
            self._to(State.FAULT if "estop" in fault.name else State.DAMP, str(fault))
        elif fault is not None:
            self.estop.record(fault)

        handler = {
            State.BOOT: self._boot,
            State.ZERO_TORQUE: self._zero_torque,
            State.RAMP: self._ramp,
            State.STAND: self._stand,
            State.RUN: self._run,
            State.RECOVER: self._recover,
            State.DAMP: self._damp,
            State.FAULT: self._fault,
        }[self.state]
        handler(st)
        return self.state

    def _boot(self, st: LegState) -> None:
        self.bus.write(LegCommand.zero_torque())
        if st.ok:
            self._to(State.ZERO_TORQUE, "first LowState")

    def _zero_torque(self, st: LegState) -> None:
        self.bus.write(LegCommand.zero_torque())

    def _ramp(self, st: LegState) -> None:
        """Blend from the measured pose to standing over stand_ramp_s at full gains.  The
        blend is over the TARGET, so the driver's PD does the work and the robot rises
        smoothly from whatever heap it started in."""
        if self._ramp_from is None or self.elapsed < 1e-6:
            self._ramp_from = np.asarray(st.q, float).copy()
        alpha = min(1.0, self.elapsed / self.stand_ramp_s)
        q = (1.0 - alpha) * self._ramp_from + alpha * self.map.q_default
        self._write(q)
        if alpha >= 1.0:
            self._ramp_from = None
            self._to(State.STAND, "ramp complete")

    def _stand(self, st: LegState) -> None:
        # Hold the default pose exactly.  No velocity command reaches the policy here, which
        # is what makes a false start structurally impossible rather than merely unlikely.
        self._write(self.map.q_default)
        if self.allow_run:
            self._to(State.RUN, "start gate")

    def _run(self, st: LegState) -> None:
        if self._fallen(st):
            self._to(State.RECOVER, f"body height {self._body_height(st):.2f} m")
            return
        _, q = self.policy.step(st, self.cmd)
        self._write(q)

    def _recover(self, st: LegState) -> None:
        """Self-righting is worth 10 points per occurrence versus a human touching the
        robot.  Tuck to the default pose with the rate limiter doing the work, then hand
        back to the policy; if it does not take after a few tries, stop burning the clock
        and sit in DAMP so a judge-approved reset can happen from a known posture."""
        if self.elapsed > 4.0:
            self._recover_tries += 1
            if self._recover_tries >= self.max_recover_tries:
                self.estop.record(Fault("recover_exhausted", Severity.DEGRADE,
                                        f"{self._recover_tries} attempts"))
                self._to(State.DAMP, "self-righting failed")
            else:
                self._to(State.RAMP, "retry stand")
            return
        if not self._fallen(st) and self.elapsed > 1.0:
            self._bad_since = None
            self._to(State.RUN, "recovered")
            return
        self._write(self.map.q_default)

    def _damp(self, st: LegState) -> None:
        self.bus.write(LegCommand.damping(self.damp_kd))
        self.limiter.reset(st.q if st.ok else None)

    def _fault(self, st: LegState) -> None:
        self.bus.write(LegCommand.damping(self.damp_kd))
        if not self.estop.active():
            self._to(State.DAMP, "fault cleared")

    def _write(self, q_target: np.ndarray) -> None:
        before = self.limiter.report.rate_hits.sum()
        q = self.limiter(np.asarray(q_target, float))
        fired = self.limiter.report.rate_hits.sum() > before
        self._rate_streak = self._rate_streak + 1 if fired else 0
        self.bus.write(LegCommand(q=q, kp=np.full(12, self.kp), kd=np.full(12, self.kd)))

    # ---- reporting --------------------------------------------------------------------------
    def report(self) -> dict:
        return {
            "state": self.state.value,
            "transitions": [{"t": round(t, 3), "from": a, "to": b, "why": w}
                            for t, a, b, w in self.transitions],
            "clamps": self.limiter.report.summary(self.map.names),
            "estop": self.estop.summary(),
            "recover_attempts": self._recover_tries,
        }
