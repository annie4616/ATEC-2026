"""Go / no-go before the whistle.

The preparation window is five minutes, it happens once, and the rules make it the only
chance to find out that something is wrong: after the countdown starts nothing is paused for
any reason, including our own equipment.  So the point of this file is to convert
"everything looked fine" into a list of things that were actually checked, printed where a
person standing next to a robot can read it in a few seconds.

Each check reports PASS, WARN or FAIL.  FAIL means do not start; WARN means a capability is
missing and the mission will run degraded -- no LiDAR, no arm -- which is worth knowing
before the whistle rather than discovering at the first grasp.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

import numpy as np


class R(enum.Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class Check:
    name: str
    result: R
    detail: str = ""


@dataclass
class SelfTest:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, result: R, detail: str = "") -> None:
        self.checks.append(Check(name, result, detail))

    @property
    def ok(self) -> bool:
        return not any(c.result is R.FAIL for c in self.checks)

    def render(self) -> str:
        w = max((len(c.name) for c in self.checks), default=10)
        lines = [f"  {c.result.value:4s}  {c.name:<{w}s}  {c.detail}" for c in self.checks]
        lines.append(f"  ---- {'READY' if self.ok else 'NOT READY'} "
                     f"({sum(c.result is R.PASS for c in self.checks)} pass, "
                     f"{sum(c.result is R.WARN for c in self.checks)} warn, "
                     f"{sum(c.result is R.FAIL for c in self.checks)} fail)")
        return "\n".join(lines)


def run(legs, arm=None, perception=None, cfg=None, estop=None,
        lowstate_hz: float | None = None) -> SelfTest:
    t = SelfTest()

    # --- legs -------------------------------------------------------------------------
    st = legs.read()
    t.add("leg link", R.PASS if st.ok else R.FAIL,
          f"tick={st.tick} v={st.voltage:.1f} V" if st.ok else "no LowState")
    if st.ok:
        # Encoders reading exactly zero is the classic symptom of talking to the right bus
        # with the wrong message type: plausible-looking, entirely wrong.
        t.add("encoders", R.PASS if np.abs(st.q).max() > 1e-6 else R.FAIL,
              f"|q|max={np.abs(st.q).max():.3f}")
        g = _grav(st.quat)
        t.add("IMU upright", R.PASS if g[2] < -0.9 else R.FAIL,
              f"projected gravity z={g[2]:+.3f}")
        t.add("IMU quaternion normalised",
              R.PASS if abs(float(np.linalg.norm(st.quat)) - 1.0) < 1e-2 else R.FAIL,
              f"|q|={float(np.linalg.norm(st.quat)):.4f}")
        hot = st.temperature is not None and float(np.max(st.temperature)) > 70.0
        t.add("motor temperature", R.WARN if hot else R.PASS,
              f"max={float(np.max(st.temperature)):.0f} C" if st.temperature is not None else "n/a")
        if st.voltage > 0:
            low = st.voltage < 24.0
            t.add("battery", R.WARN if low else R.PASS, f"{st.voltage:.1f} V")

    # --- state freshness -------------------------------------------------------------
    # Measured PASSIVELY, by the caller, over ticks it was going to run anyway.  The first
    # version sampled here with 50 x 4 ms sleeps, which is 200 ms of blocking inside one
    # call -- and since prep runs inside the 50 Hz control loop, that was a 200 ms hole in
    # the command stream that took three wrong guesses to track down.  A self-test that
    # perturbs the system it is testing is worse than no self-test.
    if lowstate_hz is not None:
        t.add("LowState rate", R.PASS if lowstate_hz > 30 else R.FAIL,
              f"{lowstate_hz:.0f} Hz distinct ticks")

    # --- remote ------------------------------------------------------------------------
    rc = legs.remote()
    have_rc = rc is not None and rc.ok
    t.add("remote link", R.PASS if have_rc else R.WARN,
          "" if have_rc else "no controller: the start gate and the abort button need it")

    # --- E-Stop -------------------------------------------------------------------------
    if estop is not None:
        t.add("E-Stop hardware", R.PASS if estop.hardware_present else R.WARN,
              "" if estop.hardware_present else "no hardware line; software abort only")
        t.add("E-Stop not latched", R.FAIL if estop.active() else R.PASS,
              str(estop.latched or ""))

    # --- arm ---------------------------------------------------------------------------
    if arm is not None:
        ast = arm.read()
        t.add("arm link", R.PASS if ast.ok else R.WARN, ast.fault or "")
        if ast.ok:
            t.add("arm enabled", R.PASS if ast.enabled else R.WARN, "")
            t.add("arm gripper", R.PASS if ast.gripper_stroke >= 0.0 else R.WARN,
                  f"stroke={ast.gripper_stroke*1000:.0f} mm")

    # --- perception ---------------------------------------------------------------------
    if perception is not None:
        rep = perception.report()
        for name, mb in rep.get("mailboxes", {}).items():
            fresh = mb["age_s"] is not None and mb["age_s"] < 3.0
            severity = R.PASS if fresh else (R.WARN if name != "belt" else R.WARN)
            t.add(f"perception:{name}", severity,
                  f"age={mb['age_s']}s puts={mb['put']}")
        t.add("perception errors", R.PASS if rep.get("errors", 0) == 0 else R.WARN,
              str(rep.get("errors", 0)))

    return t


def _grav(quat) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    return np.array([2.0 * (-z * x + w * y), -2.0 * (z * y + w * x),
                     1.0 - 2.0 * (w * w + z * z)])
