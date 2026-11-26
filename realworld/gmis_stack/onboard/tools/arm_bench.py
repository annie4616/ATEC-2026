#!/usr/bin/env python3
"""Test the PiPER-X arm with no legs, no DDS and no robot network.

The arm and the legs share nothing but a mini-PC.  The legs are DDS over ethernet, the arm
is CAN over `can0`, and neither driver imports the other -- they meet exactly once, in
run.py's `build()`.  That means the arm can be exercised on its own, and it needs to be:
`onboard.run` refuses to start without LowState (prep's first stage is
`legs.wait_for_state`), so with only the arm plugged in there is otherwise no way to find
out whether the CAN side works at all.

    python -m onboard.tools.arm_bench                      # mock, no hardware
    python -m onboard.tools.arm_bench --backend real       # read-only: link, enable, feedback
    python -m onboard.tools.arm_bench --backend real --move --grip   # also moves the arm

Read-only by default, and that default is the point.  On check-in day the first question is
"is the CAN link real", not "does the arm reach"; answering the first must not require
clearing space around a machine nobody has commanded yet.  --move and --grip are opt-in and
print what they are about to do before they do it.

Legs-only is the mirror image and already exists: `ARM=0 bash onboard/bin/run.sh`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onboard import cfg as C  # noqa: E402
from onboard.oio.iface import ArmCommand  # noqa: E402

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_rows: list[tuple[str, str, str]] = []


def add(name: str, result: str, detail: str = "") -> None:
    _rows.append((name, result, detail))
    print(f"  {result:4s}  {name:<24s}  {detail}")


def _fmt(v) -> str:
    return "[" + " ".join(f"{float(x):+.3f}" for x in v) + "]"


def check_link(arm, cfg, seconds: float) -> bool:
    """Is there a live CAN link, and is it streaming rather than answering once?

    A single successful read proves the SDK loaded, not that the bus is healthy -- a stale
    cached frame reads exactly like a fresh one until you look at the timestamps.  So this
    samples for a while and reports the rate of DISTINCT frames, the same way selftest.py
    treats LowState.
    """
    st = arm.read()
    add("arm link", PASS if st.ok else FAIL, st.fault or f"q={_fmt(st.q)}")
    if not st.ok:
        add("arm state rate", FAIL, "no link -- is can0 up? (piper_sdk can_activate.sh)")
        return False

    t0, n, last_t = time.monotonic(), 0, st.t
    while time.monotonic() - t0 < seconds:
        s = arm.read()
        if s.ok and s.t != last_t:
            n += 1
            last_t = s.t
        time.sleep(0.002)
    hz = n / max(time.monotonic() - t0, 1e-9)
    add("arm state rate", PASS if hz > 20 else FAIL, f"{hz:.0f} Hz distinct frames")

    st = arm.read()
    # Every joint reading exactly zero is the CAN equivalent of the encoder check in
    # selftest.py: a plausible-looking frame that means the fields were never populated.
    add("encoders", PASS if float(np.abs(st.q).max()) > 1e-6 else WARN,
        f"|q|max={float(np.abs(st.q).max()):.4f} -- all-zero may just be the true home pose")

    lim = np.asarray(cfg["joint_limits"], dtype=np.float64)
    inside = bool(np.all(st.q >= lim[:, 0] - 1e-3) and np.all(st.q <= lim[:, 1] + 1e-3))
    add("q within URDF limits", PASS if inside else FAIL, _fmt(st.q))

    stroke_ok = 0.0 <= st.gripper_stroke <= float(cfg["gripper"]["max_stroke_m"]) + 1e-4
    add("gripper feedback", PASS if stroke_ok else WARN,
        f"stroke={st.gripper_stroke * 1000:.1f} mm  current={st.gripper_current:.2f} A")
    add("joint current", PASS if float(np.max(st.current)) >= 0.0 else WARN,
        f"{_fmt(st.current)} A -- all-zero means this firmware does not report it, "
        "which disables the grasp-hold signal")
    return hz > 20


def check_enable(arm) -> bool:
    ok = arm.enable()
    add("arm enable", PASS if ok else FAIL,
        "all joints enabled" if ok else "EnableArm(7) did not take within 3 s")
    return ok


def ramp_to(arm, q_goal: np.ndarray, stroke: float, cfg, label: str) -> bool:
    """Walk the target to `q_goal` under the configured per-step cap.

    The cap is the same `limits.max_target_delta_rad` the mission uses.  A bench that moves
    faster than the mission is not a bench for the mission, and 0.2 rad/step at 50 Hz is
    already the arm's full 10 rad/s rating.
    """
    hz = float(cfg["transport"]["command_hz"])
    step = float(cfg["limits"]["max_target_delta_rad"])
    st = arm.read()
    if not st.ok:
        add(label, FAIL, "no state to ramp from")
        return False
    q = st.q.copy()
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        d = np.clip(q_goal - q, -step, step)
        q = q + d
        arm.write(ArmCommand(q=q.copy(), gripper_stroke=stroke))
        if float(np.max(np.abs(q_goal - q))) < 1e-3:
            break
        time.sleep(1.0 / hz)
    time.sleep(0.5)
    err = float(np.max(np.abs(arm.read().q - q_goal)))
    # 0.05 rad ~ 3 deg.  The arm closes its own loop and settles with a small steady-state
    # offset under gravity; this is a "did it go there" check, not a servo spec.
    add(label, PASS if err < 0.05 else WARN, f"max joint error {err:.3f} rad")
    return err < 0.05


def check_move(arm, cfg, amp: float) -> None:
    home = np.asarray(cfg["home"], dtype=np.float64)
    stroke = arm.read().gripper_stroke
    print(f"\n  -- moving to home {_fmt(home)} --")
    if not ramp_to(arm, home, stroke, cfg, "home ramp"):
        return
    lim = np.asarray(cfg["joint_limits"], dtype=np.float64)
    soft = float(cfg["limits"]["soft_limit_factor"])
    for i in range(6):
        # Probe each joint alone, both ways, clipped into the soft range.  One joint at a
        # time is what makes the answer readable: if joint 4 does not move, nothing else
        # moved to hide it.
        for sign in (+1.0, -1.0):
            q = home.copy()
            q[i] = np.clip(home[i] + sign * amp, lim[i, 0] * soft, lim[i, 1] * soft)
            ramp_to(arm, q, stroke, cfg, f"joint{i + 1} {'+' if sign > 0 else '-'}{amp:.2f}")
        ramp_to(arm, home, stroke, cfg, f"joint{i + 1} return")


def check_grip(arm, cfg) -> None:
    g = cfg["gripper"]
    hz = float(cfg["transport"]["command_hz"])
    d = float(g["max_stroke_delta_m"])
    q = arm.read().q.copy()
    print("\n  -- gripper open/close --")
    for label, goal in (("gripper open", float(g["max_stroke_m"])), ("gripper close", 0.0)):
        s = arm.read().gripper_stroke
        deadline = time.monotonic() + 10.0
        while abs(goal - s) > 1e-4 and time.monotonic() < deadline:
            s += float(np.clip(goal - s, -d, d))
            arm.write(ArmCommand(q=q.copy(), gripper_stroke=s))
            time.sleep(1.0 / hz)
        time.sleep(0.5)
        st = arm.read()
        # Closing on air must reach ~0.  Stopping short with current drawn is the hold
        # signature -- correct behaviour if something is between the fingers, and worth
        # saying out loud rather than failing.
        err = abs(st.gripper_stroke - goal)
        held = goal == 0.0 and st.gripper_current >= float(g["hold_current_a"])
        add(label, PASS if err < 5e-3 or held else WARN,
            f"stroke={st.gripper_stroke * 1000:.1f} mm  current={st.gripper_current:.2f} A"
            + ("  (stopped on an object -- that is a hold, not a failure)" if held else ""))


def main() -> int:
    cfg = C.load("piperx")
    ap = argparse.ArgumentParser(description="arm-only bench; no legs, no DDS")
    ap.add_argument("--backend", default="mock", choices=("mock", "real"))
    ap.add_argument("--can", default=None, help="override configs/piperx.yaml can_interface")
    ap.add_argument("--seconds", type=float, default=2.0, help="link sampling window")
    ap.add_argument("--move", action="store_true", help="MOVES THE ARM: home + joint sweep")
    ap.add_argument("--amp", type=float, default=0.20, help="--move sweep amplitude, rad")
    ap.add_argument("--grip", action="store_true", help="MOVES THE GRIPPER: open then close")
    a = ap.parse_args()

    if a.can:
        cfg["transport"]["can_interface"] = a.can

    if a.backend == "real":
        from onboard.oio.piper_can import PiperCanBus
        arm = PiperCanBus(cfg)
    else:
        from onboard.oio.replay import MockArmBus
        arm = MockArmBus(cfg)

    moves = a.move or a.grip
    print(f"== arm bench  backend={a.backend}  can={cfg['transport']['can_interface']}  "
          f"{'MOVES THE ARM' if moves else 'read-only'} ==")
    if moves and a.backend == "real":
        print("   the arm is about to move -- clear the space around it.  Ctrl-C to abort.")
        time.sleep(3.0)

    try:
        arm.start()
    except Exception as e:  # noqa: BLE001
        # An SDK or interface failure is the single most likely outcome on a fresh mini-PC,
        # and it is a diagnosis, not a crash.
        add("arm bus start", FAIL, f"{type(e).__name__}: {e}")
        print("\n  can0 up?   ip -details link show can0")
        print("  bring up:  bash ~/src/piper_sdk/can_activate.sh can0 1000000")
        return 1

    try:
        if check_link(arm, cfg, a.seconds) and check_enable(arm):
            if a.move:
                check_move(arm, cfg, a.amp)
            if a.grip:
                check_grip(arm, cfg)
        elif moves:
            print("\n  link or enable failed -- skipping the motion tests")
    except KeyboardInterrupt:
        print("\n  aborted")
    finally:
        # hold(), not disable: stop() freezes the arm where it is.  A disabled arm falls.
        arm.stop()

    n_fail = sum(1 for _, r, _ in _rows if r == FAIL)
    n_warn = sum(1 for _, r, _ in _rows if r == WARN)
    print(f"\n  ---- {'ARM OK' if n_fail == 0 else 'ARM NOT READY'}  "
          f"({sum(1 for _, r, _ in _rows if r == PASS)} pass, {n_warn} warn, {n_fail} fail)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
