#!/usr/bin/env python3
"""Run the onboard stack against a mock or recorded robot.

This is the answer to "how is any of this tested before check-in day".  It drives the real
control loop, the real policy, the real state machine, the real limiters and the real start
gate -- only the transport is swapped.  A regression in the parts that will run at the venue
shows up here, on a desktop, with no robot booked.

    python onboard/tools/replay_run.py                       # 30 s, clean
    python onboard/tools/replay_run.py --minutes 30          # full soak
    python onboard/tools/replay_run.py --inject inverted     # must end in DAMP
    python onboard/tools/replay_run.py --inject estop        # must latch FAULT
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onboard import cfg as C  # noqa: E402
from onboard.bringup.startgate import StartGate, TimerTrigger  # noqa: E402
from onboard.control.legs import LegPolicy  # noqa: E402
from onboard.control.states import LegSupervisor, State  # noqa: E402
from onboard.mission.budget import Budget  # noqa: E402
from onboard.oio.estop import EStop, Fault, Severity  # noqa: E402
from onboard.oio.replay import FAULTS, MockLegBus  # noqa: E402
from onboard.rt.clock import RateLoop  # noqa: E402


def run(minutes: float, inject: str, fault_at: float, seed: int, quiet: bool) -> dict:
    a2 = C.load("a2pro")
    pol = C.load("policies")
    mission = C.load("mission_hk")

    bus = MockLegBus(a2, fault=inject, fault_at_s=fault_at, seed=seed)
    bus.start()
    policy = LegPolicy(a2, pol)
    estop = EStop()
    sup = LegSupervisor(bus, policy, a2, estop)
    gate = TimerTrigger(delay_s=2.0)
    budget = Budget(total_s=minutes * 60.0,
                    reserve_for_delivery_s=float(mission["budget"]["reserve_for_delivery_s"]))

    vx = float(mission["locomotion"]["nominal_vx"])
    loop = RateLoop(float(a2["control"]["rate_hz"]), "control")
    loop.start()

    t_boot = time.monotonic()
    t_end = t_boot + minutes * 60.0 + 10.0  # + prep window
    armed = False
    last_state = None
    printed = 0.0

    while time.monotonic() < t_end:
        loop.tick()

        # Preparation window: arm the legs, wait for the stand, declare ready.
        if not armed and sup.state is State.ZERO_TORQUE:
            sup.arm()
            armed = True
        if sup.state is State.STAND and gate.state.value == "idle":
            gate.ready()

        if gate.update(bus.remote()) and not budget.started:
            budget.start()
            sup.start_run()

        # Only after the gate does anything ask the legs to move.
        if sup.state is State.RUN and budget.started and not budget.expired():
            sup.set_command(np.array([vx, 0.0, 0.0]))
        else:
            sup.set_command(np.zeros(3))
        bus.set_velocity_command(sup.cmd)

        state = sup.step()

        if state is not last_state:
            if not quiet:
                # Wall-clock since process start, not budget time: the interesting
                # transitions (ramp, stand) all happen BEFORE the whistle, and printing
                # budget time there stamps every one of them 0.00 s.
                print(f"  +{time.monotonic() - t_boot:6.2f}s  "
                      f"{last_state.value if last_state else '-':>11s} -> {state.value}")
            last_state = state

        if state is State.FAULT and estop.latched is not None:
            # A latched fault ends the autonomous run: from here only a human can proceed.
            if budget.elapsed() > (fault_at + 2.0) or not budget.started:
                break
        if budget.started and budget.expired():
            # Time is up: stop asking for velocity and let the supervisor settle.  Note this
            # REQUESTS stand rather than forcing it -- if the robot damped earlier, the end
            # of the budget is not a reason to stand it back up.
            sup.set_command(np.zeros(3))
            sup.request(State.STAND, "budget expired")
            break
        if not quiet and budget.elapsed() - printed > 10.0:
            printed = budget.elapsed()
            tr = bus.truth()
            print(f"  +{time.monotonic() - t_boot:6.1f}s  {state.value:11s} "
                  f"x={tr['xy'][0]:6.2f} m  z={tr['z']:.2f}  {loop.stats.summary()['hz']:.1f} Hz")

    truth = bus.truth()
    bus.stop()

    rep = {
        "inject": inject,
        "final_state": sup.state.value,
        "distance_m": round(float(truth["xy"][0]), 3),
        "budget": budget.report(),
        "loop": loop.stats.summary(),
        "supervisor": sup.report(),
    }
    return rep


def expected_outcome(inject: str) -> tuple[set[str], str]:
    """What each injected failure MUST produce.  Written down so the test asserts the
    safety contract rather than just 'it did not crash'."""
    return {
        "none": ({"run", "stand"}, "clean run keeps walking"),
        "lowstate_stall": ({"damp"}, "stale LowState must damp, not coast on old obs"),
        "inverted": ({"damp"}, "body past the orientation limit must damp"),
        "estop": ({"fault"}, "E-Stop must latch and stay latched"),
        "joint_fault": ({"run", "damp", "recover", "stand"}, "jammed joint: keep going or damp"),
    }.get(inject, ({"run", "stand", "damp", "fault"}, ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock", choices=("mock",))
    ap.add_argument("--minutes", type=float, default=0.5)
    ap.add_argument("--inject", default="none", choices=FAULTS)
    ap.add_argument("--fault-at", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    print(f"== replay_run  backend={a.backend}  inject={a.inject}  {a.minutes} min ==")
    rep = run(a.minutes, a.inject, a.fault_at, a.seed, a.quiet)

    ok_states, why = expected_outcome(a.inject)
    ok = rep["final_state"] in ok_states
    print(f"\nfinal state : {rep['final_state']}   expected {sorted(ok_states)}  "
          f"{'OK' if ok else 'FAIL'}   {why}")
    print(f"distance    : {rep['distance_m']} m")
    ls = rep["loop"]
    print(f"loop        : {ls['hz']:.2f} Hz  dt p99 {ls['dt_ms_p99']:.2f} ms  "
          f"max {ls['dt_ms_max']:.2f} ms  late={ls['late']} overrun={ls['overrun']}")
    cl = rep["supervisor"]["clamps"]
    print(f"clamps      : limit={cl['limit_hits']} rate={cl['rate_hits']} "
          f"nonfinite={cl['nonfinite']} worst_rate={cl['worst_rate_rad_per_step']}")
    for f in rep["supervisor"]["estop"]["faults"]:
        print(f"fault       : [{f['sev']}] {f['name']}: {f['detail']}")

    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=2, default=str) + "\n")
        print(f"wrote {a.out}")
    raise SystemExit(0 if ok else 1)
