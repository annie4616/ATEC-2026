#!/usr/bin/env python3
"""The mission entry point.  One file, two backends.

    python -m onboard.run --backend mock                 # desktop, no hardware
    python -m onboard.run --backend real --nic enp2s0    # the robot

The backend choice is the ONLY difference between a rehearsal on this laptop and the run at
the venue.  Everything above `onboard/oio/` -- the policy, the estimator, the safety
supervisor, the mission FSM, the timing -- is byte-identical in both, which is the entire
reason the stack is arranged this way: the robot is shared equipment we cannot touch until
check-in day, so the code that runs there has to have been exercised somewhere else.

Structure of a run:

    build -> prep (5 min window) -> READY -> [whistle] -> 30 min mission -> report

The control loop is one 50 Hz thread.  Perception is its own thread pinned off the control
core and communicates only through age-stamped mailboxes, so a 500 ms detector frame can
never delay a motor command.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import onboard  # noqa: E402,F401  threads + MISSION_REAL_NOISE before anything else
from onboard import cfg as C  # noqa: E402
from onboard.bringup.prep import Prep  # noqa: E402
from onboard.bringup.startgate import StartGate, TimerTrigger  # noqa: E402
from onboard.control.legs import LegPolicy  # noqa: E402
from onboard.control.states import LegSupervisor, State  # noqa: E402
from onboard.estimate.attitude import Attitude  # noqa: E402
from onboard.estimate.navstate import NavState  # noqa: E402
from onboard.estimate.odom import LegOdometry  # noqa: E402
from onboard.mission.budget import Budget  # noqa: E402
from onboard.mission.collector import Collector  # noqa: E402
from onboard.mission.deliver import Delivery  # noqa: E402
from onboard.mission.fsm import MissionFSM, Phase  # noqa: E402
from onboard.mission.grasp import GraspEpisode, GraspPolicy  # noqa: E402
from onboard.mission.graspobs import PregraspBank  # noqa: E402
from onboard.mission.ledger import Ledger  # noqa: E402
from onboard.estimate.binmem import BinMemory  # noqa: E402
from onboard.perceive.objtrack import ObjectTracker  # noqa: E402
from onboard.control.arm import ArmController  # noqa: E402
from onboard.oio.estop import EStop  # noqa: E402
from onboard.perceive.procworker import PerceptionProcess  # noqa: E402
from onboard.rt.clock import RateLoop, set_realtime  # noqa: E402


def build(backend: str, nic: str | None, fault: str, no_arm: bool,
          with_collector: bool = False, out_dir: str | None = None,
          frames_dir: str | None = None):
    a2, piper = C.load("a2pro"), C.load("piperx")
    sensors, mission, policies = C.load("sensors"), C.load("mission_hk"), C.load("policies")

    if backend == "real":
        from onboard.oio.a2_dds import A2DdsBus
        legs = A2DdsBus(a2, net_interface=nic)
        arm = None
        if not no_arm:
            from onboard.oio.piper_can import PiperCanBus
            arm = PiperCanBus(piper)
    else:
        from onboard.oio.replay import MockArmBus, MockLegBus
        legs = MockLegBus(a2, fault=fault)
        arm = None if no_arm else MockArmBus(piper)

    policy = LegPolicy(a2, policies)
    estop = EStop(a2["safety"].get("estop_gpio"))
    sup = LegSupervisor(legs, policy, a2, estop)
    att, odo = Attitude(), LegOdometry()
    nav = NavState(mission, sensors)
    # A PROCESS, not a thread: the detector's pure-Python inner loops hold the GIL, and in
    # a thread that showed up as a 213 ms hole in the 50 Hz command stream.
    percep = PerceptionProcess(sensors, backend=backend, nic=nic, frames_dir=frames_dir)
    budget = Budget(total_s=float(mission["budget"]["total_s"]),
                    reserve_for_delivery_s=float(mission["budget"]["reserve_for_delivery_s"]))
    fsm = MissionFSM(sup, nav, odo, budget, mission)

    collector = None
    if with_collector:
        if arm is None:
            print("   collector requested but the arm is disabled -- Task 2 is off")
        else:
            col = C.load("collector")
            arm_ctl = ArmController(piper)
            ep = GraspEpisode(col, piper,
                              GraspPolicy(policies["grasp"]["actor"]),
                              PregraspBank(policies["grasp"]["bank"],
                                           float(col["grasp"]["bank_min_x"])),
                              arm_ctl, arm)
            binmem = BinMemory(col)
            ledger = Ledger(Path(out_dir) / "ledger.json" if out_dir else None)
            collector = Collector(col, mission, sup, nav, odo, percep, arm, arm_ctl,
                                  ep, ObjectTracker(col), Delivery(col, binmem, arm_ctl),
                                  ledger)
            fsm.attach_collector(collector)
            print(f"   collector attached: grasp policy iter {ep.policy.iter}, "
                  f"bank {len(ep.bank)} configs, anchor "
                  f"{[round(float(v), 3) for v in ep.bank.anchor]}")
            if backend == "mock":
                print("   NOTE mock backend: recorded frames and the mock robot move "
                      "independently,\n        so approaches cannot converge -- "
                      "approach_timeout is the expected outcome.")

    return dict(collector=collector,legs=legs, arm=arm, policy=policy, estop=estop, sup=sup, att=att, odo=odo,
                nav=nav, percep=percep, budget=budget, fsm=fsm, a2=a2, mission=mission,
                sensors=sensors)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock", choices=("mock", "real"))
    ap.add_argument("--nic", default=None, help="network interface facing the robot")
    ap.add_argument("--gate", default="manual", choices=("manual", "timer"),
                    help="manual = controller combination at the whistle; timer = rehearsal only")
    ap.add_argument("--gate-delay", type=float, default=3.0)
    ap.add_argument("--minutes", type=float, default=None, help="override the 30 min budget")
    ap.add_argument("--fault", default="none", help="mock only: inject a failure")
    ap.add_argument("--no-arm", action="store_true")
    ap.add_argument("--frames", default=None,
                    help="mock only: feed perception from recorded .npz frames. NOTE the "
                         "frames follow their own recorded trajectory while the mock robot "
                         "follows the commands it is given, so targets never actually come "
                         "to the aim point -- expect every attempt to end in "
                         "approach_timeout. This exercises detection, tracking, selection, "
                         "the ledger and the timing; it does NOT test the approach servo.")
    ap.add_argument("--with-collector", action="store_true",
                    help="run Task 2 as well: pick up objects and deliver them")
    ap.add_argument("--out", default=None, help="write the run report here")
    a = ap.parse_args()

    ctx = build(a.backend, a.nic, a.fault, a.no_arm, a.with_collector,
                str(Path(a.out).parent) if a.out else None, a.frames)
    legs, sup, att, odo, nav = ctx["legs"], ctx["sup"], ctx["att"], ctx["odo"], ctx["nav"]
    percep, budget, fsm = ctx["percep"], ctx["budget"], ctx["fsm"]
    if a.minutes is not None:
        # Scale the delivery reserve with the shortened budget, or a test run can never
        # collect: the 30-minute profile reserves 240 s, which is longer than most test runs.
        scale = (a.minutes * 60.0) / max(budget.total_s, 1e-9)
        budget.total_s = a.minutes * 60.0
        budget.reserve_for_delivery_s *= scale
    complaint = budget.sane()
    if complaint:
        print(f"   BUDGET: {complaint}")

    print(f"== onboard  backend={a.backend}  gate={a.gate}  "
          f"budget={budget.total_s / 60:.0f} min ==")

    legs.start()
    if ctx["arm"] is not None:
        ctx["arm"].start()

    # Before the loop exists.  `spawn` re-imports numpy, torch and the detector in the child
    # and that measured 1.1 s -- harmless here, a 1.1 s gap in the command stream if it
    # happens once the 50 Hz loop is running.
    print("   starting perception process ...", flush=True)
    t_p = time.monotonic()
    percep.start()
    deadline = t_p + 20.0
    while time.monotonic() < deadline and percep.n_recv == 0:
        time.sleep(0.05)
    print(f"   perception up in {time.monotonic() - t_p:.1f} s "
          f"({'producing' if percep.n_recv else 'NO OUTPUT -- degraded'})")

    # AFTER legs, arm and perception are up -- not before.  Affinity and scheduling policy
    # are inherited by threads and processes created later, so calling this first put the DDS
    # writer thread, the arm thread and the spawned perception process all on core 0 with a
    # SCHED_FIFO-80 control thread ahead of them.  Measured in the container, which is the
    # first environment where the RT request is actually GRANTED (--cap-add SYS_NICE +
    # --ulimit rtprio): mock LowState collapsed 50 Hz -> 12 Hz and prep failed outright.  On
    # the host the request is denied, so the bug was invisible.  Only the control thread
    # should be pinned; everything else keeps the whole machine.
    rt = set_realtime(80, core=0)
    print(f"   realtime: {rt}")

    loop = RateLoop(float(ctx["a2"]["control"]["rate_hz"]), "control")
    loop.start()
    stopping = {"flag": False}

    def _stop(*_):
        stopping["flag"] = True
    # SIGTERM as well as SIGINT.  `docker stop` sends SIGTERM, and inside the container this
    # process is PID 1 -- the kernel delivers a signal to PID 1 only if a handler is
    # installed, so without this line SIGTERM was IGNORED, the daemon SIGKILLed us after the
    # grace period, and `_shutdown` -> `legs.stop()`'s damp-and-keep-damping never ran.  The
    # robot was left holding whatever its last command was.
    for _sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(_sig, _stop)

    def tick() -> None:
        loop.tick()
        st = legs.read()
        att.update(st)
        odo.update(st, att)
        nav.predict(st.t, odo, att)
        percep.set_pose(st.quat, odo.height)
        _consume(nav, percep, st.t)
        if a.backend == "mock" and hasattr(legs, "set_velocity_command"):
            legs.set_velocity_command(sup.cmd)
        sup.step()

    # ---- preparation -----------------------------------------------------------------
    # Everything from here is wrapped so that `_shutdown` runs on ANY exit -- a raised
    # exception in the control loop used to skip damping entirely.  `_shutdown` is
    # idempotent, so the explicit calls below still damp at the same moment they always did
    # and this is only a backstop.
    try:
        return _run(a, ctx, legs, sup, att, odo, nav, percep, budget, fsm, loop, tick,
                    stopping)
    finally:
        _shutdown(ctx, percep)


def _run(a, ctx, legs, sup, att, odo, nav, percep, budget, fsm, loop, tick, stopping) -> int:
    prep = Prep(sup, legs, arm=ctx["arm"], perception=percep, attitude=att, odom=odo,
                estop=ctx["estop"])
    res = prep.run(tick)
    print("\n-- preparation --")
    print(res.render())
    if not res.ok:
        print("\nNOT READY -- refusing to start.")
        _shutdown(ctx, percep)
        return 1

    gate = TimerTrigger(delay_s=a.gate_delay) if a.gate == "timer" else StartGate()
    gate.ready()
    print(f"\nREADY. {'auto-start' if a.gate == 'timer' else 'waiting for L2+start'}\n")

    # ---- the run ---------------------------------------------------------------------
    last_phase, last_log = None, 0.0
    while not stopping["flag"]:
        tick()
        if gate.update(legs.remote()) and not budget.started:
            budget.start()
            sup.start_run()
            print(f"  +{0.0:6.1f}s  START")
        fsm.step()
        if fsm.phase is not last_phase:
            print(f"  +{budget.elapsed():6.1f}s  {last_phase.value if last_phase else '-':>11s}"
                  f" -> {fsm.phase.value}")
            last_phase = fsm.phase
        if budget.started and budget.elapsed() - last_log > 15.0:
            last_log = budget.elapsed()
            print(f"  +{budget.elapsed():6.1f}s  {fsm.phase.value:11s} "
                  f"{odo.distance:6.2f} m  lat={nav.lat:+.2f}"
                  f"({nav.fuser.lat_sigma:.2f})  {loop.stats.summary()['hz']:.1f} Hz  "
                  f"sup={sup.state.value}")
        if fsm.phase is Phase.DONE or (budget.started and budget.expired()):
            break
        if sup.state is State.FAULT:
            print("  FAULT latched -- the run cannot continue autonomously")
            break

    if ctx.get("collector") is not None:
        print("\n-- collector --")
        # Flush unconditionally.  Ledger.flush() otherwise only runs when an ATTEMPT closes,
        # so a run with zero attempts left no ledger.json at all -- and "the file is missing"
        # reads identically to "logging is broken" to whoever is holding the robot.  Zero
        # attempts is itself the finding worth recording.
        ctx["collector"].ledger.flush()
        print(ctx["collector"].ledger.render())
        print(f"  tracker: {ctx['collector'].trk.report()}")

    report = {"backend": a.backend, "prep_s": round(res.total_s, 2),
              "collector": (None if ctx.get("collector") is None
                            else ctx["collector"].report()),
              "loop": loop.stats.summary(), "supervisor": sup.report(),
              "mission": fsm.report(), "odometry": odo.report(),
              "perception": percep.report()}
    _shutdown(ctx, percep)

    print("\n-- run --")
    ls = report["loop"]
    print(f"  loop     {ls['hz']:.2f} Hz  dt p99 {ls['dt_ms_p99']:.2f} ms  "
          f"max {ls['dt_ms_max']:.2f}  late={ls['late']}  overrun={ls['overrun']}")
    print(f"  distance {odo.distance:.1f} m   phase {fsm.phase.value}   "
          f"oob {fsm.oob_events}   stuck {fsm.stuck_events}")
    print(f"  nav      {nav.report()}")
    for f in report["supervisor"]["estop"]["faults"]:
        print(f"  fault    [{f['sev']}] {f['name']}: {f['detail']}")
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"  wrote {a.out}")
    return 0


def _consume(nav: NavState, percep: PerceptionProcess, t: float) -> None:
    """Drain the perception mailboxes into the corridor filter.

    LiDAR first, camera second, both through the EKF's innovation gate.  Neither is trusted
    on its own: the camera's p90 tail is 1.8 m at the stairs, and the LiDAR cannot tell a
    guide strip from anything else in the same height band.
    """
    r = percep.rail.get()
    if r is not None:
        nav.update_rail(r.value, t)
    b = percep.belt.get()
    if b is not None:
        nav.update_camera(b.value, t)


def _shutdown(ctx, percep) -> None:
    # Called explicitly on the normal paths (so the robot is damped BEFORE the report is
    # printed, not after) and again from main()'s finally.  The second call must be free.
    if ctx.get("_shutdown_done"):
        return
    ctx["_shutdown_done"] = True
    try:
        percep.stop()
    except Exception:  # noqa: BLE001
        pass
    if ctx.get("arm") is not None:
        try:
            ctx["arm"].stop()
        except Exception:  # noqa: BLE001
            pass
    # Legs last, and its stop() damps and keeps damping.
    ctx["legs"].stop()


if __name__ == "__main__":
    raise SystemExit(main())
