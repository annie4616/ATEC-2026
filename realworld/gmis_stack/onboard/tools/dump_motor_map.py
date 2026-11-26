#!/usr/bin/env python3
"""Discover the A2 Pro's motor index, sign and offset by moving the joints by hand.

This is the first thing to run on check-in day, and everything else is blocked behind it.

The policy's joint order -- FR, FL, RR, RL x (hip, thigh, calf) -- is character-for-character
the Unitree SDK motor order used on Go2, so `leg_joint2motor_idx` is very probably the
identity.  "Very probably" is not a basis for putting 120 N.m into twelve motors.  A2 Pro is
newer than anything the reference stacks cover, and the failure mode of a wrong map is not a
crash: the robot stands, passes a smoke test, and folds the first time it tries to turn.

Procedure, about ten minutes with the robot in zero-torque:
    python onboard/tools/dump_motor_map.py --nic <iface>
then move ONE joint at a time when prompted.  The tool watches which motor index moves,
which direction it moves in, and what it reads at the mechanical zero, and prints yaml ready
to paste into configs/a2pro.yaml.

With --watch it just streams every motor, which is what you want when the answer is
"nothing moved" and the question becomes whether the message type is right at all.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onboard import cfg as C  # noqa: E402

JOINTS = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
          "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]
# Which way the joint should read when moved in its POSITIVE URDF direction:
#   hip   +x : abduct outward-left
#   thigh +y : swing the knee backward/up
#   calf  +y : extend the shin
HINT = {"hip": "abduct the leg outward (away from the body)",
        "thigh": "swing the thigh so the knee moves BACKWARD",
        "calf": "straighten the shin"}


def raw_motors(bus, n: int = 12) -> np.ndarray:
    """Motor-order joint positions, bypassing the map -- this tool exists to build it."""
    st = bus.read()
    return st.q if st.ok else np.full(n, np.nan)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nic", required=True)
    ap.add_argument("--watch", action="store_true", help="just stream every motor")
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="rad of motion before a joint counts as moved")
    ap.add_argument("--out", default=None, help="write the discovered yaml here")
    a = ap.parse_args()

    from onboard.oio.a2_dds import A2DdsBus

    a2 = C.load("a2pro")
    # Identity map while probing: we are measuring the map, so we must not apply one.
    a2["leg_joint2motor_idx"] = list(range(12))
    a2["joint_sign"] = [1] * 12
    a2["joint_offset"] = [0.0] * 12

    bus = A2DdsBus(a2, net_interface=a.nic)
    bus.start()
    print(f"waiting for LowState on {a.nic} (msg_type={bus.msg_type}) ...")
    if not bus.wait_for_state(10.0):
        print("NO LowState. Check: the NIC, that sport_mode/high-level control is off, and\n"
              "whether transport.msg_type should be 'hg' instead of 'go'.", file=sys.stderr)
        return 1
    print("connected.\n")

    if a.watch:
        try:
            while True:
                q = raw_motors(bus)
                print("  ".join(f"{i:2d}:{v:+.3f}" for i, v in enumerate(q)), end="\r",
                      flush=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print()
        finally:
            bus.stop()
        return 0

    idx, sign, offset = [None] * 12, [1] * 12, [0.0] * 12
    print("Put the robot in a safe posture. For each joint: move it clearly in the named\n"
          "direction, hold, then press Enter. Press Enter with no motion to skip.\n")
    for k, name in enumerate(JOINTS):
        kind = name.split("_")[1]
        base = raw_motors(bus)
        input(f"[{k + 1:2d}/12] {name:9s} -- {HINT[kind]}, then Enter: ")
        now = raw_motors(bus)
        d = now - base
        d[~np.isfinite(d)] = 0.0
        j = int(np.argmax(np.abs(d)))
        if abs(d[j]) < a.threshold:
            print(f"        no motor moved more than {a.threshold} rad -- skipped\n")
            continue
        # Second-largest tells us whether the reading is clean or two motors moved together
        # (a leg lifted by hand moves three joints at once if you are not careful).
        second = float(np.sort(np.abs(d))[-2])
        idx[k] = j
        sign[k] = 1 if d[j] > 0 else -1
        offset[k] = float(now[j] - sign[k] * 0.0)
        print(f"        motor {j:2d}  delta {d[j]:+.3f} rad  sign {sign[k]:+d}"
              f"   (next largest {second:.3f})"
              f"{'   <-- AMBIGUOUS, redo this joint' if second > 0.4 * abs(d[j]) else ''}\n")

    bus.stop()

    missing = [JOINTS[i] for i, v in enumerate(idx) if v is None]
    dup = len(set(v for v in idx if v is not None)) != len([v for v in idx if v is not None])
    print("\n== discovered map ==")
    for k, name in enumerate(JOINTS):
        print(f"  {name:9s} motor={idx[k]}  sign={sign[k]:+d}")
    if missing:
        print(f"\nINCOMPLETE: no motion recorded for {', '.join(missing)}")
    if dup:
        print("\nCONFLICT: two joints mapped to the same motor -- redo those two")

    yaml = ("leg_joint2motor_idx: [" + ", ".join(str(v if v is not None else -1) for v in idx)
            + "]\njoint_sign:          [" + ", ".join(f"{v:+d}" for v in sign) + "]\n")
    print("\n-- paste into configs/a2pro.yaml --\n" + yaml)
    if a.out:
        Path(a.out).write_text(yaml)
        print(f"wrote {a.out}")
    return 0 if (not missing and not dup) else 1


if __name__ == "__main__":
    raise SystemExit(main())
