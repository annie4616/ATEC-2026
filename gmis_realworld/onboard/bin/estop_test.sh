#!/usr/bin/env bash
# Verify the E-Stop path BEFORE anything else on check-in day.
#
# The rules require a working E-Stop and count its use as a reset (-10, three per run), so
# it is both a safety device and a scored one.  This drives the robot to a stand and then
# asks the operator to trigger each abort path in turn, confirming the software sees it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$HERE"
NIC="${NIC:?set NIC}"
exec "${PY:-python3}" - "$NIC" <<'PY'
import sys, time
from onboard import cfg as C
from onboard.oio.a2_dds import A2DdsBus
from onboard.oio.estop import EStop

a2 = C.load("a2pro")
bus = A2DdsBus(a2, net_interface=sys.argv[1]); bus.start()
es = EStop(a2["safety"].get("estop_gpio"))
print("hardware E-Stop line configured:", es.hardware_present)
if not bus.wait_for_state(5.0):
    print("no LowState"); raise SystemExit(1)
print("\n1) press the controller abort button (select). waiting 20 s ...")
seen = False
t0 = time.monotonic()
while time.monotonic() - t0 < 20 and not seen:
    seen = bus.remote().pressed("select"); time.sleep(0.02)
print("   controller abort seen:", seen)
if es.hardware_present:
    print("\n2) trip the hardware E-Stop. waiting 20 s ...")
    t0 = time.monotonic(); hw = False
    while time.monotonic() - t0 < 20 and not hw:
        hw = es.hardware_tripped(); time.sleep(0.05)
    print("   hardware trip seen:", hw)
bus.stop()
print("\nE-Stop check complete. Both paths must be confirmed before the run.")
PY
