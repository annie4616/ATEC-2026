#!/usr/bin/env bash
# Put the robot into damping and HOLD it there until interrupted.
#
# Damping is a state, not a packet.  unitree_rl_gym publishes its damping command once on
# exit, so a single dropped DDS frame leaves the robot holding its last position target --
# which, if it was mid-stride, is not a posture anyone wants it holding.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$HERE"
NIC="${NIC:?set NIC}"
exec "${PY:-python3}" - "$NIC" <<'PY'
import sys, time
from onboard import cfg as C
from onboard.oio.a2_dds import A2DdsBus
bus = A2DdsBus(C.load("a2pro"), net_interface=sys.argv[1])
bus.start()
bus.wait_for_state(5.0)
bus.damp()
print("damping at", C.load("a2pro")["safety"]["damp_kd"], "kd -- ctrl-C to release")
try:
    while True:
        time.sleep(0.5)
        print(f"  tx={bus.stats()['tx']} rx={bus.stats()['rx']}", end="\r", flush=True)
except KeyboardInterrupt:
    print("\nreleasing")
finally:
    bus.stop()
PY
