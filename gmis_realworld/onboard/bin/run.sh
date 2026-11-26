#!/usr/bin/env bash
# The competition run.  Everything else is rehearsal for this.
#
#   NIC=enp2s0 bash onboard/bin/run.sh
#
# Preconditions, all checked by the script:
#   * the robot's high-level controller (sport_mode) is OFF, or rt/lowcmd is ignored
#   * the CAN interface is up if the arm is being used
#   * a controller is paired, because the start gate and the abort button both need it
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"

NIC="${NIC:?set NIC to the interface facing the robot, e.g. NIC=enp2s0}"
PY="${PY:-python3}"
OUT="${OUT:-onboard/logs/$(date +%m%d_%H%M)}"
mkdir -p "$OUT"

# Verify the bundle before anything is powered.  A mismatch here means the weights on this
# machine are not the weights that were escrowed.
"$PY" onboard/tools/bundle_artifacts.py --verify

if [ "${ARM:-1}" = "1" ]; then
  ip link show can0 >/dev/null 2>&1 || {
    echo "can0 is not up -- run the piper_sdk can_activate.sh, or set ARM=0" >&2; exit 1; }
fi

exec "$PY" -m onboard.run --backend real --nic "$NIC" \
     --gate "${GATE:-manual}" ${ARM:+$([ "${ARM:-1}" = "0" ] && echo --no-arm)} \
     --out "$OUT/report.json" 2>&1 | tee "$OUT/run.log"
