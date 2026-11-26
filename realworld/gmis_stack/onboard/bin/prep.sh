#!/usr/bin/env bash
# Preparation only: bring everything up, print the go/no-go, and stop before the gate.
# Use during the 5-minute window and during equipment testing to time the sequence.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$HERE"
NIC="${NIC:?set NIC}"
exec "${PY:-python3}" -m onboard.run --backend real --nic "$NIC" --gate manual --minutes 0
