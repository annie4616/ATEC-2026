#!/usr/bin/env bash
# Thin wrapper so the gate can be run with a chosen interpreter:
#   PY=/path/to/plain/python bash onboard/tools/offsim_gate.sh
# Use a python WITHOUT isaaclab installed -- that is the whole point of the gate.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${PY:-python3}" "$HERE/offsim_gate.py" "$@"
