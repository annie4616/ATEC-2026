#!/usr/bin/env bash
# Thin dispatcher so `docker run atec-onboard <verb>` reads like the host scripts do.
#
# Two verbs are NOT thin, and deliberately so -- `run` and `prep` are the container's
# equivalents of onboard/bin/run.sh and onboard/bin/prep.sh, and until 2026-09-09 they were
# missing the parts of those scripts that produce evidence and accept the interface:
#   * `run` dropped --out, so ledger.json and report.json were never written even though
#     the runbook tells the operator to read them after every attempt, and dropped the
#     `tee`, so the run log died with the container.
#   * `prep` dropped --nic, so the documented command aborted on argument validation.
set -euo pipefail
cd /app

# The container runs as root, and it has to: SocketCAN's raw socket for the PiPER arm needs
# CAP_NET_RAW, which a --user process does not get.  Root-owned output in a host bind mount
# is then un-deletable by the operator -- and onboard/bin/run.sh's own `mkdir -p "$OUT"`
# fails against a root-owned onboard/logs.  A permissive umask keeps every artefact readable,
# writable and removable from the host without giving up the capability.
# (If this site never uses the arm, docker/run.sh honours DOCKER_USER="$(id -u):$(id -g)".)
umask 0000

# The arm precondition, mirroring onboard/bin/run.sh:22-25.  Without it the failure is a
# 20-line piper_sdk traceback ending in "ConnectionError: ['can0' Interface __init__ ERROR]"
# during startup -- which reads like a broken image rather than "the CAN interface is down".
_require_can() {
  [ "${ARM:-1}" = "1" ] || return 0
  ip link show "${CAN_IF:-can0}" >/dev/null 2>&1 && return 0
  echo "ERROR: ${CAN_IF:-can0} is not up." >&2
  echo "  On the mini-PC:  bash ~/src/piper_sdk/can_activate.sh can0 1000000" >&2
  echo "  Then re-run.  To run legs-only instead:  ARM=0 ..." >&2
  exit 1
}
_arm_flag() { [ "${ARM:-1}" = "1" ] || echo --no-arm; }

case "${1:-doctor}" in
  doctor)   shift; exec python onboard/tools/doctor.py "$@" ;;
  identity) shift; exec python onboard/tools/doctor.py --identity "$@" ;;
  gate)     shift; exec python onboard/tools/offsim_gate.py "$@" ;;
  docs)     shift; exec python onboard/tools/check_docs.py "$@" ;;
  bench)    shift; exec python onboard/tools/bench.py "$@" ;;
  motormap) shift; exec python onboard/tools/dump_motor_map.py "$@" ;;
  calib)    shift; exec python onboard/tools/calib_extrinsics.py "$@" ;;
  replay)   shift; exec python onboard/tools/replay_run.py "$@" ;;
  nav)      shift; exec python onboard/tools/check_nav.py "$@" ;;
  collector) shift; exec python onboard/tools/check_collector.py "$@" ;;
  parity)   shift; exec python onboard/tools/parity.py "$@" ;;

  prep)
    shift
    # NIC is forwarded by docker/run.sh.  Fail with the same message the host script gives
    # rather than argparse's, so the fix is obvious.
    : "${NIC:?set NIC to the interface facing the robot, e.g. NIC=enp2s0}"
    _require_can
    exec python -m onboard.run --backend real --gate manual --minutes 0 --nic "$NIC" \
         $(_arm_flag) "$@"
    ;;

  run)
    shift
    # Mirrors onboard/bin/run.sh:16-31.  /app/onboard/logs is the bind mount, so everything
    # written here survives --rm.
    case " $* " in *" --backend real "*) _require_can ;; esac
    OUT="${OUT:-/app/onboard/logs/$(date -u +%m%d_%H%M%S)}"
    mkdir -p "$OUT"
    echo "   run output -> ${OUT}/{report.json,ledger.json,run.log}" >&2
    # Process substitution rather than a `| tee` pipeline ON PURPOSE: with a pipeline, bash
    # stays PID 1 and SIGTERM from `docker stop` never reaches python, so the legs are never
    # damped.  `exec` here keeps python as PID 1 while its output still lands in run.log.
    exec python -m onboard.run --out "$OUT/report.json" "$@" \
         > >(tee -a "$OUT/run.log") 2>&1
    ;;

  estop)    shift; exec bash onboard/bin/estop_test.sh "$@" ;;
  damp)     shift; exec bash onboard/bin/damp.sh "$@" ;;
  shell)    shift; exec bash "$@" ;;
  *)        exec "$@" ;;
esac
