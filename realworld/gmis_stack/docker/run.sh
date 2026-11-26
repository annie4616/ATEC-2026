#!/usr/bin/env bash
# Run the on-board image with the flags the hardware actually needs.
#
#   bash docker/run.sh doctor
#   bash docker/run.sh identity                      # what is running, for the inspection desk
#   bash docker/run.sh gate
#   NIC=enp2s0 bash docker/run.sh estop              # FIRST, before anything can move
#   NIC=enp2s0 bash docker/run.sh motormap --nic enp2s0
#   NIC=enp2s0 bash docker/run.sh run --backend real --nic enp2s0 --with-collector
#
# Each flag below is here for a reason, and dropping one produces a failure that looks like
# something else:
#
#   --network host     DDS is not a TCP client.  CycloneDDS discovers peers by multicast on
#                      the robot's subnet (192.168.123.x), and SocketCAN's `can0` is a
#                      network interface too.  In a bridge network the robot is simply not
#                      there, and the symptom is "no LowState" with no error.
#   --ipc host         perception runs in a child process and the default 64 MB /dev/shm is
#                      a classic opaque multiprocessing failure.  NOTE that --ipc host makes
#                      --shm-size inert (the host's /dev/shm is used as-is), which is why
#                      --shm-size is NOT set here; it was, and it read as protection it
#                      never provided.
#   --cap-add SYS_NICE the control thread asks for SCHED_FIFO.  Without the capability it
#                      silently runs at normal priority -- survivable, but every perception
#                      burst becomes jitter on a 50 Hz motor stream.
#   --ulimit rtprio    the capability alone is not enough; the limit has to allow it.
#   --device /dev/...  the camera, when it is a UVC device rather than an SDK channel.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-atec-onboard}"
TAG="${TAG:-latest}"
LOGS="${LOGS:-$HERE/onboard/logs}"

# Preflight.  "docker: command not found" at the venue, on the one machine that matters, is
# a worse discovery than a one-line check here.
command -v docker >/dev/null 2>&1 || {
  echo "docker is not installed on this host -- see docker/README.md 'Host prerequisites'" >&2
  exit 127; }
docker image inspect "${IMAGE}:${TAG}" >/dev/null 2>&1 || {
  echo "image ${IMAGE}:${TAG} is not loaded. Build it (bash docker/build.sh) or load the" >&2
  echo "tarball you carried in (docker load < atec-onboard.tar.gz)." >&2
  exit 125; }

mkdir -p "$LOGS"

ARGS=(
  --rm
  --network host
  --ipc host
  --cap-add SYS_NICE
  --ulimit rtprio=99
  --ulimit memlock=-1
  # SIGTERM must reach python so the legs get damped; see onboard/run.py's handler.  20 s is
  # comfortably longer than the damp-and-hold in A2DdsBus.stop().
  --stop-signal SIGTERM
  --stop-timeout 20
  # The stack writes to onboard/logs/ RELATIVE TO THE REPO ROOT -- run reports, the ledger,
  # calibration captures.  Inside the image that is /app/onboard/logs, so that is what has
  # to be bound.  Binding it at /logs (as this did until 2026-09-09) put the mount somewhere
  # nothing writes: every calibration view and every ledger.json died with the container.
  # /logs still resolves -- it is a symlink to this path in the image.
  -v "$LOGS:/app/onboard/logs"
)
# Interactive only when there is a terminal: `docker run -it` under CI hangs.
[ -t 0 ] && ARGS+=(-it)

# Root by default, and deliberately: the PiPER arm speaks SocketCAN, whose raw socket needs
# CAP_NET_RAW -- a --user process does not have it, and the failure is at the first CAN
# frame, not at start-up.  Output ownership is handled by the entrypoint's umask instead.
# Set DOCKER_USER="$(id -u):$(id -g)" if this run does not touch the arm.
[ -n "${DOCKER_USER:-}" ] && { ARGS+=(--user "$DOCKER_USER"); echo "   user: $DOCKER_USER" >&2; }

# ENV PASSTHROUGH.  The runtime reads MISSION_* / ONBOARD_* knobs (see onboard/__init__.py
# and course_mission/perception.py), and until 2026-09-09 exactly one of them was forwarded
# -- so changing any other one meant editing source and rebuilding the image between venue
# runs.  Everything forwarded is ECHOED, because under the 0907 rules (IV.5, code and
# parameters locked at inspection) an unlogged parameter change is the thing that voids a
# run.  `identity` hashes the effective config for the same reason.
ARGS+=(-e "MISSION_REAL_NOISE=${MISSION_REAL_NOISE:-0}")
FWD=()
for v in $(compgen -v | grep -E '^(MISSION_|ONBOARD_)' | sort || true); do
  [ "$v" = "MISSION_REAL_NOISE" ] && continue
  ARGS+=(-e "$v=${!v}"); FWD+=("$v=${!v}")
done
for v in NIC PY OUT ARM CAN_IF FRAMES; do
  [ -n "${!v:-}" ] && { ARGS+=(-e "$v=${!v}"); FWD+=("$v=${!v}"); }
done
[ "${#FWD[@]}" -gt 0 ] && printf '   env: %s\n' "${FWD[*]}" >&2

# Cameras and CAN adapters, only if they exist.  Mounting a device that is not there is an
# error; silently not mounting one that is would be worse, so say which were found.
FOUND=()
for d in /dev/video0 /dev/video1 /dev/bus/usb; do
  [ -e "$d" ] && { ARGS+=(--device "$d"); FOUND+=("$d"); }
done
[ "${#FOUND[@]}" -gt 0 ] && echo "   devices: ${FOUND[*]}" >&2 || \
  echo "   devices: none found (no camera or USB-CAN attached)" >&2

# Recorded frames for the offline checks, when the caller points at them.
[ -n "${FRAMES:-}" ] && ARGS+=(-v "${FRAMES}:/frames:ro")

# CONFIG OVERRIDE.  The configs are baked into the image so that a digest pins the whole
# behaviour -- but at the venue the motor map, the camera intrinsics and the mount pose are
# MEASURED, and rebuilding a 2 GB image between a checkerboard and a stand-up test is not a
# workflow.  So: measure into a directory, mount it over the baked-in one, and rebuild once
# at the end so the digest you escrow matches what actually ran.
#   CONFIGS=$PWD/venue-configs bash docker/run.sh run ...
# `run.sh identity` prints a sha256 over whatever configs are EFFECTIVE, so an override is a
# recorded fact rather than an invisible one.
if [ -n "${CONFIGS:-}" ]; then
  [ -d "$CONFIGS" ] || { echo "CONFIGS=$CONFIGS is not a directory" >&2; exit 2; }
  ARGS+=(-v "$(cd "$CONFIGS" && pwd):/app/onboard/configs:ro")
  echo "   configs: $CONFIGS  (overriding the image's own)" >&2
fi

exec docker run "${ARGS[@]}" "${IMAGE}:${TAG}" "$@"
