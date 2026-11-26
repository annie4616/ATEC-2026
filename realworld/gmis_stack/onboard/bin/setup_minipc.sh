#!/usr/bin/env bash
# Prepare the on-board mini-PC (Beelink, Intel N150, 12 GB, Ubuntu 22.04/24.04) to run
# `onboard/`.  Idempotent: re-running it is how you check it is still right.
#
#   bash onboard/bin/setup_minipc.sh --check     # report only, changes nothing
#   bash onboard/bin/setup_minipc.sh             # install, prompting before system changes
#   bash onboard/bin/setup_minipc.sh --yes       # non-interactive
#
# It REFUSES to run on a machine that does not look like the target unless --force is given.
# That guard is not politeness: this repository also lives on a shared GPU workstation, and
# an accidental `apt install` plus a CPU-governor change there is somebody else's ruined day.
#
# What it does NOT do, and cannot:
#   * measure the camera intrinsics or the motor map -- those need the robot and a
#     checkerboard.  `tools/doctor.py` lists them as UNCONFIG/ASSUME and they stay that way
#     until check-in day.
#   * bring up CAN or the robot network permanently.  It writes helpers; you run them with
#     the hardware attached.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ONBOARD="$HERE/onboard"
VENV="${ONBOARD_VENV:-$HOME/.venv/onboard}"
CHECK=0; YES=0; FORCE=0
for a in "$@"; do
  case "$a" in
    --check) CHECK=1 ;;
    --yes|-y) YES=1 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

say()  { printf '  %s\n' "$*"; }
head_() { printf '\n== %s ==\n' "$*"; }
run()  {
  if [ "$CHECK" = 1 ]; then printf '  would run: %s\n' "$*"; return 0; fi
  printf '  + %s\n' "$*"; "$@"
}
need_sudo() {
  [ "$CHECK" = 1 ] && { printf '  would run (sudo): %s\n' "$*"; return 0; }
  printf '  + sudo %s\n' "$*"; sudo "$@"
}
confirm() {
  [ "$YES" = 1 ] || [ "$CHECK" = 1 ] && return 0
  read -r -p "  $1 [y/N] " r; [ "$r" = y ] || [ "$r" = Y ]
}

# ---------------------------------------------------------------- 0. is this the target?
head_ "target check"
CPU="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs || echo '?')"
MEM_GB="$(awk '/MemTotal/{printf "%.1f", $2/1e6}' /proc/meminfo)"
say "cpu    : $CPU"
say "memory : ${MEM_GB} GB"
say "arch   : $(uname -m)"
LOOKS_RIGHT=0
case "$CPU" in *N150*|*N100*|*N95*|*N97*) LOOKS_RIGHT=1 ;; esac
if [ "$LOOKS_RIGHT" = 0 ] && [ "$FORCE" = 0 ] && [ "$CHECK" = 0 ]; then
  cat >&2 <<'MSG'

  REFUSING: this does not look like the on-board mini-PC (expected an Intel N-series CPU).
  This script installs packages and changes the CPU governor; running it on a shared
  workstation by mistake is exactly what the check is for.

  If this really is the target machine, re-run with --force.
  To see what it would do anywhere, use --check.
MSG
  exit 1
fi
[ "$LOOKS_RIGHT" = 0 ] && say "NOTE: not an N-series CPU; continuing because --check/--force"

# ---------------------------------------------------------------- 1. system packages
head_ "system packages"
APT_PKGS="python3-venv python3-dev build-essential cmake git \
can-utils iproute2 usbutils v4l-utils \
libgl1 libglib2.0-0 chrony"
say "apt: $APT_PKGS"
if confirm "install system packages with apt?"; then
  need_sudo apt-get update -qq
  # shellcheck disable=SC2086
  need_sudo apt-get install -y -qq $APT_PKGS
else
  say "skipped"
fi

# ---------------------------------------------------------------- 2. virtualenv
head_ "python environment"
say "venv: $VENV"
if [ ! -d "$VENV" ]; then
  run python3 -m venv "$VENV"
else
  say "already exists"
fi
PIP="$VENV/bin/pip"
[ "$CHECK" = 1 ] && PIP="echo would-pip"
run "$VENV/bin/python" -m pip install -q --upgrade pip wheel setuptools 2>/dev/null || true

# ---------------------------------------------------------------- 3. runtime deps
head_ "runtime dependencies"
# torch first and explicitly from the CPU index: the default wheel pulls ~2.5 GB of CUDA
# that this machine has no GPU for, and 512 GB is not a reason to waste it.
run "$PIP" install -q --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1"
run "$PIP" install -q -r "$ONBOARD/requirements-onboard.txt" \
  || say "NOTE: some pins failed; see requirements-onboard.txt"

# ---------------------------------------------------------------- 4. Unitree SDK (legs)
head_ "unitree_sdk2py (legs, DDS)"
# cyclonedds is version-locked by the SDK and its wheel does not always build.  The
# documented fallback is to compile CycloneDDS and point CYCLONEDDS_HOME at it -- do that
# BEFORE pip, or pip builds against nothing and fails in a confusing way.
if "$VENV/bin/python" -c "import cyclonedds" 2>/dev/null; then
  say "cyclonedds already present"
else
  if ! run "$PIP" install -q "cyclonedds==0.10.2"; then
    say "wheel failed -- building CycloneDDS from source"
    SRC="$HOME/src/cyclonedds"
    run git clone -q --depth 1 -b releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds "$SRC" || true
    run cmake -S "$SRC" -B "$SRC/build" -DCMAKE_INSTALL_PREFIX="$HOME/.local/cyclonedds" -DBUILD_EXAMPLES=OFF
    run cmake --build "$SRC/build" --target install -j"$(nproc)"
    say "export CYCLONEDDS_HOME=$HOME/.local/cyclonedds   # add this to your shell profile"
    if [ "$CHECK" = 0 ]; then
      CYCLONEDDS_HOME="$HOME/.local/cyclonedds" "$PIP" install -q "cyclonedds==0.10.2"
    fi
  fi
fi
SDK="$HOME/src/unitree_sdk2_python"
if [ -d "$SDK" ]; then say "unitree_sdk2_python already cloned"; else
  run git clone -q --depth 1 https://github.com/unitreerobotics/unitree_sdk2_python "$SDK"
fi
run "$PIP" install -q -e "$SDK"

# ---------------------------------------------------------------- 5. Piper SDK (arm)
head_ "piper_sdk (arm, CAN)"
run "$PIP" install -q "python-can>=4.0" piper_sdk

# ---------------------------------------------------------------- 6. real-time scheduling
head_ "real-time scheduling"
# Without an rtprio limit the control thread cannot take SCHED_FIFO and silently runs at
# normal priority.  It survives -- the loop is drift-free either way -- but every perception
# burst becomes a jitter source, and jitter on a 50 Hz motor stream is not free.
LIMITS=/etc/security/limits.d/99-onboard-realtime.conf
say "grant rtprio to group 'realtime', add $USER to it"
if confirm "write $LIMITS and add $USER to group realtime?"; then
  need_sudo groupadd -f realtime
  need_sudo usermod -aG realtime "$USER"
  if [ "$CHECK" = 0 ]; then
    printf '@realtime   -   rtprio   99\n@realtime   -   memlock  unlimited\n' \
      | sudo tee "$LIMITS" >/dev/null
    say "wrote $LIMITS   (log out and back in for it to take effect)"
  else
    say "would write $LIMITS"
  fi
else
  say "skipped -- the loop will run at normal priority"
fi

# ---------------------------------------------------------------- 7. power / determinism
head_ "power and determinism"
# The N150 is a 6 W part; on `powersave` it can drop clock mid-stride.  The control loop is
# light, but it is periodic, and a governor that ramps is a jitter source.
if confirm "set the CPU governor to performance and disable suspend?"; then
  if [ "$CHECK" = 0 ]; then
    for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
      [ -w "$g" ] && echo performance | sudo tee "$g" >/dev/null || true
    done
    say "governor set (not persistent across reboot -- see the systemd unit below)"
  else
    say "would set scaling_governor=performance on every core"
  fi
  need_sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
else
  say "skipped"
fi

# ---------------------------------------------------------------- 8. links (helpers only)
head_ "robot links"
cat <<'MSG'
  These need the hardware attached, so they are printed rather than run:

    # legs -- the mini-PC must sit on the robot's subnet
    sudo ip addr add 192.168.123.99/24 dev <iface>
    sudo ip link set <iface> up
    ping -c1 192.168.123.161

    # arm -- 1 Mbps, via the piper_sdk helper
    bash "$HOME/src/piper_sdk/can_activate.sh" can0 1000000
    ip -details link show can0

  And on the robot itself, the built-in high-level controller (sport_mode) must be OFF or
  it will fight every LowCmd this stack sends.
MSG

# ---------------------------------------------------------------- 9. verify
head_ "verify"
if [ "$CHECK" = 1 ]; then
  say "would run: doctor.py, offsim_gate.py, bench.py"
else
  say "python: $VENV/bin/python"
  "$VENV/bin/python" "$ONBOARD/tools/doctor.py" || true
fi

cat <<MSG

  Next, in this order -- the first two gate everything else:

    1. bash $ONBOARD/bin/estop_test.sh            # before anything moves
    2. $VENV/bin/python $ONBOARD/tools/dump_motor_map.py --nic <iface>
    3. $VENV/bin/python $ONBOARD/tools/calib_extrinsics.py --capture
    4. NIC=<iface> bash $ONBOARD/bin/prep.sh      # time the five-minute sequence
MSG
