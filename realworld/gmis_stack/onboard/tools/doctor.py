#!/usr/bin/env python3
"""Deployment readiness: what is installed, what is configured, what is still assumed.

The question this answers is "can this machine run the robot yet", and the honest answer has
three parts that are easy to conflate:

  * **Missing** -- a package or a device that is not there.  Fixable by `bin/setup_minipc.sh`.
  * **Unconfigured** -- a value nobody has measured yet (camera intrinsics, the E-Stop line).
    Fixable only at the venue, with the robot and a checkerboard.
  * **Assumed** -- a value that IS filled in, from a convention, and has never been checked
    against this robot.  The motor map is the dangerous one: if it is wrong the robot still
    stands, still passes a smoke test, and folds the first time it turns.

The third category is why this file exists.  A green "all packages installed" is not
readiness, and a checklist that does not distinguish "absent" from "unverified" will let an
assumption walk onto the course.

    python onboard/tools/doctor.py            # report
    python onboard/tools/doctor.py --strict   # non-zero exit unless RUN-READY
    python onboard/tools/doctor.py --identity # what is running, for the inspection desk
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ONBOARD = Path(__file__).resolve().parents[1]
REPO = ONBOARD.parent
sys.path.insert(0, str(REPO))

OK, MISSING, UNCONFIGURED, ASSUMED, WARN = "OK", "MISSING", "UNCONFIG", "ASSUMED", "WARN"
_ICON = {OK: "  ok  ", MISSING: " MISS ", UNCONFIGURED: "UNCONF", ASSUMED: "ASSUME",
         WARN: " warn "}

rows: list[tuple[str, str, str, str]] = []


def add(section: str, name: str, state: str, detail: str = "") -> None:
    rows.append((section, name, state, detail))


def have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def version(mod: str) -> str:
    try:
        m = __import__(mod)
        return str(getattr(m, "__version__", "?"))
    except Exception:  # noqa: BLE001
        return "?"


# ---- 1. machine ------------------------------------------------------------------------
def check_machine() -> None:
    s = "machine"
    add(s, "architecture", OK if platform.machine() == "x86_64" else WARN, platform.machine())
    try:
        cpu = [l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
               if l.startswith("model name")]
        name = cpu[0] if cpu else "?"
    except OSError:
        name = "?"
    add(s, "cpu", OK if "N150" in name else WARN, f"{name} x{os.cpu_count()}")
    try:
        kb = next(int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()
                  if l.startswith("MemTotal"))
        gb = kb / 1e6
        add(s, "memory", OK if gb >= 7.0 else WARN, f"{gb:.1f} GB")
    except (OSError, StopIteration):
        add(s, "memory", WARN, "unreadable")
    free = shutil.disk_usage(REPO).free / 1e9
    add(s, "disk free", OK if free > 10 else WARN, f"{free:.0f} GB")
    gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov.exists():
        g = gov.read_text().strip()
        # On a 6 W part, `powersave` can halve the clock mid-stride; the control loop is not
        # heavy but it is periodic, and a governor that ramps is a source of jitter.
        add(s, "cpu governor", OK if g == "performance" else WARN, g)


# ---- 2. python packages ----------------------------------------------------------------
def check_python() -> None:
    s = "python"
    v = sys.version_info
    add(s, "interpreter", OK if v >= (3, 10) else WARN,
        f"{v.major}.{v.minor}.{v.micro} @ {sys.prefix}")
    add(s, "in a venv", OK if sys.prefix != sys.base_prefix else WARN,
        "system python -- setup_minipc.sh builds a venv")
    for mod, why in (("numpy", "everything"), ("yaml", "configs"),
                     ("torch", "policies"), ("scipy", "detector, 10x faster labelling"),
                     ("cv2", "camera rectification and JPEG decode")):
        add(s, mod, OK if have(mod) else MISSING, f"{version(mod) if have(mod) else why}")
    if have("torch"):
        import torch
        add(s, "torch is CPU build", OK if "+cu" not in torch.__version__ else WARN,
            torch.__version__)


# ---- 3. robot SDKs ----------------------------------------------------------------------
def check_sdk() -> None:
    s = "sdk"
    for mod, why in (("unitree_sdk2py", "legs over DDS"),
                     ("cyclonedds", "the DDS transport itself"),
                     ("piper_sdk", "arm over CAN"),
                     ("can", "python-can, the CAN layer")):
        add(s, mod, OK if have(mod) else MISSING, version(mod) if have(mod) else why)
    if have("cyclonedds"):
        v = version("cyclonedds")
        add(s, "cyclonedds pinned 0.10.2", OK if v.startswith("0.10.2") else WARN, v)


# ---- 4. links ----------------------------------------------------------------------------
def check_links() -> None:
    s = "links"
    try:
        out = subprocess.run(["ip", "-o", "link"], capture_output=True, text=True, timeout=5).stdout
        nics = [l.split(":")[1].strip() for l in out.splitlines() if ":" in l]
    except Exception:  # noqa: BLE001
        nics = []
    add(s, "network interfaces", OK if nics else WARN, ", ".join(nics[:6]))
    try:
        addrs = subprocess.run(["ip", "-o", "-4", "addr"], capture_output=True,
                               text=True, timeout=5).stdout
        on_robot_net = "192.168.123." in addrs
    except Exception:  # noqa: BLE001
        on_robot_net = False
    add(s, "on the robot subnet", OK if on_robot_net else UNCONFIGURED,
        "need an address on 192.168.123.x to reach the robot at .161")
    can_up = any("can" in n for n in nics)
    add(s, "CAN interface", OK if can_up else UNCONFIGURED,
        "can0 not up -- run the piper_sdk can_activate.sh")


# ---- 5. configuration --------------------------------------------------------------------
def check_config() -> None:
    s = "config"
    from onboard import cfg as C
    a2, piper, sensors = C.load("a2pro"), C.load("piperx"), C.load("sensors")

    add(s, "net_interface", OK if a2["transport"]["net_interface"] else UNCONFIGURED,
        "set it, or pass --nic")
    add(s, "msg_type", ASSUMED, f"{a2['transport']['msg_type']!r} -- Go2 convention; "
                                "A2 may speak unitree_hg. dump_motor_map.py settles it")
    idx = list(a2["leg_joint2motor_idx"])
    is_identity = idx == list(range(12))
    add(s, "motor map", ASSUMED if is_identity else OK,
        "identity, from the Go2 convention -- A WRONG MAP DOES NOT CRASH: the robot stands, "
        "passes a smoke test, and folds on the first turn"
        if is_identity else f"measured: {idx}")
    signs = list(a2["joint_sign"])
    all_pos = set(signs) == {1}
    add(s, "joint sign/offset", ASSUMED if all_pos else OK,
        "all +1, unmeasured" if all_pos else f"measured: {signs}")
    add(s, "E-Stop line", OK if a2["safety"].get("estop_gpio") else UNCONFIGURED,
        "no hardware line configured; software abort only")

    src = sensors["camera"].get("source") or {}
    add(s, "camera intrinsics", OK if src.get("K") else UNCONFIGURED,
        "calib_extrinsics.py --intrinsics; without it the frame is assumed pinhole")
    add(s, "camera extrinsics", ASSUMED, "catalogue mount + 22 deg; 2 deg of pitch error "
                                         "is 0.31 m of range error at 2 m")
    add(s, "camera backend", ASSUMED if sensors["camera"]["backend"] == "a2_sdk" else OK,
        f"{sensors['camera']['backend']!r} -- the a2_sdk path uses Go2's VideoClient")
    dsrc = sensors["camera"]["depth"]["source"]
    add(s, "depth source", WARN if dsrc == "ground" else OK,
        f"{dsrc!r} -- ground plane alone retains 32 % of object detections; "
        "LiDAR 32 lines retains 57 %")
    lb = (sensors.get("lidar") or {}).get("backend", "none")
    add(s, "lidar", WARN if lb == "none" else OK, f"backend {lb!r}")


# ---- 6. artifacts and self-tests ----------------------------------------------------------
def check_stack() -> None:
    s = "stack"
    r = subprocess.run([sys.executable, str(ONBOARD / "tools" / "bundle_artifacts.py"),
                        "--verify"], capture_output=True, text=True)
    add(s, "policy artifacts", OK if r.returncode == 0 else MISSING,
        (r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else "")
    r = subprocess.run([sys.executable, str(ONBOARD / "tools" / "offsim_gate.py")],
                       capture_output=True, text=True)
    add(s, "offsim gate", OK if r.returncode == 0 else MISSING,
        (r.stdout or "").strip().splitlines()[-1] if r.stdout else "")
    # SCHED_FIFO without root needs an rtprio limit; without it the loop runs at normal
    # priority, which is survivable but is a jitter source worth knowing about.
    try:
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_RTPRIO)
        add(s, "realtime priority", OK if soft > 0 else WARN,
            f"RLIMIT_RTPRIO soft={soft}; setup_minipc.sh installs a limits.d rule")
    except (ImportError, AttributeError, OSError):
        add(s, "realtime priority", WARN, "unreadable")


VERDICTS = {
    "RUN-READY": "every hardware path is installed, configured and verified",
    "BENCH-READY": "the stack runs against mocks; hardware paths are not installed",
    "FIELD-PENDING": "installed, but values remain unmeasured or unverified",
}


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_tree(root: Path) -> tuple[str, int]:
    """Order-independent digest over a directory: sha256 of sorted (relpath, filehash)."""
    h = hashlib.sha256()
    n = 0
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(f.relative_to(root).as_posix().encode())
        h.update(_sha256_file(f).encode())
        n += 1
    return h.hexdigest(), n


def identity() -> int:
    """The four values the inspection desk asks for, and the one it should.

    Rule IV.5 of the 0907 guide locks code, algorithm parameters and system configuration at
    inspection.  Nothing in this stack could previously state any of them: the Dockerfile
    wrote /app/BUILD_INFO and /opt/venv/SDK_STATUS and NOTHING EVER READ EITHER, and no tool
    printed a config digest at all.

    The config digest covers the EFFECTIVE configs -- the venue bind mount, when one is
    mounted over the baked-in directory.  That is the point: a measured override becomes a
    recorded fact instead of an invisible one, and the same command re-run after the
    end-of-day rebuild should print the same digest with no mount in play.
    """
    out: dict[str, object] = {}

    build_info = Path("/app/BUILD_INFO")
    in_image = build_info.exists()
    out["context"] = "container image" if in_image else "host checkout"
    if in_image:
        for line in build_info.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k] = v
    else:
        try:
            rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                 capture_output=True, text=True).stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                   capture_output=True, text=True).stdout.strip()
            out["git_rev"] = f"{rev}{'-dirty' if dirty else ''}" or "unknown"
        except Exception:  # noqa: BLE001
            out["git_rev"] = "unknown"

    cfg_dir = ONBOARD / "configs"
    if cfg_dir.is_dir():
        digest, n = _sha256_tree(cfg_dir)
        out["configs_sha256"] = digest
        out["configs_files"] = n
        out["configs_path"] = str(cfg_dir)

    man = ONBOARD / "artifacts" / "manifest.json"
    if man.exists():
        out["artifacts_manifest_sha256"] = _sha256_file(man)
        try:
            entries = json.loads(man.read_text()).get("entries", {})
            out["artifacts"] = {k: v.get("sha256", "")[:16] for k, v in sorted(entries.items())}
        except Exception:  # noqa: BLE001
            pass

    sdk_status = Path("/opt/venv/SDK_STATUS")
    if sdk_status.exists():
        skipped = [l for l in sdk_status.read_text().splitlines() if l.strip()]
        out["sdk_skipped"] = skipped or "none (all robot SDKs installed)"

    width = max(len(k) for k in out)
    print("== identity ==")
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"  {k:<{width}} :")
            for kk, vv in v.items():
                print(f"  {'':<{width}}   {kk:<12} {vv}")
        elif isinstance(v, list):
            print(f"  {k:<{width}} : {', '.join(v)}")
        else:
            print(f"  {k:<{width}} : {v}")
    print()
    print("  The image DIGEST is not visible from inside the container; take it from the")
    print("  host:  docker image inspect atec-onboard:latest --format '{{.Id}}'")
    return 0


def main(strict: bool) -> int:
    for fn in (check_machine, check_python, check_sdk, check_links, check_config, check_stack):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            add(fn.__name__, "check failed", WARN, f"{type(e).__name__}: {e}")

    width = max(len(n) for _, n, _, _ in rows)
    last = None
    for sec, name, state, detail in rows:
        if sec != last:
            print(f"\n== {sec} ==")
            last = sec
        print(f"  [{_ICON[state]}] {name:<{width}} {detail}")

    n = {k: sum(1 for _, _, s, _ in rows if s == k)
         for k in (OK, MISSING, UNCONFIGURED, ASSUMED, WARN)}
    if n[MISSING]:
        verdict = "BENCH-READY"
    elif n[UNCONFIGURED] or n[ASSUMED]:
        verdict = "FIELD-PENDING"
    else:
        verdict = "RUN-READY"

    print(f"\n{'-' * 72}")
    print(f"  ok {n[OK]}   missing {n[MISSING]}   unconfigured {n[UNCONFIGURED]}   "
          f"assumed {n[ASSUMED]}   warn {n[WARN]}")
    print(f"  VERDICT: {verdict} -- {VERDICTS[verdict]}")
    if n[MISSING]:
        print("\n  next: bash onboard/bin/setup_minipc.sh")
    if n[ASSUMED]:
        print("  the ASSUMED rows are the dangerous ones -- they are filled in and untested.")
        print("  settle them with tools/dump_motor_map.py and tools/calib_extrinsics.py")
    return 0 if (verdict == "RUN-READY" or not strict) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--identity", action="store_true",
                    help="print git rev / config digest / artifact digests and exit")
    _a = ap.parse_args()
    raise SystemExit(identity() if _a.identity else main(_a.strict))
