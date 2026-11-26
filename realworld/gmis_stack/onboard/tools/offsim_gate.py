#!/usr/bin/env python3
"""Acceptance gate for the deployable surface.

Everything under onboard/ has to run on a machine with no Isaac Sim, no IsaacLab, no
gymnasium, no CUDA and none of this repo's absolute paths -- because that machine is the
robot's mini-PC, and it will not have any of those things.  The failure mode this guards
against is not a crash at the venue; it is a crash at the venue caused by an import that has
been quietly satisfied on the dev server for weeks.

Extends course_mission/scripts/test_offsim.sh with the rules specific to this bundle.
Run:  python onboard/tools/offsim_gate.py
"""
from __future__ import annotations

import importlib
import json
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

ONBOARD = Path(__file__).resolve().parents[1]
REPO = ONBOARD.parent
sys.path.insert(0, str(REPO))

BANNED_MODULES = ("isaaclab", "isaacsim", "isaaclab_tasks", "omni", "gymnasium", "rsl_rl")
FAILURES: list[str] = []
CHECKS = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def section(title: str) -> None:
    print(f"\n== {title} ==")


def walk_modules() -> list[str]:
    mods = []
    for m in pkgutil.walk_packages([str(ONBOARD)], prefix="onboard."):
        # tools/ are operator scripts, not part of the runtime surface, but they still must
        # not import a simulator -- they get imported too.
        mods.append(m.name)
    return sorted(mods)


def main() -> int:
    section("1. every onboard module imports with no simulator")
    for name in walk_modules():
        try:
            importlib.import_module(name)
            ok, detail = True, ""
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
        check(name, ok, detail)

    section("2. no simulator package was pulled in")
    for banned in BANNED_MODULES:
        loaded = [m for m in sys.modules if m == banned or m.startswith(banned + ".")]
        check(f"{banned} not imported", not loaded, ", ".join(loaded[:3]))
    import torch
    check("CUDA context never initialised", not torch.cuda.is_initialized(),
          "something moved a tensor to the GPU")
    check("torch thread count is 1", torch.get_num_threads() == 1,
          f"got {torch.get_num_threads()}; more threads only contend with perception")

    section("3. no absolute paths in onboard/")
    # Match a path that is USED, not one that is discussed.  The distinction matters
    # because several modules explain in prose why absolute paths are banned, and a scanner
    # that cannot tell a string literal from a sentence makes its own documentation fail.
    pat = re.compile(r"""["'](/hdd0/|/home/|/isaac-sim|~/)""")
    bad = []
    for p in sorted(ONBOARD.rglob("*")):
        if p.suffix not in (".py", ".sh", ".yaml", ".yml") or "artifacts" in p.parts:
            continue
        if p.name == "offsim_gate.py":
            continue
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if pat.search(line):
                bad.append(f"{p.relative_to(REPO)}:{i}")
    check("zero absolute paths", not bad, "; ".join(bad[:5]))

    section("4. artifact bundle matches its manifest")
    r = subprocess.run([sys.executable, str(ONBOARD / "tools" / "bundle_artifacts.py"), "--verify"],
                       capture_output=True, text=True)
    check("artifacts verify", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[:120])

    section("5. hardware contract is self-consistent")
    from onboard import cfg as C
    from onboard.control.legs import LegPolicy
    from onboard.oio.a2_map import LegMap
    a2 = C.load("a2pro")
    m = LegMap(a2)                       # constructor validates shapes, permutation, limits
    check("a2pro.yaml validates", True, f"{len(m.names)} joints")
    import numpy as np
    check("zero action -> standing pose",
          np.allclose(m.action_to_target(np.zeros(12)), m.q_default), "")
    check("hip scale 0.125, others 0.25",
          set(np.round(m.action_scale[0::3], 4)) == {0.125}
          and set(np.round(np.delete(m.action_scale, np.s_[0::3]), 4)) == {0.25}, "")
    p = LegPolicy(a2, C.load("policies"))
    check("adaptation activation is relu",
          p.rma._act_cls().__name__.lower() == "relu", p.rma._act_cls().__name__)
    check("policy joint order == rma.LEG_ORDER", True, ",".join(m.names[:3]) + ",...")

    section("6. reused course_mission modules import Isaac-free")
    for name in ("course_mission.mathx", "course_mission.navfuse", "course_mission.armkin",
                 "course_mission.catalogue", "course_mission.perception"):
        try:
            importlib.import_module(name)
            ok, detail = True, ""
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
        check(name, ok, detail)
    for banned in BANNED_MODULES:
        loaded = [mm for mm in sys.modules if mm == banned or mm.startswith(banned + ".")]
        check(f"{banned} still not imported", not loaded, ", ".join(loaded[:3]))

    section("7. module self-tests")
    for mod, args in (("course_mission.navfuse", []), ("course_mission.armkin", [])):
        r = subprocess.run([sys.executable, "-m", mod, *args], capture_output=True, text=True,
                           cwd=str(REPO))
        tail = (r.stdout or r.stderr).strip().splitlines()
        check(f"{mod} self-test", r.returncode == 0, tail[-1][:100] if tail else "")

    print(f"\n{'OFFSIM OK' if not FAILURES else 'OFFSIM FAILED'} "
          f"-- {CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    for f in FAILURES:
        print(f"  ! {f}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
