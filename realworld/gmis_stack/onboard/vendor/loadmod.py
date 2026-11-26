"""Load a single module file without executing its package __init__.

`eunha_rl/epkg/rma.py` is pure torch and imports cleanly on a machine with no simulator.
`eunha_rl/epkg/__init__.py` does `import gymnasium as gym`.  So a normal `import epkg.rma`
fails on the mini-PC for a reason that has nothing to do with the file we want -- the same
trap DEPLOY.md found blocking nine other modules.

Rather than fork rma.py (and then have two copies of the RMA contract drifting apart) or
install a training dependency on the robot, load the file directly.  `check_mathx.py`
already does exactly this to reach IsaacLab's math.py without booting a simulator.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_file(path: str | Path, name: str | None = None) -> ModuleType:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"cannot load module: {p}")
    mod_name = name or f"_onboard_{p.stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"no loader for {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod        # registered before exec so dataclasses/pickle resolve
    spec.loader.exec_module(mod)
    return mod
