"""Config loading, and the repo-relative path rule.

Nothing under onboard/ may contain an absolute path.  39 of the 40 shell scripts in
course_mission/scripts/ hardcode this server's home directory, and solution_v5_venue.py
pushes four sibling trees onto sys.path -- fine here, useless on the robot, where none of
those directories exist.  This module is the only place that resolves a location.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ONBOARD = Path(__file__).resolve().parent
REPO = ONBOARD.parent


def onboard_path(rel: str | os.PathLike) -> Path:
    """Resolve a path stated relative to onboard/.  Absolute inputs are rejected loudly:
    a config that works only on the machine that wrote it is worse than a crash."""
    p = Path(rel)
    if p.is_absolute():
        raise ValueError(f"absolute path in onboard config: {rel}")
    return ONBOARD / p


class Cfg(dict):
    """dict with attribute access, so `cfg.control.rate_hz` reads like the yaml looks."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v

    def get_path(self, k: str) -> Path:
        return onboard_path(self[k])


def load(name: str) -> Cfg:
    """Load onboard/configs/<name>.yaml (or a path relative to onboard/)."""
    p = onboard_path(name if "/" in str(name) else f"configs/{name}.yaml")
    with p.open() as fh:
        return Cfg(yaml.safe_load(fh))


def load_all() -> Cfg:
    return Cfg({n: load(n) for n in ("a2pro", "piperx", "sensors", "policies", "mission_hk")})
