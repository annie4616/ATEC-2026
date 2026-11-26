"""Process-wide runtime configuration, applied on the first `import onboard.*`.

Two settings have to be right *before* anything else loads, and both are the kind of thing
that is invisible when wrong:

**Thread count.**  torch defaults to one compute thread per core.  On this dev box that is
192; on the N150 it is 4 -- and those 4 are the same cores perception is running on.  A
three-layer MLP gains nothing from parallelism (the leg step measures 153 us single-
threaded) and loses plenty to contention with the detector.  Setting it after the policy is
constructed is too late: any torch op executed during loading already used every core.

**Sensor-noise injection.**  `course_mission/deploy.py` reads `MISSION_REAL_NOISE` at
import time and `perception.py` binds it as a module constant, so setting it later has no
effect at all -- a fact that already produced one wrong measurement (a verification agent
toggled it inside a live process and concluded the noise was nearly free).  It is a pure
simulator cost: on real frames it adds `np.random.normal` over the whole image and makes
belt/bin detection about 3.5x slower for nothing.  On the robot it must be off, and this is
the only place early enough to turn it off.

Overridable from the environment so a diagnostic run can put them back.
"""
from __future__ import annotations

import os

# The robot's camera supplies its own noise; we do not need to add any.
os.environ.setdefault("MISSION_REAL_NOISE", "0")
os.environ.setdefault("MISSION_COMPUTE", "n150")

_THREADS = os.environ.setdefault("ONBOARD_TORCH_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

try:  # torch is present on the robot, but the config layer must import without it
    import torch as _torch

    _torch.set_num_threads(int(_THREADS))
    _torch.set_num_interop_threads(1)
except (ImportError, RuntimeError):
    # RuntimeError: interop threads can only be set once per process; harmless if a
    # parent already did it.
    pass
