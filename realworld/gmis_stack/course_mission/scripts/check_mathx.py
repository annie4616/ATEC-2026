"""Prove course_mission/mathx.py is bit-comparable to isaaclab.utils.math.

Run in the isaaclab env (no GPU, no SimulationApp needed — isaaclab.utils.math imports
only numpy/torch):

    /home/yewon/.conda/envs/isaaclab/bin/python course_mission/scripts/check_mathx.py

A vendored maths library that is subtly wrong is worse than no vendored library: every
consumer keeps running and every frame transform is quietly off. This compares all nine
helpers on random batched inputs and reports the worst absolute deviation.
"""
import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("mathx", _HERE / "mathx.py")
mx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mx)                      # by path: no course_mission/__init__

# Load IsaacLab's math.py BY PATH. `import isaaclab.utils.math` runs isaaclab/__init__,
# which refuses to load without a live SimulationApp — but math.py itself imports only
# logging/math/typing/numpy/torch, so the reference implementation is reachable without
# starting a simulator at all. That is also the point being proved.
_REF = Path("/hdd0/yewon/ATEC2026_Simulation_Challenge/IsaacLab/source/isaaclab/isaaclab/utils/math.py")
_rspec = importlib.util.spec_from_file_location("isaaclab_math_ref", _REF)
ref = importlib.util.module_from_spec(_rspec)
_rspec.loader.exec_module(ref)

torch.manual_seed(0)
N = 512


def rq(n=N):
    q = torch.randn(n, 4, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def rv(n=N):
    return torch.randn(n, 3, dtype=torch.float64)


def cmp(name, a, b):
    if isinstance(a, tuple):
        d = max(float((x - y).abs().max()) for x, y in zip(a, b))
    else:
        d = float((a - b).abs().max())
    print(f"{name:28s} max|diff| = {d:.3e}   {'OK' if d < 1e-9 else 'MISMATCH'}")
    return d < 1e-9


q1, q2, v = rq(), rq(), rv()
ok = True
ok &= cmp("quat_mul", mx.quat_mul(q1, q2), ref.quat_mul(q1, q2))
ok &= cmp("quat_inv", mx.quat_inv(q1), ref.quat_inv(q1))
ok &= cmp("quat_apply", mx.quat_apply(q1, v), ref.quat_apply(q1, v))
ok &= cmp("quat_apply_inverse", mx.quat_apply_inverse(q1, v), ref.quat_apply_inverse(q1, v))
ok &= cmp("matrix_from_quat", mx.matrix_from_quat(q1), ref.matrix_from_quat(q1))
ok &= cmp("euler_xyz_from_quat", mx.euler_xyz_from_quat(q1), ref.euler_xyz_from_quat(q1))
ok &= cmp("skew_symmetric_matrix", mx.skew_symmetric_matrix(v), ref.skew_symmetric_matrix(v))
t1, t2 = rv(), rv()
a = mx.combine_frame_transforms(t1, q1, t2, q2)
b = ref.combine_frame_transforms(t1, q1, t2, q2)
ok &= cmp("combine_frame_transforms.t", a[0], b[0])
ok &= cmp("combine_frame_transforms.q", a[1], b[1])
a = mx.subtract_frame_transforms(t1, q1, t2, q2)
b = ref.subtract_frame_transforms(t1, q1, t2, q2)
ok &= cmp("subtract_frame_transforms.t", a[0], b[0])
ok &= cmp("subtract_frame_transforms.q", a[1], b[1])
# the no-second-frame overloads the mission relies on
a = mx.combine_frame_transforms(t1, q1)
b = ref.combine_frame_transforms(t1, q1)
ok &= cmp("combine (t only).t", a[0], b[0])
a = mx.subtract_frame_transforms(t1, q1)
b = ref.subtract_frame_transforms(t1, q1)
ok &= cmp("subtract (t only).t", a[0], b[0])

# NON-UNIT input. The first version of mathx.matrix_from_quat hardcoded the factor 2.0
# (valid only for unit quaternions) and every unit-input comparison above still matched to
# 1e-16, so the test passed on a function that was wrong for anything else.
qn = torch.randn(N, 4, dtype=torch.float64) * 1.7
ok &= cmp("matrix_from_quat (non-unit)", mx.matrix_from_quat(qn), ref.matrix_from_quat(qn))
ok &= cmp("quat_inv (non-unit)", mx.quat_inv(qn), ref.quat_inv(qn))
ok &= cmp("matrix_from_quat(quat_inv)", mx.matrix_from_quat(mx.quat_inv(qn)),
          ref.matrix_from_quat(ref.quat_inv(qn)))

print("ALL MATCH" if ok else "SOME MISMATCH")
sys.exit(0 if ok else 1)
