"""The nine IsaacLab math helpers the mission actually uses, without IsaacLab.

`runner.py`, `mission.py` and `perception.py` reach into `isaaclab.utils.math` in 28
places, and those 28 calls are the only reason two otherwise-portable modules need a
simulator installed. Isaac Lab does not install on a GPU-less mini-PC, so on the robot
this import is the difference between "the stack starts" and "the stack does not exist".

Everything here is quaternion algebra in Isaac Lab's convention -- **(w, x, y, z)**,
batched along dim 0 -- and is verified against the real implementations by
`scripts/check_mathx.py`, which imports both and compares on random inputs. Nothing in
this file touches PhysX, CUDA or the filesystem: it is torch-cpu arithmetic that happens
to have been living inside a simulator package.

Only what is used is here. Adding a helper means adding its check.
"""

from __future__ import annotations

import torch


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 * q2, both (..., 4) in (w, x, y, z)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ), dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Inverse of a quaternion — conjugate over the squared norm.

    The eps clamp matches Isaac Lab exactly; without it a zero quaternion divides by zero
    here and returns inf where the original returns a finite (meaningless) value, which is
    the difference between a NaN that propagates silently and one that does not.
    """
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by q. Uses the cross-product form, not a 3x3 build: fewer flops and it
    keeps the batch shape rules identical to Isaac Lab's."""
    shape = v.shape
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    xyz = q[:, 1:]
    t = xyz.cross(v, dim=-1) * 2.0
    return (v + q[:, 0:1] * t + xyz.cross(t, dim=-1)).reshape(shape)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by the inverse of q — i.e. world vector -> body frame."""
    shape = v.shape
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    xyz = -q[:, 1:]                      # conjugate; q is unit in every call site
    t = xyz.cross(v, dim=-1) * 2.0
    return (v + q[:, 0:1] * t + xyz.cross(t, dim=-1)).reshape(shape)


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) -> (..., 3, 3) rotation matrix."""
    shape = q.shape[:-1]
    qn = q.reshape(-1, 4)
    w, x, y, z = qn.unbind(-1)
    # SCALE-INVARIANT, like Isaac Lab's (which divides by the squared norm rather than
    # assuming unit length). Hardcoding 2.0 agrees exactly on unit input — so the original
    # numerical check passed — and diverges on anything else, which is precisely the case
    # a vendored copy exists to survive.
    two = 2.0 / qn.pow(2).sum(-1)
    m = torch.stack([
        1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w),
        two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w),
        two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y),
    ], dim=-1)
    return m.reshape(*shape, 3, 3)


def euler_xyz_from_quat(q: torch.Tensor, wrap_to_2pi: bool = False):
    """(roll, pitch, yaw) as a 3-tuple of (...,) tensors, extrinsic XYZ.

    Returned as a TUPLE, matching Isaac Lab: the mission indexes `[2][0]` for yaw and a
    stacked tensor would silently give a different number.

    RANGE IS (-pi, pi], NOT [0, 2pi). The first draft of this file wrapped to [0, 2pi)
    and every other helper still matched to 1e-16, so nothing would have looked broken —
    but `runner.base_pose()` feeds this yaw straight into heading arithmetic, and a 2*pi
    offset there turns every steering error into a full-circle correction. Caught only
    because scripts/check_mathx.py compares against the real implementation.
    """
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = torch.where(sinp.abs() >= 1.0,
                        torch.copysign(torch.full_like(sinp, torch.pi / 2), sinp),
                        torch.asin(sinp.clamp(-1.0, 1.0)))

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    if wrap_to_2pi:
        return roll % (2 * torch.pi), pitch % (2 * torch.pi), yaw % (2 * torch.pi)
    return roll, pitch, yaw


def combine_frame_transforms(t01: torch.Tensor, q01: torch.Tensor,
                             t12: torch.Tensor | None = None,
                             q12: torch.Tensor | None = None):
    """Compose 0->1 with 1->2, returning the 0->2 pose."""
    if t12 is not None:
        t02 = t01 + quat_apply(q01, t12)
    else:
        t02 = t01.clone()
    q02 = quat_mul(q01, q12) if q12 is not None else q01.clone()
    return t02, q02


def subtract_frame_transforms(t01: torch.Tensor, q01: torch.Tensor,
                              t02: torch.Tensor | None = None,
                              q02: torch.Tensor | None = None):
    """Express pose 0->2 relative to frame 1, i.e. the 1->2 pose."""
    q10 = quat_inv(q01)
    t12 = quat_apply(q10, t02 - t01) if t02 is not None else quat_apply(q10, -t01)
    q12 = quat_mul(q10, q02) if q02 is not None else q10
    return t12, q12


def skew_symmetric_matrix(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3) with M @ a == v x a."""
    shape = v.shape[:-1]
    vv = v.reshape(-1, 3)
    z = torch.zeros(vv.shape[0], device=v.device, dtype=v.dtype)
    m = torch.stack([
        z, -vv[:, 2], vv[:, 1],
        vv[:, 2], z, -vv[:, 0],
        -vv[:, 1], vv[:, 0], z,
    ], dim=-1)
    return m.reshape(*shape, 3, 3)
