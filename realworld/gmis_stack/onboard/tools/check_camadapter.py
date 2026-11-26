#!/usr/bin/env python3
"""Prove the hardware camera pose equals the simulator's.

`perception.cam_pose_body` takes two WORLD poses.  On the robot there is no world pose, so
CameraAdapter.pose() builds the same frame from the IMU quaternion and the fixed mount.
"Same" has to be demonstrated, not asserted: a silent disagreement here rotates every
unprojected detection, and the symptom is a detector that finds the right colours in the
wrong places -- which reads as a detector problem for as long as you let it.

Deploy-side twin of scripts/check_mathx.py and scripts/check_armkin.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onboard import cfg as C  # noqa: E402
from onboard.perceive.camadapter import CameraAdapter, quat_to_mat  # noqa: E402
from course_mission.perception import cam_pose_body  # noqa: E402


def main(n: int = 500, seed: int = 0) -> int:
    ad = CameraAdapter(C.load("sensors"))
    rng = np.random.default_rng(seed)
    ep, eR = [], []
    for _ in range(n):
        # A plausible body attitude: full yaw, and roll/pitch out to +-25 deg, which covers
        # the stairs and the grass slope with room to spare.
        yaw = rng.uniform(-np.pi, np.pi)
        roll = rng.uniform(-0.44, 0.44)
        pitch = rng.uniform(-0.44, 0.44)
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        q = np.array([cr * cp * cy + sr * sp * sy,
                      sr * cp * cy - cr * sp * sy,
                      cr * sp * cy + sr * cp * sy,
                      cr * cp * sy - sr * sp * cy])
        R_wb = quat_to_mat(q)
        base_p = rng.uniform(-20, 20, 3)

        # The simulator's route: build the mount's WORLD pose, then project to body frame.
        mount_p_w = base_p + R_wb @ ad.mount_p
        mount_R_ros_w = R_wb @ ad.mount_R_ros
        p_ref, R_ref = cam_pose_body(mount_p_w, mount_R_ros_w, base_p, yaw)

        # The robot's route: IMU quaternion + fixed mount, no world pose anywhere.
        p_got, R_got = ad.pose(q)

        ep.append(float(np.abs(p_got - p_ref).max()))
        eR.append(float(np.abs(R_got - R_ref).max()))

    ep, eR = np.asarray(ep), np.asarray(eR)
    print(f"camera pose vs perception.cam_pose_body over {n} attitudes:")
    print(f"  position  max |diff| = {ep.max():.3e} m")
    print(f"  rotation  max |diff| = {eR.max():.3e}")
    ok = ep.max() < 1e-9 and eR.max() < 1e-9
    print("CAMADAPTER OK" if ok else "CAMADAPTER MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
