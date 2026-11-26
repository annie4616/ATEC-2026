"""Depth for a camera that may not have any.

REAL-HARDWARE CONFIRMATION #1 is whether the A2 Pro's built-in forward camera returns depth
at all.  Unitree quadrupeds typically pair a wide-angle RGB camera with a separate 3D LiDAR
rather than an RGB-D module, so "no depth channel" is the likely answer, and three of the
detectors (`detect_arrays`, `detect_posts`, `detect_corridor`) take a depth image.

Rather than fork perception.py, synthesise the array it expects.  Two sources, in order of
preference:

  lidar   Project the point cloud into the virtual pinhole.  Real range, sparse (a spinning
          LiDAR gives thousands of points against 230k pixels), so it is dilated to fill.
  ground  Intersect every ray with the estimated ground plane.  Exact for anything lying on
          the ground -- which is every target object in this task -- and wrong for anything
          standing up, where it reports the range to the ground BEHIND the object.

The ground fallback is not a hack bolted on for hardware: perception.py already prefers
ray/ground intersection over the depth channel for position (`detect_belt` uses no depth at
all, by design, because "the depth camera is the sensor we trust least outdoors").  This
just makes that preference total when it has to be.
"""
from __future__ import annotations

import numpy as np


def ray_directions(K: np.ndarray, width: int, height: int) -> np.ndarray:
    """Unit-z camera rays for every pixel, in ROS camera axes.  Cached by the caller: this
    is 230k pixels of trig-free arithmetic but it is still 3 MB of float64 per call."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.arange(width, dtype=np.float64)
    v = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    return np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1)


class DepthSynth:
    def __init__(self, sensors_cfg, K: np.ndarray, width: int, height: int) -> None:
        self.K = np.asarray(K, dtype=np.float64)
        self.w, self.h = int(width), int(height)
        self.max_range = float(sensors_cfg["camera"]["depth"].get("max_range_m", 10.0))
        self._rays = ray_directions(self.K, self.w, self.h)
        # Hole-fill radius for the projected cloud.  Measured on 80 real frames at 32 scan
        # lines: k=1 retains 54.1 % of the depth-camera's detections, k=2 56.8, k=3 59.5,
        # k=4 67.6, k=5 73.0.  It rises because a sparse cloud leaves an object's blob full
        # of NaN and the detector needs enough valid pixels inside the blob to measure it.
        self.dilate = int(sensors_cfg["camera"]["depth"].get("lidar_dilate", 5))
        li = sensors_cfg.get("lidar") or {}
        ex = (li.get("extrinsics") or {})
        self.lidar_p = np.asarray(ex.get("xyz", [0.0, 0.0, 0.0]), dtype=np.float64)
        self.lidar_rpy = np.asarray(ex.get("rpy", [0.0, 0.0, 0.0]), dtype=np.float64)

    # ---- ground-plane depth -------------------------------------------------------------
    def from_ground(self, cam_p: np.ndarray, cam_R: np.ndarray, ground_z: float) -> np.ndarray:
        """z-depth to the ground plane for every pixel.

        Returns DISTANCE TO THE IMAGE PLANE, not radial range, because that is what the
        simulator's `distance_to_image_plane` channel is and what the detectors unproject
        with.  Getting that wrong scales every range by 1/cos(angle from the optical axis)
        -- up to 20 % at the frame corners, which is exactly the size of error that makes a
        grasp miss without ever looking like a bug.

        Rays that point up, or hit the plane beyond max_range, come back NaN: "no return"
        is information, and a fabricated 10 m is not.
        """
        rays_b = self._rays @ np.asarray(cam_R, dtype=np.float64).T     # camera -> body
        dz = rays_b[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (float(ground_z) - float(cam_p[2])) / dz                # ray scale
        t = np.where(dz < -1e-6, t, np.nan)                             # must point down
        # `t` scales a ray whose camera-z component is 1, so t IS the z-depth.
        d = np.where(np.isfinite(t) & (t > 0.05) & (t < self.max_range), t, np.nan)
        return d.astype(np.float32)

    # ---- LiDAR-projected depth ----------------------------------------------------------
    def from_lidar(self, points_sensor: np.ndarray, cam_p: np.ndarray, cam_R: np.ndarray,
                   dilate: int | None = None) -> np.ndarray:
        """Project a cloud into the pinhole -> sparse z-depth, dilated to fill.

        `points_sensor` is in the LiDAR's own frame (x forward, y left, z up), which is what
        the device natively provides -- no world pose, no map.
        """
        P = np.asarray(points_sensor, dtype=np.float64)
        if P.size == 0:
            return np.full((self.h, self.w), np.nan, np.float32)

        R_bl = _rpy_to_mat(self.lidar_rpy)
        p_body = P @ R_bl.T + self.lidar_p                       # LiDAR -> body
        p_cam = (p_body - np.asarray(cam_p, float)) @ np.asarray(cam_R, float)  # body -> cam

        z = p_cam[:, 2]
        keep = (z > 0.15) & (z < self.max_range)
        if not keep.any():
            return np.full((self.h, self.w), np.nan, np.float32)
        p_cam, z = p_cam[keep], z[keep]

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        u = np.rint(p_cam[:, 0] / z * fx + cx).astype(np.int64)
        v = np.rint(p_cam[:, 1] / z * fy + cy).astype(np.int64)
        inside = (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        u, v, z = u[inside], v[inside], z[inside]
        if u.size == 0:
            return np.full((self.h, self.w), np.nan, np.float32)

        # Nearest surface wins per pixel: sort far-to-near so the nearest write lands last.
        order = np.argsort(-z)
        depth = np.full((self.h, self.w), np.nan, np.float32)
        depth[v[order], u[order]] = z[order].astype(np.float32)
        k = self.dilate if dilate is None else int(dilate)
        return _dilate_nanmin(depth, k) if k >= 1 else depth

    def synth(self, source: str, cam_p, cam_R, ground_z: float,
              cloud: np.ndarray | None = None) -> np.ndarray | None:
        """Dispatch on sensors.yaml's `camera.depth.source`.

        SPARSE LIDAR IS LEFT SPARSE.  The first version of this filled the LiDAR's holes
        with the ground plane, on the reasoning that a NaN pixel makes the detector drop a
        blob while the plane is right for anything lying on the ground.  Measured against
        80 real recorded frames, that reasoning is backwards:

            true depth                        100.0 %  of detections retained
            LiDAR 16 lines, holes left NaN     48.6 %
            LiDAR 16 lines + ground-plane fill 18.9 %      <- the "improvement"
            LiDAR 32 lines, holes left NaN     56.8 %
            LiDAR 32 lines + ground-plane fill 21.6 %
            ground plane alone                 32.4 %

        The fill is worse than either input alone because `detect_arrays` discriminates on
        HEIGHT ABOVE GROUND: a plane-filled pixel is by construction at zero height, so
        filling an object's holes with the plane flattens the very object the detector is
        looking for.  A NaN is honest -- the detector treats it as no information -- and a
        plane sample is a confident wrong answer.
        """
        if source == "lidar" and cloud is not None and len(cloud):
            return self.from_lidar(cloud, cam_p, cam_R)
        if source in ("ground", "lidar"):
            # `lidar` with no cloud this tick: fall back rather than blind the detector.
            return self.from_ground(cam_p, cam_R, ground_z)
        return None


def _rpy_to_mat(rpy) -> np.ndarray:
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _dilate_nanmin(d: np.ndarray, k: int) -> np.ndarray:
    """Grow each sparse return over a (2k+1) square, keeping the NEAREST value.

    Nearest rather than mean, because a pixel straddling an object edge should report the
    object, not a blend of the object and the ground three metres behind it.

    Separable: min over a square window is min-over-rows then min-over-columns, so this is
    2*(2k+1) shifted comparisons instead of (2k+1)^2.  At the k=5 that the measurements
    favour that is 22 passes instead of 121, which is the difference between the hole-fill
    being free and it being comparable to the detector it feeds.
    """
    if k < 1:
        return d
    out = _min1d(d, k, axis=0)
    return _min1d(out, k, axis=1)


def _min1d(d: np.ndarray, k: int, axis: int) -> np.ndarray:
    out = d.copy()
    for s in range(1, k + 1):
        for sh in (s, -s):
            m = np.full_like(d, np.nan)
            if axis == 0:
                a, b = max(0, sh), d.shape[0] + min(0, sh)
                m[a:b, :] = d[a - sh:b - sh, :]
            else:
                a, b = max(0, sh), d.shape[1] + min(0, sh)
                m[:, a:b] = d[:, a - sh:b - sh]
            out = np.fmin(out, m)
    return out
