"""Real camera -> exactly the arrays `course_mission.perception` was tuned on.

perception.py is 1652 lines of detector whose every constant -- colour prototypes, blob
span thresholds, lateral gates, the ray/ground unprojection -- was fitted against ONE
camera: 640x360, fx = fy = 488.7, principal point at the image centre, mounted at
(0.3381, 0.0336, 0.0525) in the base frame and pitched 22 deg down.  Hand it a different
field of view and none of those numbers mean what they meant, in ways that look like a
detector that has simply got worse rather than a geometry error.

So this module does not adapt the detector to the camera.  It adapts the camera to the
detector: whatever the real lens is, the frame handed on is remapped into that virtual
pinhole.  Unitree forward cameras are typically wide-angle, so expect an undistort-and-crop.

The other half is the pose.  perception works in the GRAVITY-ALIGNED BODY frame -- origin at
the base, x along the base's yaw, z up -- with the body's roll and pitch left inside the
camera rotation, which is precisely what an IMU gives you.  In simulation that frame is
produced by `cam_pose_body()` from two world poses we do not have on hardware.  Here it is
built from the IMU quaternion and the fixed mount, and `tools/check_camadapter.py` proves
the two agree.
"""
from __future__ import annotations

import numpy as np

from onboard.oio.iface import Frame

# ROS camera axes (x right, y down, z forward) expressed in the body convention
# (x forward, y left, z up).  Same matrix as mission.HeadCamMount._ROS_IN_WORLD; the
# detector's rays are built in ROS axes and rotated by cam_rot, so this must match or every
# unprojection is rotated 90 degrees and nothing is where it is reported to be.
ROS_IN_BODY = np.array([[0.0, 0.0, 1.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0]])


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def quat_to_mat(q) -> np.ndarray:
    """(w, x, y, z) -> 3x3.  Same convention as IsaacLab and the Unitree IMU."""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def yaw_of(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


class CameraAdapter:
    """Rectify frames into the virtual pinhole and compute the detector's camera pose."""

    def __init__(self, sensors_cfg) -> None:
        cam = sensors_cfg["camera"]
        vp = cam["virtual_pinhole"]
        self.width = int(vp["width"])
        self.height = int(vp["height"])
        self.K = np.array([[float(vp["fx"]), 0.0, float(vp["cx"])],
                           [0.0, float(vp["fy"]), float(vp["cy"])],
                           [0.0, 0.0, 1.0]], dtype=np.float64)

        ex = cam["extrinsics"]
        self.mount_p = np.asarray(ex["xyz"], dtype=np.float64)
        # Mount orientation in the BODY convention.  Positive pitch is nose-down, matching
        # launch.HEAD_ROT = quaternion about +y by +22 deg.
        self.mount_R = (rot_z(np.deg2rad(float(ex.get("yaw_deg", 0.0))))
                        @ rot_y(np.deg2rad(float(ex["pitch_deg"])))
                        @ rot_x(np.deg2rad(float(ex.get("roll_deg", 0.0)))))
        self.mount_R_ros = self.mount_R @ ROS_IN_BODY

        self.depth_source = str(cam["depth"]["source"])
        self.depth_units = float(cam["depth"].get("units_per_m", 1.0))
        self.max_range = float(cam["depth"].get("max_range_m", 10.0))

        src = cam.get("source") or {}
        self.src_model = src.get("model")
        self._map1 = self._map2 = None
        self._src_shape = None
        self._src = src

    # ---- pose -------------------------------------------------------------------------
    def pose(self, quat) -> tuple[np.ndarray, np.ndarray]:
        """IMU orientation -> (cam_p, cam_R) in the gravity-aligned body frame.

        Reproduces `perception.cam_pose_body` without needing a world pose.  Yaw is removed
        because the detector's measurements are all relative -- lateral offset from the
        belt, bearing to a bin -- and a heading we cannot observe absolutely would only
        inject drift into quantities that do not need it.  Roll and pitch stay, because
        they are real and they are what tilts the ground plane.
        """
        R_wb = quat_to_mat(quat)
        R_gb = rot_z(-yaw_of(R_wb)) @ R_wb          # roll/pitch only
        return R_gb @ self.mount_p, R_gb @ self.mount_R_ros

    @staticmethod
    def ground_z(body_height: float) -> float:
        """The detector's `ground_z` is the ground's height in the same base-origin frame
        as `cam_p`, so it is NEGATIVE the body height.  Getting the sign wrong here puts
        the belt plane above the camera and every fix disappears."""
        return -float(body_height)

    # ---- image ------------------------------------------------------------------------
    def rectify(self, rgb: np.ndarray, depth: np.ndarray | None = None):
        """Real image -> virtual-pinhole image.

        With no source model declared the frame is assumed already pinhole and is only
        resized, which is the identity when it is already 640x360 -- so the replay path and
        the sim-recorded frames go through untouched.
        """
        if rgb.shape[0] == self.height and rgb.shape[1] == self.width and not self.src_model:
            return rgb, depth
        if self.src_model in ("pinhole", "fisheye"):
            m1, m2 = self._remap_tables(rgb.shape)
            import cv2  # noqa: PLC0415  only needed on the undistort path
            rgb = cv2.remap(rgb, m1, m2, cv2.INTER_LINEAR)
            if depth is not None:
                # NEAREST for depth: interpolating across a depth discontinuity invents a
                # surface halfway between the object and the background.
                depth = cv2.remap(depth, m1, m2, cv2.INTER_NEAREST)
            return rgb, depth
        return _resize_nn(rgb, self.height, self.width), (
            None if depth is None else _resize_nn(depth, self.height, self.width))

    def _remap_tables(self, shape):
        if self._map1 is not None and self._src_shape == shape:
            return self._map1, self._map2
        import cv2  # noqa: PLC0415
        fx, fy, cx, cy = (float(v) for v in self._src["K"])
        Ks = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        d = np.asarray(self._src.get("dist") or [0, 0, 0, 0], dtype=np.float64)
        size = (self.width, self.height)
        if self.src_model == "fisheye":
            self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
                Ks, d[:4].reshape(4, 1), np.eye(3), self.K, size, cv2.CV_16SC2)
        else:
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                Ks, d, np.eye(3), self.K, size, cv2.CV_16SC2)
        self._src_shape = shape
        return self._map1, self._map2

    def scale_depth(self, depth: np.ndarray | None) -> np.ndarray | None:
        """SDK units -> metres, invalid -> NaN.  Zero is the usual 'no return' code and a
        0.0 m reading unprojects to a point inside the camera, which the detector will
        happily classify as an object at your feet."""
        if depth is None:
            return None
        d = np.asarray(depth, dtype=np.float32)
        if self.depth_units != 1.0:
            d = d / self.depth_units
        bad = (d <= 1e-3) | (d > self.max_range) | ~np.isfinite(d)
        if bad.any():
            d = d.copy()
            d[bad] = np.nan
        return d

    def prepare(self, frame: Frame, quat, body_height: float) -> dict:
        """Everything the detectors need, in one dict, in their units and their frame."""
        rgb, depth = self.rectify(frame.rgb, frame.depth)
        cam_p, cam_rot = self.pose(quat)
        return {
            "rgb": rgb,
            "depth": self.scale_depth(depth),
            "K": self.K,
            "cam_p": cam_p,
            "cam_rot": cam_rot,
            "ground_z": self.ground_z(body_height),
            "t": frame.t,
            "tick": frame.tick,
        }


def _resize_nn(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour resize with no OpenCV dependency, so the pure-numpy path stays
    pure numpy on a machine where cv2 may not be installed."""
    ys = (np.arange(h) * (img.shape[0] / h)).astype(np.int64).clip(0, img.shape[0] - 1)
    xs = (np.arange(w) * (img.shape[1] / w)).astype(np.int64).clip(0, img.shape[1] - 1)
    return img[ys][:, xs]
