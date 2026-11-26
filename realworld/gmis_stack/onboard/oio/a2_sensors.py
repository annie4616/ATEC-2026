"""A2 Pro on-board camera and LiDAR.

The SDK surface for these is REAL-HARDWARE CONFIRMATION territory: Unitree exposes the
forward camera over its own video channel on some platforms and as a plain V4L2 device on
others, and whether the A2's LiDAR is reachable at all through the Python SDK is unknown
until we have the robot.  So this file is a thin, swappable shim with three backends and a
declared preference, rather than a commitment to one API we cannot test.

Whatever the source turns out to be, `perceive/camadapter.py` rectifies it into the virtual
pinhole the detector was tuned on, so nothing downstream has to change.
"""
from __future__ import annotations

import time

import numpy as np

from onboard.oio.iface import CameraSource, Cloud, Frame, LidarSource, NullLidar


class V4l2Camera(CameraSource):
    """Plain UVC capture.  The fallback that works if the SDK does not cooperate: a USB
    camera on the mini-PC needs nothing from the robot at all."""

    def __init__(self, sensors_cfg, device: str | int = 0) -> None:
        cam = sensors_cfg["camera"]
        src = cam.get("source") or {}
        self.device = device
        self.width = int(src.get("width") or cam["virtual_pinhole"]["width"])
        self.height = int(src.get("height") or cam["virtual_pinhole"]["height"])
        self._cap = None
        self.tick = 0

    def start(self) -> None:
        import cv2  # noqa: PLC0415
        self._cap = cv2.VideoCapture(self.device)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # One-frame buffer: a queued frame is a frame whose pose stamp is already wrong.
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001
            pass

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def grab(self) -> Frame | None:
        if self._cap is None:
            return None
        ok, bgr = self._cap.read()
        if not ok:
            return None
        return Frame(t=time.monotonic(), tick=self.tick, rgb=bgr[:, :, ::-1].copy(),
                     depth=None, K=np.eye(3), ok=True)


class A2SdkCamera(CameraSource):
    """The robot's own forward camera through unitree_sdk2py's VideoClient.

    Returns JPEG-ish frames that have to be decoded, which is why the decode cost is
    measured rather than assumed -- on an N150 a 720p decode is not free relative to a 2 Hz
    detection budget.
    """

    def __init__(self, sensors_cfg, nic: str | None = None) -> None:
        self.nic = nic
        self._client = None
        self.tick = 0
        self.errors = 0

    def start(self) -> None:
        from unitree_sdk2py.go2.video.video_client import VideoClient  # noqa: PLC0415
        self._client = VideoClient()
        self._client.SetTimeout(3.0)
        self._client.Init()

    def stop(self) -> None:
        self._client = None

    def grab(self) -> Frame | None:
        if self._client is None:
            return None
        try:
            code, data = self._client.GetImageSample()
            if code != 0 or not data:
                return None
            import cv2  # noqa: PLC0415
            bgr = cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            return Frame(t=time.monotonic(), tick=self.tick, rgb=bgr[:, :, ::-1].copy(),
                         depth=None, K=np.eye(3), ok=True)
        except Exception:  # noqa: BLE001
            self.errors += 1
            return None


class A2SdkLidar(LidarSource):
    """Point cloud from the robot's LiDAR.

    Unknown until we have the hardware whether this is exposed at all.  If it is not,
    `build_lidar` hands back a NullLidar and navigation runs on the camera belt detector
    alone -- which is the path that was actually validated in simulation, so the degradation
    is a known quantity rather than a surprise.
    """

    def __init__(self, sensors_cfg, topic: str = "rt/utlidar/cloud") -> None:
        self.topic = topic
        self._sub = None
        self._latest = None
        self.tick = 0

    def start(self) -> None:
        from unitree_sdk2py.core.channel import ChannelSubscriber  # noqa: PLC0415
        from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_  # noqa: PLC0415
        self._sub = ChannelSubscriber(self.topic, PointCloud2_)
        self._sub.Init(self._on_cloud, 5)

    def stop(self) -> None:
        if self._sub is not None:
            self._sub.Close()

    def _on_cloud(self, msg) -> None:
        try:
            pts = np.frombuffer(bytes(msg.data), np.float32)
            stride = int(msg.point_step // 4)
            self._latest = pts.reshape(-1, stride)[:, :3].astype(np.float64)
        except Exception:  # noqa: BLE001
            self._latest = None

    def grab(self) -> Cloud | None:
        c = self._latest
        self._latest = None            # consume: a repeat is not a new measurement
        if c is None:
            return None
        return Cloud(t=time.monotonic(), tick=self.tick, points=c, ok=True)


def build_camera(sensors_cfg) -> CameraSource:
    b = str(sensors_cfg["camera"].get("backend", "a2_sdk"))
    if b == "v4l2":
        return V4l2Camera(sensors_cfg, sensors_cfg["camera"].get("device") or 0)
    if b == "a2_sdk":
        return A2SdkCamera(sensors_cfg)
    from onboard.oio.replay import ReplayCamera
    return ReplayCamera(sensors_cfg)


def build_lidar(sensors_cfg) -> LidarSource:
    b = str((sensors_cfg.get("lidar") or {}).get("backend", "none"))
    if b == "a2_sdk":
        return A2SdkLidar(sensors_cfg)
    if b == "replay":
        from onboard.oio.replay import ReplayLidar
        return ReplayLidar()
    return NullLidar()
