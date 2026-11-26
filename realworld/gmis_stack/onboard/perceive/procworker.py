"""Perception in its own PROCESS, because a thread is not enough.

The worker started life as a thread and looked fine: the control loop held 50.00 Hz, the
detectors ran, nothing crashed.  Then a single 213 ms control tick showed up, once per run,
and none of the tick's own work accounted for it -- read, estimate, consume and supervise
together measured 0.25 ms on exactly the tick that took 213.

The time was spent not being scheduled.  `perception`'s connected-components and blob
routines are pure-Python loops, so they hold the GIL for as long as they run, and numpy only
releases it for the big vectorised calls in between.  A 200 ms detector frame in a thread is
therefore a 200 ms hole in the 50 Hz motor command stream -- ten missed commands, at a
moment chosen by whatever the detector happens to be looking at.

No amount of tuning fixes that inside one interpreter.  So the detector goes in a child
process: the GIL is no longer shared, and the parent's only perception work is draining
small result dicts off a queue.  The parent-side interface is unchanged -- same mailboxes,
same `report()` -- so `run.py` swaps one class for the other.

The pose the frame is unprojected with travels the other way through a 5-float shared array
(quaternion + body height), written by the control loop every tick.  It is deliberately not
a queue: the child wants the newest pose, never a backlog, and a torn read costs a fraction
of a degree of attitude on one frame.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time

import numpy as np

from onboard.rt.mailbox import Mailbox

_QUEUE_MAX = 8


class PerceptionProcess:
    def __init__(self, sensors_cfg, backend: str = "replay", frames_dir: str | None = None,
                 nic: str | None = None) -> None:
        self.cfg = sensors_cfg
        self.backend = backend
        self.frames_dir = frames_dir
        self.nic = nic

        p = sensors_cfg["perception"]
        max_age = float(p.get("max_age_s", 1.5))
        self.belt = Mailbox("belt", max_age)
        self.rail = Mailbox("rail", max_age)
        self.bins = Mailbox("bins", max_age * 4)
        self.objects = Mailbox("objects", max_age * 2)
        self._boxes = {"belt": self.belt, "rail": self.rail,
                       "bins": self.bins, "objects": self.objects}

        ctx = mp.get_context("spawn")     # fork would inherit torch's threads and the loop
        self._q: mp.Queue = ctx.Queue(maxsize=_QUEUE_MAX)
        self._pose = ctx.Array("d", [1.0, 0.0, 0.0, 0.0, 0.413], lock=False)
        self._stop = ctx.Event()
        self._proc: mp.Process | None = None
        self._drain: threading.Thread | None = None
        self._running = False
        self.errors = 0
        self.timing: dict[str, dict] = {}
        self.n_recv = 0

    # ---- lifecycle -----------------------------------------------------------------------
    def start(self) -> None:
        ctx = mp.get_context("spawn")
        self._proc = ctx.Process(target=_child_main, name="perception",
                                 args=(self.cfg, self.backend, self.frames_dir, self.nic,
                                       self._q, self._pose, self._stop), daemon=True)
        self._proc.start()
        self._running = True
        self._drain = threading.Thread(target=self._drain_loop, name="percep_drain",
                                       daemon=True)
        self._drain.start()

    def stop(self) -> None:
        """Shut down without deadlocking on the queue.

        A multiprocessing.Queue that still holds data keeps its feeder thread alive, and
        joining a child whose queue has not been drained hangs -- which it did: the run
        completed, printed nothing further, and sat there until it was killed.  So: tell the
        child to stop, keep draining while it winds down, then cancel the queue's join
        thread before touching the process at all.
        """
        self._stop.set()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and self._proc is not None and self._proc.is_alive():
            try:
                self._q.get(timeout=0.05)
            except (queue.Empty, EOFError, OSError):
                pass
        self._running = False
        try:
            self._q.cancel_join_thread()
            self._q.close()
        except Exception:  # noqa: BLE001
            pass
        if self._proc is not None:
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=1.0)
        if self._drain is not None:
            self._drain.join(timeout=1.0)

    def set_pose(self, quat, height: float) -> None:
        """Called from the control loop every tick.  Five float stores, no lock."""
        self._pose[0] = float(quat[0])
        self._pose[1] = float(quat[1])
        self._pose[2] = float(quat[2])
        self._pose[3] = float(quat[3])
        self._pose[4] = float(height)

    # ---- parent side -----------------------------------------------------------------------
    def _drain_loop(self) -> None:
        while self._running:
            try:
                msg = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                break
            kind = msg.get("kind")
            if kind == "timing":
                self.timing = msg["timing"]
                self.errors = msg.get("errors", self.errors)
                continue
            box = self._boxes.get(kind)
            if box is not None:
                box.put(msg["value"], t=msg["t"], tick=msg["tick"])
                self.n_recv += 1

    def report(self) -> dict:
        alive = self._proc is not None and self._proc.is_alive()
        return {"errors": self.errors, "alive": alive, "received": self.n_recv,
                "mailboxes": {n: b.report() for n, b in self._boxes.items()},
                **self.timing}

    @staticmethod
    def _warm() -> None:
        """Nothing to warm in the parent any more -- the detector never runs here.  Kept so
        `run.py` can call it uniformly for either worker."""
        return


# ==========================================================================================
# child process
# ==========================================================================================
def _child_main(cfg, backend, frames_dir, nic, q, pose, stop) -> None:
    import os

    os.environ.setdefault("MISSION_REAL_NOISE", "0")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "2")

    from onboard.perceive.camadapter import CameraAdapter
    from onboard.perceive.depthsynth import DepthSynth
    from onboard.perceive.railnav import RailDetector
    from onboard.rt.clock import pin_perception

    pin_perception()
    camera, lidar = _build_sources(cfg, backend, frames_dir, nic)
    camera.start()
    if lidar is not None:
        lidar.start()

    adapter = CameraAdapter(cfg)
    depth = DepthSynth(cfg, adapter.K, adapter.width, adapter.height)
    rails = RailDetector(cfg)

    from course_mission.perception import Perception, detect_belt, detect_bins

    p = cfg["perception"]
    detect_dt = 1.0 / float(p.get("detect_hz", 2.0))
    belt_dt = 1.0 / float(p.get("belt_hz", 10.0))
    timing = {k: [] for k in ("belt", "bins", "objects", "rail")}
    errors = 0
    cloud = None
    next_belt = next_obj = next_timing = 0.0

    def send(kind, value, t, tick) -> None:
        try:
            q.put_nowait({"kind": kind, "value": value, "t": t, "tick": tick})
        except queue.Full:
            pass          # the consumer wants the newest, never a backlog

    while not stop.is_set():
        now = time.monotonic()
        try:
            if lidar is not None:
                c = lidar.grab()
                if c is not None:
                    cloud = c.points
                    t0 = time.perf_counter()
                    fix = rails.detect(c.points)
                    timing["rail"].append(time.perf_counter() - t0)
                    if fix is not None:
                        send("rail", fix, c.t, c.tick)

            frame = camera.grab()
            if frame is None:
                time.sleep(0.002)
                continue

            quat = np.array([pose[0], pose[1], pose[2], pose[3]])
            prep = adapter.prepare(frame, quat, float(pose[4]))
            if prep["depth"] is None:
                prep["depth"] = depth.synth(adapter.depth_source, prep["cam_p"],
                                            prep["cam_rot"], prep["ground_z"], cloud)

            if now >= next_belt:
                next_belt = now + belt_dt
                t0 = time.perf_counter()
                b = detect_belt(prep["rgb"], prep["K"], prep["cam_p"], prep["cam_rot"],
                                prep["ground_z"], add_noise=False)
                timing["belt"].append(time.perf_counter() - t0)
                if b is not None:
                    send("belt", b, prep["t"], prep["tick"])
                t0 = time.perf_counter()
                try:
                    bn = detect_bins(prep["rgb"], prep["K"], prep["cam_p"], prep["cam_rot"],
                                     prep["ground_z"], add_noise=False, depth=prep["depth"])
                except Exception:  # noqa: BLE001
                    bn = None
                timing["bins"].append(time.perf_counter() - t0)
                if bn:
                    send("bins", bn, prep["t"], prep["tick"])

            if now >= next_obj:
                next_obj = now + detect_dt
                t0 = time.perf_counter()
                dets = Perception.detect_arrays(prep["rgb"], prep["depth"], prep["K"],
                                                prep["cam_p"], prep["cam_rot"],
                                                prep["ground_z"], add_noise=False)
                timing["objects"].append(time.perf_counter() - t0)
                send("objects", dets, prep["t"], prep["tick"])

            if now >= next_timing:
                next_timing = now + 5.0
                q.put_nowait({"kind": "timing", "errors": errors,
                              "timing": _summarise(timing)})
        except queue.Full:
            pass
        except Exception as e:  # noqa: BLE001
            errors += 1
            if errors <= 3:
                print(f"[perception] {type(e).__name__}: {e}", flush=True)
        time.sleep(0.001)

    camera.stop()
    if lidar is not None:
        lidar.stop()


def _build_sources(cfg, backend, frames_dir, nic):
    if backend == "real":
        from onboard.oio.a2_sensors import build_camera, build_lidar
        return build_camera(cfg), build_lidar(cfg)
    from onboard.oio.replay import ReplayCamera, ReplayLidar
    return ReplayCamera(cfg, frames_dir, fps=8.0), ReplayLidar()


def _summarise(timing: dict) -> dict:
    out = {}
    for k, v in timing.items():
        if v:
            a = np.asarray(v[-400:]) * 1e3
            out[f"{k}_ms"] = {"n": len(v), "p50": round(float(np.percentile(a, 50)), 2),
                              "p90": round(float(np.percentile(a, 90)), 2),
                              "max": round(float(a.max()), 2)}
    return out
