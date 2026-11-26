"""How much does the robot tremble standing still, and how much of it is the noise?

Stands with a zero command and measures base motion and joint velocity, with the
deployment noise on and off, in one process so the comparison is exact.
"""
import sys
from pathlib import Path

_S = Path(__file__).resolve()
RW = _S.parents[2]
for _p in (str(RW / "source" / "atec_rl_lab"), str(RW / "eunha_rl"), str(RW / "a2pro_rl"), str(RW)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

# Shared machine: GPUs 1-7 are other people's training runs. Pin to GPU 0 BEFORE any
# CUDA-aware import so forgetting the CUDA_VISIBLE_DEVICES prefix cannot land this on
# someone else's device.
import os as _os
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from isaaclab.app import AppLauncher  # noqa: E402
import argparse  # noqa: E402

parser = argparse.ArgumentParser()
for k, d in (("--speed", "s075"), ("--depth", "0.24"), ("--walk_ckpt", ""), ("--grasp_ckpt", "")):
    parser.add_argument(k, type=str, default=d)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--per_type", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
a = parser.parse_args(); a.enable_cameras = True
app = AppLauncher(a).app

import numpy as np, torch, gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import course_mission  # noqa: F401,E402
import course_mission.deploy as deploy  # noqa: E402
from course_mission.launch import TASK, head_cam_cfg, resolve_ckpts  # noqa: E402
from course_mission.mission import LegDriver  # noqa: E402

dev = a.device or "cuda:0"
cfg = parse_env_cfg(TASK, device=dev, num_envs=1)
cfg.scene.head_cam = head_cam_cfg()
env = gym.make(TASK, cfg=cfg, render_mode=None); u = env.unwrapped; env.reset()
robot = u.scene["robot"]
ck = resolve_ckpts(a)
legs = LegDriver(robot, dev, ck["walk"], ck["crouch"], ck["adapt"])
act = torch.zeros(1, u.action_manager.total_action_dim, device=dev)
zero = torch.zeros(1, 3, device=dev)

def measure(tag, steps=500):
    import course_mission.mission as mission
    pos, qd, act_delta = [], [], []
    prev = None
    with torch.inference_mode():
        for i in range(steps):
            a12 = legs.step(zero)
            act[:, :12] = a12
            env.step(act)
            if i < 120:                      # let it settle first
                prev = a12.clone(); continue
            d = robot.data
            pos.append(d.root_pos_w[0].cpu().numpy().copy())
            qd.append(float(d.joint_vel[0, legs.leg_ids].abs().mean()))
            if prev is not None:
                act_delta.append(float((a12 - prev).abs().mean()))
            prev = a12.clone()
    p = np.array(pos)
    print(f"[IDLE] {tag:14s} base xy wander {1000*np.linalg.norm(p[:,:2].std(axis=0)):6.1f} mm | "
          f"base z std {1000*p[:,2].std():5.1f} mm | mean |joint vel| {np.mean(qd):5.3f} rad/s | "
          f"action chatter {np.mean(act_delta):.4f}", flush=True)

measure("noise ON")
deploy.REAL_NOISE = False
import course_mission.mission as m
m.REAL_NOISE = False
measure("noise OFF")
deploy.REAL_NOISE = True
m.REAL_NOISE = True
measure("noise ON again")
print("IDLE_DONE", flush=True)
env.close(); app.close()
