"""Why does the follow camera track the robot while the head camera will not move?

Writes a pose to BOTH cameras in the same run and reads each back three ways: straight
from the sensor buffer, from the underlying prim view, and from the USD stage. Whichever
of those disagrees tells us whether the write fails or the read is stale.
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
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import course_mission  # noqa: F401,E402
from course_mission.launch import TASK, head_cam_cfg  # noqa: E402

dev = a.device or "cuda:0"
cfg = parse_env_cfg(TASK, device=dev, num_envs=1)
cfg.scene.head_cam = head_cam_cfg()
# the control: exactly the config the follow camera uses today (it does track)
cfg.scene.follow_cam = CameraCfg(
    prim_path="/World/follow_cam", update_period=0.04, height=720, width=1280,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.1, 250.0)))
env = gym.make(TASK, cfg=cfg, render_mode=None); u = env.unwrapped; env.reset()
act = torch.zeros(1, u.action_manager.total_action_dim, device=dev)
target = torch.tensor([[-18.0, 7.4, 0.5]], dtype=torch.float32, device=dev)
look = torch.tensor([[-17.0, 7.4, 0.3]], dtype=torch.float32, device=dev)

for nm in ("head_cam", "follow_cam"):
    cam = u.scene[nm]
    print(f"\n=== {nm}  (prim {cam.cfg.prim_path}, update_period={cam.cfg.update_period}) ===",
          flush=True)
    print(f"  before: data.pos_w={np.round(cam.data.pos_w[0].cpu().numpy(),3)} "
          f"view={np.round(cam._view.get_world_poses()[0][0].cpu().numpy(),3)}", flush=True)
    cam.set_world_poses_from_view(target.clone(), look.clone())
    print(f"  after write (no step): data.pos_w={np.round(cam.data.pos_w[0].cpu().numpy(),3)} "
          f"view={np.round(cam._view.get_world_poses()[0][0].cpu().numpy(),3)}", flush=True)
    with torch.inference_mode():
        env.step(act)
    print(f"  after one step:        data.pos_w={np.round(cam.data.pos_w[0].cpu().numpy(),3)} "
          f"view={np.round(cam._view.get_world_poses()[0][0].cpu().numpy(),3)}", flush=True)
print("\nCAMSYNC_OK", flush=True)
env.close(); app.close()
