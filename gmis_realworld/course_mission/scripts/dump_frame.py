"""Dump one head-camera frame (rgb + depth + intrinsics + poses) to an npz so the
unprojection can be debugged offline without holding a GPU."""
import argparse, sys
from pathlib import Path
_S = Path(__file__).resolve(); RW = _S.parents[2]
for _p in (str(RW/"source"/"atec_rl_lab"), str(RW/"eunha_rl"), str(RW/"a2pro_rl"), str(RW)):
    if _p in sys.path: sys.path.remove(_p)
    sys.path.insert(0, _p)
# Shared machine: GPUs 1-7 are other people's training runs. Pin to GPU 0 BEFORE any
# CUDA-aware import so forgetting the CUDA_VISIBLE_DEVICES prefix cannot land this on
# someone else's device.
import os as _os
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--per_type", type=int, default=4)
parser.add_argument("--steps", type=int, default=250)
AppLauncher.add_app_launcher_args(parser)
a = parser.parse_args(); a.enable_cameras = True
app = AppLauncher(a).app
import numpy as np, torch, gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.utils import parse_env_cfg
import course_mission
from course_mission.launch import head_cam_cfg, HEAD_POS, HEAD_ROT, TASK
cfg = parse_env_cfg(TASK, device=a.device or "cuda:0", num_envs=1)
cfg.layout_seed = a.seed; cfg.per_type = a.per_type
cfg.scene.head_cam = head_cam_cfg()
env = gym.make(TASK, cfg=cfg, render_mode=None); u = env.unwrapped; env.reset()
act = torch.zeros(1, u.action_manager.total_action_dim, device=u.device)
for _ in range(a.steps): env.step(act)
cam = u.scene["head_cam"]; rb = u.scene["robot"]
o = {n: u.scene[n].data.root_pos_w[0].cpu().numpy() for n, *_ in cfg.mission_layout}
np.savez(str(RW/"course_mission"/"logs"/"frame.npz"),
    rgb=cam.data.output["rgb"][0, ..., :3].cpu().numpy(),
    dip=cam.data.output["distance_to_image_plane"][0].cpu().numpy(),
    dtc=cam.data.output["distance_to_camera"][0].cpu().numpy(),
    K=cam.data.intrinsic_matrices[0].cpu().numpy(),
    cam_pos=cam.data.pos_w[0].cpu().numpy(),
    quat_ros=cam.data.quat_w_ros[0].cpu().numpy(),
    quat_world=cam.data.quat_w_world[0].cpu().numpy(),
    quat_opengl=cam.data.quat_w_opengl[0].cpu().numpy(),
    root_pos=rb.data.root_pos_w[0].cpu().numpy(),
    root_quat=rb.data.root_quat_w[0].cpu().numpy(),
    names=np.array(list(o.keys())), objs=np.array(list(o.values())))
print("[DUMP] saved frame.npz", flush=True)
env.close(); app.close()
