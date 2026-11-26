"""TEMP standalone test: can a closed-gripper 'flip' stand up a lying sugar box?

Forces object_1 to spawn flat, then runs a scripted press-and-roll flip with the
gripper closed, recording a video. Throwaway diagnostic — safe to delete.

Run:
  CUDA_VISIBLE_DEVICES=7 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=/hdd0/sohyun \
  TMPDIR=/hdd0/sohyun/atec_2026_tesk_e/tmp \
  python scripts/test_flip.py --headless --enable_cameras
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import numpy as np
import torch
import imageio

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv

from atec_rl_lab.tasks.task_e.env_cfg import TaskEEnvPiperCfg, TABLE_TOP_Z, TABLE_CENTER_X, TABLE_CENTER_Y
from atec_rl_lab.utils import CartesianController

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "act"))
from task_e.config import (
    EE_BODY_NAME, ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES,
    GRIPPER_CLOSE_POS, GRIPPER_OPEN_POS, ACTION_SCALE, CARRY_Z,
    DEFAULT_PLACE_QUAT_W, ACT_STIFFNESS, ACT_DAMPING, ACT_EFFORT_LIMIT, ACT_VEL_LIMIT,
)

DEV = "cuda"

# ---- build env (1 env, actuator overrides, camera) ----
cfg = TaskEEnvPiperCfg()
cfg.seed = 12345
cfg.scene.num_envs = 1
cfg.episode_length_s = 60.0
cfg.scene.robot.actuators["default"] = ImplicitActuatorCfg(
    joint_names_expr=[".*"],
    effort_limit=ACT_EFFORT_LIMIT, velocity_limit=ACT_VEL_LIMIT,
    stiffness=ACT_STIFFNESS, damping=ACT_DAMPING,
)
env = ManagerBasedRLEnv(cfg)
robot = env.unwrapped.scene.articulations["robot"]
cam = env.unwrapped.scene["video_cam"]
arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
gripper_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)
default_jpos = robot.data.default_joint_pos.clone()

ik = CartesianController(
    robot=robot, ee_body_name=EE_BODY_NAME, arm_joint_names=ARM_JOINT_NAMES,
    num_envs=1, device=DEV, command_type="position", lambda_val=0.05, max_joint_delta=0.2,
)

env.reset()
robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))

# ---- force object_1 FLAT (largest face down) near table center ----
obj = env.unwrapped.scene.rigid_objects["object_1"]
state = obj.data.default_root_state[0:1].clone()
state[0, 0] = TABLE_CENTER_X
state[0, 1] = TABLE_CENTER_Y + 0.20
state[0, 2] = TABLE_TOP_Z + 0.02   # rest on largest (thin) face -> ~0.039 tall (truly flat)
# try to lay on the LARGEST face (90deg about X) so half_h ~0.02 (vs edge ~0.045)
state[0, 3:7] = torch.tensor([0.707, 0.707, 0.0, 0.0], device=DEV)
state[0, 7:] = 0.0
obj.write_root_state_to_sim(state)
env.unwrapped.scene.write_data_to_sim()
env.unwrapped.sim.forward()
robot.update(dt=env.unwrapped.physics_dt)
ik.reset()

frames = []
DEF_Q = torch.tensor([DEFAULT_PLACE_QUAT_W], dtype=torch.float32, device=DEV)
g_close = torch.tensor([GRIPPER_CLOSE_POS], dtype=torch.float32, device=DEV)
g_open = torch.tensor([GRIPPER_OPEN_POS], dtype=torch.float32, device=DEV)


def drive(ee_pos, n, grip, quat=DEF_Q, record=False, label=None):
    ee = torch.tensor([ee_pos], dtype=torch.float32, device=DEV)
    minz = 9.9
    for _ in range(n):
        arm_des = ik.compute(ee, quat)
        tgt = robot.data.joint_pos.clone()
        tgt[:, arm_ids] = arm_des
        tgt[:, gripper_ids] = grip
        env.step((tgt - default_jpos) / ACTION_SCALE)
        robot.update(dt=env.unwrapped.physics_dt)
        minz = min(minz, float(ik.ee_pos_w[0, 2].item()))
        if record:
            frames.append(cam.data.output["rgb"][0, :, :, :3].cpu().numpy())
    if label:
        print(f"[FLIP] {label}: min gripper_base z={minz:.3f} (table top={float(TABLE_TOP_Z):.3f}, "
              f"=> {minz - float(TABLE_TOP_Z):+.3f} above table)", flush=True)


# go home (not recorded)
home = [TABLE_CENTER_X + 0.45, TABLE_CENTER_Y, CARRY_Z]

# Search candidate orientations to GUARANTEE a flat start (smallest settled half_h).
def place_box(q):
    st = obj.data.default_root_state[0:1].clone()
    st[0, 0] = TABLE_CENTER_X
    st[0, 1] = TABLE_CENTER_Y + 0.20
    st[0, 2] = TABLE_TOP_Z + 0.05
    st[0, 3:7] = torch.tensor(q, dtype=torch.float32, device=DEV)
    st[0, 7:] = 0.0
    obj.write_root_state_to_sim(st)
    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.forward()
    ik.reset()

CANDS = [[0.0,0.707,0.0,0.707],[0.707,0.707,0.0,0.0],[0.707,0.0,0.707,0.0],[1.0,0.0,0.0,0.0],[0.5,0.5,0.5,0.5]]
best_q, best_hh = CANDS[0], 999.0
for q in CANDS:
    place_box(q)
    drive(home, 110, g_open)
    hh = float(obj.data.root_com_pos_w[0, 2].item() - TABLE_TOP_Z)
    print(f"[FLIP] cand q={q} -> half_h={hh:.3f}", flush=True)
    if hh < best_hh:
        best_hh, best_q = hh, q
    if hh < 0.028:
        break
if best_hh >= 0.028:        # none truly flat; re-place flattest and settle
    place_box(best_q)
    drive(home, 110, g_open)

bx = obj.data.root_com_pos_w[0].cpu().numpy()
half_h = float(bx[2] - TABLE_TOP_Z)
print(f"[FLIP] START box center={bx.round(3)} half_h={half_h:.3f} (flat if <0.03)", flush=True)

cx, cy = float(bx[0]), float(bx[1])
top = TABLE_TOP_Z

# ---- Go to the box's LEFT (+Y) side, push -Y until contacting the box, then lean & lift to stand it up.
# z commanded LOW (top-0.10) to beat the ~+0.1 IK z-offset so the gripper reaches the box's short side face.
drive([cx, cy - 0.10, top + 0.10], 70,  g_close, record=True)   # go to left (+Y) of box, above
drive([cx, cy - 0.10, top - 0.10], 90,  g_close, record=True)   # descend to floor (top-0.10 is the reachable contact depth)
drive([cx, cy + 0.02, top - 0.10], 130, g_close, record=True, label="push")   # push -Y along floor THROUGH the box footprint
drive([cx, cy - 0.00, top + 0.18], 120, g_close, record=True)   # lean against box and LIFT -> tip it up
drive([cx, cy - 0.10, top + 0.25], 60,  g_open,  record=True)   # retract

bx2 = obj.data.root_com_pos_w[0].cpu().numpy()
half_h2 = float(bx2[2] - TABLE_TOP_Z)
print(f"[FLIP] AFTER box center={bx2.round(3)} half_h={half_h2:.3f} "
      f"(>0.04 => stood up onto edge!)", flush=True)

out = "datasets/test_flip"
os.makedirs(out + "/videos", exist_ok=True)
imageio.mimwrite(out + "/videos/flip.mp4", frames, fps=50, quality=7)
print(f"[FLIP] video -> {out}/videos/flip.mp4 ({len(frames)} frames)", flush=True)

env.close()
simulation_app.close()
