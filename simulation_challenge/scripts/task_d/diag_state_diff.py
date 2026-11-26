"""Measure full-state at DEMO start vs SERVER(air-drop+settle) start. Run twice:
  --mode demo   -> reset + _reset_asset_root + 10 warmup ; snapshot
  --mode server -> plain reset + N settle(walk@zero) ; snapshot at N=0,30,60,90
Single env per run (abs paths). Compare the two output files afterward."""
import os, sys
from isaaclab.app import AppLauncher
ap = __import__("argparse").ArgumentParser(); AppLauncher.add_app_launcher_args(ap)
ap.add_argument("--mode", type=str, default="demo")
a = ap.parse_args(); a.enable_cameras = True; a.headless = True
app = AppLauncher(a); sim = app.app

import numpy as np, torch
from isaaclab.envs import ManagerBasedRLEnv
import atec_rl_lab.tasks  # noqa
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "act"))
from task_d.collector import _reset_asset_root, WARMUP_STEPS
from task_d.walking_oracle import WalkingOracle

OUT = f"/hdd0/sohyun/tmp/diag_state_{a.mode}.txt"
fh = open(OUT, "w")
def w(*x): fh.write(" ".join(str(s) for s in x)+"\n"); fh.flush()
POL = "/hdd0/sohyun/atec_2026_tesk_e/ATEC2026_Simulation_Challenge/logs/rsl_rl/unitree_b2_rough/2026-06-18_04-37-10/exported/policy.pt"

def grab(robot, box):
    d = robot.data
    return {
        "root_pos": d.root_pos_w[0,:3].detach().cpu().numpy().copy(),
        "root_quat": d.root_quat_w[0,:4].detach().cpu().numpy().copy(),
        "lin_vel": d.root_lin_vel_w[0,:3].detach().cpu().numpy().copy(),
        "ang_vel": d.root_ang_vel_w[0,:3].detach().cpu().numpy().copy(),
        "jpos": d.joint_pos[0].detach().cpu().numpy().copy(),
        "jvel": d.joint_vel[0].detach().cpu().numpy().copy(),
        "box_pos": box.data.root_pos_w[0,:3].detach().cpu().numpy().copy(),
    }

def dump(tag, S):
    w(f"=== {tag} ===")
    for k,v in S.items(): w(f"  {k}: {np.round(v,4).tolist()}")

cfg = TaskDEnvB2Cfg(); cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg)
robot = env.unwrapped.scene["robot"]; box = env.unwrapped.scene["box"]
obs,_ = env.reset()
orc = WalkingOracle(POL, robot, "cuda"); orc.reset()

if a.mode == "demo":
    _reset_asset_root(robot); _reset_asset_root(box)
    for _ in range(WARMUP_STEPS):
        obs,*_ = env.step(orc.compute_action())
    dump("DEMO start (reset_root+10 warmup)", grab(robot, box))
else:
    orc.set_velocity_command(0.0,0.0,0.0)
    dump("SERVER N=0 (just plain reset)", grab(robot, box))
    for n in range(1, 91):
        obs,*_ = env.step(orc.compute_action())
        if n in (30,60,90):
            dump(f"SERVER N={n} (plain reset + {n} settle@zero)", grab(robot, box))
fh.close(); env.close(); sim.close()
