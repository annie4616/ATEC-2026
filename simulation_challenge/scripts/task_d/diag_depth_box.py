"""Probe head_depth for a simple box-distance signal. From demo start, at push/cross
steps, log the true box pos and depth-image statistics in the forward-central region, to
design a robust 'distance to box ahead' + 'box settled' detector. Single env, abs paths."""
import os, sys
from isaaclab.app import AppLauncher
ap = __import__("argparse").ArgumentParser(); AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args(); a.enable_cameras = True; a.headless = True
app = AppLauncher(a); sim = app.app

import numpy as np, torch
from isaaclab.envs import ManagerBasedRLEnv
import atec_rl_lab.tasks  # noqa
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "act"))
from task_d.collector import _reset_asset_root, WARMUP_STEPS
from task_d.walking_oracle import WalkingOracle

OUT = "/hdd0/sohyun/tmp/diag_depth_box.txt"
fh = open(OUT, "w")
def w(*x): fh.write(" ".join(str(s) for s in x)+"\n"); fh.flush()
POL = "/hdd0/sohyun/atec_2026_tesk_e/ATEC2026_Simulation_Challenge/logs/rsl_rl/unitree_b2_rough/2026-06-18_04-37-10/exported/policy.pt"
SUB = "/hdd0/sohyun/atec_2026_tesk_e/ATEC2026_Simulation_Challenge/submission_mem"
velcmd = np.load(SUB+"/vel_cmd_seq.npy").astype(np.float32)

cfg = TaskDEnvB2Cfg(); cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg)
robot = env.unwrapped.scene["robot"]; box = env.unwrapped.scene["box"]
obs, _ = env.reset(); _reset_asset_root(robot); _reset_asset_root(box)
orc = WalkingOracle(POL, robot, "cuda"); orc.reset()
for _ in range(WARMUP_STEPS):
    obs, *_ = env.step(orc.compute_action())

def depth_stats(o):
    d = o["image"]["head_depth"][0].detach().cpu().numpy().squeeze()  # (480,640)
    d = np.where(np.isfinite(d), d, 0.0)
    H, W = d.shape
    # central forward strip (lower-center, where the box ahead/below appears)
    cx0, cx1 = W//2-80, W//2+80
    cy0, cy1 = int(H*0.55), int(H*0.95)
    crop = d[cy0:cy1, cx0:cx1]
    valid = crop[(crop > 0.05) & (crop < 9.0)]
    if valid.size == 0:
        return None
    return float(np.median(valid)), float(valid.min()), float(np.percentile(valid,25)), valid.size

w("head_depth probe: forward-central strip median/min distance vs true box")
for t in range(0, 1000):
    orc.set_velocity_command(float(velcmd[t,0]), float(velcmd[t,1]), float(velcmd[t,2]))
    obs, *_ = env.step(orc.compute_action())
    if t % 50 == 0 or t in (750,800,850,900,950):
        bx = box.data.root_pos_w[0]; rp = robot.data.root_pos_w[0]
        st = depth_stats(obs)
        bxr = bx[0].item()-rp[0].item()   # box x relative to robot (world)
        s = f"t={t:4d} box=({bx[0].item():+.2f},{bx[1].item():+.2f},{bx[2].item():+.2f}) " \
            f"robot_x={rp[0].item():+.2f} box_rel_x={bxr:+.2f}"
        if st: s += f" | depth med={st[0]:.2f} min={st[1]:.2f} p25={st[2]:.2f} n={st[3]}"
        else:  s += " | depth: no valid"
        w(s)
fh.close(); env.close(); sim.close()
