"""Derive a box detector from extero. Dump, at a known box position, the lidar's
ray_hits_w (world xyz of every hit), the sensor pos, and extero (height_scan), so we can
(a) confirm extero index ordering, (b) identify which rays hit the BOX (by comparing
hit xyz to the true box AABB), and (c) design an extero-only rule to localize the box.
Single env, abs paths, log file."""
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

OUT = "/hdd0/sohyun/tmp/diag_lidar_rays.txt"
fh = open(OUT, "w")
def w(*x): fh.write(" ".join(str(s) for s in x)+"\n"); fh.flush()
POL = "/hdd0/sohyun/atec_2026_tesk_e/ATEC2026_Simulation_Challenge/logs/rsl_rl/unitree_b2_rough/2026-06-18_04-37-10/exported/policy.pt"
SUB = "/hdd0/sohyun/atec_2026_tesk_e/ATEC2026_Simulation_Challenge/submission_mem"
velcmd = np.load(SUB+"/vel_cmd_seq.npy").astype(np.float32)

cfg = TaskDEnvB2Cfg(); cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg)
robot = env.unwrapped.scene["robot"]; box = env.unwrapped.scene["box"]
lidar = env.unwrapped.scene.sensors["lidar_sensor"]
obs, _ = env.reset(); _reset_asset_root(robot); _reset_asset_root(box)
orc = WalkingOracle(POL, robot, "cuda"); orc.reset()
for _ in range(WARMUP_STEPS):
    obs, *_ = env.step(orc.compute_action())

# push to a step where the box is clearly ahead in the pit region (t800-ish), and a
# settled step (t900). dump rays at both.
BOX_HALF = (0.4, 0.5, 0.3)   # x,y,z half extents
for tstop in (800, 900):
    while True:
        t = orc_step = None
        break
# run to t=800 and t=900, dumping at each
dump = {800, 900}
for t in range(0, 901):
    orc.set_velocity_command(float(velcmd[t,0]), float(velcmd[t,1]), float(velcmd[t,2]))
    obs, *_ = env.step(orc.compute_action())
    if t in dump:
        bx = box.data.root_pos_w[0].detach().cpu().numpy()
        sp = lidar.data.pos_w[0].detach().cpu().numpy()
        hits = lidar.data.ray_hits_w[0].detach().cpu().numpy()   # (5760,3)
        ext = obs["extero"][0].detach().cpu().numpy()            # (5760,)
        w(f"\n=== t={t} box={np.round(bx,2).tolist()} sensor_pos={np.round(sp,2).tolist()} "
          f"num_rays={hits.shape[0]} ===")
        # which rays hit the box AABB?
        bxlo = bx - np.array(BOX_HALF) - 0.05
        bxhi = bx + np.array(BOX_HALF) + 0.05
        finite = np.isfinite(hits).all(axis=1)
        inbox = finite & np.all(hits >= bxlo, axis=1) & np.all(hits <= bxhi, axis=1)
        w(f"  finite rays={finite.sum()} box-hit rays={inbox.sum()}")
        if inbox.sum() > 0:
            idx = np.where(inbox)[0]
            w(f"  box-hit ray indices (first 30): {idx[:30].tolist()}")
            w(f"  box-hit index range: {idx.min()}..{idx.max()}  channels(idx//360)={np.unique(idx//360).tolist()}")
            w(f"  extero at box-hit rays: min={ext[idx].min():.2f} max={ext[idx].max():.2f} "
              f"mean={ext[idx].mean():.2f}")
            w(f"  hit xyz mean={np.round(hits[idx].mean(0),2).tolist()}")
        # verify extero == sensor_z - hit_z - 0.5
        recon = sp[2] - hits[:,2] - 0.5
        err = np.nanmax(np.abs(np.where(finite, recon-ext, 0)))
        w(f"  extero==sensor_z-hit_z-0.5 check maxerr={err:.4f}")
        # ray ordering hint: is it channel-major or azimuth-major? show hit pattern
        w(f"  extero[0:5]={np.round(ext[0:5],2).tolist()} extero[360:365]={np.round(ext[360:365],2).tolist()}")
fh.close(); env.close(); sim.close()
