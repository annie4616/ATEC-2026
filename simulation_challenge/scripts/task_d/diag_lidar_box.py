"""Can we localize the BOX from the submission obs? The Task D lidar is a MultiMeshRayCaster
that includes the Box mesh (track_mesh_transforms), so the box SHOULD appear in extero.
From the demo (oracle 36pt) start, at cross-relevant steps, dump: obs keys, extero shape,
true box pos (for calibration), and the lidar return distribution to see if the box is a
detectable cluster. Single env, absolute paths, log file."""
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

OUT = "/hdd0/sohyun/tmp/diag_lidar_box.txt"
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

w("obs keys:", list(obs.keys()))
for k, v in obs.items():
    if hasattr(v, "shape"):
        w(f"  obs['{k}'] shape={tuple(v.shape)}")
    elif isinstance(v, dict):
        for kk, vv in v.items():
            w(f"  obs['{k}']['{kk}'] shape={tuple(vv.shape) if hasattr(vv,'shape') else '?'}")

T = velcmd.shape[0]
dump_steps = {750, 800, 850, 900, 950}
for t in range(0, 1000):
    orc.set_velocity_command(float(velcmd[t,0]), float(velcmd[t,1]), float(velcmd[t,2]))
    obs, *_ = env.step(orc.compute_action())
    if t in dump_steps:
        bx = box.data.root_pos_w[0]; rp = robot.data.root_pos_w[0]
        w(f"\n=== t={t} true box=({bx[0].item():+.2f},{bx[1].item():+.2f},{bx[2].item():+.2f}) "
          f"robot=({rp[0].item():+.2f},{rp[1].item():+.2f},{rp[2].item():+.2f}) ===")
        ext = obs.get("extero", None)
        if ext is not None:
            e = ext[0].detach().cpu().numpy()
            w(f"  extero shape={e.shape} min={e.min():.2f} max={e.max():.2f} "
              f"mean={e.mean():.3f} nonzero={(np.abs(e)>0.01).sum()}")
            # height_scan = sensor_z - hit_z - 0.5; box top rises -> distinct values.
            # report histogram of extreme returns (potential box hits)
            hi = np.sort(e)[-20:]
            lo = np.sort(e)[:20]
            w(f"  top20 extero: {np.round(hi,2).tolist()}")
            w(f"  bot20 extero: {np.round(lo,2).tolist()}")
fh.close(); env.close(); sim.close()
