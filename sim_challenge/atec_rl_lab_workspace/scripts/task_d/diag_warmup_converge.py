"""Single-env: from the SERVER air-drop (plain reset), run oracle warmup and measure
convergence of robot joint pose to the DEMO recording-start reference + root x/z, for
both zero-vel and forward warmup. One env only (3x env creation with cameras hangs)."""
import os, sys
from isaaclab.app import AppLauncher
ap = __import__("argparse").ArgumentParser(); AppLauncher.add_app_launcher_args(ap)
ap.add_argument("--vel", type=str, default="zero")   # zero | fwd
a = ap.parse_args(); a.enable_cameras = True; a.headless = True
app = AppLauncher(a); sim = app.app

import numpy as np, torch
from isaaclab.envs import ManagerBasedRLEnv
import atec_rl_lab.tasks  # noqa
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "act"))
from task_d.walking_oracle import WalkingOracle

OUT = f"/hdd0/sohyun/tmp/diag_wc_{a.vel}.txt"
fh = open(OUT, "w")
def w(*x): fh.write(" ".join(str(s) for s in x)+"\n"); fh.flush()

# DEMO recording-start joint_pos_rel reference (captured earlier; == proprio_jprel_start.npy)
demo_jp = np.array([0.102,-0.091,0.230,-0.094,-0.089,0.211,0.092,-0.048,0.116,
                    -0.064,-0.097,0.116], dtype=np.float32)
vel = [0.6,0.0,0.0] if a.vel=="fwd" else [0.0,0.0,0.0]

cfg = TaskDEnvB2Cfg(); cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg)
robot = env.unwrapped.scene["robot"]
obs, _ = env.reset()   # PLAIN reset = air-drop
_POL = "/hdd0/sohyun/atec_2026_tesk_e/ATEC2026_Simulation_Challenge/logs/rsl_rl/unitree_b2_rough/2026-06-18_04-37-10/exported/policy.pt"
orc = WalkingOracle(_POL, robot, "cuda")
orc.reset()
w(f"=== AIR-DROP + warmup @ {a.vel} {vel} ===  (demo target jp[:12] fixed)")
w("  N | root_x  z      | max|jp-demo|")
for n in range(0, 151):
    if n in (0,5,10,15,20,30,40,50,70,100,150):
        jp = obs["proprio"][0,12:24].detach().cpu().numpy()
        rx = robot.data.root_pos_w[0,0].item(); rz = robot.data.root_pos_w[0,2].item()
        d = float(np.max(np.abs(jp-demo_jp)))
        w(f"  {n:3d} | {rx:+.3f} {rz:+.3f} | {d:.3f}")
    orc.set_velocity_command(*vel)
    obs, *_ = env.step(orc.compute_action())
fh.close(); env.close(); sim.close()
