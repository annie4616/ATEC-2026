"""Can a walking-policy GUIDE converge the dead-reckoned base to the demo start
(-3.0005, -0.0002, yaw 0) tightly (pos<2cm, yaw<1deg, speed<0.05)? From air-drop, run a
position+yaw P-controller via the walking policy and log convergence over time, plus the
TRUE root pose (to check odometry accuracy). This tells us if 'feedback-guide until at the
start, then deterministic replay' is feasible. Single env, abs paths."""
import os, sys, math
from isaaclab.app import AppLauncher
ap = __import__("argparse").ArgumentParser(); AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args(); a.enable_cameras = True; a.headless = True
app = AppLauncher(a); sim = app.app

import numpy as np, torch
from isaaclab.envs import ManagerBasedRLEnv
import atec_rl_lab.tasks  # noqa
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "act"))
from task_d.walking_oracle import WalkingOracle

OUT = "/hdd0/sohyun/tmp/diag_guide.txt"
fh = open(OUT, "w")
def w(*x): fh.write(" ".join(str(s) for s in x)+"\n"); fh.flush()
POL = "/hdd0/sohyun/atec_2026_tesk_e/ATEC2026_Simulation_Challenge/logs/rsl_rl/unitree_b2_rough/2026-06-18_04-37-10/exported/policy.pt"

TGT = (-3.0005, -0.0002, 0.0)
cfg = TaskDEnvB2Cfg(); cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg)
robot = env.unwrapped.scene["robot"]
obs,_ = env.reset()
orc = WalkingOracle(POL, robot, "cuda"); orc.reset()
STEP_DT = 0.02
rx, ry, ryaw = -3.0, 0.0, 0.0  # odom (spawn)
ix = iy = iyaw = 0.0           # integral accumulators
KI = float(os.environ.get("GUIDE_KI", "0.6"))      # integral gain
KP = float(os.environ.get("GUIDE_KP", "1.5"))

def yawnorm(e): return (e + math.pi) % (2*math.pi) - math.pi

w(f"guide PI convergence (KP={KP} KI={KI}): odom vs target; also TRUE root pose")
for n in range(400):
    p = obs["proprio"][0]
    mvx, mvy, mwz = float(p[0]), float(p[1]), float(p[5])
    ryaw += mwz*STEP_DT
    c,s = math.cos(ryaw), math.sin(ryaw)
    rx += (c*mvx - s*mvy)*STEP_DT
    ry += (s*mvx + c*mvy)*STEP_DT
    ex, ey, eyaw = TGT[0]-rx, TGT[1]-ry, yawnorm(TGT[2]-ryaw)
    dist = math.hypot(ex, ey); spd = math.hypot(mvx, mvy)
    if n % 25 == 0 or (n>40 and dist<0.02 and abs(eyaw)<0.017 and spd<0.05):
        tr = robot.data.root_pos_w[0]; tq = robot.data.root_quat_w[0]
        # true yaw from quat (w,x,y,z)
        qw,qx,qy,qz = [float(v) for v in tq]
        tyaw = math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
        w(f"n={n:3d} odom=({rx:+.3f},{ry:+.3f},{ryaw:+.3f}) err(d={dist:.3f},yaw={eyaw:+.3f}) spd={spd:.3f}"
          f" | TRUE=({tr[0].item():+.3f},{tr[1].item():+.3f},yaw{tyaw:+.3f})")
    # P-control: only correct after landing (n>30)
    if n < 30:
        leg = orc.compute_action()  # let it land (zero vel default? oracle uses FORWARD)
        # actually set zero vel for settle
        orc.set_velocity_command(0.0,0.0,0.0); leg = orc.compute_action()
    else:
        bex = c*ex + s*ey; bey = -s*ex + c*ey
        # integral accumulate (body frame for x/y, world for yaw); anti-windup clamp
        ix = max(-0.5, min(0.5, ix + bex*STEP_DT))
        iy = max(-0.5, min(0.5, iy + bey*STEP_DT))
        iyaw = max(-0.5, min(0.5, iyaw + eyaw*STEP_DT))
        vx = max(-0.6,min(0.6, KP*bex + KI*ix))
        vy = max(-0.6,min(0.6, KP*bey + KI*iy))
        wz = max(-0.8,min(0.8, 2.0*eyaw + KI*iyaw))
        orc.set_velocity_command(vx,vy,wz); leg = orc.compute_action()
    obs,*_ = env.step(leg)
fh.close(); env.close(); sim.close()
