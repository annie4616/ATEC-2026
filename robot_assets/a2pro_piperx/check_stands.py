"""Physics-only stand check (NO rendering/cameras — the rendering.kit hangs in this env).
Spawns A2+PiperX, holds default joint targets, reports base height over time.
If base z stays ~0.5 the robot stands; if it collapses toward 0 it fell."""
import argparse
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=3.0)
ap.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True  # no enable_cameras -> uses the fast headless kit, no render hang
app_launcher = AppLauncher(args)
app = app_launcher.app

import os
os.makedirs(os.path.dirname(args.out), exist_ok=True)
def log(m):
    with open(args.out, "a") as f:
        f.write(str(m) + "\n")

open(args.out, "w").close()
log("app launched")

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from atec_rl_lab.assets.robots import UNITREE_A2_PIPER_CFG

sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
robot = Articulation(UNITREE_A2_PIPER_CFG.replace(prim_path="/World/Robot"))
sim.reset()
log("reset done")

default_pos = robot.data.default_joint_pos.clone()
z0 = float(robot.data.root_pos_w[0, 2].item())
log(f"z0={z0:.3f}")
steps = int(args.seconds / sim.get_physics_dt())
for i in range(steps):
    robot.set_joint_position_target(default_pos)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())
    if i % 100 == 0:
        z = float(robot.data.root_pos_w[0, 2].item())
        # base tilt: projected gravity z ~ -1 when upright
        gz = float(robot.data.projected_gravity_b[0, 2].item())
        log(f"step {i:4d}  z={z:.3f}  grav_z={gz:.3f}")

zf = float(robot.data.root_pos_w[0, 2].item())
gzf = float(robot.data.projected_gravity_b[0, 2].item())
verdict = "STANDS" if (zf > 0.30 and gzf < -0.85) else "FELL/UNSTABLE"
log(f"FINAL z={zf:.3f} grav_z={gzf:.3f}  -> {verdict}")
app.close()
