"""Spawn the composed USD as an IsaacLab Articulation and dump its DoF / body layout.
This is the go/no-go check: it must load as ONE articulation with 20 DoF (12 leg + 8 arm)."""
import argparse
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(headless=True).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext

sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
cfg = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=args.usd, activate_contact_sensors=False),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
    ),
    actuators={},
)
robot = Articulation(cfg)
sim.reset()

out = []
out.append(f"num_instances: {robot.num_instances}")
out.append(f"num_joints (DoF): {robot.num_joints}")
out.append(f"num_bodies: {robot.num_bodies}")
out.append(f"joint_names ({len(robot.joint_names)}):")
for n in robot.joint_names:
    out.append("    " + n)
out.append(f"body_names ({len(robot.body_names)}):")
for n in robot.body_names:
    out.append("    " + n)
txt = "\n".join(out)
with open(args.usd + ".artic.txt", "w") as f:
    f.write(txt + "\n")
print("=== ARTICULATION LOADED ===", flush=True)
print(txt, flush=True)
app.close()
