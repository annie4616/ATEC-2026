"""Spawn A2 Pro + Piper X, hold default joint targets, save frames via an IsaacLab Camera
sensor (direct numpy -> PNG). Checks the robot stands. Progress printed to a file."""
import argparse
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=2.5)
ap.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
# Pass the args namespace (the known-working launch path in this project); setting kwargs
# directly on AppLauncher hangs on this box. Force headless + cameras via the namespace.
args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
app = app_launcher.app

import os
os.makedirs(args.out, exist_ok=True)
LOG = open(os.path.join(args.out, "run.log"), "w")
def log(m):
    LOG.write(str(m) + "\n"); LOG.flush()

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from atec_rl_lab.assets.robots import UNITREE_A2_PIPER_CFG

log("launched app")
sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))

robot = Articulation(UNITREE_A2_PIPER_CFG.replace(prim_path="/World/Robot"))

cam_cfg = CameraCfg(
    prim_path="/World/cam",
    height=720, width=960,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=24.0),
)
cam = Camera(cam_cfg)
log("built robot + camera")

sim.reset()
# place the camera: front-side view of the robot at origin
from isaaclab.utils.math import quat_from_euler_xyz
eye = torch.tensor([[1.8, -1.8, 1.0]], device=sim.device)
tgt = torch.tensor([[0.0, 0.0, 0.35]], device=sim.device)
cam.set_world_poses_from_view(eye, tgt)

default_pos = robot.data.default_joint_pos.clone()
z0 = float(robot.data.root_pos_w[0, 2].item())
log(f"reset done, base z0={z0:.3f}")

steps = int(args.seconds / sim.get_physics_dt())
saved = 0
for i in range(steps):
    robot.set_joint_position_target(default_pos)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())
    cam.update(sim.get_physics_dt())
    if i % 10 == 0:  # ~20 fps at dt=0.005
        rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        from PIL import Image
        Image.fromarray(rgb).save(os.path.join(args.out, f"frame_{saved:04d}.png"))
        saved += 1
        if i % 100 == 0:
            log(f"step {i}/{steps} z={float(robot.data.root_pos_w[0,2].item()):.3f} saved={saved}")

z1 = float(robot.data.root_pos_w[0, 2].item())
log(f"DONE base z start={z0:.3f} end={z1:.3f} frames={saved}  {'STANDS' if z1>0.25 else 'FELL'}")
LOG.close()
app.close()
