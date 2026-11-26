"""Spawn A2 Pro + Piper X, hold the LEGS at their standing default but drive the ARM to a
'stowed behind the back' target, render a few views, and report whether the arm clears the
body/ground and the robot still stands. Arm-stow candidate values are passed on the CLI so
we can iterate the pose without editing the robot cfg.

Piper X arm geometry (from URDF):
  - arm base mounted at A2 base_link (0.20, 0, 0.10)  = forward-top of the trunk
  - joint1 : yaw about +z   limit [-2.618, 2.618]   (rotate the whole arm left/right/back)
  - joint2 : shoulder pitch limit [0, 3.14]
  - joint3 : elbow           limit [-2.967, 0]
  - joint4,5,6 : wrist       [-1.57,1.57] / [-1.57,1.57] / [-3.14,3.14]
  - joint7,8 : prismatic gripper
"""
import argparse
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=2.5)
ap.add_argument("--out", type=str, required=True)
# arm-stow target (radians). Default: yaw the arm back near its limit, fold shoulder+elbow
# so the arm lies low over the trunk pointing backward.
ap.add_argument("--j1", type=float, default=2.6)
ap.add_argument("--j2", type=float, default=2.6)
ap.add_argument("--j3", type=float, default=-2.6)
ap.add_argument("--j4", type=float, default=0.0)
ap.add_argument("--j5", type=float, default=0.0)
ap.add_argument("--j6", type=float, default=0.0)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
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

# resolve arm joint indices by name so we override only the arm
names = robot.data.joint_names
log(f"joint order: {names}")
arm_targets = {"joint1": args.j1, "joint2": args.j2, "joint3": args.j3,
               "joint4": args.j4, "joint5": args.j5, "joint6": args.j6}
name_to_idx = {n: i for i, n in enumerate(names)}

default_pos = robot.data.default_joint_pos.clone()
target = default_pos.clone()
for jn, val in arm_targets.items():
    if jn in name_to_idx:
        target[:, name_to_idx[jn]] = val
        log(f"  set {jn} (idx {name_to_idx[jn]}) -> {val}")
    else:
        log(f"  WARN missing joint {jn}")

z0 = float(robot.data.root_pos_w[0, 2].item())
log(f"reset done, base z0={z0:.3f}")

# three views: front-side, rear (behind), top-down — to see the stowed arm clearly
from isaaclab.utils.math import quat_from_euler_xyz
views = {
    "sideA": (torch.tensor([[1.8, -1.8, 1.0]], device=sim.device), torch.tensor([[0.0, 0.0, 0.35]], device=sim.device)),
    "rear":  (torch.tensor([[-1.9, 0.0, 0.9]], device=sim.device), torch.tensor([[0.0, 0.0, 0.40]], device=sim.device)),
    "top":   (torch.tensor([[0.0, -0.01, 2.0]], device=sim.device), torch.tensor([[0.0, 0.0, 0.40]], device=sim.device)),
}

steps = int(args.seconds / sim.get_physics_dt())
for i in range(steps):
    robot.set_joint_position_target(target)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())

# settle done; now snap each view
for vname, (eye, tgt) in views.items():
    cam.set_world_poses_from_view(eye, tgt)
    for _ in range(3):
        sim.step(); cam.update(sim.get_physics_dt())
    rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
    from PIL import Image
    Image.fromarray(rgb).save(os.path.join(args.out, f"stow_{vname}.png"))
    log(f"saved view {vname}")

# report final arm joint pos vs target (did PD reach the stow pose within limits?)
final = robot.data.joint_pos[0].detach().cpu().numpy()
for jn in arm_targets:
    idx = name_to_idx.get(jn)
    if idx is not None:
        log(f"  {jn}: target={arm_targets[jn]:+.3f} reached={final[idx]:+.3f}")

z1 = float(robot.data.root_pos_w[0, 2].item())
log(f"DONE base z start={z0:.3f} end={z1:.3f}  {'STANDS' if z1>0.25 else 'FELL'}")
LOG.close()
app.close()
