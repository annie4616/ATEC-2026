"""Sweep several arm-stow candidate poses for A2 Pro + Piper X in ONE process; for each,
drive legs to standing default + arm to the candidate, settle, and render side+top+rear.
Goal: arm folded LOW and BACK over the trunk (not raised into the air), clearing legs/ground.

Arm mount = A2 base_link (0.20, 0, 0.10), forward-top. Chain reach ~0.63 m, so a truly
compact stow needs joint2 low (shoulder down toward the trunk) + joint3 elbow fully folded.
"""
import argparse
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import os
os.makedirs(args.out, exist_ok=True)
LOG = open(os.path.join(args.out, "run.log"), "w")
def log(m): LOG.write(str(m) + "\n"); LOG.flush()

import numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from atec_rl_lab.assets.robots import UNITREE_A2_PIPER_CFG

# candidate poses: (name, j1, j2, j3, j4, j5, j6)
# joint2 in [0,3.14] (shoulder), joint3 in [-2.967,0] (elbow). Small joint2 = arm laid
# back toward horizontal; large joint2 = arm up. We want arm BACK + LOW, so try yaw back
# (j1 high) with a modest shoulder and a hard-folded elbow.
# joint2 large (>pi/2) throws the whole arm backward-over-the-trunk; combined with elbow
# fully folded (j3=-2.9) the forearm doubles back so the gripper tucks near the trunk top.
# Sweep j2 in the high range to find the flattest "laid back over the spine" pose.
CANDS = [
    ("F_j2_180",  2.6, 3.1, -2.9, 0.0, 0.0, 0.0),   # shoulder max back, elbow max fold
    ("G_j2_160",  2.6, 2.9, -2.9, 0.0, 0.0, 0.0),
    ("H_j2_140",  2.6, 2.4, -2.9, 0.0, 0.0, 0.0),
    ("I_j2_180_e0",2.6, 3.1, 0.0, 0.0, 0.0, 0.0),   # arm straight back, elbow open
    ("J_wrist_dn",2.6, 3.1, -2.9, 0.0, 1.2, 0.0),   # add wrist pitch to drop gripper down
]

sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))
robot = Articulation(UNITREE_A2_PIPER_CFG.replace(prim_path="/World/Robot"))
cam = Camera(CameraCfg(prim_path="/World/cam", height=720, width=960, data_types=["rgb"],
                       spawn=sim_utils.PinholeCameraCfg(focal_length=24.0)))
sim.reset()
names = robot.data.joint_names
n2i = {n: i for i, n in enumerate(names)}
log(f"joint order: {names}")

views = {
    "side": (torch.tensor([[1.8, -1.8, 1.0]], device=sim.device), torch.tensor([[0.0, 0.0, 0.35]], device=sim.device)),
    "top":  (torch.tensor([[0.0, -0.01, 2.0]], device=sim.device), torch.tensor([[0.0, 0.0, 0.40]], device=sim.device)),
    "rear": (torch.tensor([[-1.9, 0.0, 0.9]], device=sim.device), torch.tensor([[0.0, 0.0, 0.40]], device=sim.device)),
}
from PIL import Image

for (cname, j1, j2, j3, j4, j5, j6) in CANDS:
    # reset robot to default state each candidate
    root = robot.data.default_root_state.clone()
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    tgt = robot.data.default_joint_pos.clone()
    for jn, v in zip(["joint1","joint2","joint3","joint4","joint5","joint6"], [j1,j2,j3,j4,j5,j6]):
        tgt[:, n2i[jn]] = v
    for _ in range(400):  # settle ~2s
        robot.set_joint_position_target(tgt)
        robot.write_data_to_sim(); sim.step(); robot.update(sim.get_physics_dt())
    z = float(robot.data.root_pos_w[0, 2].item())
    fp = robot.data.joint_pos[0].detach().cpu().numpy()
    reach = {jn: float(fp[n2i[jn]]) for jn in ["joint1","joint2","joint3"]}
    log(f"[{cname}] target j1/2/3={j1}/{j2}/{j3} reached={reach} baseZ={z:.3f} {'STAND' if z>0.25 else 'FELL'}")
    for vn, (eye, t) in views.items():
        cam.set_world_poses_from_view(eye, t)
        for _ in range(3): sim.step(); cam.update(sim.get_physics_dt())
        rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(rgb).save(os.path.join(args.out, f"{cname}_{vn}.png"))
    log(f"  saved views for {cname}")

log("ALL DONE")
LOG.close(); app.close()
