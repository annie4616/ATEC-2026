"""Verify the A2+PiperX legs actually COLLIDE with the ground (don't pass through).
Drops the robot from a height with zero joint torque (limp) and checks:
  - does the base settle at a sensible height (feet stop it) rather than fall to z~0?
  - do the feet register contact forces from the ground?
If the robot passes through, base z -> ~0 (or negative) and contact forces stay 0."""
import argparse
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app

open(args.out, "w").close()
def log(m):
    with open(args.out, "a") as f: f.write(str(m) + "\n")

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationContext
from atec_rl_lab.assets.robots import UNITREE_A2_PIPER_CFG

sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

# spawn a bit high so it drops onto the ground
cfg = UNITREE_A2_PIPER_CFG.replace(prim_path="/World/Robot")
cfg.init_state.pos = (0.0, 0.0, 0.7)
robot = Articulation(cfg)

# contact sensor on the feet
cs_cfg = ContactSensorCfg(prim_path="/World/Robot/.*_foot", history_length=1, track_air_time=False)
contact = ContactSensor(cs_cfg)

sim.reset()
log("reset; dropping robot from z=0.7 with ZERO joint effort (limp)")

# apply zero effort so the robot is passive — only ground collision stops it
zero = torch.zeros((1, robot.num_joints), device=sim.device)
for i in range(600):  # 3 s
    robot.set_joint_effort_target(zero)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())
    contact.update(sim.get_physics_dt())
    if i % 100 == 0:
        z = float(robot.data.root_pos_w[0, 2].item())
        # net contact force magnitude on the feet
        f = contact.data.net_forces_w  # (1, num_feet, 3)
        fmag = float(f.norm(dim=-1).sum().item()) if f is not None else -1.0
        log(f"step {i:4d}  base_z={z:.3f}  total_foot_contact_force={fmag:.1f} N")

z = float(robot.data.root_pos_w[0, 2].item())
f = contact.data.net_forces_w
fmag = float(f.norm(dim=-1).sum().item()) if f is not None else -1.0
# limp robot: legs splay, base rests LOW but should be > 0 (feet/body on ground, not through it)
collides = (z > 0.05) and (fmag > 1.0)
log(f"FINAL base_z={z:.3f}  foot_contact_force={fmag:.1f} N")
log(f"VERDICT: {'COLLIDES (feet hit ground, no pass-through)' if collides else 'SUSPECT pass-through / no contact'}")
app.close()
