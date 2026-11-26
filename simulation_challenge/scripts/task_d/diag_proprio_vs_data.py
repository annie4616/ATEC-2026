"""Compare the walking-policy obs built from proprio (the submission's path) vs from
robot.data (the oracle's path), step by step from the demo start, to find which field
diverges. Drives the env with the robot.data-oracle (ground truth) so both see the same
trajectory; logs per-field max abs diff of the reconstructed policy_obs."""
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

OUT = "/hdd0/sohyun/tmp/diag_pvd.txt"
fh = open(OUT, "w")
def w(*x): fh.write(" ".join(str(s) for s in x)+"\n"); fh.flush()

SUB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "submission_mem")
velcmd = np.load(os.path.join(SUB, "vel_cmd_seq.npy")).astype(np.float32)
LEG_PERM = [1, 5, 10, 0, 4, 9, 3, 7, 12, 2, 6, 11]
ENV_TO_TRAIN = torch.tensor([4.,2.,2.,4.,2.,2.,4.,2.,2.,4.,2.,2.], device="cuda").view(1,-1)

cfg = TaskDEnvB2Cfg(); cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg)
robot = env.unwrapped.scene["robot"]; box = env.unwrapped.scene["box"]
obs, _ = env.reset(); _reset_asset_root(robot); _reset_asset_root(box)
orc = WalkingOracle("logs/rsl_rl/unitree_b2_rough/2026-06-18_04-37-10/exported/policy.pt", robot, "cuda")
orc.reset()
for _ in range(WARMUP_STEPS):
    obs, *_ = env.step(orc.compute_action())

leg_perm = torch.tensor(LEG_PERM, device="cuda", dtype=torch.long)
for t in range(60):
    p = obs["proprio"].to("cuda", torch.float32).view(1, -1)
    adim = (p.shape[-1]-12)//3
    # proprio-reconstructed fields
    pr_ang = p[:, 3:6]
    pr_grav = p[:, 9:12]
    pr_jp = p[:, 12:12+adim][:, leg_perm]
    pr_jv = p[:, 12+adim:12+2*adim][:, leg_perm]
    pr_la = p[:, 12+2*adim:12+3*adim][:, :12]
    # robot.data fields (oracle path)
    d = robot.data
    rd_ang = d.root_ang_vel_b[:, :3]
    rd_grav = d.projected_gravity_b[:, :3]
    jpr = d.joint_pos - d.default_joint_pos
    rd_jp = jpr[:, orc.leg_ids]
    rd_jv = d.joint_vel[:, orc.leg_ids]
    rd_la = orc._last_leg_action_env   # env-side last leg action (oracle order)
    if t < 6 or t % 10 == 0:
        w(f"t={t:3d} ang={float((pr_ang-rd_ang).abs().max()):.4f} "
          f"grav={float((pr_grav-rd_grav).abs().max()):.4f} "
          f"jp={float((pr_jp-rd_jp).abs().max()):.4f} "
          f"jv={float((pr_jv-rd_jv).abs().max()):.4f} "
          f"la={float((pr_la-rd_la).abs().max()):.4f}")
    # step with ground-truth oracle
    orc.set_velocity_command(float(velcmd[t,0]), float(velcmd[t,1]), float(velcmd[t,2]))
    obs, *_ = env.step(orc.compute_action())

fh.close(); env.close(); sim.close()
