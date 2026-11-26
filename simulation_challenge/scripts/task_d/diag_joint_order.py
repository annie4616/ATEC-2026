"""Diagnose joint ordering: proprio joint_pos_rel order vs articulation order vs
the action(joint_leg/joint_arm) order, and whether feeding seq[0] from the demo start
holds the robot (i.e. replay-only, no alignment) reproduces the 36-point run.

Also dumps, at the demo start: proprio joint_pos_rel, articulation joint names, the
oracle's resolved leg_ids/arm_ids, and what action seq[0] is -- so we can see if
action index k corresponds to proprio index k."""
import os, sys
from isaaclab.app import AppLauncher
p = __import__("argparse").ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.enable_cameras = True; a.headless = True
app = AppLauncher(a); sim = app.app

import numpy as np, torch
from isaaclab.envs import ManagerBasedRLEnv
import atec_rl_lab.tasks  # noqa
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "act"))
from task_d.collector import _reset_asset_root, WARMUP_STEPS
from task_d.walking_oracle import WalkingOracle, LEG_JOINT_NAMES, ARM_JOINT_NAMES

cfg = TaskDEnvB2Cfg(); cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg)
robot = env.unwrapped.scene["robot"]; box = env.unwrapped.scene["box"]
obs, _ = env.reset(); _reset_asset_root(robot); _reset_asset_root(box)
orc = WalkingOracle("logs/rsl_rl/unitree_b2_rough/2026-06-18_04-37-10/exported/policy.pt", robot, "cuda")
orc.reset()
for _ in range(WARMUP_STEPS):
    obs, *_ = env.step(orc.compute_action())

# Articulation joint names (robot.data order)
art_names = robot.data.joint_names
leg_ids, _ = robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)
arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
print("=== articulation joint_names (robot.data order) ===")
for i, nm in enumerate(art_names):
    print(f"  [{i:2d}] {nm}")
print("oracle leg_ids (LEG_JOINT_NAMES->art idx):", list(leg_ids))
print("oracle arm_ids (ARM_JOINT_NAMES->art idx):", list(arm_ids))

# proprio joint_pos_rel (the obs the solution sees)
prop = obs["proprio"][0].detach().cpu().numpy()
adim = (prop.shape[0] - 12) // 3
jprel_proprio = prop[12:12 + adim]
# articulation joint_pos_rel (ground truth, in robot.data order)
jprel_art = (robot.data.joint_pos - robot.data.default_joint_pos)[0].detach().cpu().numpy()
print("\n=== joint_pos_rel: proprio[12:] vs articulation (robot.data) ===")
print("proprio jprel:", np.round(jprel_proprio, 3).tolist())
print("art     jprel:", np.round(jprel_art, 3).tolist())
print("max|proprio-art| (same-index):", float(np.max(np.abs(jprel_proprio - jprel_art))))

# What the demo's recorded seq[0] is, and the oracle action right now
seq = np.load("submission_mem/joint_action_seq.npy").astype(np.float32)
oracle_act = orc.compute_action()[0].detach().cpu().numpy()
print("\n=== action at demo start ===")
print("seq[0]      :", np.round(seq[0], 3).tolist())
print("oracle now  :", np.round(oracle_act, 3).tolist())
print("max|seq0-oracle|:", float(np.max(np.abs(seq[0] - oracle_act))))
env.close()
sim.close()
