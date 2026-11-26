"""Decisive test for the closed-loop-oracle replay strategy.

From the DEMO start (reset_root + 10 warmup), drive a proprio-reconstructed walking
oracle with the recorded per-step vel_cmd_seq and check:
  (a) does proprio joint order == articulation leg-first order? (the suspected bug)
  (b) does the proprio-reconstructed oracle action match the recorded seq[t]?
  (c) does driving robot.data-oracle with vel_cmd_seq cross the pit (sanity)?

Writes findings to a RESULT file (flushed), not stdout, so nothing is lost."""
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
from task_d.walking_oracle import WalkingOracle, LEG_JOINT_NAMES, ARM_JOINT_NAMES

OUT = "/hdd0/sohyun/tmp/diag_oracle_result.txt"
fh = open(OUT, "w")
def w(*x):
    fh.write(" ".join(str(s) for s in x) + "\n"); fh.flush()

SUB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "submission_mem")
seq = np.load(os.path.join(SUB, "joint_action_seq.npy")).astype(np.float32)       # (T,20)
velcmd = np.load(os.path.join(SUB, "vel_cmd_seq.npy")).astype(np.float32)          # (T,3)
T = seq.shape[0]

cfg = TaskDEnvB2Cfg(); cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg)
robot = env.unwrapped.scene["robot"]; box = env.unwrapped.scene["box"]
obs, _ = env.reset(); _reset_asset_root(robot); _reset_asset_root(box)
orc = WalkingOracle("logs/rsl_rl/unitree_b2_rough/2026-06-18_04-37-10/exported/policy.pt", robot, "cuda")
orc.reset()
for _ in range(WARMUP_STEPS):
    obs, *_ = env.step(orc.compute_action())

# (a) joint order
art_names = list(robot.data.joint_names)
leg_ids, _ = robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)
arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
w("=== (a) JOINT ORDER ===")
w("articulation joint_names:", art_names)
w("oracle leg_ids:", list(leg_ids), "arm_ids:", list(arm_ids))
w("leg_ids == [0..11]?", list(leg_ids) == list(range(12)))
w("arm_ids == [12..19]?", list(arm_ids) == list(range(12, 20)))
# proprio joint order: proprio uses joint_names='.*' preserve_order -> articulation order.
prop = obs["proprio"][0].detach().cpu().numpy()
adim = (prop.shape[0] - 12) // 3
jp_proprio = prop[12:12 + adim]
jp_art = (robot.data.joint_pos - robot.data.default_joint_pos)[0].detach().cpu().numpy()
w("proprio[12:] == articulation jprel (same order)? maxdiff=",
  float(np.max(np.abs(jp_proprio - jp_art))))

# (b) reconstructed-oracle action vs recorded seq, over the first 30 steps,
#     driving the REAL oracle with vel_cmd_seq and comparing to seq.
w("\n=== (b) oracle(robot.data) vs recorded seq, driven by vel_cmd_seq ===")
diffs = []
for t in range(min(40, T)):
    orc.set_velocity_command(float(velcmd[t, 0]), float(velcmd[t, 1]), float(velcmd[t, 2]))
    act = orc.compute_action()[0].detach().cpu().numpy()
    d = float(np.max(np.abs(act - seq[t])))
    diffs.append(d)
    if t < 8 or t % 10 == 0:
        w(f"  t={t:3d} vel={np.round(velcmd[t],2).tolist()} maxdiff(act,seq)={d:.4f}")
    obs, _, term, trunc, _ = env.step(torch.tensor(act, device="cuda").view(1, -1))
w("first-40 mean maxdiff:", float(np.mean(diffs)), "max:", float(np.max(diffs)))
rp = robot.data.root_pos_w[0]
w("robot after 40 oracle steps: x=%.3f z=%.3f" % (rp[0].item(), rp[2].item()))

fh.close()
env.close(); sim.close()
