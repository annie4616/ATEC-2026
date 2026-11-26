"""Dump the pure-B2 (12-DoF) articulation joint order used by the parkour teacher.

Run with PARKOUR_ROBOT=b2 so default_cfg loads UNITREE_B2_CFG (legs only).
Writes the articulation joint_names order to a JSON so we can build the
proprio->teacher and teacher->action permutations for the Task A (B2Piper) env.
"""
import os, sys, json
from isaaclab.app import AppLauncher
p = __import__("argparse").ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.enable_cameras = True; a.headless = True
app = AppLauncher(a); sim = app.app

import torch  # noqa
import gymnasium as gym  # noqa
import parkour_tasks  # noqa: F401  (registers Isaac-Extreme-Parkour-* envs)
from isaaclab_tasks.utils import parse_env_cfg

OUT = "/hdd0/sohyun/atec_2026_tesk_e/diag_b2_12dof_order.json"
TASK = "Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0"

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1, use_fabric=True)
env = gym.make(TASK, cfg=cfg)
robot = env.unwrapped.scene["robot"]
env.reset()

art = list(robot.data.joint_names)
result = {"articulation_order": art, "num_joints": len(art)}
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(f"[DIAG] njoints={len(art)} wrote {OUT}", flush=True)
print(art, flush=True)

env.close()
sim.close()
