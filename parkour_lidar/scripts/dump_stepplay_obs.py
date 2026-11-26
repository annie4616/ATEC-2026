"""Dump the REAL 753-dim teacher obs from the parkour StepPlay env (12-DoF B2),
plus the robot base state, for the first few steps. Saves to JSON so we can
compare against the Task A adapter's synthesized obs term-by-term.

Run with PARKOUR_ROBOT=b2 PARKOUR_NJOINTS=12.
"""
import os, sys, json
from isaaclab.app import AppLauncher
p = __import__("argparse").ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.enable_cameras = True; a.headless = True
app = AppLauncher(a); sim = app.app

import torch  # noqa
import gymnasium as gym  # noqa
import parkour_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

OUT = "/hdd0/sohyun/atec_2026_tesk_e/stepplay_obs_dump.json"
TASK = "Isaac-Extreme-Parkour-Teacher-Unitree-Go2-StepPlay-v0"

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1, use_fabric=True)
env = gym.make(TASK, cfg=cfg)
robot = env.unwrapped.scene["robot"]
obs, _ = env.reset()

def obs_vec(o):
    # obs may be dict of groups; the teacher policy obs group name is 'policy'
    if isinstance(o, dict):
        for k in ("policy", "obs", "teacher"):
            if k in o:
                return o[k]
        # fallback: first value
        return next(iter(o.values()))
    return o

dump = {"joint_names": list(robot.data.joint_names), "steps": []}
default_jp = robot.data.default_joint_pos[0].detach().cpu().tolist()
dump["default_joint_pos"] = default_jp

for step in range(6):
    v = obs_vec(obs)[0].detach().cpu()
    rec = {
        "step": step,
        "obs_dim": int(v.shape[-1]),
        "proprio53": v[0:53].tolist(),
        "scan132_stats": {
            "min": float(v[53:185].min()), "max": float(v[53:185].max()),
            "mean": float(v[53:185].mean()),
            "first12": v[53:65].tolist(),   # first y-row (x sweep)
        },
        "priv_e9": v[185:194].tolist(),
        "base_pos_w": robot.data.root_pos_w[0].detach().cpu().tolist(),
        "base_quat_w": robot.data.root_quat_w[0].detach().cpu().tolist(),
        "base_lin_vel_b": robot.data.root_lin_vel_b[0].detach().cpu().tolist(),
        "base_ang_vel_b": robot.data.root_ang_vel_b[0].detach().cpu().tolist(),
        "joint_pos": robot.data.joint_pos[0].detach().cpu().tolist(),
    }
    dump["steps"].append(rec)
    # zero action to keep it near spawn for the first couple steps
    act = torch.zeros((1, robot.num_joints), device="cuda:0")
    obs, *_ = env.step(act)

with open(OUT, "w") as f:
    json.dump(dump, f, indent=2)
print(f"[DUMP] wrote {OUT}", flush=True)

env.close()
sim.close()
