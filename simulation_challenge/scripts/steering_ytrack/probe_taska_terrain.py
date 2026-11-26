"""Measure the REAL Task A terrain height profile along the robot's path (y=0),
by raycasting /World/ground. Tells us exactly what terrain sits at each x, so we
stop guessing where the robot falls.
"""
import os, sys, json
from isaaclab.app import AppLauncher
p = __import__("argparse").ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.enable_cameras = True; a.headless = True
app = AppLauncher(a); sim = app.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.warp import raycast_mesh
from isaaclab.sensors.ray_caster.ray_caster import RayCaster
import atec_rl_lab.tasks  # noqa
from atec_rl_lab.tasks.task_a.env_cfg import TaskAEnvB2Cfg

OUT = "/hdd0/sohyun/atec_2026_tesk_e/taska_terrain_profile.json"

cfg = TaskAEnvB2Cfg(); cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg)
env.reset()

mesh = RayCaster.meshes.get("/World/ground")
if mesh is None and RayCaster.meshes:
    mesh = next(iter(RayCaster.meshes.values()))
assert mesh is not None, "no ground warp mesh"

# sample x from -145 .. +150 at y=0, cast down from z=+10
xs = [round(-145 + 0.5 * i, 2) for i in range(int((150 - (-145)) / 0.5) + 1)]
starts = torch.tensor([[x, 0.0, 10.0] for x in xs], device="cuda:0", dtype=torch.float32)
dirs = torch.zeros_like(starts); dirs[:, 2] = -1.0
hits = raycast_mesh(starts, dirs, mesh)[0]        # (N,3)
h = hits[:, 2]
h = torch.where(torch.isfinite(h), h, torch.zeros_like(h))
profile = [[x, round(float(z), 3)] for x, z in zip(xs, h.tolist())]

# also dump the terrain_sequence order that was actually generated
try:
    ti = env.unwrapped.scene.terrain
    gen_types = getattr(ti.terrain_generator, "sub_terrain_types", None)
except Exception:
    gen_types = None

with open(OUT, "w") as f:
    json.dump({"profile_x_h": profile, "sub_terrain_types": gen_types}, f, indent=2)

# print a compact summary: height at key x's + where big steps are
print("[PROFILE] x -> terrain height (y=0):", flush=True)
prev = None
for x, z in profile:
    if x % 5 == 0:
        print(f"  x={x:+7.1f}  h={z:+.3f}", flush=True)
    if prev is not None and abs(z - prev[1]) > 0.15:
        print(f"    STEP {prev[1]:+.2f}->{z:+.2f} between x={prev[0]:+.1f} and {x:+.1f}", flush=True)
    prev = [x, z]
print(f"[DONE] wrote {OUT}", flush=True)

env.close()
sim.close()
