"""Probe the TRUE Task D terrain heights along the box's push lane (y~1.5) and the
robot lane, by ray-casting the scene ground mesh from above. Prints a height profile so
the hard-coded terrain model in submission_taskd_teacher/solution.py can be corrected."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="ATEC-TaskD-B2Piper")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch
import numpy as np
import gymnasium as gym
import warp as wp
from isaaclab_tasks.utils import parse_env_cfg
import atec_rl_lab.tasks  # noqa: F401 register

cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env = gym.make(args.task, cfg=cfg)
env.reset()
scene = env.unwrapped.scene
origin = scene.env_origins[0].cpu().numpy()
print("env_origin:", origin)

# Use the terrain warp mesh to raycast straight down at a grid of world points.
from isaaclab.sensors.ray_caster import raycast_mesh
ti = scene.terrain
# collect the warp meshes
mesh_ids = ti.warp_meshes if hasattr(ti, "warp_meshes") else None
print("terrain attrs:", [a for a in dir(ti) if "mesh" in a.lower()])

def height_at(xs, ys, mesh):
    n = len(xs)
    starts = torch.tensor([[x, y, 10.0] for x, y in zip(xs, ys)], device=env.unwrapped.device, dtype=torch.float32)
    dirs = torch.tensor([[0.0, 0.0, -1.0]] * n, device=env.unwrapped.device, dtype=torch.float32)
    hits = raycast_mesh(starts.unsqueeze(0), dirs.unsqueeze(0), mesh)[0][0]
    return hits[:, 2].cpu().numpy()

# find a mesh handle
mesh = None
for attr in ["warp_meshes", "meshes", "_terrain_meshes"]:
    m = getattr(ti, attr, None)
    if m:
        mesh = list(m.values())[0] if isinstance(m, dict) else m[0]
        print("using mesh from", attr); break

xs = [round(x, 2) for x in np.arange(-3.5, 4.01, 0.25)]
for lane_y in [1.5, 0.0, -0.45]:
    try:
        hs = height_at(xs, [lane_y] * len(xs), mesh)
        print(f"\n--- lane y={lane_y} (world x -> height) ---")
        print(" ".join(f"{x:+.2f}:{h:+.2f}" for x, h in zip(xs, hs)))
    except Exception as e:
        print(f"lane {lane_y} raycast err: {e}")

env.close()
app.close()
