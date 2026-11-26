"""Raycast the Task A /World/ground into a 2-D (x,y) height grid and dump it to JSON.
The full-course maze uses this so each wall segment can be placed at the terrain height
under it (walls otherwise bury on up-slopes / float on down-slopes). Grid covers the whole
course x in [-145,150] and the corridor band y in [-10,10].
"""
import os, json
from isaaclab.app import AppLauncher
p = __import__("argparse").ArgumentParser()
p.add_argument("--corridor", action="store_true",
               help="probe the CORRIDOR-SYMMETRIC terrain (t_ramp/t_stairs) instead of the "
                    "default pyramid terrain; writes taska_terrain_grid_sym.json")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.enable_cameras = True; a.headless = True
app = AppLauncher(a); sim = app.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.warp import raycast_mesh
from isaaclab.sensors.ray_caster.ray_caster import RayCaster
import atec_rl_lab.tasks  # noqa
from atec_rl_lab.tasks.task_a.env_cfg import TaskAEnvB2Cfg

if a.corridor:
    from atec_rl_lab.tasks.task_a.terrain import TASK_A_CORRIDOR_TERRAIN_CFG
    OUT = "/hdd0/sohyun/atec_2026_tesk_e/taska_terrain_grid_sym.json"
else:
    OUT = "/hdd0/sohyun/atec_2026_tesk_e/taska_terrain_grid.json"

cfg = TaskAEnvB2Cfg(); cfg.scene.num_envs = 1
if a.corridor:
    cfg.scene.terrain = TASK_A_CORRIDOR_TERRAIN_CFG
    cfg.sim.physics_material = cfg.scene.terrain.physics_material
env = ManagerBasedRLEnv(cfg)
env.reset()

mesh = RayCaster.meshes.get("/World/ground")
if mesh is None and RayCaster.meshes:
    mesh = next(iter(RayCaster.meshes.values()))
assert mesh is not None, "no ground warp mesh"

X0, X1, DX = -145.0, 150.0, 0.5
Y0, Y1, DY = -10.0, 10.0, 0.5
xs = [round(X0 + DX * i, 2) for i in range(int((X1 - X0) / DX) + 1)]
ys = [round(Y0 + DY * j, 2) for j in range(int((Y1 - Y0) / DY) + 1)]

starts = torch.tensor([[x, y, 10.0] for x in xs for y in ys],
                      device="cuda:0", dtype=torch.float32)
dirs = torch.zeros_like(starts); dirs[:, 2] = -1.0
hits = raycast_mesh(starts, dirs, mesh)[0]           # (N,3)
h = hits[:, 2]
h = torch.where(torch.isfinite(h), h, torch.zeros_like(h)).cpu().tolist()

# grid[i][j] = height at (xs[i], ys[j])
grid = []
k = 0
for _ in xs:
    row = []
    for _ in ys:
        row.append(round(h[k], 3)); k += 1
    grid.append(row)

with open(OUT, "w") as f:
    json.dump({"x0": X0, "dx": DX, "nx": len(xs),
               "y0": Y0, "dy": DY, "ny": len(ys), "grid": grid}, f)
print(f"[DONE] wrote {OUT}  ({len(xs)}x{len(ys)} grid)", flush=True)

env.close()
sim.close()
