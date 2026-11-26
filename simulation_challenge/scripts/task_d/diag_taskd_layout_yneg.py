"""Diagnostic: print the WORLD-frame layout of the Task D scene (robot, box, pit).

We need the pit's true world x and floor z to place the box as a stepping-stone
inside the pit. Rather than guess from terrain-local math, build the real eval env
(TaskDEnvB2Cfg), step once, and read back the actual positions + a height scan so
we can locate the pit edges empirically.

Run:
  CUDA_VISIBLE_DEVICES=7 TMPDIR=/hdd0/sohyun/tmp PYTHONPATH=/hdd0/sohyun \
    python scripts/diag_taskd_layout.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg


def main() -> None:
    print(">>> [diag] building env...", flush=True)
    cfg = TaskDEnvB2Cfg()
    cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg)
    print(">>> [diag] env built, resetting...", flush=True)
    env.reset()
    # one zero-action step so positions settle to the spawned layout
    n_act = env.action_manager.total_action_dim
    print(f">>> [diag] action dim = {n_act}, stepping...", flush=True)
    act = torch.zeros((1, n_act), device=env.unwrapped.device)
    env.step(act)
    print(">>> [diag] stepped, reading positions...", flush=True)

    scene = env.unwrapped.scene
    robot = scene["robot"]
    box = scene["box"]
    env_origin = scene.env_origins[0].tolist()

    rp = robot.data.root_pos_w[0].tolist()
    bp = box.data.root_pos_w[0].tolist()

    print("\n================ TASK D LAYOUT (world frame) ================")
    print(f"env_origin       : {env_origin}")
    print(f"robot root_pos_w : x={rp[0]:.3f}  y={rp[1]:.3f}  z={rp[2]:.3f}")
    print(f"box   root_pos_w : x={bp[0]:.3f}  y={bp[1]:.3f}  z={bp[2]:.3f}")

    # Probe the pit floor by DROPPING the box at a series of x positions along the
    # traverse (y=0). Where it settles low (z << 0) is the pit; where it rests on
    # the surface (z ~ box_half) is solid ground. This needs no raycast API — just
    # physics. box is 0.6 tall => box_half = 0.3, so resting-on-ground z ~ 0.3.
    print("\nprobe pit floor: drop box at each x (y=0), read settled z:", flush=True)
    # Use sim.step directly (NOT env.step) so the env's reset events don't
    # teleport the box back every step. We write the box pose, settle with raw
    # physics, then read it back.
    sim = env.unwrapped.sim
    for x_probe in [-1.5,-1.0,-0.5,-0.2,0.0,0.2,0.5,1.0,1.5,2.0]:
        print(f"   -> probing x={x_probe:+.2f} ...", flush=True)
        root = box.data.root_state_w.clone()
        root[0, 0] = float(x_probe)
        root[0, 1] = -0.45
        root[0, 2] = 2.0
        root[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=root.device)
        root[0, 7:] = 0.0
        box.write_root_state_to_sim(root)
        box.write_data_to_sim()
        for _ in range(100):
            sim.step(render=False)
            box.update(dt=sim.get_physics_dt())
        bz = box.data.root_pos_w[0, 2].item()
        bx = box.data.root_pos_w[0, 0].item()
        tag = "PIT" if bz < 0.0 else "ground"
        print(f"      settled x={bx:+.2f} z={bz:+.3f}  [{tag}]", flush=True)

    print("============================================================\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
