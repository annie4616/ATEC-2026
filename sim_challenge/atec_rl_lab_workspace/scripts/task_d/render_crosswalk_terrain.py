"""Render the Task D CrossWalk training terrain to PNG images for inspection.

Spawns a single CrossWalk env, resets it (box settled half-sunk in the pit, robot
on the -y lane in front), and saves a few camera views so we can visually confirm
the demo-matched geometry: pit (full y width), platform (+y half only), box filling
the pit center, robot on the platform-free -y lane.

Run (env_isaaclab):
  CUDA_VISIBLE_DEVICES=2 TMPDIR=/hdd0/sohyun/tmp PYTHONPATH=/hdd0/sohyun \
  python scripts/render_crosswalk_terrain.py --out /hdd0/sohyun/tmp/crosswalk_terrain
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default="/hdd0/sohyun/tmp/crosswalk_terrain")
parser.add_argument("--task", type=str, default="ATEC-TaskD-RL-CrossWalk-Unitree-B2-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import importlib.util
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.envs import ManagerBasedRLEnv


def _load_crosswalk_cfg_cls():
    """Load the CrossWalk env cfg from its file WITHOUT importing the package
    __init__ chain (which pulls in rsl_rl agent cfgs and a brittle VecEnv import).
    The taskd MDP modules it references are imported normally by the file."""
    base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "source", "atec_rl_lab", "atec_rl_lab", "train", "locomotion", "taskd",
        "config", "quadruped", "unitree_b2", "taskd_crosswalk_env_cfg.py",
    )
    spec = importlib.util.spec_from_file_location("taskd_crosswalk_env_cfg", base)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.UnitreeB2TaskDCrossWalkEnvCfg


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    cfg_cls = _load_crosswalk_cfg_cls()
    env_cfg = cfg_cls()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()
    # step a few times so the box settles and physics shows the real resting pose
    act = torch.zeros((1, env.action_manager.total_action_dim), device=args_cli.device)
    for _ in range(30):
        env.step(act)

    robot = env.scene["robot"]
    box = env.scene["box"]
    rp = robot.data.root_pos_w[0].cpu().numpy()
    bp = box.data.root_pos_w[0].cpu().numpy()
    print(f"[geom] robot world pos = ({rp[0]:+.2f},{rp[1]:+.2f},{rp[2]:+.2f})", flush=True)
    print(f"[geom] box   world pos = ({bp[0]:+.2f},{bp[1]:+.2f},{bp[2]:+.2f})", flush=True)

    # Spawn standalone cameras at fixed world poses looking at the pit (world x~0).
    views = {
        "side":   (dict(pos=(0.0, -4.5, 2.0),  look=(0.0, 0.0, -0.3))),   # from -y, see -y lane + pit cross-section
        "iso":    (dict(pos=(-3.5, -3.5, 3.0), look=(0.5, 0.0, -0.3))),   # isometric behind/-y
        "top":    (dict(pos=(0.0, 0.0, 7.0),   look=(0.0, 0.0, 0.0))),    # top-down: pit (full y) + platform (+y half)
        "behind": (dict(pos=(-3.0, -0.45, 1.6),look=(1.5, -0.45, -0.2))), # robot's eye down the -y lane toward the cross
    }
    for name, v in views.items():
        cam_cfg = CameraCfg(
            prim_path=f"/World/RenderCam_{name}",
            height=720, width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.1, 100.0)),
        )
        cam = Camera(cam_cfg)
        cam._initialize_impl()
        eye = torch.tensor([v["pos"]], device=args_cli.device, dtype=torch.float32)
        tgt = torch.tensor([v["look"]], device=args_cli.device, dtype=torch.float32)
        cam.set_world_poses_from_view(eye, tgt)
        for _ in range(6):
            env.sim.step()
            cam.update(dt=0.0)
        img = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        try:
            from PIL import Image
            Image.fromarray(img).save(os.path.join(args_cli.out, f"terrain_{name}.png"))
        except Exception:
            np.save(os.path.join(args_cli.out, f"terrain_{name}.npy"), img)
        print(f"[saved] {name}: {os.path.join(args_cli.out, f'terrain_{name}.png')}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
