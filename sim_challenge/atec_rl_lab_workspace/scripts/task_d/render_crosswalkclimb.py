"""Save a still image of the CrossWalkClimb Task D env (box settled in the pit,
robot in front about to cross + climb the far rim). Single env, head-on pit view
(same camera as play.py's Cross video). Steps a few frames so the reset poses
settle, then writes a PNG. Single fresh process (avoids the repeated-env render
hang). Abs paths."""
import os
from isaaclab.app import AppLauncher
ap = __import__("argparse").ArgumentParser(); AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args(); a.enable_cameras = True; a.headless = True
app = AppLauncher(a); sim = app.app

import numpy as np, torch
from isaaclab.envs import ManagerBasedRLEnv
import atec_rl_lab.tasks  # noqa  (registers eval tasks / scene assets)
from atec_rl_lab.train.locomotion.taskd.config.quadruped.unitree_b2.taskd_crosswalkclimb_env_cfg import (
    UnitreeB2TaskDCrossWalkClimbEnvCfg,
)

OUT = "/hdd0/sohyun/tmp/crosswalkclimb_env.png"

env_cfg = UnitreeB2TaskDCrossWalkClimbEnvCfg()
env_cfg.scene.num_envs = 1
# Head-on pit view (robot faces +x toward the pit); behind & above, looking forward.
env_cfg.viewer.origin_type = "asset_root"
env_cfg.viewer.asset_name = "robot"
env_cfg.viewer.env_index = 0
env_cfg.viewer.eye = (-3.0, 1.5, 3.5)
env_cfg.viewer.lookat = (1.0, 0.0, -0.3)

env = ManagerBasedRLEnv(env_cfg, render_mode="rgb_array")
env.reset()
# step a few frames with zero action so box/robot settle at their reset poses
act = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
frame = None
for i in range(12):
    env.step(act)
    frame = env.render()
if frame is not None:
    try:
        import imageio.v2 as imageio
        imageio.imwrite(OUT, np.asarray(frame))
    except Exception:
        from PIL import Image
        Image.fromarray(np.asarray(frame)).save(OUT)
    print("SAVED", OUT, np.asarray(frame).shape)
else:
    print("NO_FRAME")
env.close(); sim.close()
