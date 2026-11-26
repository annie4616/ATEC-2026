"""Get the LIVE proprio joint order from the running ObservationManager term (not a
re-resolved SceneEntityCfg, which gave a wrong order). Compute the correct LEG_PERM by
empirically matching proprio joint_pos to robot.data per-joint values."""
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

OUT = "/hdd0/sohyun/tmp/diag_live_order.txt"
fh = open(OUT, "w")
def w(*x): fh.write(" ".join(str(s) for s in x)+"\n"); fh.flush()

cfg = TaskDEnvB2Cfg(); cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg)
robot = env.unwrapped.scene["robot"]; box = env.unwrapped.scene["box"]
obs, _ = env.reset(); _reset_asset_root(robot); _reset_asset_root(box)
orc = WalkingOracle("logs/rsl_rl/unitree_b2_rough/2026-06-18_04-37-10/exported/policy.pt", robot, "cuda")
orc.reset()
# advance a few steps so joints have distinct nonzero values (helps matching)
for _ in range(WARMUP_STEPS):
    obs, *_ = env.step(orc.compute_action())
orc.set_velocity_command(1.0, 0.5, 0.3)
for _ in range(5):
    obs, *_ = env.step(orc.compute_action())

art_names = list(robot.data.joint_names)
w("articulation order:", art_names)

# Try to read the live term's joint order from the ObservationManager.
try:
    om = env.unwrapped.observation_manager
    grp = om._group_obs_term_cfgs["proprio"]
    names = om._group_obs_term_names["proprio"]
    w("proprio term names:", names)
    for nm, cfg_t in zip(names, grp):
        ac = getattr(cfg_t, "params", {}).get("asset_cfg", None)
        if ac is not None and hasattr(ac, "joint_ids"):
            jids = ac.joint_ids
            jids = list(jids) if not isinstance(jids, slice) else list(range(len(art_names)))
            w(f"  term '{nm}' joint_ids:", jids, "-> names:", [art_names[i] for i in jids])
except Exception as e:
    import traceback; w("OM introspection failed:", repr(e)); w(traceback.format_exc())

# EMPIRICAL perm: for each oracle leg joint, find which proprio index has the matching
# joint_pos_rel value (robust to any ordering).
p = obs["proprio"][0].detach().cpu().numpy()
adim = (p.shape[0]-12)//3
jp_proprio = p[12:12+adim]
jpr_art = (robot.data.joint_pos - robot.data.default_joint_pos)[0].detach().cpu().numpy()
# robot.data leg order (oracle):
leg_vals = jpr_art[orc.leg_ids]   # oracle-order leg jprel ground truth
emp_leg_perm = []
for v in leg_vals:
    # index in proprio whose value best matches v
    j = int(np.argmin(np.abs(jp_proprio - v)))
    emp_leg_perm.append(j)
w("\nempirical LEG_PERM (proprio idx matching oracle leg order):", emp_leg_perm)
w("match residual:", float(np.max(np.abs(jp_proprio[emp_leg_perm] - leg_vals))))
# arm
arm_vals = jpr_art[orc.arm_ids]
emp_arm_perm = [int(np.argmin(np.abs(jp_proprio - v))) for v in arm_vals]
w("empirical ARM_PERM:", emp_arm_perm)
# Also dump full proprio jp vs art jp to see the actual permutation
w("\nproprio jp:", np.round(jp_proprio,4).tolist())
w("art     jp:", np.round(jpr_art,4).tolist())
w("orc.leg_ids:", list(orc.leg_ids), "orc.arm_ids:", list(orc.arm_ids))

fh.close(); env.close(); sim.close()
