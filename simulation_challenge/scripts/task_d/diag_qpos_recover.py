import argparse, os, sys, types
_SRC=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"source","atec_rl_lab")
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); a=p.parse_args(); a.enable_cameras=True
app=AppLauncher(a); sim=app.app
import numpy as np, torch
from isaaclab.envs import ManagerBasedRLEnv
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"act"))
from task_d.collector import _reset_asset_root, WARMUP_STEPS
from task_d.walking_oracle import WalkingOracle
cfg=TaskDEnvB2Cfg(); cfg.scene.num_envs=1
env=ManagerBasedRLEnv(cfg); robot=env.unwrapped.scene["robot"]; box=env.unwrapped.scene["box"]
obs,_=env.reset(); _reset_asset_root(robot); _reset_asset_root(box)
orc=WalkingOracle("logs/rsl_rl/unitree_b2_rough/2026-06-18_04-37-10/exported/policy.pt",robot,"cuda"); orc.reset()
for _ in range(WARMUP_STEPS):
    obs,*_=env.step(orc.compute_action())
# warmup 후(=데모 시작 시점) 실제 qpos vs proprio복원 vs qpos_start
actual=robot.data.joint_pos[0].cpu().numpy()
prop=obs["proprio"][0].detach().cpu().numpy()
adim=(prop.shape[0]-12)//3
dft=np.array([0.1,-0.1,0.1,-0.1,0.8,0.8,1.0,1.0,0.0,-1.5,-1.5,-1.5,-1.5,0,0,0,0,0,0,0],dtype=np.float32)
recov=prop[12:12+adim]+dft[:adim]
qs=np.load("submission_mem/qpos_start.npy")
print(">>> actual  joint_pos[:13]:", np.round(actual[:13],3).tolist())
print(">>> recov(prop+dft)[:13] :", np.round(recov[:13],3).tolist())
print(">>> qpos_start[:13]      :", np.round(qs[:13],3).tolist())
print(">>> |recov-actual| max:", round(float(np.abs(recov-actual).max()),4))
print(">>> |actual-qpos_start| max:", round(float(np.abs(actual-qs).max()),4), "(작아야 정렬 즉시통과)")
print(">>> |recov-qpos_start| max:", round(float(np.abs(recov-qs).max()),4))
env.close()
