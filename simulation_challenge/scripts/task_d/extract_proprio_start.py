"""warmup 후(=데모 수집 시작 시점)의 proprio_rel(joint_pos) 벡터를 추출해 저장.
이걸 solution_mem의 정렬 기준으로 쓴다(순서/default 복원 불필요, eval proprio와 그대로 비교)."""
import argparse, os, sys
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
prop=obs["proprio"][0].detach().cpu().numpy().astype(np.float32)   # full proprio at demo start
adim=(prop.shape[0]-12)//3
jp_rel = prop[12:12+adim]                                          # joint_pos_rel (정렬 기준)
print(">>> proprio dim:", prop.shape[0], "adim:", adim)
print(">>> demo-start joint_pos_rel[:13]:", np.round(jp_rel[:13],3).tolist())
np.save("submission_mem/proprio_jprel_start.npy", jp_rel)
print(">>> saved submission_mem/proprio_jprel_start.npy", jp_rel.shape)
env.close()
