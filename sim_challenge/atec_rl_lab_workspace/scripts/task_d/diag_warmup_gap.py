import argparse, os, sys
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); a=p.parse_args(); a.enable_cameras=True
app=AppLauncher(a); sim=app.app
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"act"))
from task_d.collector import _reset_asset_root, WARMUP_STEPS
from task_d.walking_oracle import WalkingOracle
cfg=TaskDEnvB2Cfg(); cfg.scene.num_envs=1
env=ManagerBasedRLEnv(cfg); robot=env.unwrapped.scene["robot"]; box=env.unwrapped.scene["box"]
ref=np.load("submission_mem/proprio_jprel_start.npy")
# case1: env.reset()만 (서버가 reset만 하는 시나리오)
obs,_=env.reset()
jr_resetonly = obs["proprio"][0].detach().cpu().numpy()[12:32]
print(">>> reset-only jprel[:13]:", np.round(jr_resetonly[:13],3).tolist())
print(">>> |reset-only - demo_start| max:", round(float(np.abs(jr_resetonly-ref).max()),4))
# case2: reset + asset_root + warmup (데모와 동일)
_reset_asset_root(robot); _reset_asset_root(box)
orc=WalkingOracle("logs/rsl_rl/unitree_b2_rough/2026-06-18_04-37-10/exported/policy.pt",robot,"cuda"); orc.reset()
for _ in range(WARMUP_STEPS): obs,*_=env.step(orc.compute_action())
jr_warm = obs["proprio"][0].detach().cpu().numpy()[12:32]
print(">>> warmup jprel[:13]:", np.round(jr_warm[:13],3).tolist())
print(">>> |warmup - demo_start| max:", round(float(np.abs(jr_warm-ref).max()),4), "(0이어야 정상)")
env.close()
