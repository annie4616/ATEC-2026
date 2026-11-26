import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import torch
from isaaclab.envs import ManagerBasedRLEnv
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg
def main():
    cfg = TaskDEnvB2Cfg(); cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg); env.reset()
    env.step(torch.zeros((1, env.action_manager.total_action_dim), device=env.unwrapped.device))
    scene = env.unwrapped.scene; box = scene["box"]; sim = env.unwrapped.sim
    org = scene.env_origins[0]
    print("ENV_ORIGIN", org.tolist(), flush=True)
    # Fine sweep right around the pit: x from -0.5..+2.5 step 0.25 at y=-0.45 (pit lane)
    import numpy as np
    for x in np.arange(-0.5, 2.51, 0.25):
        root = box.data.root_state_w.clone()
        root[0,0]=float(x); root[0,1]=-0.45; root[0,2]=2.5
        root[0,3:7]=torch.tensor([1.,0,0,0],device=root.device); root[0,7:]=0.0
        box.write_root_state_to_sim(root); box.write_data_to_sim()
        for _ in range(200):
            sim.step(render=False); box.update(dt=sim.get_physics_dt())
        bz=box.data.root_pos_w[0,2].item(); bx=box.data.root_pos_w[0,0].item()
        tag="PIT" if bz<0 else ("low" if bz<0.4 else "ground")
        print(f"x={x:+.2f} -> settled x={bx:+.2f} z={bz:+.3f} [{tag}]", flush=True)
    env.close()
if __name__=="__main__":
    main(); simulation_app.close()
