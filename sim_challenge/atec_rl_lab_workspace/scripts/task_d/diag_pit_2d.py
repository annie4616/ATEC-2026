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
    n_act = env.action_manager.total_action_dim
    env.step(torch.zeros((1, n_act), device=env.unwrapped.device))
    scene = env.unwrapped.scene; box = scene["box"]; sim = env.unwrapped.sim
    print("ENV_ORIGIN", scene.env_origins[0].tolist(), flush=True)
    print("2D PIT MAP (settled z; <0 = PIT). rows=y, cols=x", flush=True)
    ys = [-3.5,-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0]
    xs = [-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0]
    for y in ys:
        row=[]
        for x in xs:
            root = box.data.root_state_w.clone()
            root[0,0]=float(x); root[0,1]=float(y); root[0,2]=3.0
            root[0,3:7]=torch.tensor([1.,0,0,0],device=root.device); root[0,7:]=0.0
            box.write_root_state_to_sim(root); box.write_data_to_sim()
            for _ in range(120):
                sim.step(render=False); box.update(dt=sim.get_physics_dt())
            row.append(box.data.root_pos_w[0,2].item())
        print(f"y={y:+5.1f} | " + " ".join(f"{z:+5.2f}" for z in row), flush=True)
    print("xs        | " + " ".join(f"{x:+5.1f}" for x in xs), flush=True)
    env.close()
if __name__=="__main__":
    main(); simulation_app.close()
