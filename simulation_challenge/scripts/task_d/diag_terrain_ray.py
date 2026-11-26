import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import torch, numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg
import warp as wp
from isaaclab.utils.warp import raycast_mesh
def main():
    cfg = TaskDEnvB2Cfg(); cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg); env.reset()
    env.step(torch.zeros((1, env.action_manager.total_action_dim), device=env.unwrapped.device))
    scene = env.unwrapped.scene
    print("ENV_ORIGIN", scene.env_origins[0].tolist(), flush=True)
    # get the terrain warp mesh
    ti = scene.terrain
    mesh = list(ti.warp_meshes.values())[0]
    dev = env.unwrapped.device
    xs = np.arange(-3.0, 5.01, 0.5)
    ys = [-2.0,-1.0,-0.5,0.0,0.5,1.5]
    print("TERRAIN HEIGHT (ray down from z=5). rows=y cols=x", flush=True)
    for y in ys:
        pts = torch.tensor([[float(x), float(y), 5.0] for x in xs], device=dev, dtype=torch.float32)
        dirs = torch.tensor([[0.,0.,-1.]]*len(xs), device=dev, dtype=torch.float32)
        hits = raycast_mesh(pts.unsqueeze(0), dirs.unsqueeze(0), mesh.id)[0][0]
        zr = hits[:,2].cpu().numpy()
        print(f"y={y:+5.1f} | " + " ".join(f"{z:+5.2f}" for z in zr), flush=True)
    print("xs        | " + " ".join(f"{x:+5.1f}" for x in xs), flush=True)
    env.close()
if __name__=="__main__":
    main(); simulation_app.close()
