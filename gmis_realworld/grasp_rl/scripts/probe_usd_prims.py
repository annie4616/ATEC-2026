"""Inspect the spawned USD object's prim tree: where are the RigidBody/Collision APIs,
and what does the finger contact sensor actually resolve? (debugging force_matrix=0)"""

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
GRASP_RL = _SCRIPT.parents[1]
RW = _SCRIPT.parents[2]
for _p in (str(RW / "source" / "atec_rl_lab"), str(GRASP_RL)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

# Shared machine: GPUs 1-7 are other people's training runs. Pin to GPU 0 BEFORE any
# CUDA-aware import so forgetting the CUDA_VISIBLE_DEVICES prefix cannot land this on
# someone else's device.
import os as _os
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--object_usd", type=str,
                    default=str(RW / "atec_robot_model" / "objects" / "task_b" / "011_banana.usd"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402

import graspenv  # noqa: F401, E402
from graspenv.grasp_env_cfg import B2PiperGraspStage2EnvCfg  # noqa: E402


def main():
    cfg = B2PiperGraspStage2EnvCfg()
    cfg.scene.num_envs = 2
    cfg.sim.device = args_cli.device or "cuda:0"
    cfg.scene.object.spawn = sim_utils.UsdFileCfg(
        usd_path=args_cli.object_usd,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
    )
    cfg.events.object_scale = None
    cfg.events.object_mass = None
    cfg.events.object_material = None

    env = gym.make("ATEC-Grasp-B2Piper-Stage2-v0", cfg=cfg)
    env.reset()
    uenv = env.unwrapped

    from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402

    stage = uenv.scene.stage
    root = stage.GetPrimAtPath("/World/envs/env_0/Object")
    print("RESULT| Object subtree:", flush=True)
    for prim in Usd.PrimRange(root):
        apis = []
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            apis.append("RigidBody")
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            apis.append("Collision")
        if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            apis.append("ContactReport")
        print(f"RESULT|   {prim.GetPath()}  [{prim.GetTypeName()}]  {apis}", flush=True)

    sensor = uenv.scene["contact_finger_l"]
    print(f"RESULT| sensor body_names={sensor.body_names}", flush=True)
    print(f"RESULT| sensor cfg filter={sensor.cfg.filter_prim_paths_expr}", flush=True)
    fm = sensor.data.force_matrix_w
    print(f"RESULT| force_matrix shape={None if fm is None else tuple(fm.shape)}", flush=True)
    try:
        view = sensor.contact_physx_view
        print(f"RESULT| physx view filter_count={view.filter_count}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"RESULT| physx view introspect failed: {e!r}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
