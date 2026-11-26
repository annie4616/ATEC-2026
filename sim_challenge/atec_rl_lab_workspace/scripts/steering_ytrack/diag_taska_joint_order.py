"""Dump Task A B2Piper joint orderings so we can map the parkour teacher correctly.

Prints:
  (1) articulation joint order  = robot.data.joint_names  (what parkour joint_names=[".*"] used)
  (2) proprio obs joint order   = the env's proprio.joint_pos asset_cfg joint_names
  (3) action term joint order   = joint_leg + joint_arm joint_names
and the index maps between them.
"""
import os, sys
from isaaclab.app import AppLauncher
p = __import__("argparse").ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.enable_cameras = True; a.headless = True
app = AppLauncher(a); sim = app.app

import json  # noqa
import torch  # noqa
from isaaclab.envs import ManagerBasedRLEnv
import atec_rl_lab.tasks  # noqa
from atec_rl_lab.tasks.task_a.env_cfg import TaskAEnvB2Cfg

OUT_JSON = "/hdd0/sohyun/atec_2026_tesk_e/diag_taska_joints.json"
result = {}

cfg = TaskAEnvB2Cfg(); cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg)
robot = env.unwrapped.scene["robot"]
obs, _ = env.reset()

art = list(robot.data.joint_names)
result["articulation_order"] = art
print("=== (1) articulation order robot.data.joint_names (parkour .* order) ===")
for i, nm in enumerate(art):
    print(f"  [{i:2d}] {nm}")

# proprio obs joint order
try:
    prop_names = cfg.observations.proprio.joint_pos.params["asset_cfg"].joint_names
except Exception as e:
    prop_names = f"(err {e})"
print("\n=== (2) proprio.joint_pos asset_cfg.joint_names ===")
print(prop_names)

# action term joint order
leg_names = cfg.actions.joint_leg.joint_names
arm_names = cfg.actions.joint_arm.joint_names if cfg.actions.joint_arm else None
print("\n=== (3) action joint_leg.joint_names ===")
print(leg_names)
print("=== (3) action joint_arm.joint_names ===")
print(arm_names)

# proprio preserve_order?
try:
    print("\nproprio joint_pos preserve_order:",
          cfg.observations.proprio.joint_pos.params.get("asset_cfg").__dict__.get("preserve_order"))
except Exception:
    pass

# index maps: articulation index -> position in the action leg list, etc.
def idx_of(names, name):
    return names.index(name) if name in names else None

print("\n=== map: articulation index -> action(leg+arm concat) index ===")
action_order = list(leg_names) + (list(arm_names) if arm_names else [])
art_to_action = [idx_of(action_order, nm) for nm in art]
print("art_to_action:", art_to_action)
print("action_order:", action_order)

# resolve what find_joints('.*') returns WITHOUT preserve_order (parkour used this)
ids_all, names_all = robot.find_joints([".*"])
print("\n=== find_joints('.*') (no preserve_order) -> parkour training order ===")
print("names:", list(names_all))
print("ids  :", list(ids_all))

result["proprio_order"] = list(prop_names) if not isinstance(prop_names, str) else prop_names
result["action_leg_order"] = list(leg_names)
result["action_arm_order"] = list(arm_names) if arm_names else None
result["action_order_concat"] = action_order
result["art_to_action"] = art_to_action
result["parkour_dotstar_names"] = list(names_all)
result["parkour_dotstar_ids"] = [int(i) for i in ids_all]

with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n[DIAG] wrote {OUT_JSON}", flush=True)

env.close()
sim.close()
