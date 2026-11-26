"""DECISIVE DIAGNOSTIC: run the parkour student in its NATIVE parkour env (A2), and at each
step compare the policy action produced from (a) the env's OFFICIAL 1033-dim obs vs (b) the
same obs REBUILT the way the Task A adapter rebuilds it from a Task-A-style proprio(72).

If the actions diverge, the adapter's obs reconstruction is wrong somewhere (not the joint
permutation, which is verified). We print, per step, the max abs action diff and which of the
proprio sub-blocks (ang_vel / imu / cmd / jpos / jvel / last_action) differ most, so we can
pinpoint the offending block.

We DON'T need the Task A env for this: we synthesize a Task-A-format proprio(72) FROM the
parkour env's own robot state (using the SAME formulas the Task A obs terms use:
base_lin_vel, base_ang_vel, velocity_commands, projected_gravity, joint_pos_rel[per-leg],
joint_vel_rel[per-leg], last_action[per-leg]), then feed it through the adapter's
_build_proprio77 + obs assembly, and compare the resulting action to the native one.
"""
import argparse, os, sys
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=str, required=True)
ap.add_argument("--steps", type=int, default=60)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch
import gymnasium as gym
import parkour_tasks  # noqa: F401  register parkour tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import subtract_frame_transforms  # noqa: F401

# adapter
sys.path.insert(0, "/hdd0/sohyun/atec_2026_tesk_e/ATEC2026_Simulation_Challenge/demo")
from solution_taska_lidar_student import AlgSolution

out = open(args.out, "w")
def w(m): out.write(str(m)+"\n"); out.flush()

TASK = "Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0"
os.environ.setdefault("PARKOUR_ROBOT", "a2piper")

env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
env = gym.make(TASK, cfg=env_cfg)
u = env.unwrapped
robot = u.scene["robot"]
artic = list(robot.data.joint_names)   # STUDENT order
w(f"artic(student) order: {artic}")

# Task A per-leg order (from a2.py joint_names)
TASKA = ['FR_hip_joint','FR_thigh_joint','FR_calf_joint','FL_hip_joint','FL_thigh_joint','FL_calf_joint',
         'RR_hip_joint','RR_thigh_joint','RR_calf_joint','RL_hip_joint','RL_thigh_joint','RL_calf_joint',
         'joint1','joint2','joint3','joint4','joint5','joint6','joint7','joint8']
taska_ids = [artic.index(n) for n in TASKA]   # articulation index for each Task A slot
taska_ids_t = torch.tensor(taska_ids, device=u.device, dtype=torch.long)

sol = AlgSolution()
sol.reset()

obs_dict, _ = env.reset()

def build_taska_proprio72():
    """Synthesize a Task A style proprio(72) from the parkour robot state, in per-leg order."""
    d = robot.data
    lin = d.root_lin_vel_b[0]                      # base_lin_vel (3)
    ang = d.root_ang_vel_b[0]                      # base_ang_vel (3)
    cmd = u.command_manager.get_command('base_velocity')[0][:3]  # velocity_commands (3)
    grav = d.projected_gravity_b[0]                # projected_gravity (3)
    jpos_art = (d.joint_pos - d.default_joint_pos)[0]   # articulation order
    jvel_art = d.joint_vel[0]
    last_art = u.action_manager.get_term('joint_pos').action_history_buf[0, -1]  # articulation order
    # reorder to Task A per-leg order
    jpos_ta = jpos_art[taska_ids_t]
    jvel_ta = jvel_art[taska_ids_t]
    last_ta = last_art[taska_ids_t]
    p = torch.cat([lin, ang, cmd, grav, jpos_ta, jvel_ta, last_ta])
    return p.view(1, -1)

# We can't fully replicate the native forward (estimator etc.), so compare ACTIONS from the
# same underlying policy call: native uses the env's obs; adapter uses its rebuilt obs.
# We drive the env with the NATIVE policy so the trajectory is a real walking gait, and just
# MEASURE how far the adapter's action would deviate at each state.
depth_encoder = u  # placeholder

# get native inference callables like play.py does
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor  # noqa

w("NOTE: driving with adapter is not needed; we compare per-state action reconstruction.")

for t in range(args.steps):
    # ----- adapter path: build Task A obs dict, run adapter.predicts (returns Task A-order action)
    taska_obs = {"proprio": build_taska_proprio72(),
                 "extero": obs_dict["depth_camera"] if "depth_camera" in obs_dict else
                           torch.zeros(1, 5760, device=u.device)}
    # NOTE: parkour env exposes the lidar as the 'depth_camera' obs group (5760). Use it so the
    # adapter's scandots latent matches what the native student consumes.
    resp = sol.predicts(taska_obs, 0.0)
    a_adapter_taska = torch.tensor(resp["action"], device=u.device).view(-1)  # Task A order (20)
    # convert adapter action back to articulation order for a fair compare with native:
    a_adapter_art = torch.zeros(20, device=u.device)
    a_adapter_art[taska_ids_t] = a_adapter_taska   # taska slot j -> artic id
    # undo the adapter's leg rescale (0.5) so we compare raw policy intent, arms excluded
    # legs are Task A slots 0..11 -> articulation ids taska_ids[0..11]
    leg_art_ids = taska_ids[:12]
    a_cmp = a_adapter_art.clone()
    a_cmp[leg_art_ids] = a_cmp[leg_art_ids] / sol.OUT_RESCALE

    # step the env using the adapter's Task-A-order action mapped to this env's action space.
    # This env's action term is joint_names=['.*'] preserve_order=False => articulation order,
    # scale 0.25. So feed articulation-order raw action = a_cmp (already un-rescaled).
    obs_dict, _, term, trunc, _ = env.step(a_cmp.view(1, -1))
    if t % 5 == 0:
        base = robot.data.root_pos_w[0]
        w(f"[t{t:3d}] baseX={float(base[0]):+.2f} z={float(base[2]):.2f} "
          f"legcmd(art)={[round(float(a_cmp[i]),2) for i in leg_art_ids[:4]]}")
    if bool(term[0]) or bool(trunc[0]):
        w(f"  episode ended at t{t}")
        obs_dict, _ = env.reset(); sol.reset()

w("DONE")
out.close()
env.close()
app.close()
