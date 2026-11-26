"""Hand-scripted box-CLIMB sequence experiment (Task D).

The walking policy can't climb the settled box, so here we author a leg-joint
sequence directly (via WalkingOracle.set_climb_action) and watch — on video —
whether the robot mounts the box and clears the far step.

Flow per attempt:
  1. push the box into the pit + approach it (PushIntoPitCrossController, as usual),
  2. once close, run a staged CLIMB sequence (reach -> plant -> push -> pull),
  3. record a third-person video and report the robot's max x / whether it crossed.

Leg action layout (LEG_JOINT_NAMES order): per leg (hip, thigh, calf) for
FR, FL, RR, RL — 12 values, each an OFFSET from the default pose
(default thigh ~0.8 front / 1.0 rear, calf ~-1.5). Positive thigh lifts the leg
forward/up; positive calf extends (straightens) the shin.

Usage (defaults to the ROUGH walking policy, unitree_b2_rough 2026-06-18):
  python scripts/act/experiment_climb.py \
      --output_dir ./datasets/atec_task_d_climb --max_steps 1400 --headless --enable_cameras
  # or point at a different checkpoint:
  python scripts/act/experiment_climb.py --policy_path <policy.pt> ...
  # legs-only baseline (arm reach disabled):
  python scripts/act/experiment_climb.py --no_arm ...
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Box-climb sequence experiment for Task D.")
# Default to the ROUGH walking policy (the pre-fast-speed one). The fast (vx<=3.5)
# policy is reverted here per request — it learned to stand still at high commands.
# 45-D legs-only interface = compatible with WalkingOracle. Override with --policy_path.
parser.add_argument(
    "--policy_path",
    type=str,
    default="logs/rsl_rl/unitree_b2_rough/2026-06-18_04-37-10/exported/policy.pt",
)
parser.add_argument("--output_dir", type=str, default="./datasets/atec_task_d_climb")
parser.add_argument("--run_tag", type=str, default="")
parser.add_argument("--max_steps", type=int, default=1400)
parser.add_argument("--diag", action="store_true", default=False)
# Toggle the ARM-FORWARD reach during the climb. With it OFF the legs do the climb
# alone (arm held at default) — useful as a baseline to isolate the arm's effect.
parser.add_argument("--no_arm", action="store_true", default=False,
                    help="Disable the arm-forward reach; legs-only climb.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
import numpy as np
import imageio

from isaaclab.envs import ManagerBasedRLEnv
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg

from task_d.walking_oracle import WalkingOracle
from task_d.collector import _reset_asset_root
from task_d.push_controller_cross import PushIntoPitCrossController

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_utils import camera_follow  # noqa: E402

SUCCESS_X = 3.5

# --- CLIMB SEQUENCE (edit these to tune the motion) -------------------------
# Each stage: (name, duration_steps, leg_offsets[12], arm_offsets[8]).
# leg order: FR(h,t,c) FL(h,t,c) RR(h,t,c) RL(h,t,c).
# arm order: arm_joint1..arm_joint8 (1=base yaw, 2=shoulder pitch, 3=elbow,
#            4-6=wrist, 7-8=gripper). All are OFFSETS from the default pose;
# the env applies joint_target = default + ACTION_SCALE(0.5) * offset.
def _legs(fr=(0, 0, 0), fl=(0, 0, 0), rr=(0, 0, 0), rl=(0, 0, 0)):
    return list(fr) + list(fl) + list(rr) + list(rl)


def _arm(j1=0.0, j2=0.0, j3=0.0, j4=0.0, j5=0.0, j6=0.0, j7=0.0, j8=0.0):
    return [j1, j2, j3, j4, j5, j6, j7, j8]


# ARM-FORWARD reach: the climb fails because the robot can't get its rear over the
# 0.4 m step (front legs up, rear stuck, body pitched up -> slides back / falls in).
# So during the climb we REACH THE ARM FORWARD onto the box top to act as a 5th
# support/pull point, helping carry the body's weight over the step. The dominant
# motion is shoulder pitch (j2) forward + elbow (j3) extension so the gripper goes
# out and down ahead of the body. (Signs/magnitudes are a first guess — tune on video.)
_ARM_REACH = _arm(j1=1.0, j2=0.8, j3=0.6, j4=0.0, j5=0.3, j6=0.0)
# _ARM_PRESS = _arm(j1=0.0, j2=1.6, j3=1.0, j4=0.0, j5=0.2, j6=0.0)  # press harder/further
_ARM_PRESS = _arm(j1=0.0, j2=0.0, j3=0.0, j4=0.0, j5=0.0, j6=0.0) 

CLIMB_STAGES = [
    # REACH: lift FRONT legs up/forward toward the box top (thigh+, calf extend)
    # AND extend the arm forward onto the box top to plant a support point.
    ("reach", 12, _legs(fr=(1.0, 1.0, 1.2), fl=(1.0, 1.0, 1.2)), _ARM_REACH),
    # PRESS+PUSH: drive the arm further forward/down (pull the body over) while the
    # REAR legs extend hard to push the body up/forward onto the box.
    ("push", 60, _legs(rr=(0.0, -0.6, 1.2), rl=(0.0, -0.6, 1.2),
                        fr=(0.0, 0.2, 0.4), fl=(0.0, 0.2, 0.4)), _ARM_PRESS),
    # PULL: tuck the REAR legs up onto the box (thigh+, calf fold) to finish, arm
    # still planted forward for support.
    ("pull", 60, _legs(rr=(0.0, 1.0, 1.0), rl=(0.0, 1.0, 1.0)), _ARM_PRESS),
]


def build_env():
    cfg = TaskDEnvB2Cfg()
    cfg.seed           = int(time.time_ns() % (2**31))
    cfg.scene.num_envs = 1
    return ManagerBasedRLEnv(cfg, render_mode="rgb_array")


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)
    env   = build_env()
    dev   = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    box   = env.unwrapped.scene["box"]
    oracle = WalkingOracle(args_cli.policy_path, robot, dev)

    env.reset()
    _reset_asset_root(robot)
    _reset_asset_root(box)
    oracle.reset()

    # SIMPLIFIED SETUP for climb testing: skip the whole push approach. Place the box
    # SETTLED in the pit and the robot RIGHT IN FRONT of it, then trigger the climb
    # immediately. This isolates the climb sequence so we can tune the leg motion
    # without the push-controller approach getting in the way.
    import torch
    org = env.unwrapped.scene.env_origins[0]
    # box settled in pit (env-local 0.2, -0.45, top ~ -0.18 -> center -0.48)
    bs = box.data.root_state_w[0:1].clone()
    bs[0, 0] = org[0] + 4.4; bs[0, 1] = org[1] - 0.45; bs[0, 2] = org[2] - 0.7
    bs[0, 3:7] = torch.tensor([1., 0, 0, 0], device=dev); bs[0, 7:] = 0
    box.write_root_state_to_sim(bs)
    # Geometry (env-local x; pit ~ x in [-0.7, +0.7], box center +0.2, box top z=-0.18):
    #   pit RIM (near edge) ~ env-local x = -0.7.
    # Robot starts well behind the rim (env-local -1.5) so it WALKS up and CLIMBS right at
    # the rim — not deep inside near the box (env-local +0.2) where it just sank into the
    # pit and crouched. Facing +x toward the box.
    rs = robot.data.root_state_w[0:1].clone()
    rs[0, 0] = org[0] + 2.0; rs[0, 1] = org[1] - 0.45; rs[0, 2] = org[2] + 0.55
    rs[0, 3:7] = torch.tensor([1., 0, 0, 0], device=dev); rs[0, 7:] = 0
    robot.write_root_state_to_sim(rs)
    # Let the robot SETTLE longer after the teleport (10 steps wasn't enough — it
    # collapsed before walking). Hold a zero velocity command while it stabilizes.
    oracle.set_velocity_command(0.0, 0.0, 0.0)
    for _ in range(50):
        env.step(oracle.compute_action())

    # Walk (rough policy) all the way to the PIT START at env-local x = -0.7, then climb.
    # The robot uses the walking policy to approach the near rim; the climb sequence only
    # fires once it has actually reached the pit edge.
    CLIMB_TRIGGER_X = org[0].item() + 3.1

    frames = []
    max_x = -99.0
    climb_active = False
    stage_idx = 0
    stage_t = 0
    crossed = False

    # Box settled pose to PIN every step (this eval env doesn't make the box kinematic,
    # so it would otherwise fall through the pit floor and vanish as a stepping stone).
    box_pin = box.data.root_state_w[0:1].clone()
    box_pin[0, 0] = org[0] + 4.4; box_pin[0, 1] = org[1] - 0.45; box_pin[0, 2] = org[2] - 0.7
    box_pin[0, 3:7] = torch.tensor([1., 0, 0, 0], device=dev); box_pin[0, 7:] = 0

    for step in range(args_cli.max_steps):
        # Pin the box in place so it stays a solid stepping stone the robot can climb.
        box.write_root_state_to_sim(box_pin.clone())

        rx = robot.data.root_pos_w[0, 0].item()
        bx = box.data.root_pos_w[0, 0].item()
        bz = box.data.root_pos_w[0, 2].item()

        # Trigger the climb only once the robot has WALKED UP TO the pit rim / box front.
        if not climb_active and rx >= CLIMB_TRIGGER_X:
            climb_active = True
            stage_idx, stage_t = 0, 0

        # if not climb_active:
            # APPROACH: walk forward (+x) toward the rim using the walking policy.
        if not climb_active:
            oracle.set_climb_action(None)
            oracle.set_arm_action(None)   # arm held at default while approaching
            # 로봇 yaw 계산해서 +x로 heading 유지 (vy 대신 wz로 방향 잡기)
            q = robot.data.root_quat_w[0]
            w_, x_, y_, z_ = q[0].item(), q[1].item(), q[2].item(), q[3].item()
            ryaw = np.arctan2(2.0*(w_*z_ + x_*y_), 1.0 - 2.0*(y_*y_ + z_*z_))
            wz = 0.0 if abs(ryaw) < 0.05 else (-1.5 if ryaw > 0 else 1.5)
            oracle.set_velocity_command(3.0, -0.70, wz)   # vy=0, wz로 heading 보정

        else:
            # run the staged climb sequence (legs + arm both hand-scripted).
            # --no_arm holds the arm at default so the legs climb alone (baseline).
            if stage_idx < len(CLIMB_STAGES):
                name, dur, legs, arm = CLIMB_STAGES[stage_idx]
                oracle.set_climb_action(legs)
                oracle.set_arm_action(None if args_cli.no_arm else arm)
                stage_t += 1
                if stage_t >= dur:
                    stage_idx += 1
                    stage_t = 0
            else:
                # after the sequence, drive forward to clear the far platform.
                # Keep the arm reached forward for support while finishing the climb
                # (unless --no_arm).
                oracle.set_climb_action(None)
                oracle.set_arm_action(None if args_cli.no_arm else _ARM_PRESS)
                oracle.set_velocity_command(2.0, 0.0, 0.0)

        action = oracle.compute_action()
        camera_follow(env)
        frame = env.render()
        if frame is not None:
            frames.append(frame[:, :, :3])
        env.step(action)

        rx = robot.data.root_pos_w[0, 0].item()
        max_x = max(max_x, rx)
        if args_cli.diag and step % 25 == 0:
            stg = CLIMB_STAGES[stage_idx][0] if (climb_active and stage_idx < len(CLIMB_STAGES)) else ("post" if climb_active else "approach")
            print(f"[step {step:4d}] robot=({rx:+.2f},{robot.data.root_pos_w[0,1].item():+.2f},"
                  f"{robot.data.root_pos_w[0,2].item():+.2f}) box=({bx:+.2f},{bz:+.2f}) climb={stg}")
        if rx > SUCCESS_X:
            crossed = True
            break

    tag = f"_{args_cli.run_tag}" if args_cli.run_tag else ""
    vp = os.path.join(args_cli.output_dir, f"climb{tag}.mp4")
    if frames:
        imageio.mimwrite(vp, frames, fps=50, quality=7)
    print(f"\n[RESULT] max_x={max_x:+.2f} crossed={crossed} -> {vp}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
