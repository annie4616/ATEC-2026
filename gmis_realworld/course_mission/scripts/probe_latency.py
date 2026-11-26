"""How much control latency can the crouch survive?

The deep squat (0.24 m) is the mission's most marginal manoeuvre, and it is the first
thing that broke when the deployment model added a control delay: at 2 steps (40 ms) the
roll went from 0.03 rad to 1.8 rad — the robot on its side. This sweeps the delay in one
process so the tolerance is measured rather than guessed, and reports walking stability
at the same delays for contrast.

Result feeds COMPUTE_LIMITS["n150"]["action_delay_steps"] in mission.py.
"""

import argparse
import sys
from pathlib import Path

_S = Path(__file__).resolve()
RW = _S.parents[2]
for _p in (str(RW / "source" / "atec_rl_lab"), str(RW / "eunha_rl"), str(RW / "a2pro_rl"), str(RW)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

# Shared machine: GPUs 1-7 are other people's training runs. Pin to GPU 0 BEFORE any
# CUDA-aware import so forgetting the CUDA_VISIBLE_DEVICES prefix cannot land this on
# someone else's device.
import os as _os
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Crouch/walk stability vs control latency.")
parser.add_argument("--speed", type=str, default="s075")
parser.add_argument("--depth", type=str, default="0.24")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--per_type", type=int, default=4)
parser.add_argument("--walk_ckpt", type=str, default="")
parser.add_argument("--grasp_ckpt", type=str, default="")
parser.add_argument("--delays", type=str, default="0,1,2,3")
parser.add_argument("--sections", type=str, default="grass,mud,pebble")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as mu  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import course_mission  # noqa: F401, E402
import course_sim.terrain as ct  # noqa: E402
from course_mission.launch import head_cam_cfg, HEAD_POS, HEAD_ROT, TASK, resolve_ckpts  # noqa: E402
from course_mission.runner import MissionRunner, Profile, ground_z_arc  # noqa: E402


def main():
    device = args_cli.device or "cuda:0"
    ck = resolve_ckpts(args_cli)
    env_cfg = parse_env_cfg(TASK, device=device, num_envs=1)
    env_cfg.layout_seed = args_cli.seed
    env_cfg.per_type = args_cli.per_type
    env_cfg.scene.head_cam = head_cam_cfg()

    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env.reset()
    prof = Profile(name="latency", max_terrain_rank=3)
    r = MissionRunner(env, uenv, env_cfg, prof, ck,
                      {"head": uenv.scene["head_cam"]}, {}, Path("/tmp"), device)
    r.cfg_depth = float(args_cli.depth)
    robot = uenv.scene["robot"]

    def place(section):
        ax = float(ct.SECTION_CENTERS[section])
        ex, ey = ct.curve_map(ax, 0.0)
        root = torch.tensor([[float(ex), float(ey), ground_z_arc(ax) + 0.50]], device=device)
        quat = mu.quat_from_euler_xyz(
            torch.zeros(1, device=device), torch.zeros(1, device=device),
            torch.tensor([float(ct.curve_theta(ax))], device=device))
        robot.write_root_pose_to_sim(torch.cat([root, quat], dim=1))
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
        r.legs.fresh = True
        r._act_queue = []
        for _ in range(90):
            r._write_arm(r.travel_q, open_=False)
            r._step(r.legs.step(torch.zeros(1, 3, device=device)))

    rows = []
    with torch.inference_mode():
        for delay in [int(x) for x in args_cli.delays.split(",")]:
            r.action_delay = delay
            for section in args_cli.sections.split(","):
                place(section)
                # walk first: does the 0.6 m/s gait survive this delay?
                cmd = torch.zeros(1, 3, device=device)
                cmd[0, 0] = 0.6
                rolls = []
                for _ in range(250):
                    r._write_arm(r.travel_q, open_=False)
                    r._step(r.legs.step(cmd))
                    rr, pp, _ = mu.euler_xyz_from_quat(robot.data.root_quat_w)
                    rolls.append(abs(float(mu.wrap_to_pi(rr)[0])))
                walk_roll = float(np.percentile(rolls, 90))
                walk_z = float(robot.data.root_pos_w[0, 2])

                place(section)
                stable = r.crouch_and_check(section)
                d = robot.data
                rr, pp, _ = mu.euler_xyz_from_quat(d.root_quat_w)
                rows.append((delay, section, stable, walk_roll, walk_z,
                             abs(float(mu.wrap_to_pi(rr)[0])),
                             abs(float(mu.wrap_to_pi(pp)[0])),
                             float(d.root_pos_w[0, 2])))
                print(f"[LAT] delay={delay} ({delay * 20} ms) {section:7s} "
                      f"walk: p90|roll|={walk_roll:.3f} z={walk_z:.3f} | "
                      f"crouch: {'STABLE' if stable else 'UNSTABLE'} "
                      f"roll={rows[-1][5]:.3f} pitch={rows[-1][6]:.3f} z={rows[-1][7]:.3f}",
                      flush=True)
                r.stand_up()

    print("[LAT] ===== summary =====", flush=True)
    for delay in sorted({x[0] for x in rows}):
        sub = [x for x in rows if x[0] == delay]
        ok = sum(x[2] for x in sub)
        print(f"[LAT] delay {delay} ({delay * 20:3d} ms): crouch stable {ok}/{len(sub)} | "
              f"worst crouch roll {max(x[5] for x in sub):.3f} rad | "
              f"worst walk p90 roll {max(x[3] for x in sub):.3f} rad", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
