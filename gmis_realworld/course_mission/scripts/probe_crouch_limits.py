"""Measure LEG joint loading against hardware limits during walk and crouch.

Nothing in any run so far has recorded what the crouch asks of the leg joints — the only
joint instrumentation in the stack is the ARM's (jv_peak/jt_peak inside run_grasp). The
simulator also CLAMPS applied torque at effort_limit_sim, so "over the limit" cannot
appear in sim data at all; the real-robot red flags are

  * SATURATION: |tau| pinned at the limit (the motor has nothing left — on hardware the
    posture sags, the driver heats, or the protection trips),
  * RANGE MARGIN: joint angle close to its URDF stop (a controller that lives at a stop
    has no authority in one direction and bangs the mechanical end),
  * VELOCITY vs the URDF ceiling, and mechanical power |tau*omega| (battery/thermal).

Protocol: teleport to each terrain, 400 steps standing walk-in-place (baseline), then
crouch to the mission depth and hold ~600 steps with the same stability gate the mission
uses, then stand. Per phase, per joint group (hip/thigh/calf): peak |tau|, saturation
fraction, min range margin, peak |omega|, peak and mean total power.

    bash course_mission/scripts/run_probe_crouch_limits.sh
"""
import argparse
import os as _os
import sys
from pathlib import Path as _P

_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
_RW = _P(__file__).resolve().parents[2]
for _p in (str(_RW / "source" / "atec_rl_lab"), str(_RW / "eunha_rl"),
           str(_RW / "a2pro_rl"), str(_RW)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--speed", type=str, default="s075")
parser.add_argument("--depth", type=str, default="0.24")
parser.add_argument("--per_type", type=int, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--walk_ckpt", type=str, default="")
parser.add_argument("--grasp_ckpt", type=str, default="")
parser.add_argument("--hold", type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import gymnasium as gym          # noqa: E402
import numpy as np               # noqa: E402
import torch                     # noqa: E402

import isaaclab.utils.math as mu  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import course_sim.terrain as ct  # noqa: E402
from course_mission.launch import TASK, resolve_ckpts  # noqa: E402
from course_mission.mission import LEG_ORDER, LegDriver, arc_of  # noqa: E402
from course_mission.runner import ground_z_arc  # noqa: E402

# Hardware envelopes, from a2pro_rl/a2pkg/a2.py (actuator cfg) and a2.urdf (module
# docstring): efforts 120/120/180 Nm, velocities 22/22/14.7 rad/s, ranges hip +-1.01,
# thigh front [-2.34, 3.15] / hind [-1.56, 3.94], calf [-2.77, -0.54].
EFFORT = {"hip": 120.0, "thigh": 120.0, "calf": 180.0}
VEL = {"hip": 22.0, "thigh": 22.0, "calf": 14.7}
RANGE = {
    "hip": (-1.01, 1.01),
    "thigh_front": (-2.34, 3.15), "thigh_hind": (-1.56, 3.94),
    "calf": (-2.77, -0.54),
}


def _group(name):
    if "_hip_" in name:
        return "hip"
    if "_calf_" in name:
        return "calf"
    return "thigh_front" if name.startswith(("FR", "FL")) else "thigh_hind"


class Meter:
    def __init__(self, names):
        self.names = names
        self.reset()

    def reset(self):
        self.n = 0
        self.tau_peak = {}
        self.sat = {}
        self.margin = {}
        self.vel_peak = {}
        self.pow_peak = 0.0
        self.pow_sum = 0.0

    def step(self, q, qd, tau):
        self.n += 1
        p_tot = 0.0
        for i, nm in enumerate(self.names):
            g = _group(nm)
            eg = "thigh" if g.startswith("thigh") else g
            t = abs(float(tau[i]))
            self.tau_peak[g] = max(self.tau_peak.get(g, 0.0), t)
            self.sat[g] = self.sat.get(g, 0) + (1 if t >= 0.98 * EFFORT[eg] else 0)
            lo, hi = RANGE[g]
            m = min(float(q[i]) - lo, hi - float(q[i]))
            self.margin[g] = min(self.margin.get(g, 9.9), m)
            self.vel_peak[g] = max(self.vel_peak.get(g, 0.0), abs(float(qd[i])))
            p_tot += t * abs(float(qd[i]))
        self.pow_peak = max(self.pow_peak, p_tot)
        self.pow_sum += p_tot

    def report(self, label):
        print(f"[LIMITS] --- {label} ({self.n} steps) ---", flush=True)
        for g in ("hip", "thigh_front", "thigh_hind", "calf"):
            eg = "thigh" if g.startswith("thigh") else g
            n_j = sum(1 for nm in self.names if _group(nm) == g)
            sat_pct = 100.0 * self.sat.get(g, 0) / max(self.n * n_j, 1)
            print(f"[LIMITS] {g:12s} peak|tau| {self.tau_peak.get(g,0):6.1f}/{EFFORT[eg]:.0f} Nm"
                  f"  sat {sat_pct:5.1f}%  min-margin {self.margin.get(g,9.9):+6.3f} rad"
                  f"  peak|w| {self.vel_peak.get(g,0):5.1f}/{VEL[eg]:.1f} rad/s", flush=True)
        print(f"[LIMITS] {'power':12s} peak {self.pow_peak:6.0f} W   "
              f"mean {self.pow_sum/max(self.n,1):6.0f} W", flush=True)


def main():
    device = "cuda:0"
    ck = resolve_ckpts(args_cli)
    env_cfg = parse_env_cfg(TASK, device=device, num_envs=1)
    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env.reset()
    robot = uenv.scene["robot"]
    legs = LegDriver(robot, device, ck["walk"], ck["crouch"], ck["adapt"])
    names = list(robot.joint_names)
    leg_ids = [names.index(j) for j in LEG_ORDER]
    act = torch.zeros(1, uenv.action_manager.total_action_dim, device=device)
    meter = Meter(LEG_ORDER)
    depth = float(args_cli.depth)

    def step(cmd, measure=False):
        act[:, :12] = legs.step(cmd)
        env.step(act)
        if measure:
            d = robot.data
            tau = getattr(d, "applied_torque", None)
            meter.step(d.joint_pos[0, leg_ids].cpu().numpy(),
                       d.joint_vel[0, leg_ids].cpu().numpy(),
                       tau[0, leg_ids].cpu().numpy() if tau is not None
                       else np.zeros(12))

    zero = torch.zeros(1, 3, device=device)
    crouch = torch.zeros(1, 3, device=device)
    crouch[0, 0] = 1.0

    with torch.inference_mode():
        for _ in range(150):
            step(zero)
        for sec in ("grass", "mud", "pebble"):
            ax = ct.SECTION_CENTERS[sec]
            ex, ey = ct.curve_map(ax, 0.0)
            th = float(ct.curve_theta(ax))
            root = torch.tensor([[float(ex), float(ey), ground_z_arc(ax) + 0.50]],
                                device=device)
            quat = mu.quat_from_euler_xyz(torch.zeros(1, device=device),
                                          torch.zeros(1, device=device),
                                          torch.tensor([th], device=device))
            robot.write_root_pose_to_sim(torch.cat([root, quat], dim=1))
            robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
            legs.fresh = True
            for _ in range(120):                    # settle after the teleport
                step(zero)

            meter.reset()
            for _ in range(400):                    # standing baseline
                step(zero, measure=True)
            meter.report(f"{sec} STAND")

            legs.set_mode("crouch", blend_steps=60)
            for _ in range(120):                    # descend, not measured
                step(crouch)
            meter.reset()
            for _ in range(args_cli.hold):          # held crouch — the load case
                step(crouch, measure=True)
            zvals = float(robot.data.root_pos_w[0, 2]) - ground_z_arc(ax)
            meter.report(f"{sec} CROUCH h={zvals:.3f} (target {depth})")

            legs.set_mode("walk", blend_steps=60)
            meter.reset()
            for _ in range(200):                    # stand back up — transient loads
                step(zero, measure=True)
            meter.report(f"{sec} STAND-UP transient")

    print("[LIMITS] DONE", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
