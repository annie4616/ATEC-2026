"""Extract the settled crouch pose (12 leg joint angles + base height) from each crouch policy.

Adapted from scripts/rsl_rl/play.py (checkpoint loading pattern). One sim session iterates
every crouch_sweep depth checkpoint: command c=1, let the policy settle, average the last
seconds, and write grasp_rl/banks/crouch_poses.json.

All paths resolve INSIDE real_world_preliminary/ATEC2026_Simulation_Challenge (rule 3);
crouch_sweep/runs is accessed read-only (rule 4).

Run:
  cd real_world_preliminary/ATEC2026_Simulation_Challenge
  CUDA_VISIBLE_DEVICES=5 ~/.conda/envs/isaaclab/bin/python \
      grasp_rl/scripts/extract_crouch_pose.py --headless --device cuda:0
"""

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
GRASP_RL = _SCRIPT.parents[1]              # grasp_rl/
RW = _SCRIPT.parents[2]                    # real_world_preliminary/ATEC2026_Simulation_Challenge
# our atec_rl_lab copy and our crouchpkg copy MUST win over any editable install elsewhere
for _p in (str(RW / "source" / "atec_rl_lab"), str(GRASP_RL)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Extract settled crouch poses from crouch_sweep checkpoints.")
parser.add_argument("--runs_root", type=str, default=str(RW / "logs" / "rsl_rl" / "crouch_sweep" / "runs"))
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--settle_time", type=float, default=2.0, help="Stand phase before commanding crouch [s].")
parser.add_argument("--crouch_time", type=float, default=6.0, help="Crouch phase length [s].")
parser.add_argument("--measure_time", type=float, default=2.0, help="Averaging window at the end [s].")
parser.add_argument("--out", type=str, default=str(GRASP_RL / "banks" / "crouch_poses.json"))
parser.add_argument("--extra_runs", type=str, default="",
                    help="Extra tag:run_dir_name pairs, comma-separated (e.g. '0.3:b2_crouch_cycle_2026-06-30_06-36-28').")
parser.add_argument("--skip_scan", action="store_true", help="Only process --extra_runs; merge into existing json.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import atec_rl_lab  # noqa: E402
import crouchpkg  # noqa: F401, E402  (registers ATEC-Crouch-B2-v0 from OUR copy)
from crouchpkg import mdp_crouch  # noqa: E402
from crouchpkg.crouch_env_cfg import B2CrouchEnvCfg, STAND_HEIGHT  # noqa: E402

# rule-3 guard: refuse to run against any atec_rl_lab outside this repo
assert str(RW) in atec_rl_lab.__file__, f"atec_rl_lab resolved OUTSIDE the working repo: {atec_rl_lab.__file__}"
assert str(GRASP_RL) in mdp_crouch.__file__, f"crouchpkg resolved outside grasp_rl: {mdp_crouch.__file__}"


def _disable_all_terms(cfg_obj):
    if cfg_obj is None:
        return
    for attr in dir(cfg_obj):
        if attr.startswith("__"):
            continue
        value = getattr(cfg_obj, attr)
        if value is not None and not callable(value):
            setattr(cfg_obj, attr, None)


def find_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {run_dir}")
    return ckpts[-1]


def main():
    device = args_cli.device if getattr(args_cli, "device", None) else "cuda:0"
    runs_root = Path(args_cli.runs_root)
    depth_runs = {}
    if not args_cli.skip_scan:
        for d in sorted(runs_root.iterdir()):
            if d.is_dir() and d.name.startswith("crouch0."):
                tag = d.name.split("_")[0].replace("crouch", "")   # "0.26"
                depth_runs[tag] = d
    for pair in filter(None, args_cli.extra_runs.split(",")):
        tag, name = pair.split(":", 1)
        depth_runs[tag] = runs_root / name
    if not depth_runs:
        raise SystemExit(f"No crouch run dirs selected under {runs_root}")
    print(f"[EXTRACT] depths found: {sorted(depth_runs)}", flush=True)

    # deterministic settle: no events at all, no noise, no terminations
    env_cfg = B2CrouchEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.observations.policy.enable_corruption = False
    _disable_all_terms(env_cfg.events)
    _disable_all_terms(env_cfg.terminations)
    _disable_all_terms(env_cfg.curriculum)
    env_cfg.episode_length_s = 1000.0
    env_cfg.commands.base_velocity.debug_vis = False

    agent_cfg = load_cfg_from_registry("ATEC-Crouch-B2-v0", "rsl_rl_cfg_entry_point")
    agent_cfg.device = device

    env = gym.make("ATEC-Crouch-B2-v0", cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dt = uenv.step_dt
    joint_names = list(robot.joint_names)
    cmd = mdp_crouch._get_cmd(uenv)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    n_settle = int(round(args_cli.settle_time / dt))
    n_crouch = int(round(args_cli.crouch_time / dt))
    n_meas = int(round(args_cli.measure_time / dt))

    out_path = Path(args_cli.out)
    if out_path.exists():
        results = json.loads(out_path.read_text())  # merge into existing entries
    else:
        results = {"stand_height": STAND_HEIGHT, "control_dt": dt, "joint_names": joint_names, "poses": {}}
    with torch.inference_mode():
        for tag in sorted(depth_runs, key=float, reverse=True):
            ckpt = find_checkpoint(depth_runs[tag])
            runner.load(str(ckpt))
            policy = runner.get_inference_policy(device=uenv.device)
            env.reset()
            obs = env.get_observations()

            q_buf, z_buf, g_buf = [], [], []
            for phase_c, steps in ((0.0, n_settle), (1.0, n_crouch)):
                cmd[:] = phase_c
                obs = env.get_observations()
                for i in range(steps):
                    actions = policy(obs)
                    obs, _, _, _ = env.step(actions)
                    if phase_c > 0.5 and i >= steps - n_meas:
                        q_buf.append(robot.data.joint_pos.cpu().clone())
                        z_buf.append((robot.data.root_pos_w[:, 2] - uenv.scene.env_origins[:, 2]).cpu().clone())
                        g_buf.append(robot.data.projected_gravity_b[:, 2].cpu().clone())

            q = torch.stack(q_buf)            # [T, N, J]
            z = torch.stack(z_buf)            # [T, N]
            q_mean = q.mean(dim=(0, 1))
            q_std_t = q.std(dim=0).mean().item()   # temporal jitter (settled if small)
            entry = {
                "checkpoint": str(ckpt),
                "base_height_mean": round(z.mean().item(), 4),
                "base_height_std": round(z.std().item(), 4),
                "projected_gravity_z": round(torch.stack(g_buf).mean().item(), 4),
                "joint_pos": {n: round(q_mean[j].item(), 5) for j, n in enumerate(joint_names)},
                "joint_temporal_std": round(q_std_t, 5),
            }
            results["poses"][tag] = entry
            print(f"[EXTRACT] depth {tag}: z={entry['base_height_mean']:.3f}±{entry['base_height_std']:.3f} "
                  f"jitter={q_std_t:.4f} grav_z={entry['projected_gravity_z']:.3f}", flush=True)

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[EXTRACT] written: {args_cli.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
