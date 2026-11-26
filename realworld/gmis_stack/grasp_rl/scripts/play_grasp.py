"""Play a trained grasp policy and record a video (few envs, static close-up camera).

Successful episodes terminate early and auto-reset with a new random object, so the video
shows a continuous stream of grasp attempts. Also prints the success count for the rollout.

Run:
  cd real_world_preliminary/ATEC2026_Simulation_Challenge
  CUDA_VISIBLE_DEVICES=5 ~/.conda/envs/isaaclab/bin/python \
      grasp_rl/scripts/play_grasp.py --headless --device cuda:0 \
      --checkpoint grasp_rl/runs/<run>/model_4999.pt --video
"""

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

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="ATEC-Grasp-B2Piper-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--object_usd", type=str, default=None,
                    help="Evaluate on a specific USD object (e.g. task_b assets) instead of the "
                         "procedural primitives; disables size/mass/material randomization.")
parser.add_argument("--object_quat", type=str, default=None,
                    help="Spawn orientation for --object_usd as 'w,x,y,z' (e.g. lay the narrow axis "
                         "horizontal: '0.7071,0,0.7071,0').")
parser.add_argument("--shape", type=str, default="both", choices=["both", "cylinder", "cuboid"],
                    help="Restrict procedural objects to one primitive (e.g. cylinder for the can).")
parser.add_argument("--crouch_depth", type=str, default="0.2")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=800, help="Control steps to run/record (@50 Hz).")
parser.add_argument("--width_range", type=str, default="0.030,0.060")
parser.add_argument("--length_range", type=str, default=None, help="Object length (y) range; default = width_range.")
parser.add_argument("--height_range", type=str, default="0.060,0.130")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--out_dir", type=str, default=str(GRASP_RL / "videos"))
parser.add_argument("--heur", type=str, default="",
                    help="Comma list of inference-time heuristics: 'shield' (geometric camera "
                         "keep-out action override), 'open' (force gripper open while the TCP is "
                         "far from the estimated grasp point), 'reopen' (detect air-grasps via "
                         "finger width and retry). Obs-side heuristics use env vars "
                         "GRASP_PN_EMA / GRASP_DIMS_PRIOR.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
import os as _os

_os.environ["GRASP_CROUCH_DEPTH"] = args_cli.crouch_depth
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import atec_rl_lab  # noqa: E402
import graspenv  # noqa: F401, E402
from graspenv.agents import GraspPPORunnerCfg  # noqa: E402
from graspenv.grasp_env_cfg import _CUBE_BASE, B2PiperGraspEnvCfg, B2PiperGraspStage2EnvCfg  # noqa: E402

assert str(RW) in atec_rl_lab.__file__, f"atec_rl_lab outside repo: {atec_rl_lab.__file__}"


def main():
    device = args_cli.device if getattr(args_cli, "device", None) else "cuda:0"
    w_lo, w_hi = (float(s) for s in args_cli.width_range.split(","))

    env_cfg = B2PiperGraspStage2EnvCfg() if "Stage2" in args_cli.task else B2PiperGraspEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.object_usd:
        import isaaclab.sim as sim_utils

        # real asset: keep authored mass/materials, apply only the locked grasp-physics values
        env_cfg.scene.object.spawn = sim_utils.UsdFileCfg(
            usd_path=args_cli.object_usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        )
        env_cfg.events.object_scale = None
        env_cfg.events.object_mass = None
        env_cfg.events.object_material = None
        if args_cli.object_quat:
            quat = tuple(float(s) for s in args_cli.object_quat.split(","))
            env_cfg.scene.object.init_state = env_cfg.scene.object.init_state.replace(rot=quat)
    env_cfg.sim.device = device
    if args_cli.shape != "both" and args_cli.object_usd is None:
        keep = "CylinderCfg" if args_cli.shape == "cylinder" else "CuboidCfg"
        env_cfg.scene.object.spawn.assets_cfg = [
            a for a in env_cfg.scene.object.spawn.assets_cfg if type(a).__name__ == keep
        ]
    if env_cfg.events.object_scale is not None:
        l_lo, l_hi = (float(s) for s in (args_cli.length_range or args_cli.width_range).split(","))
        h_lo, h_hi = (float(s) for s in args_cli.height_range.split(","))
        bx, _, bz = _CUBE_BASE
        env_cfg.events.object_scale.params["scale_range"]["x"] = (w_lo / bx, w_hi / bx)
        env_cfg.events.object_scale.params["scale_range"]["y"] = (l_lo / bx, l_hi / bx)
        env_cfg.events.object_scale.params["scale_range"]["z"] = (h_lo / bz, h_hi / bz)
    env_cfg.viewer.eye = (2.6, 2.6, 1.4)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.25)
    # terrain-noise mode: env origins sit far from the world origin (32x32 grid) —
    # anchor the camera to env_0's origin so the robot stays in frame
    if float(os.environ.get("GRASP_TERRAIN_NOISE", "0") or 0) > 0:
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        os.makedirs(args_cli.out_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env, video_folder=args_cli.out_dir, name_prefix="grasp_play",
            step_trigger=lambda step: step == 0, video_length=args_cli.steps, disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env)
    uenv = env.unwrapped

    agent_cfg = GraspPPORunnerCfg()
    agent_cfg.device = device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=uenv.device)

    from graspenv.mdp.utils import (
        KEEPOUT_HI,
        KEEPOUT_LO,
        _keepout_points_b,
        camera_keepout_violation,
        grasp_point_w,
        tcp_pos_w,
    )

    heur = set(filter(None, args_cli.heur.split(",")))
    if heur:
        print(f"[PLAY] heuristics active: {sorted(heur)}", flush=True)
    n_env = args_cli.num_envs
    # shield margin: only slightly outside the REAL camera box — the training-time shaped
    # penalty already owns the wide margin; a hard override there hijacks normal approaches
    _sh_lo = torch.tensor(KEEPOUT_LO, device=uenv.device) - 0.010
    _sh_hi = torch.tensor(KEEPOUT_HI, device=uenv.device) + 0.010
    air_cnt = torch.zeros(n_env, dtype=torch.long, device=uenv.device)
    reopen_left = torch.zeros(n_env, dtype=torch.long, device=uenv.device)
    shield_steps, reopen_events = 0, 0
    robot = uenv.scene["robot"]
    _j7 = robot.joint_names.index("arm_joint7")
    _j8 = robot.joint_names.index("arm_joint8")

    def apply_heuristics(actions: torch.Tensor) -> torch.Tensor:
        nonlocal shield_steps, reopen_events, air_cnt, reopen_left
        if "open" in heur:
            # keep the gripper open until close to the (perceived) grasp point: prevents
            # a half-closed hand from bumping the object during the approach
            tgt = getattr(uenv, "_pn_point_w", None)
            tgt = tgt if tgt is not None else grasp_point_w(uenv)
            far = (tcp_pos_w(uenv) - tgt).norm(dim=1) > 0.10
            actions[far, 6] = 1.0
        if "reopen" in heur:
            # air-grasp: commanded closed but the fingers met each other -> nothing inside;
            # force a reopen so the policy gets a second attempt within the episode
            width = robot.data.joint_pos[:, _j7] - robot.data.joint_pos[:, _j8]
            air = (actions[:, 6] < 0) & (width < 0.006)
            air_cnt = torch.where(air, air_cnt + 1, torch.zeros_like(air_cnt))
            trigger = air_cnt >= 10
            reopen_events += int(trigger.sum())
            reopen_left = torch.where(trigger, torch.full_like(reopen_left, 15), reopen_left)
            air_cnt[trigger] = 0
            forced = reopen_left > 0
            actions[forced, 6] = 1.0
            reopen_left = (reopen_left - 1).clamp(min=0)
        if "shield" in heur:
            # geometric camera shield: if any monitored arm point is inside the inflated
            # keep-out box, override the IK delta (base frame) with the shortest exit push
            pts = _keepout_points_b(uenv)                                   # [N, P, 3]
            inside = ((pts > _sh_lo) & (pts < _sh_hi)).all(dim=-1)          # [N, P]
            bad = inside.any(dim=1)
            if bad.any():
                shield_steps += int(bad.sum())
                d6 = torch.cat([pts - _sh_lo, _sh_hi - pts], dim=-1)        # [N, P, 6]
                d6 = torch.where(inside.unsqueeze(-1), d6, torch.full_like(d6, 1e6))
                face = d6.view(pts.shape[0], -1).argmin(dim=1) % 6
                axis = face % 3
                sign = torch.where(face < 3, -1.0, 1.0)
                push = torch.zeros(pts.shape[0], 3, device=pts.device)
                push[torch.arange(pts.shape[0], device=pts.device), axis] = sign
                # GENTLE override: strip only the inward component of the policy delta and
                # add a moderate outward push — a full hijack (2 cm/step along one axis)
                # punches the gripper through objects and destabilizes the approach
                a = actions[bad, 0:3]
                inward = (a * push[bad]).sum(dim=-1, keepdim=True).clamp(max=0.0)
                actions[bad, 0:3] = a - inward * push[bad] + 0.5 * push[bad]
        return actions

    obs = env.get_observations()
    successes, episodes = 0, 0
    term_counts = {k: 0 for k in uenv.termination_manager.active_terms}
    z_min, d_min, f_max = 10.0, 10.0, 0.0
    keepout_steps = 0
    obj_zmin, obj_zmax = 10.0, -10.0
    # per-EPISODE closest TCP-to-grasp-point distance, keyed by how the episode ended:
    # a time-out that never came within 10 cm is the mission's "flail" (policy saturates,
    # TCP never reaches) — tells whether the flail is intrinsic to the policy or an
    # artefact of the mission plant
    ep_dmin = torch.full((n_env,), 10.0, device=uenv.device)
    ep_hist = {}
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            actions = policy(obs)
            if heur:
                actions = apply_heuristics(actions.clone())
            obs, _, dones, _ = env.step(actions)
            for k in term_counts:
                term_counts[k] += int(uenv.termination_manager.get_term(k).sum())
            succ = uenv.termination_manager.get_term("grasp_success")
            successes += int(succ.sum())
            episodes += int(dones.sum())
            z_min = min(z_min, float((uenv.scene["robot"].data.root_pos_w[:, 2] - uenv.scene.env_origins[:, 2]).min()))
            d_now = (tcp_pos_w(uenv) - grasp_point_w(uenv)).norm(dim=1)
            d_min = min(d_min, float(d_now.min()))
            dn = dones.bool().view(-1)         # the wrapper hands dones back as long
            if dn.any():
                for k in term_counts:
                    tk = uenv.termination_manager.get_term(k).bool().view(-1) & dn
                    for d in ep_dmin[tk].tolist():
                        ep_hist.setdefault(k, []).append(d)
                ep_dmin[dn] = 10.0             # the reset env starts a fresh episode
            ep_dmin = torch.minimum(ep_dmin, d_now)
            fl = uenv.scene["contact_finger_l"].data.force_matrix_w
            if fl is not None:
                f_max = max(f_max, float(fl.norm(dim=-1).max()))
            nf = uenv.scene["contact_finger_l"].data.net_forces_w
            if not hasattr(main, "_net_max"):
                main._net_max = 0.0
            main._net_max = max(main._net_max, float(nf.norm(dim=-1).max()))
            keepout_steps += int((camera_keepout_violation(uenv) > 0).sum())
            oz = uenv.scene["object"].data.root_pos_w[:, 2] - uenv.scene.env_origins[:, 2]
            obj_zmin, obj_zmax = min(obj_zmin, float(oz.min())), max(obj_zmax, float(oz.max()))
    print(f"[PLAY] terminations: {term_counts} | min base z seen: {z_min:.3f}", flush=True)
    print(f"[PLAY] diag: min|tcp-grasp_pt|={d_min:.3f} m | max finger-object force={f_max:.2f} N "
          f"| max finger NET force={getattr(main, '_net_max', 0.0):.2f} N "
          f"| object z range {obj_zmin:.3f}-{obj_zmax:.3f} "
          f"| camera keep-out env-steps: {keepout_steps}", flush=True)
    if heur:
        print(f"[PLAY] heur diag: shield overrides {shield_steps} env-steps | reopen events {reopen_events}", flush=True)
    for k, ds in sorted(ep_hist.items()):
        ds = sorted(ds)
        far = sum(1 for d in ds if d > 0.10)
        med = ds[len(ds) // 2]
        print(f"[PLAY] ep-min|tcp-grasp_pt| {k}: n={len(ds)} never-within-10cm={far} "
              f"median={med:.3f} q10={ds[len(ds) // 10]:.3f} q90={ds[(9 * len(ds)) // 10]:.3f}",
              flush=True)
    print(f"[PLAY] rollout: {episodes} episodes finished, {successes} grasp successes "
          f"({successes / max(episodes, 1):.1%})", flush=True)
    if "base_fell" in uenv.termination_manager.active_terms:
        z = uenv.scene["robot"].data.root_pos_w[:, 2] - uenv.scene.env_origins[:, 2]
        print(f"[PLAY] base height now: {z.mean():.3f}±{z.std():.3f} m (free base)", flush=True)
    if args_cli.video:
        print(f"[PLAY] video dir: {args_cli.out_dir}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
