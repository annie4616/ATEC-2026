# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
try:
    # Not present in newer IsaacLab; only needed for --use_pretrained_checkpoint,
    # which we don't use (we load our own trained checkpoint).
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    def get_published_pretrained_checkpoint(*args, **kwargs):
        raise RuntimeError("pretrained_checkpoint not available in this IsaacLab version")
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import ParkourRslRlOnPolicyRunnerCfg

from scripts.rsl_rl.exporter import (
export_teacher_policy_as_jit, 
export_teacher_policy_as_onnx,
export_deploy_policy_as_jit, 
export_deploy_policy_as_onnx,
)
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg



def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: ParkourRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(ppo_runner)
    # obtain the trained policy for inference

    estimator = ppo_runner.get_estimator_inference_policy(device=env.device) 
    if agent_cfg.algorithm.class_name == "DistillationWithExtractor":
        policy = ppo_runner.get_inference_depth_policy(device=env.unwrapped.device)
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
        policy_nn = ppo_runner.alg.depth_actor
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported_deploy")
        export_deploy_policy_as_jit(policy_nn, 
                                    estimator,
                                    depth_encoder,
                                    ppo_runner.obs_normalizer, 
                                    path=export_model_dir, 
                                    filename="policy.pt")
        # ONNX export of the LiDAR encoder trips over AdaptiveAvgPool1d (opset 11 can't
        # export adaptive pooling whose output size isn't a factor of the input). The JIT
        # exports above (policy.pt + depth_latest.pt) are the deploy-critical artifacts and
        # already succeeded, so a failed ONNX pass must not abort the play/video run.
        try:
            export_deploy_policy_as_onnx(
                                policy_nn,
                                estimator,
                                depth_encoder,
                                agent_cfg,
                                normalizer=ppo_runner.obs_normalizer,
                                path=export_model_dir,
                                filename="policy.onnx"
                            )
        except Exception as _onnx_err:
            print(f"[WARN] ONNX deploy export skipped (JIT export already saved): {_onnx_err}")

    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        policy_nn = ppo_runner.alg.policy
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported_teacher")
        export_teacher_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_teacher_policy_as_onnx(
            policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )
        # Also export a SELF-CONTAINED teacher that runs the estimator internally (fills
        # priv_explicit from proprio), so a Task A deploy adapter needn't reproduce it.
        try:
            from scripts.rsl_rl.exporter import export_teacher_with_estimator_as_jit
            _ep = agent_cfg.to_dict()["estimator"]
            export_teacher_with_estimator_as_jit(
                policy_nn, estimator, ppo_runner.obs_normalizer,
                num_prop=_ep["num_prop"], num_scan=_ep["num_scan"],
                num_priv_explicit=_ep["num_priv_explicit"],
                path=export_model_dir, filename="policy_with_est.pt")
        except Exception as _ee:
            print(f"[WARN] teacher+estimator export skipped: {_ee}")

    dt = env.unwrapped.step_dt
    estimator_paras = agent_cfg.to_dict()["estimator"]
    num_prop = estimator_paras["num_prop"]
    num_scan = estimator_paras["num_scan"]
    num_priv_explicit = estimator_paras["num_priv_explicit"]
    # reset environment
    obs, extras = env.get_observations()
    timestep = 0
    # --- DIAG_PIT: track base x-advance and z to see if the robot truly CROSSES the
    # TASKD_PIT (z recovers to rim ~0.5 on the far side) or stalls in the pit (z low). ---
    _diag = os.environ.get("DIAG_PIT", "0") == "1"
    if _diag:
        _rob = env.unwrapped.scene["robot"]; _org = env.unwrapped.scene.env_origins
        _Nd = min(8, env.num_envs)
        _sx = (_rob.data.root_pos_w[:, 0] - _org[:, 0]).clone()
        _maxa = torch.full((env.num_envs,), -1e9, device=env.device)
        _zatm = torch.zeros(env.num_envs, device=env.device)
        _minz = torch.full((env.num_envs,), 1e9, device=env.device)
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        if agent_cfg.algorithm.class_name != "DistillationWithExtractor":
            with torch.inference_mode():
                # agent stepping
                obs[:, num_prop+num_scan:num_prop+num_scan+num_priv_explicit] = estimator.inference(obs[:, :num_prop])
                actions = policy(obs, hist_encoding = True)
                import os as _os
                globals()["_soc"]=globals().get("_soc",0)+1
                if _os.environ.get("SAVE_OBS")=="1" and globals()["_soc"]==30:
                    torch.save({"obs": obs[0].detach().cpu(), "act": actions[0].detach().cpu(),
                                "num_prop": num_prop, "num_scan": num_scan,
                                "num_priv_explicit": num_priv_explicit},
                               "/hdd0/sohyun/tmp/teacher_obs_ref.pt")
                    print("[SAVED_OBS] t=30 obs+act saved", flush=True)
            # env stepping
        else:
            depth_camera = extras["observations"]['depth_camera'].to(env.device)
            with torch.inference_mode():
                if env.unwrapped.common_step_counter %5 == 0:
                    obs_student = obs[:, :num_prop].clone()
                    obs_student[:, 6:8] = 0
                    depth_latent_and_yaw = depth_encoder(depth_camera, obs_student)
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = depth_latent_and_yaw[:, -2:]
                obs[:, 6:8] = 1.5*yaw
                # obs[:, num_prop+num_scan:num_prop+num_scan+num_priv_explicit] = estimator.inference(obs[:, :num_prop])
                actions = policy(obs, hist_encoding=True, scandots_latent=depth_latent)
        obs, _, _, extras = env.step(actions)
        if _diag:
            _rx = _rob.data.root_pos_w[:, 0] - _org[:, 0]
            _rz = _rob.data.root_pos_w[:, 2]
            _adv = _rx - _sx
            _upd = _adv > _maxa
            _maxa = torch.where(_upd, _adv, _maxa)
            _zatm = torch.where(_upd, _rz, _zatm)
            _minz = torch.minimum(_minz, _rz)
            if timestep % 40 == 0:
                # also log y-drift and yaw to check sideways drift in the TRAINING env
                _ry = _rob.data.root_pos_w[:, 1] - _org[:, 1]
                _q = _rob.data.root_quat_w
                import math as _mm
                _yaw = torch.atan2(2*(_q[:,0]*_q[:,3] + _q[:,1]*_q[:,2]),
                                   1 - 2*(_q[:,2]**2 + _q[:,3]**2))
                _ln = " ".join(f"e{i}:x{float(_adv[i]):+.2f},y{float(_ry[i]):+.2f},"
                               f"yaw{float(_yaw[i]):+.2f}" for i in range(_Nd))
                print(f"[DIAG t{timestep:3d}] {_ln}", flush=True)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                if _diag:
                    print("\n[DIAG SUMMARY] (TRUE cross = max_adv past far rim AND z_at_maxadv ~0.5)", flush=True)
                    for i in range(_Nd):
                        print(f"[DIAG] env{i}: max_adv={float(_maxa[i]):+.2f} "
                              f"z_at_maxadv={float(_zatm[i]):.2f} min_z_pitdip={float(_minz[i]):.2f}", flush=True)
                break
        elif _diag:
            timestep += 1
            if timestep >= 400:
                print("\n[DIAG SUMMARY]", flush=True)
                for i in range(_Nd):
                    print(f"[DIAG] env{i}: max_adv={float(_maxa[i]):+.2f} "
                          f"z_at_maxadv={float(_zatm[i]):.2f} min_z_pitdip={float(_minz[i]):.2f}", flush=True)
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
