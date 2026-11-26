"""graspenv — registers ATEC-Grasp-B2Piper-v0 (B2+Piper grasp+lift, stage 1 fixed base).

Import AFTER Isaac Sim is launched. Mirrors the crouchpkg registration pattern; everything
resolves inside real_world_preliminary (atec_rl_lab must come from this repo's source/).
"""

import gymnasium as gym

gym.register(
    id="ATEC-Grasp-B2Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "graspenv.grasp_env_cfg:B2PiperGraspEnvCfg",
        "rsl_rl_cfg_entry_point": "graspenv.agents:GraspPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Grasp-B2Piper-Stage2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "graspenv.grasp_env_cfg:B2PiperGraspStage2EnvCfg",
        "rsl_rl_cfg_entry_point": "graspenv.agents:GraspPPORunnerCfg",
    },
)
