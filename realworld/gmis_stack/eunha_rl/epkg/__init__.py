"""epkg — Eunha-lineage A2 envs (RMA walk checkpoints -> crouch -> grasp).

Policy obs = stock 45-d proprio + 8-d frozen-adaptation latent = 53, matching the
Phase-1 actor input of from_eunha/speed_*/model_14999.pt exactly (layout verified
against from_eunha/speed_1.0/env.yaml).
"""
import gymnasium as gym

import atec_rl_lab.train  # noqa: F401  (parent assets/markers must register first)

gym.register(
    id="ATEC-Crouch-A2-EUNHA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "epkg.crouch_cfg:A2CrouchEunhaEnvCfg",
        "rsl_rl_cfg_entry_point": "epkg.agents:EunhaCrouchPPORunnerCfg",
    },
)
gym.register(
    id="ATEC-Crouch-A2-EUNHA-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "epkg.crouch_rough_cfg:A2CrouchEunhaRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "epkg.agents:EunhaCrouchPPORunnerCfg",
    },
)
gym.register(
    id="ATEC-Velocity-Flat-A2-EUNHA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "epkg.flat_cfg:A2FlatEunhaEnvCfg",
        "rsl_rl_cfg_entry_point": "epkg.agents:EunhaFlatPPORunnerCfg",
    },
)
