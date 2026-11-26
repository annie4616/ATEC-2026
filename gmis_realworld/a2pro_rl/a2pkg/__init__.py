"""a2pkg — Unitree A2 Pro locomotion tasks (plain flat velocity + WTW gait variant).

Self-contained under a2pro_rl/; registering NEW task ids, nothing existing is modified.
"""

import gymnasium as gym

import atec_rl_lab.train  # noqa: F401  base assets/tasks first

gym.register(
    id="ATEC-Velocity-Flat-A2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "a2pkg.flat_env_cfg:UnitreeA2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": "a2pkg.agents:UnitreeA2FlatPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Velocity-Flat-A2-WTW-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "a2pkg.wtw_env_cfg:UnitreeA2FlatWTWEnvCfg",
        "rsl_rl_cfg_entry_point": "a2pkg.agents:UnitreeA2FlatWTWPPORunnerCfg",
    },
)
