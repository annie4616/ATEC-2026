"""crouchpkg — external registration of the ATEC-Crouch-B2-v0 task.

Self-contained under logs/rsl_rl/260630/; never modifies the base repo. Importing this
package (after Isaac Sim is launched) registers the gym task. Mirrors the proven
for_claude/env pattern.
"""

import gymnasium as gym

import atec_rl_lab.train  # noqa: F401  ensure base ATEC-* assets/tasks are registered first

gym.register(
    id="ATEC-Crouch-B2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "crouchpkg.crouch_env_cfg:B2CrouchEnvCfg",
        "rsl_rl_cfg_entry_point": "crouchpkg.agents:B2CrouchPPORunnerCfg",
    },
)
