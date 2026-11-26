# Reference: https://github.com/fan-ziqi/robot_lab

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="ATEC-Isaac-Velocity-Flat-Unitree-B2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeB2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{ agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2FlatPPORunnerCfg"
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-Rough-Unitree-B2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeB2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{ agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2RoughPPORunnerCfg"
    },
)

# High-speed rough walking (forward vx up to 3.5 m/s in-distribution). Same
# robot/obs/action interface as the rough env, so its checkpoint warm-starts this.
gym.register(
    id="ATEC-Isaac-Velocity-Fast-Unitree-B2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fast_env_cfg:UnitreeB2FastEnvCfg",
        "rsl_rl_cfg_entry_point": f"{ agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2FastPPORunnerCfg"
    },
)

# Heading-hardened fast walker: same ~2.6 m/s fast env but with a much stronger
# heading-holding objective to kill the learned leftward yaw drift (the "limping"),
# WITHOUT hard joint-symmetry penalties that would fight terrain adaptation on ramps.
gym.register(
    id="ATEC-Isaac-Velocity-FastSym-Unitree-B2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fast_sym_env_cfg:UnitreeB2FastSymEnvCfg",
        "rsl_rl_cfg_entry_point": f"{ agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2FastSymPPORunnerCfg"
    },
)

# Grid-adaptive fast walker: MIT "Rapid Locomotion" grid curriculum to break the
# vx≈2.63 wall the global-mean curriculum stalled at. Per-env (vx,wz) bins expand to
# neighbors on success so a slow bin can't block a fast one.
gym.register(
    id="ATEC-Isaac-Velocity-FastGrid-Unitree-B2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fast_grid_env_cfg:UnitreeB2FastGridEnvCfg",
        "rsl_rl_cfg_entry_point": f"{ agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2FastGridPPORunnerCfg"
    },
)

# Fall-recovery / get-up: from a fallen/flipped pose, roll upright and stand.
# Same robot/obs/action interface as rough, so it warm-starts from the rough
# checkpoint and its policy is swap-compatible with the walker at deploy time.
gym.register(
    id="ATEC-Isaac-Velocity-Recovery-Unitree-B2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:UnitreeB2RecoveryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{ agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2RecoveryPPORunnerCfg"
    },
)
