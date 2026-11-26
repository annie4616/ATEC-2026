import gymnasium as gym

# `agents` pulls in rsl_rl (via RslRlVecEnvWrapper). Training imports rsl_rl first,
# so it's available there; but a bare env-render/eval process that only needs the
# env_cfg shouldn't require rsl_rl. Guard the import so those processes still load
# the task registrations (env_cfg_entry_point), just without the rsl_rl agent cfg.
try:
    from . import agents
except ModuleNotFoundError:
    agents = None

##
# Register Gym environment for Task D RL (Unitree B2).
##

gym.register(
    id="ATEC-TaskD-RL-Unitree-B2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.taskd_env_cfg:UnitreeB2TaskDEnvCfg",
        "rsl_rl_cfg_entry_point": f"atec_rl_lab.train.locomotion.taskd.config.quadruped.unitree_b2.agents.rsl_rl_ppo_cfg:UnitreeB2TaskDPPORunnerCfg",
    },
)

# Flat-ground walking pretrain (same robot/obs/action interface as Task D, so its
# checkpoint warm-starts ATEC-TaskD-RL-Unitree-B2-v0).
gym.register(
    id="ATEC-TaskD-RL-Unitree-B2-FlatPretrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_pretrain_env_cfg:UnitreeB2FlatPretrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"atec_rl_lab.train.locomotion.taskd.config.quadruped.unitree_b2.agents.rsl_rl_ppo_cfg:UnitreeB2FlatPretrainPPORunnerCfg",
    },
)

# Hierarchical CROSSING-ONLY Task D: box starts settled in the pit (oracle end
# state), robot in front; RL learns only the pit traversal. Same robot/obs/action
# interface, so the integrated/flat checkpoint warm-starts it.
gym.register(
    id="ATEC-TaskD-RL-Cross-Unitree-B2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.taskd_cross_env_cfg:UnitreeB2TaskDCrossEnvCfg",
        "rsl_rl_cfg_entry_point": f"atec_rl_lab.train.locomotion.taskd.config.quadruped.unitree_b2.agents.rsl_rl_ppo_cfg:UnitreeB2TaskDPPORunnerCfg",
    },
)

# WALK-ACROSS crossing (the REAL 36-point path): box settled half-sunk in the pit
# center (demo pose x~-0.25), robot walks OVER the box on the -y lane (NO jump, NO
# 1 m platform). Demo-matched rewards (dense +x progress + walk-across milestones,
# no jump shaping). See taskd_crosswalk_env_cfg.py.
gym.register(
    id="ATEC-TaskD-RL-CrossWalk-Unitree-B2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.taskd_crosswalk_env_cfg:UnitreeB2TaskDCrossWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"atec_rl_lab.train.locomotion.taskd.config.quadruped.unitree_b2.agents.rsl_rl_ppo_cfg:UnitreeB2TaskDPPORunnerCfg",
    },
)

# WALK-ACROSS crossing + dense FAR-RIM CLIMB shaping. Same as CrossWalk but adds
# farrim_climb_band (dense base-height gradient over the 0.4 m far rim) and softens
# stand_still, to break the farrim-milestone=0 plateau (robot stepped onto the box
# but never climbed back out). See taskd_crosswalkclimb_env_cfg.py.
gym.register(
    id="ATEC-TaskD-RL-CrossWalkClimb-Unitree-B2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.taskd_crosswalkclimb_env_cfg:UnitreeB2TaskDCrossWalkClimbEnvCfg",
        "rsl_rl_cfg_entry_point": f"atec_rl_lab.train.locomotion.taskd.config.quadruped.unitree_b2.agents.rsl_rl_ppo_cfg:UnitreeB2TaskDPPORunnerCfg",
    },
)
