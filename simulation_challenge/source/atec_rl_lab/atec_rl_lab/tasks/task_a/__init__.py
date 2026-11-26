import gymnasium as gym

gym.register(
    id = "ATEC-TaskA-G1",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvG1Cfg"
    },
)

gym.register(
    id = "ATEC-TaskA-Tron1Piper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvTron1Cfg"
    },
)

gym.register(
    id = "ATEC-TaskA-Tron2ALegged",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvTron2ALeggedCfg"
    },
)

gym.register(
    id = "ATEC-TaskA-Tron2AWheel",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvTron2AWheelCfg"
    },
)

gym.register(
    id = "ATEC-TaskA-B2Piper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvB2Cfg"
    },
)

gym.register(
    id = "ATEC-TaskA-A2Piper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvA2Cfg"
    },
)

gym.register(
    id = "ATEC-TaskA-B2wPiper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvB2WCfg"
    },
)

gym.register(
    id = "ATEC-TaskA-B2Piper-CurvedStrip",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvB2CurvedStripCfg"
    },
)

gym.register(
    id = "ATEC-TaskA-B2Piper-BentGoal",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvB2BentGoalCfg"
    },
)

gym.register(
    id = "ATEC-TaskA-B2Piper-Maze",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvB2MazeCfg"
    },
)

gym.register(
    id = "ATEC-TaskA-B2Piper-MazeViPlanner",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvB2MazeViPlannerCfg"
    },
)

gym.register(
    id = "ATEC-TaskA-B2Piper-RedLineCornerViPlanner",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvB2RedLineCornerViPlannerCfg"
    },
)

gym.register(
    id = "ATEC-TaskA-B2Piper-MazeFullViPlanner",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvB2MazeFullViPlannerCfg"
    },
)

gym.register(
    id = "ATEC-TaskA-B2Piper-MazeRoughWideViPlanner",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvB2MazeRoughWideViPlannerCfg"
    },
)

gym.register(
    id = "ATEC-TaskA-B2Piper-MazeSymFullViPlanner",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvB2MazeSymFullViPlannerCfg"
    },
)

from .env_cfg import (TaskAEnvCfg, TaskAEnvB2Cfg, TaskAEnvTron2ALeggedCfg, TaskAEnvTron2AWheelCfg,
                      TaskAEnvB2CurvedStripCfg, TaskAEnvB2BentGoalCfg, TaskAEnvB2MazeCfg, TaskAEnvB2MazeViPlannerCfg,
                      TaskAEnvB2MazeFullViPlannerCfg, TaskAEnvB2MazeRoughWideViPlannerCfg,
                      TaskAEnvB2MazeSymFullViPlannerCfg, TaskAEnvB2RedLineCornerViPlannerCfg)
__all__ = ['TaskAEnvCfg', 'TaskAEnvB2Cfg', 'TaskAEnvTron2ALeggedCfg', 'TaskAEnvTron2AWheelCfg',
           'TaskAEnvB2CurvedStripCfg', 'TaskAEnvB2BentGoalCfg', 'TaskAEnvB2MazeCfg', 'TaskAEnvB2MazeViPlannerCfg',
           'TaskAEnvB2MazeFullViPlannerCfg', 'TaskAEnvB2MazeRoughWideViPlannerCfg',
           'TaskAEnvB2MazeSymFullViPlannerCfg', 'TaskAEnvB2RedLineCornerViPlannerCfg']
