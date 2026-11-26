# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for velocity-based locomotion environments."""

# We leave this file empty since we don't want to expose any configs in this package directly.
# We still need this file to import the "config" module in the parent package.

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2TeacherParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2TeacherParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

# Task D pit (down-then-up box-filled-pit cross) — train the teacher to CROSS the pit
# (it only learned to climb UP a single STEP; the pit needs down-then-up). Verified
# parkour rewards, terrain swapped to TASKD_PIT.
gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-TaskDPitTrain-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2TeacherParkourEnvCfg_TASKDPIT_TRAIN",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-TaskDPitPlay-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2TeacherParkourEnvCfg_TASKDPIT_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

# STEP-only play (single up-step), step height via PARKOUR_PLAY_DIFF (0.6->0.4m).
gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-StepPlay-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2TeacherParkourEnvCfg_STEP_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

# STAIRCASE-only play (5 consecutive up-steps <=0.17 m = the course staircase) to inspect
# the teacher climbing a full 5-step staircase in isolation.
gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-StaircasePlay-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2TeacherParkourEnvCfg_STAIRCASE_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

# REAL-geometry Task D pit (pit 1.35, box-stone 0.8) — matched to deployment.
gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-TaskDPitRealTrain-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2TeacherParkourEnvCfg_TASKDPITREAL_TRAIN",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-TaskDPitRealPlay-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2TeacherParkourEnvCfg_TASKDPITREAL_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Eval-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:UnitreeGo2TeacherParkourEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Student-Unitree-Go2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:UnitreeGo2StudentParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:UnitreeGo2ParkourStudentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:UnitreeGo2StudentParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:UnitreeGo2ParkourStudentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Student-Unitree-Go2-Eval-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:UnitreeGo2StudentParkourEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:UnitreeGo2ParkourStudentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)
