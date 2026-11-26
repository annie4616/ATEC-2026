# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
RslRlOnPolicyRunnerCfg, 
RslRlPpoActorCriticCfg, 
RslRlPpoAlgorithmCfg,
)
#########################
# Policy configurations #
#########################

import os as _os
# B2Piper retraining: the teacher is trained natively on B2 + Piper arm = 20 DoF
# (deployment embodiment). Obs joint terms (joint_pos, joint_vel, action) and the
# priv_latent stiffness/damping each scale with joint count, so the dims below grow
# from the 12-DoF B2 baseline. Set PARKOUR_NJOINTS=12 to fall back to the B2 teacher.
_NJ = int(_os.environ.get("PARKOUR_NJOINTS", "20"))   # 20 = B2Piper, 12 = B2

@configclass
class ParkourRslRlBaseCfg:
    # priv_explicit = base_lin_vel(3) + 2 zero-blocks(3+3) = 9  (joint-independent)
    num_priv_explicit: int = 3 + 3 + 3 # 9
    # priv_latent = mass(4) + friction(1) + stiffness(NJ) + damping(NJ)
    num_priv_latent: int = 4 + 1 + _NJ + _NJ          # B2:29  B2Piper:45
    # prop = ang_vel(3)+imu(2)+yaw/cmd block(3)+env_idx(4) + joint_pos(NJ)+joint_vel(NJ)
    #        +action_hist(NJ) + contact_fill(4)... encoded as 3+2+3+4 + 3*NJ + 5
    num_prop: int = 3 + 2 + 3 + 4 + 3 * _NJ + 5       # B2:53  B2Piper:77
    num_scan: int = 132
    num_hist: int = 10
    
@configclass
class ParkourRslRlStateHistEncoderCfg(ParkourRslRlBaseCfg):
    class_name: str = "StateHistoryEncoder" 
    channel_size: int = 10 
    
@configclass
class ParkourRslRlDepthEncoderCfg(ParkourRslRlBaseCfg):
    # LiDAR student: encode the 16x360 LiDAR scan (5760 rays) to the scandots latent instead
    # of a depth image. LidarScanBackbone has the same (scandots_output_dim, hidden_dims)
    # construction contract, so RecurrentDepthBackbone wraps it unchanged.
    # LIDAR_BACKBONE env selects the depth backbone class without editing the cfg:
    #   "LidarScanBackbone"    (default) conv stem over the raw scan
    #   "WarpScandotsBackbone" warp-reprojection (rolling world heightmap -> teacher 132 scandots
    #                          -> 132->32 MLP); needs absolute world pose, injected by the runner.
    backbone_class_name: str = _os.environ.get("LIDAR_BACKBONE", "LidarScanBackbone")
    encoder_class_name: str = "RecurrentDepthBackbone"
    # LiDAR student: the backbone consumes a flat (batch, num_rays) scan, num_rays = 16ch x 360deg
    # = 5760 (matches the obs term 'depth_camera' shape logged at env build). ONNX export builds a
    # dummy input as torch.zeros(1, *depth_shape), so a 1-tuple (5760,) yields the correct 2D
    # (1, 5760) that conv1d expects — a 2-tuple like the old (87,58) depth image gives a 4D tensor
    # and crashes the LiDAR conv1d. JIT export ignores this (scripts, no dummy trace).
    depth_shape: tuple[int] = (5760,)
    hidden_dims: int = 512
    learning_rate: float = 1.e-3
    num_steps_per_env: int = 24 * 5

@configclass
class ParkourRslRlEstimatorCfg(ParkourRslRlBaseCfg):
    class_name: str = "DefaultEstimator" 
    train_with_estimated_states: bool = True 
    learning_rate: float = 1.e-4 
    hidden_dims: list[int] = MISSING 
    
@configclass
class ParkourRslRlActorCfg(ParkourRslRlBaseCfg):
    class_name: str = "Actor"
    state_history_encoder: ParkourRslRlStateHistEncoderCfg = MISSING


@configclass
class ParkourRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = 'ActorCriticRMA'
    tanh_encoder_output: bool = False 
    scan_encoder_dims: list[int] = MISSING
    priv_encoder_dims: list[int] = MISSING
    actor: ParkourRslRlActorCfg = MISSING

@configclass
class ParkourRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = 'PPOWithExtractor'
    dagger_update_freq: int = 1
    priv_reg_coef_schedual: list[float]= [0, 0.1, 2000, 3000]

@configclass
class ParkourRslRlDistillationAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "DistillationWithExtractor"

@configclass
class ParkourRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    policy: ParkourRslRlPpoActorCriticCfg = MISSING
    estimator: ParkourRslRlEstimatorCfg = MISSING
    depth_encoder: ParkourRslRlDepthEncoderCfg | None = None
    algorithm: ParkourRslRlPpoAlgorithmCfg | ParkourRslRlDistillationAlgorithmCfg = MISSING

