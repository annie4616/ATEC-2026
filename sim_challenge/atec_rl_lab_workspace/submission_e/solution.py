"""ATEC 2026 Task-E submission (ACT policy).

Loads the trained ACT checkpoint shipped next to this file as ``policy_act.pt``
(a copy of runs/act-task-e-rgb-100demos-seed1/checkpoints/final.pt) and serves
joint-position actions through the official AlgSolution interface.

Deployment pipeline per episode:
    1. startup zero actions (let the sim settle)
    2. PD move-to-home using proprioception only
    3. ACT inference with temporal ensembling
"""

import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF

current_path = os.path.dirname(os.path.abspath(__file__))
if current_path not in sys.path:
    sys.path.insert(0, current_path)

from act.detr.backbone import build_backbone
from act.detr.transformer import build_transformer
from act.detr.detr_vae import build_encoder, DETRVAE

# Checkpoint bundled with the submission (final.pt copied to this name).
CHECKPOINT_NAME = "policy_act.pt"


@dataclass
class Args:
    torch_deterministic: bool = True
    cuda: bool = True
    temporal_agg: bool = True

    # Backbone
    position_embedding: str = "sine"
    backbone: str = "resnet18"
    lr_backbone: float = 1e-5
    masks: bool = False
    dilation: bool = False
    include_depth: bool = False
    include_rgb: bool = True

    # Transformer
    enc_layers: int = 2
    dec_layers: int = 4
    dim_feedforward: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    num_queries: int = 30
    pre_norm: bool = False


class Agent(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, args: Args):
        super().__init__()
        self.device = "cuda"
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.include_rgb = args.include_rgb

        backbones = [build_backbone(args)] if args.include_rgb else None
        transformer = build_transformer(args)
        encoder = build_encoder(args)

        self.model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=state_dim,
            action_dim=act_dim,
            num_queries=args.num_queries,
        )

    def _preprocess_rgb(self, obs: dict) -> None:
        if self.include_rgb and "rgb" in obs:
            obs["rgb"] = obs["rgb"].float() / 255.0
            B, N, C, H, W = obs["rgb"].shape
            obs["rgb"] = self.normalize(obs["rgb"].view(B * N, C, H, W)).view(B, N, C, H, W)

    def _model_input(self, obs: dict):
        return obs if self.include_rgb else obs["state"]

    def get_action(self, obs: dict) -> torch.Tensor:
        self._preprocess_rgb(obs)
        a_hat, _ = self.model(self._model_input(obs))
        return a_hat


class AlgSolution:

    # Proprio layout: [joint_pos(8), joint_vel(8), prev_action(8)] for Piper (8 DoF).
    _QPOS_SLICE = slice(0, 8)
    _QVEL_SLICE = slice(8, 16)

    def __init__(self):
        self.device = "cuda"
        ckpt_path = os.path.join(current_path, CHECKPOINT_NAME)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        norm_stats = ckpt["norm_stats"]
        state_dim = norm_stats["state_mean"].shape[-1]
        act_dim = norm_stats["action_mean"].shape[-1]
        weight_key = "ema_agent" if "ema_agent" in ckpt else "agent"

        train_args = Args()
        train_args.num_queries = 30
        train_args.include_rgb = any("backbone" in k for k in ckpt[weight_key].keys())

        self.agent = Agent(state_dim, act_dim, train_args).to(self.device)
        self.agent.load_state_dict(ckpt[weight_key])
        self.agent.eval()

        self.num_queries = 30
        self.temporal_agg = True
        self._k = 0.01

        self.state_mean = norm_stats["state_mean"].to(self.device)
        self.state_std = norm_stats["state_std"].to(self.device)
        self.act_mean = norm_stats["action_mean"].to(self.device)
        self.act_std = norm_stats["action_std"].to(self.device)

        self.default_joint_pos = torch.tensor(
            [[0.0, 1.2, -1.5, 0.0, 1.2, 0.0, 0.035, -0.035]],
            dtype=torch.float32,
            device=self.device,
        )

        # move-to-home parameters
        self.teleop_home_joint_pos = torch.tensor(
            [[-0.000033, 0.924525, -1.514983, 0.000011, 1.219900, -0.000033,
              0.035000, -0.035000]],
            dtype=torch.float32,
            device=self.device,
        )
        self._startup_zero_steps = 25
        self._home_qpos_tolerance = 0.10
        self._home_kp = 2.0
        self._home_kd = 0.2
        self._home_hold_steps = 5

        # per-episode state
        self._ts = 0
        self._action_history = deque(maxlen=self.num_queries)
        self._last_action_seq = None
        self._startup_step = 0
        self._home_stable_steps = 0
        self._home_done = False

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {}

    def reset(self, **kwargs) -> None:
        """Reset per-episode inference state (called by the eval harness)."""
        self._ts = 0
        self._action_history.clear()
        self._last_action_seq = None
        self._startup_step = 0
        self._home_stable_steps = 0
        self._home_done = False

    def _compute_home_action(self, proprio):
        joint_pos_rel = proprio[:, self._QPOS_SLICE]
        joint_vel_rel = proprio[:, self._QVEL_SLICE]
        qpos = joint_pos_rel + self.default_joint_pos
        qerr = self.teleop_home_joint_pos - qpos

        within_tolerance = torch.all(torch.abs(qerr) <= self._home_qpos_tolerance, dim=1)
        self._home_stable_steps = (
            self._home_stable_steps + 1 if bool(torch.all(within_tolerance)) else 0
        )

        # PD in joint space; env action scale is 0.5 (use_default_offset=True).
        u = self._home_kp * qerr - self._home_kd * joint_vel_rel
        action = torch.clamp(u / 0.5, -1.0, 1.0)

        if bool(torch.all(within_tolerance)):
            action = torch.zeros_like(action)

        home_reached = bool(torch.all(within_tolerance))
        return action, home_reached

    def predicts(self, obs, current_score):
        if not isinstance(obs, dict) or "proprio" not in obs:
            raise ValueError("Expected obs dict with 'proprio' key.")

        proprio = obs["proprio"].to(self.device)

        # Stage 1: zero actions while the sim settles.
        if self._startup_step < self._startup_zero_steps:
            self._startup_step += 1
            return {
                "action": torch.zeros((proprio.shape[0], self.agent.act_dim)).numpy().tolist(),
                "giveup": False,
            }

        # Stage 2: PD move-to-home using observations only.
        if not self._home_done:
            home_action, home_reached = self._compute_home_action(proprio)
            if home_reached:
                self._home_done = True
                self._ts = 0
                self._action_history.clear()
                self._last_action_seq = None
            return {"action": home_action.cpu().numpy().tolist(), "giveup": False}

        # Stage 3: ACT inference.
        joint_pos_rel = proprio[:, self._QPOS_SLICE]
        qpos = joint_pos_rel + self.default_joint_pos
        state = (qpos - self.state_mean) / self.state_std
        model_obs = {"state": state}

        if self.agent.include_rgb:
            rgb = obs["image"]["video_rgb"].to(self.device)
            if rgb.shape[1] == 4:
                rgb = rgb[:, :3]                       # drop alpha if RGBA/NCHW
            if rgb.ndim == 4 and rgb.shape[-1] == 4:
                rgb = rgb[..., :3]                     # drop alpha if RGBA/NHWC
            if rgb.dtype != torch.uint8:
                rgb = (rgb.float() * 255.0).clamp(0, 255).to(torch.uint8)
            if rgb.ndim == 4 and rgb.shape[1] in (3, 4):
                pass
            else:
                rgb = rgb.permute(0, 3, 1, 2)
            if rgb.shape[-2:] != (224, 224):
                rgb = TF.resize(rgb, [224, 224],
                                interpolation=TF.InterpolationMode.BILINEAR,
                                antialias=True)
            model_obs["rgb"] = rgb.unsqueeze(1)        # (num_envs, 1, 3, 224, 224) uint8

        ts = self._ts
        query_frequency = 1 if self.temporal_agg else self.num_queries

        if ts % query_frequency == 0:
            with torch.no_grad():
                action_seq = self.agent.get_action(model_obs)
            if self.temporal_agg:
                self._action_history.append(action_seq)
            else:
                self._last_action_seq = action_seq

        if self.temporal_agg:
            n = len(self._action_history)
            actions_for_curr = torch.stack(
                [seq[:, n - 1 - i, :] for i, seq in enumerate(self._action_history)],
                dim=1,
            )
            exp_weights = torch.exp(-self._k * torch.arange(n, device=self.device))
            exp_weights = (exp_weights / exp_weights.sum()).unsqueeze(0).unsqueeze(-1)
            raw_action = (actions_for_curr * exp_weights).sum(dim=1)
        else:
            raw_action = self._last_action_seq[:, ts % query_frequency]

        action = raw_action * self.act_std + self.act_mean
        self._ts += 1
        return {"action": action.tolist(), "giveup": False}
