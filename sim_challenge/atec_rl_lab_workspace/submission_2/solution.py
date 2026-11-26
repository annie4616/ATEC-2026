"""Task D submission: HIERARCHICAL box-push + walk.

ACT (image -> 3-D velocity command) sits on top of a velocity-tracking walking RL
policy. Each step:
  head_rgb + joint_pos(20)  -->  ACT  -->  (vx, vy, wz)  -->  walking policy  -->  20-D joints
The ACT decides WHERE to push the box (from the head camera); the RL policy walks.
Both run on proprio + robot-mounted cameras only (no privileged box pose), so this is
deployable on the eval server.

Files: policy.pt (walking RL, TorchScript), policy_act.pt (ACT checkpoint with
norm_stats), act/ (DETRVAE model code).
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
# Put this dir on sys.path so the local `act` package resolves whether this module is
# imported as the container's top-level `solution` (submission) or as a deep module
# like `...demo.solution` (play_atec_task harness).
if current_path not in sys.path:
    sys.path.insert(0, current_path)

from act.detr.backbone import build_backbone
from act.detr.transformer import build_transformer
from act.detr.detr_vae import build_encoder, DETRVAE


@dataclass
class ActArgs:
    position_embedding: str = 'sine'
    backbone: str = 'resnet18'
    lr_backbone: float = 1e-5
    masks: bool = False
    dilation: bool = False
    include_depth: bool = False
    include_rgb: bool = True
    enc_layers: int = 2
    dec_layers: int = 4
    dim_feedforward: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    num_queries: int = 30
    pre_norm: bool = False


class ActAgent(nn.Module):
    """ACT (CVAE) policy: image + state -> action chunk. Same architecture as the
    train_task_d.py Agent, so the checkpoint loads directly."""

    def __init__(self, state_dim: int, act_dim: int, args: ActArgs):
        super().__init__()
        self.act_dim = act_dim
        self.include_rgb = args.include_rgb
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        backbones = [build_backbone(args)] if args.include_rgb else None
        transformer = build_transformer(args)
        encoder = build_encoder(args)
        self.model = DETRVAE(backbones, transformer, encoder,
                             state_dim=state_dim, action_dim=act_dim,
                             num_queries=args.num_queries)

    def _preprocess_rgb(self, obs: dict) -> None:
        if self.include_rgb and 'rgb' in obs:
            obs['rgb'] = obs['rgb'].float() / 255.0
            B, N, C, H, W = obs['rgb'].shape
            obs['rgb'] = self.normalize(obs['rgb'].view(B * N, C, H, W)).view(B, N, C, H, W)

    def _model_input(self, obs: dict):
        return obs if self.include_rgb else obs['state']

    @torch.no_grad()
    def get_action(self, obs: dict) -> torch.Tensor:
        self._preprocess_rgb(obs)
        a_hat, _ = self.model(self._model_input(obs))
        return a_hat


class AlgSolution:

    ACTION_SCALE = 0.5
    # Post-push open-loop recovery (after the box scores +14): strafe -y to leave the
    # +y platform into the crossing lane, then walk +x forward.
    V_STRAFE = 1.5            # -y strafe speed (m/s)
    STRAFE_STEPS = 50         # ~1.5 m at 1.5 m/s * 0.02 s/step
    V_FORWARD_POST = 2.0      # +x forward speed after strafing

    def __init__(self):
        self.device = 'cuda'

        # ---- walking RL policy (velocity command -> 12-D leg action) ----
        self.policy = torch.jit.load(current_path + '/policy.pt', map_location=self.device)
        self.policy.eval()

        self.leg_action_dim = 12
        self.arm_action_dim = 8
        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))

        self.train_to_env_action_scale = torch.tensor(
            [0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5],
            device=self.device, dtype=torch.float32).view(1, -1)
        self.env_to_train_action_scale = torch.tensor(
            [4.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0],
            device=self.device, dtype=torch.float32).view(1, -1)
        self.arm_default_action = torch.zeros((1, self.arm_action_dim),
                                              device=self.device, dtype=torch.float32)

        # ---- ACT policy (head_rgb + joint_pos -> velocity command) ----
        ckpt = torch.load(current_path + '/policy_act.pt', map_location=self.device)
        ns = ckpt['norm_stats']
        self.act_state_dim = ns['state_mean'].shape[-1]   # 20 (all joint positions)
        self.act_act_dim = ns['action_mean'].shape[-1]    # 3 (vx, vy, wz)
        weight_key = 'ema_agent' if 'ema_agent' in ckpt else 'agent'
        act_args = ActArgs()
        act_args.include_rgb = any('backbone' in k for k in ckpt[weight_key].keys())
        self.act = ActAgent(self.act_state_dim, self.act_act_dim, act_args).to(self.device)
        self.act.load_state_dict(ckpt[weight_key])
        self.act.eval()
        self.act_include_rgb = act_args.include_rgb
        self.num_queries = act_args.num_queries

        self.state_mean = ns['state_mean'].to(self.device)
        self.state_std = ns['state_std'].to(self.device)
        self.act_mean = ns['action_mean'].to(self.device)
        self.act_std = ns['action_std'].to(self.device)

        # Default joint pose (B2+Piper, in the proprio/observation joint order). The
        # ACT was trained on ABSOLUTE joint positions (collector recorded
        # robot.data.joint_pos), but the proprio observation gives joint_pos_REL
        # (relative to default). So we must add the default back:
        #   absolute = proprio_joint_pos_rel + default_joint_pos
        # (verified in-env: proprio[12:12+adim] + default == robot.data.joint_pos, same
        # order). Without this the ACT saw all-zero joints and output garbage velocity.
        # EXACT default_joint_pos as read from robot.data.default_joint_pos[0] in-env
        # (proprio order; note index 8 is an interleaved arm joint = 0.0, calves at 9-12).
        self.default_joint_pos = torch.tensor(
            [[0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, 0.0, -1.5, -1.5, -1.5, -1.5,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            device=self.device, dtype=torch.float32,
        )

        # temporal-ensembling state (per episode)
        self.temporal_agg = True
        self._ts = 0
        self._all_time_actions = None   # lazily sized (max_steps known only at runtime)
        self._max_steps = 1500
        self._last_vel_cmd = torch.zeros(3, device=self.device)
        self._push_done = False         # True after box scored -> recovery walk
        self._post_step = 0             # steps elapsed in post-push recovery

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {}

    def reset(self, **kwargs) -> None:
        """New episode: clear the temporal-ensembling buffer and timestep."""
        self._ts = 0
        self._all_time_actions = None
        self._last_vel_cmd = torch.zeros(3, device=self.device)
        self._push_done = False   # back to ACT-push phase for the new episode
        self._post_step = 0
        return None

    # ---------------- ACT: head image + joints -> velocity command ----------------
    def _predict_velocity_command(self, obs, joint_pos_all: torch.Tensor) -> torch.Tensor:
        """Return a (3,) velocity command (vx, vy, wz) from the head camera + joints."""
        state = (joint_pos_all - self.state_mean) / self.state_std       # (1, 20)
        model_obs = {'state': state}

        if self.act_include_rgb:
            # eval server provides obs['image']['head_rgb'] (1,480,640,3) uint8 — the
            # same head camera the demos recorded.
            img = obs['image'].get('head_rgb', None)
            if img is None:
                img = obs['image'].get('video_rgb', None)   # fallback if no head cam
            rgb = img.to(self.device)
            if rgb.ndim == 4 and rgb.shape[-1] == 3:
                rgb = rgb.permute(0, 3, 1, 2)               # (1,3,H,W)
            if rgb.dtype != torch.uint8:
                rgb = (rgb.float()).clamp(0, 255).to(torch.uint8) if rgb.max() > 1.5 \
                    else (rgb.float() * 255).clamp(0, 255).to(torch.uint8)
            rgb = TF.resize(rgb, [224, 224], antialias=True)
            model_obs['rgb'] = rgb.unsqueeze(1)             # (1,1,3,224,224)

        a_seq = self.act.get_action(model_obs)              # (1, num_queries, 3)

        if self._all_time_actions is None:
            self._all_time_actions = torch.zeros(
                (self._max_steps, self._max_steps + self.num_queries, self.act_act_dim),
                device=self.device)
        t = min(self._ts, self._max_steps - 1)
        self._all_time_actions[t, t:t + self.num_queries] = a_seq[0]
        if self.temporal_agg:
            pop = self._all_time_actions[:t + 1, t]
            pop = pop[(pop != 0).any(dim=1)]
            if pop.numel() == 0:
                a_norm = a_seq[0, 0]
            else:
                w = torch.exp(-0.01 * torch.arange(len(pop), device=self.device))
                w = (w / w.sum()).unsqueeze(1)
                a_norm = (pop * w).sum(dim=0)
        else:
            a_norm = a_seq[0, 0]
        vel = a_norm * self.act_std[0] + self.act_mean[0]   # denormalize -> (3,)
        if os.environ.get("ACT_DEBUG") and self._ts % 50 == 0:
            _rgb = model_obs.get('rgb')
            _stat = (f"rgb[mean={_rgb.float().mean():.1f} std={_rgb.float().std():.1f}]"
                     if _rgb is not None else "no-rgb")
            print(f"[ACT t={self._ts:4d}] vel_cmd=({vel[0]:+.2f},{vel[1]:+.2f},{vel[2]:+.2f}) {_stat}",
                  flush=True)
        self._ts += 1
        return vel

    # ---------------- walking policy obs (velocity command in) --------------------
    def _extract_policy_obs(self, obs, action_dim, vel_cmd) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)
        idx = 0
        idx += 3                                            # base_lin_vel (unused)
        base_ang_vel = proprio[:, idx:idx + 3]; idx += 3
        idx += 3                                            # env velocity_commands (unused)
        projected_gravity = proprio[:, idx:idx + 3]; idx += 3
        joint_pos_all = proprio[:, idx:idx + action_dim]; idx += action_dim
        joint_vel_all = proprio[:, idx:idx + action_dim]; idx += action_dim
        actions_all = proprio[:, idx:idx + action_dim]

        joint_pos_leg = joint_pos_all[:, self.leg_joint_indices]
        joint_vel_leg = joint_vel_all[:, self.leg_joint_indices]
        actions_train_leg = actions_all[:, self.leg_joint_indices] \
            * self.env_to_train_action_scale.to(dtype=proprio.dtype)

        velocity_commands = vel_cmd.view(1, 3).to(dtype=proprio.dtype)

        return torch.cat([
            base_ang_vel * 0.25,
            projected_gravity,
            velocity_commands,
            joint_pos_leg,
            joint_vel_leg * 0.05,
            actions_train_leg,
        ], dim=-1)

    def _map_policy_action_to_env_action(self, action_train, action_dim):
        if action_train.shape[-1] != self.leg_action_dim:
            raise ValueError(f"Policy output dim {action_train.shape[-1]} != {self.leg_action_dim}")
        leg_action_env = action_train * self.train_to_env_action_scale
        action_env = torch.zeros((action_train.shape[0], action_dim),
                                 device=self.device, dtype=torch.float32)
        action_env[:, self.leg_joint_indices] = leg_action_env
        action_env[:, self.arm_joint_indices] = self.arm_default_action
        return action_env

    def predicts(self, obs, current_score):
        # NOTE: the baseline gave up as soon as current_score > 1, which made the robot
        # quit the instant the box scored (+14) — leaving the robot's own progress points
        # (+2 reach, +20 cross) on the table. Keep going so we accrue those too.

        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3     # 20

        # joint positions for the ACT state. proprio gives joint_pos_REL; the ACT was
        # trained on ABSOLUTE positions, so add the default pose back.
        # proprio order: lin_vel(3) ang_vel(3) vel_cmd(3) gravity(3) joint_pos(N) ...
        joint_pos_rel = proprio[:, 12:12 + action_dim]
        joint_pos_all = joint_pos_rel + self.default_joint_pos[:, :action_dim]

        # PHASE SWITCH: the ACT only knows how to PUSH the box (its demos end once the
        # box is in the pit), so after the push it just stalls on the +y platform. Once
        # the box has scored (current_score >= 14), run an OPEN-LOOP recovery so the robot
        # actually leaves the platform and advances:
        #   (1) STRAFE -y for STRAFE_STEPS to drop off the +y platform into the crossing
        #       lane (~1.5 m: V_STRAFE 1.5 m/s * 0.02 s * 50 = 1.5 m),
        #   (2) then walk FORWARD (+x) to reach x>-1.4 (+2) and attempt the crossing.
        # Open-loop because proprio gives no absolute robot position to track.
        if current_score >= 14.0 and not getattr(self, "_push_done", False):
            self._push_done = True
            self._post_step = 0
        if getattr(self, "_push_done", False):
            if self._post_step < self.STRAFE_STEPS:
                vel_cmd = torch.tensor([0.0, -self.V_STRAFE, 0.0],
                                       device=self.device, dtype=torch.float32)
            else:
                vel_cmd = torch.tensor([self.V_FORWARD_POST, 0.0, 0.0],
                                       device=self.device, dtype=torch.float32)
            self._post_step += 1
        else:
            # ACT: head image + joints -> velocity command
            vel_cmd = self._predict_velocity_command(obs, joint_pos_all)

        # 2) walking RL: velocity command -> 12-D leg action
        policy_obs = self._extract_policy_obs(obs, action_dim, vel_cmd)
        with torch.inference_mode():
            action_train = self.policy(policy_obs)
        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        action_train = action_train.to(device=self.device, dtype=torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        action_env = self._map_policy_action_to_env_action(action_train, action_dim)
        return {'action': action_env.cpu().numpy().tolist(), 'giveup': False}
