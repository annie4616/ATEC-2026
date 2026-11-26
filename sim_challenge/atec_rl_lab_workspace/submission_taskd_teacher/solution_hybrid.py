"""Task D submission — hard-coded box-push + ACT-STUDENT platform climb.

Two phases:

  PHASE 1 (PUSH): open-loop velocity script (measured-velocity odometry). The robot
      backs up, strafes onto the box y-line, then drives +x to shove the box against the
      platform front. A velocity-tracking walking RL policy (policy.pt) executes the
      (vx,vy,wz) command. Scores box_in_target_x (+14).

  PHASE 2 (CLIMB): an ACT STUDENT (policy_student.pt) drives the legs directly from
      head_rgb + joint_pos. The student was distilled from the parkour teacher's reliable
      0.6 m climbs (collected on a box+platform terrain matching this scene, with a
      submission-identical head_camera), so it climbs box-top(0.6) -> platform-top(1.0)
      from camera + proprio ALONE — no odometry, no synthesized scandots, no goal signal.
      Output = 20-D env action (legs + arm); arm held at default.

Switch to CLIMB once the box has scored (+14) and the robot is at the platform front.

Files: policy.pt (walking RL TorchScript, 12-D leg), policy_student.pt (ACT student
checkpoint with norm_stats), act/ (DETRVAE model code).
"""

import math
import os

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from dataclasses import dataclass

current_path = os.path.dirname(os.path.abspath(__file__))
import sys
if current_path not in sys.path:
    sys.path.insert(0, current_path)

from act.detr.backbone import build_backbone
from act.detr.transformer import build_transformer
from act.detr.detr_vae import build_encoder, DETRVAE

BOX_CY = 1.6


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
    """ACT (CVAE): head image + joint state -> action chunk. Matches train_task_d.py."""

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

    def _preprocess_rgb(self, obs):
        if self.include_rgb and 'rgb' in obs:
            obs['rgb'] = obs['rgb'].float() / 255.0
            B, N, C, H, W = obs['rgb'].shape
            obs['rgb'] = self.normalize(obs['rgb'].view(B * N, C, H, W)).view(B, N, C, H, W)

    @torch.no_grad()
    def get_action(self, obs):
        self._preprocess_rgb(obs)
        a_hat, _ = self.model(obs if self.include_rgb else obs['state'])
        return a_hat


class AlgSolution:

    # ---- walking-policy action scaling (PUSH phase) ----
    TRAIN_TO_ENV = [0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5]
    ENV_TO_TRAIN = [4.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0]
    STEP_DT = 0.02

    # ---- PUSH velocity script ----
    SPAWN_X, SPAWN_Y = -3.0, 0.0
    V_SIDE = 1.0
    V_BACK = -1.2
    V_PUSH = 1.4
    V_APPROACH = 1.6
    V_RETREAT = -1.0          # BACKOFF: walk back to open run-up room before climbing,
    BACKOFF_DIST = 0.35       # so the climb matches the parkour demos' open approach
                              # (short + slow -> less +y drift off the platform edge)
    APPROACH_STEPS = 22
    V_CLIMB = 2.0
    WALKCLIMB_STEPS = 60   # walk-forward to wedge front paws on box before student step-up          # strong forward drive during walking-policy climb       # walk forward (box enters view + momentum) before handoff

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # walking RL policy (velocity command -> 12-D leg action) for the PUSH phase
        self.policy = torch.jit.load(current_path + '/policy.pt', map_location=self.device)
        self.policy.eval()
        self.leg_idx = list(range(12))
        self.arm_idx = list(range(12, 20))
        self.train_to_env = torch.tensor(self.TRAIN_TO_ENV, device=self.device).view(1, -1)
        self.env_to_train = torch.tensor(self.ENV_TO_TRAIN, device=self.device).view(1, -1)

        # ACT student (head_rgb + joint_pos -> 20-D env action) for the CLIMB phase
        ckpt = torch.load(current_path + '/policy_student.pt', map_location=self.device)
        ns = ckpt['norm_stats']
        self.st_state_dim = ns['state_mean'].shape[-1]    # 20
        self.st_act_dim = ns['action_mean'].shape[-1]     # 20
        wkey = 'ema_agent' if 'ema_agent' in ckpt else 'agent'
        st_args = ActArgs()
        st_args.include_rgb = any('backbone' in k for k in ckpt[wkey].keys())
        self.student = ActAgent(self.st_state_dim, self.st_act_dim, st_args).to(self.device)
        self.student.load_state_dict(ckpt[wkey])
        self.student.eval()
        self.st_include_rgb = st_args.include_rgb
        self.num_queries = st_args.num_queries
        self.state_mean = ns['state_mean'].to(self.device)
        self.state_std = ns['state_std'].to(self.device)
        self.act_mean = ns['action_mean'].to(self.device)
        self.act_std = ns['action_std'].to(self.device)

        # B2 default joint pose (proprio order, 20). Used to recover ABS joint_pos from
        # proprio's REL joint_pos (the student state was trained on ABS joint_pos).
        leg_default = [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5]
        self.default_joint_pos = torch.tensor([leg_default + [0.0] * 8],
                                              device=self.device, dtype=torch.float32)
        self._init_episode_state()

    def _init_episode_state(self):
        self.rx, self.ry, self.ryaw = self.SPAWN_X, self.SPAWN_Y, 0.0
        self.box_cx = -3.0
        self.phase = "PUSH"
        self.step_i = 0
        self._ts = 0
        self._all_time_actions = None
        self._max_steps = 1500
        self.temporal_agg = True
        self._backoff_target_x = -3.0
        self._settle_step = 0

    def get_action_spec(self):
        return {}

    def reset(self, **kwargs):
        self._init_episode_state()
        return None

    # ----------------------------------------------------------------- odometry
    def _integrate_odometry(self, mvx, mvy, mwz):
        self.ryaw += mwz * self.STEP_DT
        c, s = math.cos(self.ryaw), math.sin(self.ryaw)
        self.rx += (c * mvx - s * mvy) * self.STEP_DT
        self.ry += (s * mvx + c * mvy) * self.STEP_DT
        if self.phase == "PUSH" and mvx > 0.1 and self.rx > -3.2:
            self.box_cx = min(max(self.box_cx, self.rx + 0.45), -1.05)

    # ------------------------------------------------------ walking-policy obs
    def _walking_obs(self, proprio, joint_pos_rel, joint_vel, last_action, vel_cmd):
        base_ang_vel = proprio[:, 3:6]
        gravity = proprio[:, 9:12]
        actions_train_leg = last_action[:12].view(1, -1) * self.env_to_train
        return torch.cat([
            base_ang_vel * 0.25,
            gravity,
            torch.tensor(vel_cmd, device=self.device).view(1, 3),
            joint_pos_rel[:12].view(1, -1),
            (joint_vel[:12] * 0.05).view(1, -1),
            actions_train_leg,
        ], dim=-1)

    def _to_env_action(self, leg_train):
        leg_env = leg_train * self.train_to_env
        action = torch.zeros((1, 20), device=self.device, dtype=torch.float32)
        action[:, self.leg_idx] = leg_env
        action[:, self.arm_idx] = 0.0
        return action

    # -------------------------------------------------- ACT student (CLIMB)
    def _student_action(self, obs, joint_pos_abs):
        """head_rgb + abs joint_pos -> 20-D env action (temporal-ensembled)."""
        state = (joint_pos_abs - self.state_mean) / self.state_std       # (1,20)
        model_obs = {'state': state}
        if self.st_include_rgb:
            img = obs['image'].get('head_rgb', None)
            if img is None:
                img = obs['image'].get('video_rgb', None)
            rgb = img.to(self.device)
            if rgb.ndim == 4 and rgb.shape[-1] == 3:
                rgb = rgb.permute(0, 3, 1, 2)
            if rgb.dtype != torch.uint8:
                rgb = (rgb.float()).clamp(0, 255).to(torch.uint8) if rgb.max() > 1.5 \
                    else (rgb.float() * 255).clamp(0, 255).to(torch.uint8)
            rgb = TF.resize(rgb, [224, 224], antialias=True)
            model_obs['rgb'] = rgb.unsqueeze(1)

        a_seq = self.student.get_action(model_obs)          # (1, num_queries, 20)
        if self._all_time_actions is None:
            self._all_time_actions = torch.zeros(
                (self._max_steps, self._max_steps + self.num_queries, self.st_act_dim),
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
        act = a_norm * self.act_std[0] + self.act_mean[0]   # denormalize -> (20,)
        self._ts += 1
        return act.view(1, 20)

    # ------------------------------------------------------------------ predicts
    def predicts(self, obs, current_score):
        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3              # 20
        base_lin_vel = proprio[0, 0:3]
        joint_pos_rel = proprio[0, 12:12 + action_dim]
        joint_vel = proprio[0, 12 + action_dim:12 + 2 * action_dim]
        last_action = proprio[0, 12 + 2 * action_dim:12 + 3 * action_dim]
        mvx, mvy = float(base_lin_vel[0]), float(base_lin_vel[1])
        mwz = float(proprio[0, 5])

        # PUSH -> BACKOFF -> APPROACH -> CLIMB.
        #   BACKOFF: retreat to open run-up room (camera not buried in the wall).
        #   APPROACH: walk FORWARD (+x) toward the box for a few steps so (a) the box/step
        #     enters the head-camera view and (b) the robot has forward walking momentum —
        #     matching the demos' "walking toward the step" state. Hand to the student
        #     WHILE MOVING (a stationary settle made the camera see only the robot's own
        #     shadow on flat ground -> OOD -> student froze).
        if self.phase == "PUSH" and current_score >= 14.0 and self.rx >= -1.95:
            self.phase = "BACKOFF"
            self._backoff_target_x = self.rx - self.BACKOFF_DIST
        if self.phase == "BACKOFF" and self.rx <= self._backoff_target_x:
            self.phase = "APPROACH"
            self._settle_step = 0
        if self.phase == "APPROACH":
            self._settle_step += 1
            if self._settle_step >= self.APPROACH_STEPS:
                self.phase = "CLIMB"

        if self.phase in ("PUSH", "BACKOFF", "APPROACH"):
            if self.phase == "APPROACH":
                # walk forward toward the box (box enters view, momentum builds), y-trim
                vy = max(-0.4, min(0.4, 3.0 * (BOX_CY - self.ry)))
                vx, wz = self.V_APPROACH, max(-0.4, min(0.4, -1.5 * self.ryaw))
            elif self.phase == "BACKOFF":
                # retreat straight on the box y-line: STRONG y-trim back to BOX_CY so the
                # robot doesn't drift +y off the platform edge while walking backward.
                vy = max(-0.7, min(0.7, 4.0 * (BOX_CY - self.ry)))
                vx, wz = self.V_RETREAT, 0.0
            else:
                behind_x = self.rx <= (self.box_cx - 0.4 - 0.45)
                on_y = abs(self.ry - BOX_CY) <= 0.12
                if not behind_x and not on_y:
                    vx, vy, wz = self.V_BACK, 0.0, 0.0
                elif not on_y:
                    vy = self.V_SIDE if self.ry < BOX_CY else -self.V_SIDE
                    if abs(self.ry - BOX_CY) < 0.4:
                        vy *= 0.4
                    vx, wz = 0.0, 0.0
                else:
                    v = self.V_PUSH if self.box_cx < -1.10 else self.V_APPROACH
                    vy = max(-0.4, min(0.4, 2.0 * (BOX_CY - self.ry)))
                    vx, wz = v, 0.0
            walking_obs = self._walking_obs(proprio, joint_pos_rel, joint_vel,
                                            last_action, (vx, vy, wz))
            with torch.inference_mode():
                leg_train = self.policy(walking_obs)
            if not isinstance(leg_train, torch.Tensor):
                leg_train = torch.as_tensor(leg_train, device=self.device)
            leg_train = leg_train.to(self.device, torch.float32)
            if leg_train.ndim == 1:
                leg_train = leg_train.unsqueeze(0)
            action = self._to_env_action(leg_train)
            self._integrate_odometry(mvx, mvy, mwz)

        else:  # CLIMB — two stage: (a) walking policy drives forward to the stable
               # wedged "front-paws-on-box" pose, then (b) the student does the step-up
               # from that clean in-distribution start.
            self._climb_step = getattr(self, "_climb_step", 0) + 1
            if self._climb_step <= self.WALKCLIMB_STEPS:
                vx, vy, wz = self.V_CLIMB, max(-0.4, min(0.4, 2.0 * (BOX_CY - self.ry))), \
                             max(-0.4, min(0.4, -1.5 * self.ryaw))
                walking_obs = self._walking_obs(proprio, joint_pos_rel, joint_vel,
                                                last_action, (vx, vy, wz))
                with torch.inference_mode():
                    leg_train = self.policy(walking_obs)
                if not isinstance(leg_train, torch.Tensor):
                    leg_train = torch.as_tensor(leg_train, device=self.device)
                leg_train = leg_train.to(self.device, torch.float32)
                if leg_train.ndim == 1:
                    leg_train = leg_train.unsqueeze(0)
                action = self._to_env_action(leg_train)
                self._integrate_odometry(mvx, mvy, mwz)
            else:
                joint_pos_abs = (joint_pos_rel + self.default_joint_pos[0, :action_dim]).view(1, -1)
                action = self._student_action(obs, joint_pos_abs)

        return {'action': action.cpu().numpy().tolist(), 'giveup': False}
