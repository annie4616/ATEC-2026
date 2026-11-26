"""Task D submission: ACT box-push (Phase 1, == submission_2) -> PARKOUR crossing (Phase 2).

Pipeline (per step):
  Phase 1 (PUSH, until the box scores +14):  IDENTICAL to submission_2/solution.py
      head_rgb + joint_pos(20) --> ACT(vel) --> (vx, vy, wz) --> walking policy --> 20-D joints
  Phase 2 (CROSS, once current_score >= 14):
      proprio + SYNTHESIZED scandots --> [teacher-obs adapter, 1017-D] --> parkour --> 20-D joints
      Parkour teacher = B2Piper, model_12700 (2026-06-29_08-27-56), exported_teacher/policy.pt.

Scandots: the parkour teacher needs a 132-D height "scandot" grid that the eval server
does NOT provide. We SYNTHESIZE it (ported from submission_taskd_teacher/solution_teacher_b2piper.py):
  - dead-reckon the robot base pose (rx, ry, ryaw, rz) by integrating the MEASURED base
    velocity (proprio[0:3] lin, proprio[3:6] ang) every step, from the known spawn;
  - track the box (box_cx, box_cy) by dead-reckoning during the push;
  - evaluate a hard-coded Task D terrain height under a 12x11 grid (res 0.15, sensor at
    base_link + (0.375,0)) that rotates with yaw; scan = clip(rz - h - 0.3, -1, 1).
This is the previously-built scandot synthesis, now wired to the new 1017-D checkpoint
(priv_latent = 29; the older teacher_b2piper used 45 -> 1033).

Files: policy.pt (walking RL), policy_act.pt (velcmd ACT), policy_parkour.pt (parkour
       model_12700), act/ (DETRVAE model code).
"""

import os
import sys
import math
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


# =============================================================================
# Hard-coded Task D terrain (world frame) — same as solution_teacher_b2piper.py.
#   Robot spawns world (-3.0, 0.0, ~0.5), yaw 0 (+x). Box spawns (-3.0, +1.6).
#   Platform (+y): x in [-0.65,+0.65], y in [+0.5,+3.5], top z = 1.0 (off path).
#   Pit (-y side, y < +0.5): x in [-0.65,+0.65], floor z = -1.0.
#   After the push the box sinks into the -y-lane pit, TOP at z = -0.4 (0.4 m step).
# =============================================================================
PLATFORM_X0, PLATFORM_X1 = -0.65, 0.65
PLATFORM_Y0, PLATFORM_Y1 = 0.5, 3.5
PLATFORM_TOP_Z = 1.0
PIT_X0, PIT_X1 = -0.65, 0.65
PIT_Y_MAX = 0.5
PIT_FLOOR_Z = -1.0
BOX_HALF_X, BOX_HALF_Y = 0.4, 0.5
BOX_TOP_Z = -0.4
BOX_PARKED_CX = 0.0
BOX_PARKED_CY = -0.45
CROSS_LANE_Y = -0.45
SCAN_Z_OFFSET = 0.3                  # clip(base_z - terrain_h - 0.3, -1, 1)


def terrain_height(x: float, y: float, box_cx: float, box_cy: float) -> float:
    """World-frame ground height at (x, y), incl. the box stepping-stone."""
    if (box_cx - BOX_HALF_X) <= x <= (box_cx + BOX_HALF_X) and \
       (box_cy - BOX_HALF_Y) <= y <= (box_cy + BOX_HALF_Y):
        return BOX_TOP_Z
    if PLATFORM_X0 <= x <= PLATFORM_X1 and PLATFORM_Y0 <= y <= PLATFORM_Y1:
        return PLATFORM_TOP_Z
    if PIT_X0 <= x <= PIT_X1 and y < PIT_Y_MAX:
        return PIT_FLOOR_Z
    return 0.0


# Scan grid (teacher height-scanner): 12 x-cols x 11 y-rows = 132, y-outer/x-inner,
# sensor at base_link + (0.375, 0), rotates with yaw.
SCAN_OFFSET_X = 0.375
_SCAN_XS = [round(-0.825 + 0.15 * i, 4) for i in range(12)]
_SCAN_YS = [round(-0.75 + 0.15 * j, 4) for j in range(11)]
_SCAN_LOCAL = [(sx + SCAN_OFFSET_X, sy) for sy in _SCAN_YS for sx in _SCAN_XS]   # 132 (x,y)


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
    """ACT (CVAE) policy: image + state -> action chunk."""

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

    # ---- Phase 1 (== submission_2): default env action scale 0.5 -----------------
    ACTION_SCALE = 0.5

    # ---- parkour teacher obs layout (B2Piper, 20 DoF, 1017-D) --------------------
    #   num_prop 77 ; scandots 132 ; priv_explicit 9 ; priv_latent 29 ; history 10*77.
    NUM_PROP = 3 + 2 + 3 + 4 + 3 * 20 + 5      # 77
    NUM_SCAN = 132
    NUM_PRIV_E = 9
    NUM_PRIV_L = 29                             # new checkpoint (older b2piper used 45)
    NUM_HIST = 10
    NUM_JOINTS = 20
    PARKOUR_OBS_DIM = NUM_PROP + NUM_SCAN + NUM_PRIV_E \
        + NUM_PRIV_L + NUM_HIST * NUM_PROP      # 1017
    PARKOUR_TRAIN_SCALE = 0.25
    PARKOUR_OUT_RESCALE = PARKOUR_TRAIN_SCALE / ACTION_SCALE   # 0.5
    CROSS_VX = 1.0                              # forward vel command fed to the teacher

    STEP_DT = 0.02                              # decimation 4 * sim.dt 0.005
    SPAWN_X, SPAWN_Y, SPAWN_Z = -3.0, 0.0, 0.5
    BOX_SPAWN_CX, BOX_SPAWN_CY = -3.0, 1.6

    def __init__(self):
        self.device = 'cuda'

        # ---- walking RL policy (velocity command -> 12-D leg action) -------------
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

        # ---- ACT policy (head_rgb + joint_pos -> velocity command) ---------------
        ckpt = torch.load(current_path + '/policy_act.pt', map_location=self.device)
        ns = ckpt['norm_stats']
        self.act_state_dim = ns['state_mean'].shape[-1]   # 20
        self.act_act_dim = ns['action_mean'].shape[-1]    # 3
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

        # ---- PARKOUR teacher policy (1017-D obs -> 20-D joint action) ------------
        self.parkour = torch.jit.load(current_path + '/policy_parkour.pt',
                                      map_location=self.device)
        self.parkour.eval()

        # precompute scan grid local coords (132, 2)
        self.scan_local = torch.tensor(_SCAN_LOCAL, device=self.device, dtype=torch.float32)

        # Default joint pose (B2+Piper, proprio order).
        self.default_joint_pos = torch.tensor(
            [[0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, 0.0, -1.5, -1.5, -1.5, -1.5,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            device=self.device, dtype=torch.float32,
        )

        # temporal-ensembling state (per episode)
        self.temporal_agg = True
        self._init_episode_state()

    def _init_episode_state(self):
        self._ts = 0
        self._all_time_actions = None
        self._max_steps = 1500
        self._last_vel_cmd = torch.zeros(3, device=self.device)
        self._push_done = False
        # odometry (dead-reckoned base pose, world frame) + box tracking.
        self.rx, self.ry, self.ryaw, self.rz = self.SPAWN_X, self.SPAWN_Y, 0.0, self.SPAWN_Z
        self.box_cx, self.box_cy = self.BOX_SPAWN_CX, self.BOX_SPAWN_CY
        # parkour history + teacher's own previous action (fed back into proprio).
        self._parkour_hist = None
        self._teacher_last_action = torch.zeros(self.NUM_JOINTS, device=self.device)

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {}                               # == submission_2: default action config

    def reset(self, **kwargs) -> None:
        self._init_episode_state()
        return None

    # ---------------------------------------------------------------- odometry
    def _integrate_odometry(self, obs):
        """Advance dead-reckoned base pose from MEASURED base velocity (proprio)."""
        proprio = obs["proprio"].to(self.device)[0]
        vx, vy = float(proprio[0]), float(proprio[1])
        wz = float(proprio[5])
        self.ryaw += wz * self.STEP_DT
        c, s = math.cos(self.ryaw), math.sin(self.ryaw)
        self.rx += (c * vx - s * vy) * self.STEP_DT
        self.ry += (s * vx + c * vy) * self.STEP_DT
        # Box dead-reckoning during the push (rides ~0.45 m ahead, drifts to pit target).
        if not self._push_done:
            speed = math.hypot(vx, vy)
            if speed > 0.1:
                tgt_cx = self.rx + c * 0.45
                tgt_cy = self.ry + s * 0.45
                self.box_cx = max(min(tgt_cx, BOX_PARKED_CX), self.box_cx) \
                    if self.rx < BOX_PARKED_CX else BOX_PARKED_CX
                if self.rx > -0.8:
                    self.box_cy = BOX_PARKED_CY
                else:
                    self.box_cy = 0.7 * self.box_cy + 0.3 * tgt_cy

    def _synth_scan(self) -> torch.Tensor:
        """132-D height scan from hard-coded terrain + odometry pose."""
        c, s = math.cos(self.ryaw), math.sin(self.ryaw)
        lx = self.scan_local[:, 0]
        ly = self.scan_local[:, 1]
        wx = (self.rx + c * lx - s * ly).tolist()
        wy = (self.ry + s * lx + c * ly).tolist()
        heights = torch.tensor(
            [terrain_height(x, y, self.box_cx, self.box_cy) for x, y in zip(wx, wy)],
            device=self.device, dtype=torch.float32)
        return torch.clip(self.rz - heights - SCAN_Z_OFFSET, -1.0, 1.0)

    # ============================ PHASE 1: ACT push (== submission_2) =============
    def _predict_velocity_command(self, obs, joint_pos_all: torch.Tensor) -> torch.Tensor:
        state = (joint_pos_all - self.state_mean) / self.state_std
        model_obs = {'state': state}
        if self.act_include_rgb:
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

        a_seq = self.act.get_action(model_obs)
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
        vel = a_norm * self.act_std[0] + self.act_mean[0]
        if os.environ.get("ACT_DEBUG") and self._ts % 50 == 0:
            print(f"[ACT t={self._ts:4d}] vel_cmd=({vel[0]:+.2f},{vel[1]:+.2f},{vel[2]:+.2f})",
                  flush=True)
        return vel

    def _extract_policy_obs(self, obs, action_dim, vel_cmd) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)
        idx = 0
        idx += 3
        base_ang_vel = proprio[:, idx:idx + 3]; idx += 3
        idx += 3
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

    # ============================ PHASE 2: parkour cross ==========================
    def _build_proprio77(self, obs, joint_pos_rel, joint_vel) -> torch.Tensor:
        """Teacher 77-D proprio: delta_yaw steered to +x, env_idx=1 (step terrain),
        teacher's own previous action fed back, contacts unknown -> 0."""
        proprio = obs["proprio"].to(self.device)[0]
        base_ang_vel = proprio[3:6]
        gravity = proprio[9:12]
        roll = math.atan2(float(gravity[1]), float(gravity[2]) + 1e-6)
        pitch = math.atan2(-float(gravity[0]),
                           math.sqrt(float(gravity[1])**2 + float(gravity[2])**2) + 1e-6)
        delta_yaw = -self.ryaw
        p = torch.zeros(self.NUM_PROP, device=self.device, dtype=torch.float32)
        p[0:3] = base_ang_vel * 0.25
        p[3] = roll
        p[4] = pitch
        p[6] = delta_yaw
        p[7] = delta_yaw
        p[10] = self.CROSS_VX
        p[11] = 1.0                                       # env_idx: non-flat (step)
        p[13:33] = joint_pos_rel[:self.NUM_JOINTS]
        p[33:53] = joint_vel[:self.NUM_JOINTS] * 0.05
        p[53:73] = self._teacher_last_action
        p[73:77] = 0.0
        return p

    def _parkour_action(self, obs, action_dim) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)[0]
        base_lin_vel = proprio[0:3]
        joint_pos_rel = proprio[12:12 + action_dim]
        joint_vel = proprio[12 + action_dim:12 + 2 * action_dim]

        prop = self._build_proprio77(obs, joint_pos_rel, joint_vel)               # (77,)
        scan = self._synth_scan()                                                 # (132,)

        # history: proprio with idx6:8 (delta_yaw) zeroed, matching training.
        prop_hist = prop.clone()
        prop_hist[6:8] = 0.0
        if self._parkour_hist is None:
            self._parkour_hist = prop_hist.view(1, 1, -1).repeat(1, self.NUM_HIST, 1)
        else:
            self._parkour_hist = torch.cat(
                [self._parkour_hist[:, 1:], prop_hist.view(1, 1, -1)], dim=1)

        priv_e = torch.zeros(self.NUM_PRIV_E, device=self.device)
        priv_e[0:3] = base_lin_vel[:3] * 2.0
        priv_l = torch.zeros(self.NUM_PRIV_L, device=self.device)

        full = torch.cat([prop, scan, priv_e, priv_l,
                          self._parkour_hist.reshape(-1)]).view(1, -1)            # (1, 1017)
        with torch.inference_mode():
            act_raw = self.parkour(full)
        if not isinstance(act_raw, torch.Tensor):
            act_raw = torch.as_tensor(act_raw, device=self.device)
        act_raw = act_raw.to(self.device, torch.float32)
        if act_raw.ndim == 1:
            act_raw = act_raw.unsqueeze(0)
        self._teacher_last_action = act_raw[0].clone()
        # Spec scale is 0.5; rescale so the env reproduces the trained 0.25 delta.
        return (act_raw * self.PARKOUR_OUT_RESCALE).clamp(-9.6, 9.6)

    # ================================ main step ===================================
    def predicts(self, obs, current_score):
        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3     # 20

        # Odometry runs in BOTH phases so the pose/box are correct when Phase 2 starts.
        self._integrate_odometry(obs)

        if current_score >= 14.0:
            self._push_done = True

        if self._push_done:
            action = self._parkour_action(obs, action_dim)
            self._ts += 1
            if os.environ.get("ACT_DEBUG") and self._ts % 50 == 0:
                print(f"[PARKOUR t={self._ts:4d}] odom=({self.rx:+.2f},{self.ry:+.2f},"
                      f"yaw={self.ryaw:+.2f}) box=({self.box_cx:+.2f},{self.box_cy:+.2f})",
                      flush=True)
            return {'action': action.cpu().numpy().tolist(), 'giveup': False}

        # PHASE 1 (== submission_2): ACT head image + joints -> vel cmd -> walking policy.
        joint_pos_rel = proprio[:, 12:12 + action_dim]
        joint_pos_all = joint_pos_rel + self.default_joint_pos[:, :action_dim]
        vel_cmd = self._predict_velocity_command(obs, joint_pos_all)
        self._ts += 1

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
