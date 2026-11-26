"""Task D submission — TRACK-PUSH (+16) + ACT-STUDENT (depth+lidar) 0.4 m walk-across cross.

This is the COMBINED solution that pairs the VERIFIED parts of two prior solutions:

  * The TRACK PUSH from solution_teachercross.py — the position-tracking push that
    settles the box into the -y-lane pit and scores +16 (handoff fires at score>=14 with
    the robot at the -y lane). ALL of its track-push, dead-reckoned odometry, start
    alignment (settle/realign/warmup) and handoff machinery is kept here UNCHANGED.

  * The ACT STUDENT from solution_walkacross.py — a CVAE student (depth-as-rgb + abs
    joint_pos + downsampled lidar -> state404 -> 20-dim action, policy_student_walkacross.pt)
    trained on clean 0.4 m walk-across demos. It drives the crossing CLOSED-LOOP from the
    full deployment obs dict (proprio + extero/lidar + head_depth image).

Why this combination: solution_teachercross.py's push is correct (box in -y-lane pit, +16)
but its CROSS used a hand-synthesized 1033-dim parkour TEACHER obs that is OOD and made the
robot fall. solution_walkacross.py's student is correct but its OWN push put the box at the
+y platform front (wrong lane) so the student saw OOD terrain and froze. Here we use the
CORRECT push and the CORRECT student.

PIPELINE (MEM_MODE="studentcross", the default here):
  1. TRACK PUSH — identical to MODE=="track": _tracking_vel_cmd drives the walking RL
     policy (policy.pt, 12-D leg velocity tracker) to shove the box into the -y-lane pit
     for +16, while self.rx/ry/ryaw dead-reckoned odometry updates every step.
  2. STUDENT CROSS handoff — once current_score>=14 AND self.rx>=CROSS_TRIGGER_X (-0.80),
     set self._student_cross=True. From then on, every step we call _student_action(obs, ...)
     which feeds the FULL deployment obs (proprio rel-joints recovered to ABS joint_pos,
     downsampled lidar from extero, head_depth as depth-as-rgb image) to the ACT student
     and returns its 20-dim env action.

The student state404 layout, downsample_lidar, depth handling and temporal aggregation are
REUSED AS-IS from solution_walkacross.py (do not re-derive).

MODES (env-var MEM_MODE): "studentcross" (default here), plus all of solution.py's modes
kept intact as fallbacks: "track", "oracle", "hybrid", "replay", "replayfb".

Files: vel_cmd_seq.npy (T,3), root_pos_seq.npy (T,3), joint_action_seq.npy (T,20),
policy.pt (12-D leg velocity tracker), policy_student_walkacross.pt (ACT student checkpoint
with norm_stats), act/ (DETRVAE model code).
"""

import math
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
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

LEG_PERM = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]          # identity
TRAIN_TO_ENV = [0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5]
ENV_TO_TRAIN = [4.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0]

SPAWN_X, SPAWN_Y = -3.0, 0.0       # known deterministic spawn (robot faces +x, yaw 0)
STEP_DT = 0.02


# =============================================================================
# ACT STUDENT support (reused AS-IS from solution_walkacross.py).
# =============================================================================
# ---- lidar downsampling (MUST match train_task_d.downsample_lidar) ----
# deployment obs["extero"] = mdp.height_scan(offset=0.5) over the 16ch x 360-ray
# LidarPattern = 5760 rays, computed IDENTICALLY to collection (sensor_z-hit_z-0.5).
# Mean-pool the 360 horizontal rays per channel into 24 sectors -> 16*24 = 384.
LIDAR_CHANNELS = 16
LIDAR_DS_SECTORS = 24


def downsample_lidar(scan: torch.Tensor) -> torch.Tensor:
    """(...,5760) height-scan -> (...,384). Mirrors train_task_d.downsample_lidar."""
    a = torch.nan_to_num(scan.float(), nan=0.0, posinf=0.0, neginf=0.0)
    lead = a.shape[:-1]
    n = a.shape[-1]
    per = n // LIDAR_CHANNELS
    a = a[..., :per * LIDAR_CHANNELS].reshape(*lead, LIDAR_CHANNELS, per)
    grp = per // LIDAR_DS_SECTORS
    a = a[..., :grp * LIDAR_DS_SECTORS].reshape(
        *lead, LIDAR_CHANNELS, LIDAR_DS_SECTORS, grp).mean(-1)
    return a.reshape(*lead, LIDAR_CHANNELS * LIDAR_DS_SECTORS)


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

    # MEM_MODE: "studentcross" (default here) = track push (+16) then ACT-student cross;
    #           "track"  = position-tracking push only (robust to air-drop);
    #           "oracle" = open-loop vel_cmd replay (only works from demo start);
    #           "hybrid" = track + memorized open-loop joint_action over the cross window;
    #           "replay"/"replayfb" = pure / feedback-blended recorded joint_action replay.
    MODE = os.environ.get("MEM_MODE", "studentcross")
    # Position-feedback gain (world-error [m] -> body velocity command [m/s]).
    KP_POS = float(os.environ.get("MEM_KP", "1.5"))
    LOOKAHEAD = int(os.environ.get("MEM_LA", "1"))   # demo-path lookahead (steps)
    VXY_MAX = float(os.environ.get("MEM_VMAX", "3.0"))
    WZ_MAX = 1.5
    # Cross-window memorized replay (robot z<0.45 climbing box+platform), traj indices.
    CROSS_START = 950
    CROSS_END = 1200
    # POSITION-triggered cross handoff: once the box has scored (+14) AND the dead-reckoned
    # robot x reaches the demo's cross-start x, hand off the cross. In MODE=="hybrid" this
    # plays the memorized cross actions; in MODE=="studentcross" it switches to the student.
    CROSS_TRIGGER_X = float(os.environ.get("MEM_CROSS_X", "-0.80"))

    # ---- START ALIGNMENT (colleague's technique from demo/solution_b.py) ----
    SETTLE_LIN = 0.08            # base lin speed threshold [m/s]
    SETTLE_ANG = 0.10           # base ang speed threshold [rad/s]
    SETTLE_MIN = int(os.environ.get("MEM_SETTLE_MIN", "30"))   # let the air-drop LAND first
    SETTLE_MAX = int(os.environ.get("MEM_SETTLE_MAX", "120"))
    EASE_STEPS = int(os.environ.get("MEM_EASE", "60"))   # ease landed->demo-start
    MAX_LEG_DELTA = 0.04 / 0.5  # per-step leg action rate limit (rad/step / ACTION_SCALE)
    DO_ALIGN = os.environ.get("MEM_ALIGN", "1") == "1"
    DO_REALIGN = os.environ.get("MEM_REALIGN", "1") == "1"
    REALIGN_TGT_X = -3.0005
    REALIGN_TGT_Y = -0.0002
    REALIGN_TGT_YAW = 0.0
    REALIGN_POS_TOL = 0.03       # [m]
    REALIGN_YAW_TOL = 0.02       # [rad]
    REALIGN_MAX = int(os.environ.get("MEM_REALIGN_MAX", "300"))
    REALIGN_MIN = int(os.environ.get("MEM_REALIGN_MIN", "20"))
    WARMUP_N = int(os.environ.get("MEM_WARMUP_N", "10"))
    FB_KV = float(os.environ.get("MEM_FB_KV", "1.5"))    # pos err [m] -> corrective vel
    FB_GAIN = float(os.environ.get("MEM_FB_GAIN", "2.0"))  # err [m] -> blend weight
    FB_MAX = float(os.environ.get("MEM_FB_MAX", "0.7"))   # max corrective blend weight

    NUM_JOINTS = 20                     # 12 leg + 8 arm

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.velcmd = np.load(os.path.join(current_path, "vel_cmd_seq.npy")).astype(np.float32)   # (T,3)
        self.rootpos = np.load(os.path.join(current_path, "root_pos_seq.npy")).astype(np.float32) # (T,3)
        self.seq = np.load(os.path.join(current_path, "joint_action_seq.npy")).astype(np.float32) # (T,20)
        try:
            # demo-start leg pose (joint_pos_rel, identity order); = (target-default)
            self.jprel_start = np.load(
                os.path.join(current_path, "proprio_jprel_start.npy")).astype(np.float32)
        except Exception:
            self.jprel_start = None
        self.T = self.velcmd.shape[0]

        self.policy = torch.jit.load(current_path + '/policy.pt', map_location=self.device)
        self.policy.eval()

        self.leg_perm = torch.tensor(LEG_PERM, device=self.device, dtype=torch.long)
        self.train_to_env = torch.tensor(TRAIN_TO_ENV, device=self.device, dtype=torch.float32).view(1, -1)
        self.env_to_train = torch.tensor(ENV_TO_TRAIN, device=self.device, dtype=torch.float32).view(1, -1)

        # B2Piper default joint pose in proprio order (legs 12 + arm 8). Used to recover
        # ABS joint_pos from proprio's REL joint_pos (the student state was trained on ABS).
        leg_default = [-0.1, 0.8, -1.5,   0.1, 0.8, -1.5,
                       -0.1, 1.0, -1.5,   0.1, 1.0, -1.5]
        self.default_joint_pos = torch.tensor(
            [leg_default + [0.0] * 8], device=self.device, dtype=torch.float32)

        # ---- ACT student (depth-as-rgb + abs joint_pos + lidar -> 20-D env action) ----
        ckpt = torch.load(current_path + '/policy_student_walkacross.pt',
                          map_location=self.device)
        ns = ckpt['norm_stats']
        self.st_state_dim = ns['state_mean'].shape[-1]    # 404 (20 joints + 384 lidar)
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

        self._init_state()

    def _init_state(self):
        self._t = 0
        # dead-reckoned world pose (robot spawns at -3,0 facing +x)
        self.rx, self.ry, self.ryaw = SPAWN_X, SPAWN_Y, 0.0
        self.rz = 0.5                  # dead-reckoned base height
        self._cross_i = -1            # >=0 once the memorized cross replay is active
        # start-alignment phase machine: "settle" -> "ease" -> "run"
        self._phase = ("settle" if (self.DO_ALIGN and self.jprel_start is not None) else "run")
        self._phase_t = 0
        self._run_t = 0               # trajectory index during RUN
        self._ease_start = None       # captured landed leg pose (action units) at ease start
        self._prev_leg_cmd = None     # for rate-limiting
        # ----- STUDENT CROSS state -----
        self._student_cross = False   # True once we hand the cross off to the ACT student
        # ACT student temporal-aggregation buffers (lazily allocated on first cross step).
        self._ts = 0
        self._all_time_actions = None
        self._max_steps = 1500
        self.temporal_agg = True

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {}

    def reset(self, **kwargs) -> None:
        self._init_state()
        return None

    def _proprio(self, obs):
        pr = obs["proprio"]
        if hasattr(pr, "to"):
            return pr.to(self.device, torch.float32).view(1, -1)
        return torch.as_tensor(np.asarray(pr), device=self.device, dtype=torch.float32).view(1, -1)

    def _tracking_vel_cmd(self, meas_vx, meas_vy, meas_wz, t):
        """Advance odometry with measured body velocity, then return a body-frame velocity
        command = demo feed-forward + P correction toward the demo world position."""
        # integrate dead-reckoning
        self.ryaw += meas_wz * STEP_DT
        c, s = math.cos(self.ryaw), math.sin(self.ryaw)
        self.rx += (c * meas_vx - s * meas_vy) * STEP_DT
        self.ry += (s * meas_vx + c * meas_vy) * STEP_DT

        ti = t if t < self.T else self.T - 1
        la = min(ti + self.LOOKAHEAD, self.T - 1)
        ex = float(self.rootpos[la, 0]) - self.rx
        ey = float(self.rootpos[la, 1]) - self.ry
        # rotate world error into body frame
        bex = c * ex + s * ey
        bey = -s * ex + c * ey
        # feed-forward demo command + P correction
        vx = float(self.velcmd[ti, 0]) + self.KP_POS * bex
        vy = float(self.velcmd[ti, 1]) + self.KP_POS * bey
        wz = float(self.velcmd[ti, 2])
        # clamp
        sp = math.hypot(vx, vy)
        if sp > self.VXY_MAX:
            vx *= self.VXY_MAX / sp
            vy *= self.VXY_MAX / sp
        wz = max(-self.WZ_MAX, min(self.WZ_MAX, wz))
        return vx, vy, wz

    def _walk_leg(self, p, adim, vx=0.0, vy=0.0, wz=0.0):
        """Walking-policy leg action at a given velocity command (SETTLE uses zero)."""
        base_ang_vel = p[:, 3:6]
        projected_gravity = p[:, 9:12]
        joint_pos_leg = p[:, 12:24][:, self.leg_perm]
        joint_vel_leg = p[:, 12 + adim:12 + adim + 12][:, self.leg_perm]
        last_leg = p[:, 12 + 2 * adim:12 + 2 * adim + 12]
        actions_train_leg = last_leg * self.env_to_train
        vc = torch.tensor([[vx, vy, wz]], device=self.device, dtype=torch.float32)
        policy_obs = torch.cat([base_ang_vel * 0.25, projected_gravity, vc,
                                joint_pos_leg, joint_vel_leg * 0.05, actions_train_leg], dim=-1)
        with torch.inference_mode():
            at = self.policy(policy_obs)
        if not isinstance(at, torch.Tensor):
            at = torch.as_tensor(at, device=self.device, dtype=torch.float32)
        at = at.to(self.device, torch.float32)
        if at.ndim == 1:
            at = at.unsqueeze(0)
        return at * self.train_to_env            # (1,12) env-side leg action

    def _odom_step(self, p):
        """Advance dead-reckoned base pose from measured body velocity."""
        mvx, mvy = float(p[0, 0]), float(p[0, 1])
        mwz = float(p[0, 5])
        self.ryaw += mwz * STEP_DT
        c, s = math.cos(self.ryaw), math.sin(self.ryaw)
        self.rx += (c * mvx - s * mvy) * STEP_DT
        self.ry += (s * mvx + c * mvy) * STEP_DT

    def _emit_leg(self, leg_env, adim, rate_limit=False):
        if rate_limit and self._prev_leg_cmd is not None:
            leg_env = self._prev_leg_cmd + torch.clamp(
                leg_env - self._prev_leg_cmd, -self.MAX_LEG_DELTA, self.MAX_LEG_DELTA)
        self._prev_leg_cmd = leg_env.detach().clone()
        action = torch.zeros((1, adim), device=self.device, dtype=torch.float32)
        action[:, :12] = leg_env
        return {'action': [action.cpu().numpy()[0].tolist()], 'giveup': False}

    # ============================================================================
    # ACT STUDENT CROSS (reused AS-IS from solution_walkacross.py).
    # state404 = [joint_pos_abs(20), downsample_lidar(extero)(384)]; depth-as-rgb image;
    # temporal aggregation over num_queries.
    # ============================================================================
    def _student_action(self, obs, joint_pos_abs):
        """head_depth + abs joint_pos + downsampled lidar -> 20-D env action."""
        # state = [joint_pos_abs(20), lidar_ds(384)] = 404, matching training
        # (train_task_d concatenated downsampled lidar AFTER the 20 joints).
        if self.st_state_dim > joint_pos_abs.shape[-1]:
            extero = obs.get('extero', None)
            if extero is None:
                # robustness: zero-lidar (student still has depth vision)
                ld = torch.zeros((1, LIDAR_CHANNELS * LIDAR_DS_SECTORS),
                                 device=self.device)
            else:
                ld = downsample_lidar(extero.to(self.device).float())
                if ld.ndim == 1:
                    ld = ld.unsqueeze(0)
            state_raw = torch.cat([joint_pos_abs, ld], dim=-1)           # (1,404)
        else:
            state_raw = joint_pos_abs                                    # (1,20)
        state = (state_raw - self.state_mean) / self.state_std
        model_obs = {'state': state}
        if self.st_include_rgb:
            # DEPTH-AS-RGB: feed near-field head_depth (0-3m) as a 3-ch grayscale image,
            # matching training (depth_as_rgb). Bypasses the RGB texture/render gap.
            dp = obs['image'].get('head_depth', None)
            d = dp.to(self.device).float()
            if d.ndim == 4 and d.shape[-1] == 1:
                d = d[..., 0]
            if d.ndim == 4:
                d = d[..., 0]
            d = torch.nan_to_num(d, nan=3.0, posinf=3.0, neginf=0.0).clamp(0.0, 3.0) / 3.0
            d = d.unsqueeze(1)                                # (1,1,H,W)
            d = TF.resize(d, [224, 224], antialias=True)
            rgb = (d.repeat(1, 3, 1, 1) * 255).clamp(0, 255).to(torch.uint8)  # (1,3,224,224)
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

    def _student_cross_step(self, obs, p, adim):
        """One STUDENT CROSS step: recover ABS joint_pos from proprio REL joint_pos +
        default, feed the FULL deployment obs to the ACT student, return its 20-dim action.
        Odometry self.rx/ry/ryaw is still advanced from the measured base velocity."""
        joint_pos_rel = p[0, 12:12 + adim]
        joint_pos_abs = (joint_pos_rel + self.default_joint_pos[0, :adim]).view(1, -1)
        action = self._student_action(obs, joint_pos_abs)
        # keep odometry advancing (for diagnostics / fallback parity).
        self._odom_step(p)
        self._t += 1
        return {'action': action.cpu().numpy().tolist(), 'giveup': False}

    def predicts(self, obs, current_score):
        p = self._proprio(obs)
        P = p.shape[-1]
        adim = (P - 12) // 3

        # ===== STUDENT CROSS: once handed off, the ACT student drives every step. =====
        if self._student_cross:
            return self._student_cross_step(obs, p, adim)

        # ===== START ALIGNMENT: SETTLE -> (REALIGN) -> WARMUP -> RUN =====
        HOLD = os.environ.get("MEM_SETTLE_HOLD", "1") == "1"
        if self._phase == "settle":
            self._phase_t += 1
            self._odom_step(p)
            spd = math.hypot(float(p[0, 0]), float(p[0, 1]))
            wz = abs(float(p[0, 5]))
            if HOLD:
                leg = torch.tensor(self.jprel_start[:12] / 0.5, device=self.device,
                                   dtype=torch.float32).view(1, -1)
            else:
                leg = self._walk_leg(p, adim)        # zero-vel stand
            settled = (self._phase_t >= self.SETTLE_MIN
                       and spd < self.SETTLE_LIN and wz < self.SETTLE_ANG)
            if settled or self._phase_t >= self.SETTLE_MAX:
                self._phase = ("realign" if self.DO_REALIGN else "warmup"); self._phase_t = 0
                if os.environ.get("MEM_VERBOSE"):
                    print(f"[solution] SETTLE done at {self._t} (spd={spd:.3f} wz={wz:.3f}) "
                          f"odom=({self.rx:+.3f},{self.ry:+.3f},{self.ryaw:+.3f})", flush=True)
            self._t += 1
            return self._emit_leg(leg, adim, rate_limit=False)

        if self._phase == "realign":
            self._phase_t += 1
            self._odom_step(p)
            ex = self.REALIGN_TGT_X - self.rx
            ey = self.REALIGN_TGT_Y - self.ry
            eyaw = self.REALIGN_TGT_YAW - self.ryaw
            # normalize yaw error to [-pi,pi]
            eyaw = (eyaw + math.pi) % (2 * math.pi) - math.pi
            dist = math.hypot(ex, ey)
            aligned = (self._phase_t >= self.REALIGN_MIN
                       and dist < self.REALIGN_POS_TOL and abs(eyaw) < self.REALIGN_YAW_TOL)
            if aligned or self._phase_t >= self.REALIGN_MAX:
                self._phase = "warmup"; self._phase_t = 0
                if os.environ.get("MEM_VERBOSE"):
                    print(f"[solution] REALIGN done at {self._t} odom=({self.rx:+.3f},{self.ry:+.3f},"
                          f"{self.ryaw:+.3f}) err(d={dist:.3f},yaw={eyaw:+.3f})", flush=True)
                return self._emit_leg(self._walk_leg(p, adim), adim, rate_limit=False)
            # body-frame velocity toward target + yaw correction
            c, s = math.cos(self.ryaw), math.sin(self.ryaw)
            bex = c * ex + s * ey
            bey = -s * ex + c * ey
            vx = max(-0.6, min(0.6, 1.2 * bex))
            vy = max(-0.6, min(0.6, 1.2 * bey))
            wzc = max(-0.6, min(0.6, 1.5 * eyaw))
            return self._emit_leg(self._walk_leg(p, adim, vx, vy, wzc), adim, rate_limit=False)

        if self._phase == "warmup":
            self._phase_t += 1
            leg = self._walk_leg(p, adim, 0.6, 0.0, 0.0)
            if self._phase_t >= self.WARMUP_N:
                self._phase = "run"; self._phase_t = 0
                self._run_t = 0
                if os.environ.get("MEM_VERBOSE"):
                    print(f"[solution] WARMUP({self.WARMUP_N}) done -> replay begins", flush=True)
            return self._emit_leg(leg, adim, rate_limit=False)

        # ===== RUN: the memorized maneuver (t = trajectory index from replay start) =====
        t = self._run_t
        self._run_t += 1
        self._t += 1

        # PURE REPLAY
        if self.MODE == "replay":
            a = self.seq[t if t < self.seq.shape[0] else self.seq.shape[0] - 1]
            return {'action': [a.tolist()], 'giveup': False}

        # REPLAY + POSITION FEEDBACK
        if self.MODE == "replayfb":
            self._odom_step(p)
            ti = t if t < self.T else self.T - 1
            a_ff = self.seq[ti]                                   # (20,) recorded feed-forward
            ex = float(self.rootpos[ti, 0]) - self.rx
            ey = float(self.rootpos[ti, 1]) - self.ry
            err = math.hypot(ex, ey)
            c, s = math.cos(self.ryaw), math.sin(self.ryaw)
            bex = c * ex + s * ey
            bey = -s * ex + c * ey
            kv = self.FB_KV
            vx = max(-1.0, min(1.0, kv * bex))
            vy = max(-1.0, min(1.0, kv * bey))
            corr_leg = self._walk_leg(p, adim, vx, vy, 0.0)[0].detach().cpu().numpy()  # (12,)
            w_corr = min(self.FB_MAX, self.FB_GAIN * err)
            out = a_ff.copy()
            out[:12] = (1.0 - w_corr) * a_ff[:12] + w_corr * corr_leg
            if os.environ.get("MEM_VERBOSE") and t % 100 == 0:
                print(f"[replayfb] t={t} err={err:.3f} w={w_corr:.2f} odom=({self.rx:+.2f},{self.ry:+.2f})", flush=True)
            return {'action': [out.tolist()], 'giveup': False}

        base_lin_vel = p[:, 0:3]
        base_ang_vel = p[:, 3:6]
        projected_gravity = p[:, 9:12]
        jp = p[:, 12:12 + adim]
        jv = p[:, 12 + adim:12 + 2 * adim]
        last_action = p[:, 12 + 2 * adim:12 + 3 * adim]

        joint_pos_leg = jp[:, self.leg_perm]
        joint_vel_leg = jv[:, self.leg_perm]
        actions_train_leg = last_action[:, :12] * self.env_to_train

        # --- already replaying the memorized cross (hybrid mode)? advance it ---
        if self._cross_i >= 0:
            if self._cross_i < min(self.CROSS_END, self.seq.shape[0]):
                a = self.seq[self._cross_i]; self._cross_i += 1
                return {'action': [a.tolist()], 'giveup': False}
            # cross done -> fall through to tracking for the run-out to the finish line

        # velocity command
        if self.MODE in ("track", "hybrid", "studentcross"):
            mvx = float(base_lin_vel[0, 0]); mvy = float(base_lin_vel[0, 1])
            mwz = float(base_ang_vel[0, 2])
            vx, vy, wz = self._tracking_vel_cmd(mvx, mvy, mwz, t)   # also updates odometry
            vc = torch.tensor([vx, vy, wz], device=self.device, dtype=torch.float32).view(1, 3)
            # POSITION-triggered CROSS handoff: box scored (+14) and robot reached the
            # demo's cross-start x.
            if (self._cross_i < 0 and current_score >= 14.0
                    and self.rx >= self.CROSS_TRIGGER_X):
                if self.MODE == "studentcross":
                    # hand the cross off to the ACT student (closed-loop, depth+lidar).
                    self._student_cross = True
                    if os.environ.get("MEM_VERBOSE"):
                        print(f"[solution] STUDENT CROSS handoff fired at t={t} "
                              f"odom_rx={self.rx:+.2f} score={current_score}", flush=True)
                    return self._student_cross_step(obs, p, adim)
                elif self.MODE == "hybrid":
                    self._cross_i = self.CROSS_START
                    if os.environ.get("MEM_VERBOSE"):
                        print(f"[solution] CROSS handoff fired at t={t} odom_rx={self.rx:+.2f} "
                              f"score={current_score}", flush=True)
        else:  # "oracle": open-loop recorded command
            tt = t if t < self.T else self.T - 1
            vc = torch.tensor(self.velcmd[tt], device=self.device, dtype=torch.float32).view(1, 3)

        policy_obs = torch.cat([
            base_ang_vel * 0.25,
            projected_gravity,
            vc,
            joint_pos_leg,
            joint_vel_leg * 0.05,
            actions_train_leg,
        ], dim=-1)

        with torch.inference_mode():
            at = self.policy(policy_obs)
        if not isinstance(at, torch.Tensor):
            at = torch.as_tensor(at, device=self.device, dtype=torch.float32)
        at = at.to(self.device, torch.float32)
        if at.ndim == 1:
            at = at.unsqueeze(0)

        leg_env = at * self.train_to_env
        action = torch.zeros((1, adim), device=self.device, dtype=torch.float32)
        action[:, :12] = leg_env
        return {'action': [action.cpu().numpy()[0].tolist()], 'giveup': False}
