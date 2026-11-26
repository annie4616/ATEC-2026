"""Frozen EUNHA-lineage crouch leg controller (stage 2) — RMA variant of leg_actions.py.

The crouch policy from the eunha lineage takes obs 53 = [45 proprio | 8-d latent], the
latent coming from a frozen conv adaptation module over a 50-step history of the same
45-d proprio (RMA deployment). This term reproduces that interface at the control rate:

  obs45 = [base_ang_vel*0.25 | projected_gravity | cmd(1,0,0) | (q-q_stand) | qd*0.05 | last_action]
  latent = adaptation(history_50(obs45));  targets = q_stand + scale * actor([obs45, latent])

Opt-in via GRASP_LEG_EUNHA=1 (grasp_env_cfg hook); the default lineages are untouched.
Self-contained on purpose (no import from eunha_rl): activation via EUNHA_ADAPT_ACT.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from .leg_actions import ACTION_SCALE, LEG_ORDER, STAND_DEFAULT

HIST_LEN = 50
FEAT_DIM = 45
LATENT_DIM = 8


def _build_adaptation(ckpt_path: str, device) -> nn.Module:
    act = nn.ReLU if os.environ.get("EUNHA_ADAPT_ACT", "elu").lower() == "relu" else nn.ELU

    class _Adapt(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Sequential(nn.Linear(FEAT_DIM, 32), act())
            self.conv = nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=8, stride=4), act(),
                nn.Conv1d(32, 32, kernel_size=5, stride=1), act(),
                nn.Conv1d(32, 32, kernel_size=5, stride=1), act(),
            )
            self.head = nn.Linear(96, LATENT_DIM)

        def forward(self, hist):
            z = self.embed(hist)
            z = self.conv(z.transpose(1, 2))
            return self.head(z.flatten(1))

    net = _Adapt()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["adaptation_state_dict"] if isinstance(ck, dict) and "adaptation_state_dict" in ck else ck
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"adaptation load mismatch: {missing} {unexpected}"
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class EunhaCrouchLegAction(ActionTerm):
    """Zero-dim action term: drives the 12 legs with the frozen 53-obs eunha crouch net."""

    cfg: "EunhaCrouchLegActionCfg"

    def __init__(self, cfg: "EunhaCrouchLegActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        names = self._asset.joint_names
        self._leg_ids = torch.tensor([names.index(n) for n in LEG_ORDER], device=self.device)
        self._q_stand = torch.tensor([STAND_DEFAULT[n] for n in LEG_ORDER], device=self.device)
        self._scale = torch.tensor(
            [ACTION_SCALE["hip"] if "_hip_" in n else ACTION_SCALE["other"] for n in LEG_ORDER],
            device=self.device,
        )
        ckpt = torch.load(cfg.checkpoint, map_location=self.device, weights_only=False)
        sd = ckpt["model_state_dict"]
        net = nn.Sequential(
            nn.Linear(FEAT_DIM + LATENT_DIM, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 12),
        )
        net.load_state_dict({k.replace("actor.", ""): v for k, v in sd.items() if k.startswith("actor.")})
        self._net = net.to(self.device).eval()
        for p in self._net.parameters():
            p.requires_grad_(False)
        self._adapt = _build_adaptation(cfg.adaptation, self.device)
        self._hist = torch.zeros(self.num_envs, HIST_LEN, FEAT_DIM, device=self.device)
        self._hist_fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_action = torch.zeros(self.num_envs, 12, device=self.device)
        self._q_default = self._asset.data.default_joint_pos[:, self._leg_ids].clone()
        self._targets = self._q_default.clone()
        self._dummy = torch.zeros(self.num_envs, 0, device=self.device)
        print(f"[LEGS] frozen EUNHA crouch policy (53-obs RMA): {cfg.checkpoint}\n"
              f"[LEGS]   adaptation: {cfg.adaptation} act={os.environ.get('EUNHA_ADAPT_ACT', 'elu')}",
              flush=True)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._dummy

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._targets

    def process_actions(self, actions: torch.Tensor):
        data = self._asset.data
        q = data.joint_pos[:, self._leg_ids]
        qd = data.joint_vel[:, self._leg_ids]
        n = self.num_envs
        cmd = torch.zeros(n, 3, device=self.device)
        cmd[:, 0] = 1.0  # crouch commanded
        obs45 = torch.cat([data.root_ang_vel_b * 0.25, data.projected_gravity_b, cmd,
                           q - self._q_stand, qd * 0.05, self._last_action], dim=1)
        if self._hist_fresh.any():
            self._hist[self._hist_fresh] = obs45[self._hist_fresh].unsqueeze(1).expand(-1, HIST_LEN, -1)
            self._hist_fresh[:] = False
        self._hist = torch.roll(self._hist, shifts=-1, dims=1)
        self._hist[:, -1] = obs45
        with torch.no_grad():
            latent = self._adapt(self._hist)
            a = self._net(torch.cat([obs45, latent], dim=1))
        self._last_action = a
        self._targets = self._q_stand + self._scale * a

    def apply_actions(self):
        self._asset.set_joint_position_target(self._targets, joint_ids=self._leg_ids)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._last_action.zero_()
            self._targets[:] = self._q_default
            self._hist_fresh[:] = True
        else:
            self._last_action[env_ids] = 0.0
            self._targets[env_ids] = self._q_default[env_ids]
            self._hist_fresh[env_ids] = True


@configclass
class EunhaCrouchLegActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = EunhaCrouchLegAction
    checkpoint: str = ""
    """Path to the EUNHA-lineage crouch policy checkpoint (53-obs actor)."""
    adaptation: str = ""
    """Path to the matching adaptation.pt (frozen Phase-2 module)."""
