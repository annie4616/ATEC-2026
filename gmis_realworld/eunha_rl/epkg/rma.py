"""RMA deployment pieces for Eunha's A2 walking checkpoints.

Phase-1 actor input is 53 = [45 proprio obs | 8-d latent]; the latent comes from a
frozen Phase-2 adaptation module (RMA paper architecture) run over a 50-step history
of the SAME 45-d proprio obs:

  embed: Linear(45, 32) + act              (per step)
  conv:  Conv1d(32,32,k=8,s=4) + act, Conv1d(32,32,k=5,s=1) + act, Conv1d(32,32,k=5,s=1) + act
  head:  Linear(96, 8)                     (96 = 32 x 3 after the conv stack on T=50)

The 45-d obs layout / scales replicate the repo's A2 flat velocity env exactly
(verified identical order): [ang_vel*0.25 | gravity | cmd(3) | q-q_default | qd*0.05 |
last_action] with joint order FR,FL,RR,RL x (hip,thigh,calf).

Activation is not recoverable from the state_dict; EUNHA_ADAPT_ACT env var selects
'elu' (default, repo convention) or 'relu' — the walk validation stage tests both.
History time order: oldest -> newest along the conv T axis.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

HIST_LEN = 50
FEAT_DIM = 45
LATENT_DIM = 8

LEG_ORDER = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


def _act_cls():
    return nn.ReLU if os.environ.get("EUNHA_ADAPT_ACT", "elu").lower() == "relu" else nn.ELU


class AdaptationNet(nn.Module):
    """Frozen Phase-2 adaptation module rebuilt from adaptation.pt."""

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        super().__init__()
        act = _act_cls()
        self.embed = nn.Sequential(nn.Linear(FEAT_DIM, 32), act())
        self.conv = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=8, stride=4), act(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1), act(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1), act(),
        )
        self.head = nn.Linear(96, LATENT_DIM)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck["adaptation_state_dict"] if isinstance(ck, dict) and "adaptation_state_dict" in ck else ck
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, f"adaptation load mismatch: {missing} {unexpected}"
        assert int(ck.get("history_len", HIST_LEN)) == HIST_LEN and int(ck.get("feat_dim", FEAT_DIM)) == FEAT_DIM
        self.to(device).eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        """hist [N, HIST_LEN, FEAT_DIM] (oldest first) -> latent [N, LATENT_DIM]."""
        z = self.embed(hist)                    # [N, T, 32]
        z = self.conv(z.transpose(1, 2))        # [N, 32, T'] with T'=3
        return self.head(z.flatten(1))


def build_walk_actor(ckpt_path: str, device: str = "cpu") -> nn.Sequential:
    """Phase-1 actor MLP 53 -> [512,256,128] ELU -> 12 for manual inference (validation)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    net = nn.Sequential(
        nn.Linear(53, 512), nn.ELU(),
        nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 12),
    )
    net.load_state_dict({k.replace("actor.", ""): v for k, v in sd.items() if k.startswith("actor.")})
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


# ---------------------------------------------------------------- latent as an ObsTerm
def _base45(env: "ManagerBasedRLEnv", cmd_source: str) -> torch.Tensor:
    """Clean (no-corruption) 45-d proprio obs, exact layout/scales of the flat policy group."""
    robot = env.scene["robot"]
    data = robot.data
    if not hasattr(env, "_eunha_leg_ids"):
        names = list(robot.joint_names)
        env._eunha_leg_ids = torch.tensor([names.index(n) for n in LEG_ORDER], device=env.device)
    ids = env._eunha_leg_ids
    n = env.num_envs
    if cmd_source == "crouch":
        from a2crouchpkg import mdp_crouch
        cmd = torch.zeros(n, 3, device=env.device)
        cmd[:, 0] = mdp_crouch._get_cmd(env)
    else:
        cmd = env.command_manager.get_command("base_velocity")
    q = data.joint_pos[:, ids] - data.default_joint_pos[:, ids]
    qd = data.joint_vel[:, ids] * 0.05
    last_a = env.action_manager.action
    if last_a.shape[1] != 12:  # defensive: crouch/flat envs act on exactly the 12 legs
        last_a = last_a[:, :12]
    return torch.cat([data.root_ang_vel_b * 0.25, data.projected_gravity_b, cmd, q, qd, last_a], dim=1)


def rma_latent(env: "ManagerBasedRLEnv", cmd_source: str = "crouch") -> torch.Tensor:
    """8-d latent from the frozen adaptation module over a 50-step proprio history.

    Appended as the LAST obs term -> policy obs [45 | 8] = 53, matching Phase-1's actor
    input. Pushed once per control step (policy+critic groups share the cached value);
    freshly reset envs refill their history with the current obs (repeat-first).
    """
    if not hasattr(env, "_eunha_adapt"):
        ckpt = os.environ.get("EUNHA_ADAPT_CKPT")
        assert ckpt, "EUNHA_ADAPT_CKPT env var must point to adaptation.pt"
        env._eunha_adapt = AdaptationNet(ckpt, device=str(env.device))
        env._eunha_hist = torch.zeros(env.num_envs, HIST_LEN, FEAT_DIM, device=env.device)
        env._eunha_hist_init = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._eunha_stamp = -1
        env._eunha_latent = torch.zeros(env.num_envs, LATENT_DIM, device=env.device)
        print(f"[EUNHA] adaptation module loaded ({os.environ.get('EUNHA_ADAPT_ACT', 'elu')}): {ckpt}", flush=True)
    if env._eunha_stamp != env.common_step_counter:
        env._eunha_stamp = env.common_step_counter
        with torch.no_grad():
            cur = _base45(env, cmd_source)
            fresh = (env.episode_length_buf == 0) | ~env._eunha_hist_init
            if fresh.any():
                env._eunha_hist[fresh] = cur[fresh].unsqueeze(1).expand(-1, HIST_LEN, -1)
                env._eunha_hist_init |= fresh
            env._eunha_hist = torch.roll(env._eunha_hist, shifts=-1, dims=1)
            env._eunha_hist[:, -1] = cur
            env._eunha_latent = env._eunha_adapt(env._eunha_hist)
    return env._eunha_latent


def rma_latent_velocity(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return rma_latent(env, cmd_source="velocity")
