"""Frozen-crouch leg controller as a zero-dim ActionTerm (stage 2).

The RL policy keeps its 7-dim arm+gripper action space; the 12 leg joints are driven by
the FROZEN crouch policy (crouch_sweep depth-0.2 checkpoint, read-only) evaluated inside
this term at the control rate. This reproduces the crouch policy's training interface:

  obs 45 = [base_ang_vel*0.25 | projected_gravity | cmd(1,0,0) | (q-q_stand) | qd*0.05 | last_action]
  targets = q_stand_default + scale * action   (hip 0.125, thigh/calf 0.25)

where q_stand are the VELOCITY-env standing defaults (hip +-0.1, thigh 0.8/1.0, calf -1.5)
-- deliberately independent of this env's articulation defaults (which are the crouch pose).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# crouch policy joint order (velocity env preserve_order list)
LEG_ORDER = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
STAND_DEFAULT = {
    "FR_hip_joint": -0.1, "FL_hip_joint": 0.1, "RR_hip_joint": -0.1, "RL_hip_joint": 0.1,
    "FR_thigh_joint": 0.8, "FL_thigh_joint": 0.8, "RR_thigh_joint": 1.0, "RL_thigh_joint": 1.0,
    "FR_calf_joint": -1.5, "FL_calf_joint": -1.5, "RR_calf_joint": -1.5, "RL_calf_joint": -1.5,
}
ACTION_SCALE = {"hip": 0.125, "other": 0.25}


class FrozenCrouchLegAction(ActionTerm):
    """Zero-dim action term: consumes no policy actions, drives legs with the frozen crouch net."""

    cfg: "FrozenCrouchLegActionCfg"

    def __init__(self, cfg: "FrozenCrouchLegActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        names = self._asset.joint_names
        self._leg_ids = torch.tensor([names.index(n) for n in LEG_ORDER], device=self.device)
        self._q_stand = torch.tensor([STAND_DEFAULT[n] for n in LEG_ORDER], device=self.device)
        self._scale = torch.tensor(
            [ACTION_SCALE["hip"] if "_hip_" in n else ACTION_SCALE["other"] for n in LEG_ORDER],
            device=self.device,
        )
        # frozen crouch actor MLP: 45 (or 49 with the WTW gait clock) -> [512,256,128] ELU -> 12
        in_dim = 49 if cfg.use_clock else 45
        ckpt = torch.load(cfg.checkpoint, map_location=self.device, weights_only=False)
        sd = ckpt["model_state_dict"]
        net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 512), torch.nn.ELU(),
            torch.nn.Linear(512, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 12),
        )
        net.load_state_dict({k.replace("actor.", ""): v for k, v in sd.items() if k.startswith("actor.")})
        self._net = net.to(self.device).eval()
        for p in self._net.parameters():
            p.requires_grad_(False)
        self._last_action = torch.zeros(self.num_envs, 12, device=self.device)
        # hold the asset default (crouch) pose until the first policy step — NOT q_stand
        self._q_default = self._asset.data.default_joint_pos[:, self._leg_ids].clone()
        self._targets = self._q_default.clone()
        self._dummy = torch.zeros(self.num_envs, 0, device=self.device)
        self._dbg_count = 0
        if cfg.use_clock:
            # WTW gait clock (trot, 2 Hz, offsets FL/FR/RL/RR = 0/.5/.5/0, duration .5 ->
            # identity warp). Phase runs CONTINUOUSLY per env (wtw training never reset it).
            self._clk_offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device)
            self._clk_phase = torch.rand(self.num_envs, device=self.device)
        print(f"[LEGS] frozen crouch policy loaded ({in_dim}-obs): {cfg.checkpoint}", flush=True)

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
        parts = [data.root_ang_vel_b * 0.25, data.projected_gravity_b, cmd,
                 q - self._q_stand, qd * 0.05, self._last_action]
        if self.cfg.use_clock:
            self._clk_phase = torch.remainder(self._clk_phase + self._env.step_dt * 2.0, 1.0)
            raw = torch.remainder(self._clk_phase.unsqueeze(1) + self._clk_offsets.unsqueeze(0), 1.0)
            parts.append(torch.sin(2.0 * torch.pi * raw))
        obs = torch.cat(parts, dim=1)
        with torch.no_grad():
            a = self._net(obs)
        import os
        if os.environ.get("GRASP_LEG_DEBUG") and self._dbg_count < 6:
            self._dbg_count += 1
            e = 0
            print(f"[LEGDBG] step{self._dbg_count} angvel={obs[e,:3].tolist()} grav={obs[e,3:6].tolist()} "
                  f"cmd={obs[e,6:9].tolist()}", flush=True)
            print(f"[LEGDBG]   q-qstand={[round(v,2) for v in obs[e,9:21].tolist()]}", flush=True)
            print(f"[LEGDBG]   qd*.05={[round(v,2) for v in obs[e,21:33].tolist()]}", flush=True)
            print(f"[LEGDBG]   last_a={[round(v,2) for v in obs[e,33:45].tolist()]}", flush=True)
            print(f"[LEGDBG]   out={[round(v,2) for v in a[e].tolist()]}", flush=True)
        self._last_action = a
        self._targets = self._q_stand + self._scale * a

    def apply_actions(self):
        self._asset.set_joint_position_target(self._targets, joint_ids=self._leg_ids)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._last_action.zero_()
            self._targets[:] = self._q_default
        else:
            self._last_action[env_ids] = 0.0
            self._targets[env_ids] = self._q_default[env_ids]


@configclass
class FrozenCrouchLegActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = FrozenCrouchLegAction
    checkpoint: str = ""
    """Path to the crouch policy checkpoint (read-only source)."""
    use_clock: bool = False
    """WTW-lineage crouch policies take obs 49 = stock 45 + 4-dim gait clock."""
