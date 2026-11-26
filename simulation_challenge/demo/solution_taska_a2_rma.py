"""Task A adapter for the A2 + Piper X RMA locomotion policy (phase-1 actor + phase-2 adaptation).

This is a BLIND proprioceptive RMA walker (no lidar / no camera): the policy obs is a 45-dim
proprio vector, and the RMA adaptation module estimates an 8-dim environment latent from a 50-step
proprio history. The actor consumes [proprio(45), z_hat(8)] -> 12 leg joint targets. The arm
(joint1..8) is not actuated; it stays at its default pose.

Checkpoint (A2, eunha's run, read-only) is selected via env vars so this file needs no edits:
  TASKA_RMA_RUN / TASKA_POLICY (model_14999.pt) / TASKA_ADAPT (adaptation.pt).

Per step:
  * pull the 12 legs out of Task A's 20-joint proprio (first 12, per-leg order),
  * drop base_lin_vel, apply training obs scales (ang_vel*0.25, jvel*0.05),
  * inject a forward velocity command with a dead-reckoned yaw-hold toward +x (straight line),
  * build the 45-dim proprio, push into the 50-step history ring buffer,
  * z_hat = phi(history); leg = actor([proprio, z_hat]),
  * rescale train->env (hip 0.25 / others 0.5 -- train scale / env leg scale 0.5) and scatter into
    the 20-dim action (arm indices stay 0 => default arm pose).

Adapted from eunha's demo/solution.py (which targets the B2 run); only the default checkpoint path
differs. The A2 policy trained with the same leg action scales (hip 0.125 / others 0.25) as B2, so
TRAIN_TO_ENV is unchanged.
"""

import os
from typing import Any

import torch

from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.agents.rma_modules import (
    AdaptationModule,
)
from rsl_rl.networks import MLP

# Task A proprio joint_pos/joint_vel and the joint_leg action are both in PER-LEG training order
# (first 12 of the 20 joints; arm = last 8), so no joint permutation is needed.
LEG_IDX = list(range(12))
ARM_IDX = list(range(12, 20))
# env leg action scale is 0.5; policy trained with hip 0.125 / thigh,calf 0.25 -> multiply policy
# output by train_scale/0.5 = hip 0.25, others 0.5.
TRAIN_TO_ENV = [0.25, 0.5, 0.5] * 4

_A2_RUN_DEFAULT = (
    "/hdd0/eunha/ATEC2026_Simulation_Challenge/logs/rsl_rl/"
    "unitree_a2_rma_stable/2026-08-05_05-45-35"
)
_RUN = os.environ.get("TASKA_RMA_RUN", _A2_RUN_DEFAULT)
POLICY_PATH = os.environ.get("TASKA_POLICY", os.path.join(_RUN, "model_14999.pt"))
ADAPT_PATH = os.environ.get("TASKA_ADAPT", os.path.join(_RUN, "adaptation.pt"))

ACTOR_HIDDEN = [512, 256, 128]
ACTOR_ACT = "elu"

FORWARD_SPEED = float(os.environ.get("TASKA_SPEED", "1.0"))
STEP_DT = 0.02
KYAW = float(os.environ.get("TASKA_KYAW", "1.0"))
WZ_MAX = float(os.environ.get("TASKA_WZMAX", "0.5"))


class AlgSolution:

    def __init__(self):
        # Defer all CUDA work to the first predicts() call: the play harness builds AlgSolution
        # before the Isaac env, and an early CUDA context deadlocks Isaac Sim physics init.
        self.device = "cuda"
        self.actor = None
        self.phi = None
        self.leg_idx = None
        self.arm_idx = None
        self.t2e = None
        self.history = None
        self.history_len = None
        self.feat_dim = None
        self._prev_leg = None
        self._ryaw = 0.0

    def _ensure_loaded(self):
        if self.actor is not None:
            return
        dev = self.device

        ck = torch.load(ADAPT_PATH, map_location=dev, weights_only=False)
        self.history_len = int(ck["history_len"])
        self.feat_dim = int(ck["feat_dim"])
        latent_dim = int(ck["latent_dim"])
        self.phi = AdaptationModule(self.feat_dim, self.history_len, latent_dim).to(dev)
        self.phi.load_state_dict(ck["adaptation_state_dict"])
        self.phi.eval()

        msd = torch.load(POLICY_PATH, map_location=dev, weights_only=False)["model_state_dict"]
        actor_sd = {k[len("actor."):]: v for k, v in msd.items() if k.startswith("actor.")}
        in_dim = actor_sd["0.weight"].shape[1]
        out_dim = actor_sd[max(actor_sd, key=lambda k: int(k.split(".")[0]))].shape[0]
        assert in_dim == self.feat_dim + latent_dim, (
            f"actor input {in_dim} != proprio {self.feat_dim} + latent {latent_dim}"
        )
        self.actor = MLP(in_dim, out_dim, ACTOR_HIDDEN, ACTOR_ACT).to(dev)
        self.actor.load_state_dict(actor_sd)
        self.actor.eval()

        self.leg_idx = torch.tensor(LEG_IDX, device=dev, dtype=torch.long)
        self.arm_idx = torch.tensor(ARM_IDX, device=dev, dtype=torch.long)
        self.t2e = torch.tensor(TRAIN_TO_ENV, device=dev, dtype=torch.float32)
        self.history = torch.zeros(1, self.history_len, self.feat_dim, device=dev)
        self._prev_leg = torch.zeros(12, device=dev)
        self._ryaw = 0.0
        print(f"[RMA] loaded A2 actor {POLICY_PATH} (in={in_dim}) + phi hist={self.history_len} "
              f"feat={self.feat_dim} latent={latent_dim}", flush=True)

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {}  # default action config (leg scale 0.5, arm scale 0.5)

    def reset(self, **kwargs) -> None:
        if self.history is not None:
            self.history.zero_()
            self._prev_leg = torch.zeros(12, device=self.device)
        self._ryaw = 0.0

    def predicts(self, obs, current_score):
        self._ensure_loaded()
        p = obs["proprio"].to(self.device)[0]
        n = (p.shape[-1] - 12) // 3                 # 20 joints in Task A
        ang_vel = p[3:6]
        grav = p[9:12]
        jpos = p[12:12 + n][self.leg_idx]
        jvel = p[12 + n:12 + 2 * n][self.leg_idx]

        self._ryaw += float(ang_vel[2]) * STEP_DT
        wz = max(-WZ_MAX, min(WZ_MAX, -KYAW * self._ryaw))
        cmd = torch.tensor([FORWARD_SPEED, 0.0, wz], device=self.device)

        proprio = torch.cat([ang_vel * 0.25, grav, cmd, jpos, jvel * 0.05, self._prev_leg])

        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[0, -1, :] = proprio
        with torch.inference_mode():
            z_hat = self.phi(self.history)[0]
            leg = self.actor(torch.cat([proprio, z_hat]).view(1, -1))
        if leg.ndim > 1:
            leg = leg[0]
        self._prev_leg = leg.clone()

        action = torch.zeros(20, device=self.device)
        action[self.leg_idx] = leg * self.t2e
        return {"action": action.cpu().numpy().tolist(), "giveup": False}
