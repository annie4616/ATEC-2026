"""The leg control step: LegState in, LegCommand out, 50 Hz.

This is the whole locomotion bridge, and it is small on purpose.  The policy needs four
things -- IMU angular velocity, gravity direction, joint positions, joint velocities -- and
produces one thing: twelve position targets.  No map, no absolute pose, no body linear
velocity, nothing that needs a simulator to be true.  That is why locomotion ports cheaply
and everything else does not.

The contract, cross-checked against a2pkg/rough_env_cfg.py, epkg/rma.py and mission.py:

    obs45 = [w_b*0.25 (3) | g_b (3) | cmd (3) | q-q_def (12) | qd*0.05 (12) | a_{t-1} (12)]
    history = 50 frames, oldest first -> AdaptationNet -> latent 8
    actor([obs45, latent]) -> a (12)
    q_target = q_def + action_scale * a

Two traps are handled here explicitly:

  * `a_{t-1}` is the policy's RAW output, before action_scale.  Feeding back the scaled
    target instead changes the observation the policy sees by a factor of 4-8.
  * The adaptation activation is not stored in the checkpoint.  It is chosen by an
    environment variable read at class-construction time, and for this lineage it must be
    ReLU -- ELU collapses velocity tracking on turns (measured 0/160).  We set it from
    config before building the net, and then assert we got what we asked for.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from onboard.cfg import ONBOARD, REPO, onboard_path
from onboard.oio.a2_map import LegMap
from onboard.oio.iface import LegCommand, LegState
from onboard.vendor.loadmod import load_file

RMA_PATH = REPO / "eunha_rl" / "epkg" / "rma.py"


def projected_gravity(quat: np.ndarray) -> np.ndarray:
    """Gravity direction in the body frame from a (w, x, y, z) quaternion.

    Upright is (0, 0, -1).  Identical to the rotation matrix's third row negated, and to
    unitree_rl_gym's `get_gravity_orientation`; kept explicit because getting the
    quaternion convention wrong here is silent -- the robot just walks badly.
    """
    w, x, y, z = (float(v) for v in quat)
    return np.array([2.0 * (-z * x + w * y),
                     -2.0 * (z * y + w * x),
                     1.0 - 2.0 * (w * w + z * z)])


class LegPolicy:
    """RMA walk/crouch runner.  Stateful: keeps the 50-frame history and the last action."""

    def __init__(self, a2_cfg, pol_cfg, mode: str = "walk") -> None:
        self.map = LegMap(a2_cfg)
        self.cfg = a2_cfg
        walk = pol_cfg["walk"]
        self.hist_len = int(walk["history_len"])
        self.feat_dim = int(walk["feat_dim"])
        self.latent_dim = int(walk["latent_dim"])

        rt = pol_cfg.get("runtime", {})
        torch.set_num_threads(int(rt.get("torch_threads", 1)))
        torch.set_grad_enabled(False)
        self.device = str(rt.get("device", "cpu"))

        # MUST precede AdaptationNet(): rma._act_cls() reads this at construction.
        act = str(walk["adaptation_activation"]).lower()
        os.environ["EUNHA_ADAPT_ACT"] = act
        self.rma = load_file(RMA_PATH, "onboard_rma")
        if self.rma._act_cls().__name__.lower() != act:
            raise RuntimeError(
                f"adaptation activation is {self.rma._act_cls().__name__}, config says {act}. "
                "This lineage needs relu; elu collapses tracking on turns.")
        if (self.rma.HIST_LEN, self.rma.FEAT_DIM, self.rma.LATENT_DIM) != (
                self.hist_len, self.feat_dim, self.latent_dim):
            raise RuntimeError("policies.yaml disagrees with rma.py on the RMA dimensions")
        if list(self.rma.LEG_ORDER) != [f"{n}_joint" for n in self.map.names]:
            raise RuntimeError(
                "joint order mismatch between a2pro.yaml and rma.LEG_ORDER:\n"
                f"  yaml: {self.map.names}\n  rma : {self.rma.LEG_ORDER}")

        self.adapt = self.rma.AdaptationNet(str(onboard_path(walk["adaptation"])), self.device)
        self.actors = {"walk": self.rma.build_walk_actor(
            str(onboard_path(walk["actor"])), self.device)}
        crouch = pol_cfg.get("crouch")
        if crouch:
            self.actors["crouch"] = self.rma.build_walk_actor(
                str(onboard_path(crouch["actor"])), self.device)
            self.blend_steps = int(crouch.get("blend_steps", 60))
        else:
            self.blend_steps = 60

        self.mode = mode
        self._hist = torch.zeros(1, self.hist_len, self.feat_dim, device=self.device)
        self._last_a = np.zeros(12)
        self._prev_a = np.zeros(12)
        self._blend = 1.0
        self._fresh = True
        self.latent = np.zeros(self.latent_dim)

    # ---- mode ---------------------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        """Swap actor.  The history is refilled and the output cross-faded, because a
        step change in twelve joint targets at 50 Hz is a stumble."""
        if mode == self.mode:
            return
        if mode not in self.actors:
            raise KeyError(f"no {mode!r} actor loaded (have {sorted(self.actors)})")
        self.mode = mode
        self._fresh = True
        self._blend = 0.0

    def reset(self) -> None:
        self._fresh = True
        self._last_a[:] = 0.0
        self._prev_a[:] = 0.0
        self._blend = 1.0

    # ---- the step -----------------------------------------------------------------------
    def observe(self, st: LegState, cmd: np.ndarray) -> np.ndarray:
        g = projected_gravity(st.quat)
        return np.concatenate([
            np.asarray(st.gyro, float) * 0.25,
            g,
            np.asarray(cmd, float)[:3],
            np.asarray(st.q, float) - self.map.q_default,
            np.asarray(st.dq, float) * 0.05,
            self._last_a,
        ])

    def act(self, st: LegState, cmd: np.ndarray) -> np.ndarray:
        """-> raw policy action (12,).  Call `targets()` to turn it into joint positions."""
        obs = self.observe(st, cmd)
        t = torch.from_numpy(obs).to(torch.float32).unsqueeze(0)
        if self._fresh:
            self._hist[:] = t.unsqueeze(1).expand(-1, self.hist_len, -1)
            self._fresh = False
        else:
            self._hist = torch.roll(self._hist, -1, dims=1)
            self._hist[:, -1] = t
        latent = self.adapt(self._hist)
        a = self.actors[self.mode](torch.cat([t, latent], dim=1))[0].numpy().astype(np.float64)

        # Feed back the RAW action -- this is what a_{t-1} means in the observation.
        self._last_a = a.copy()
        self.latent = latent[0].numpy().astype(np.float64)

        if self._blend < 1.0:
            a = self._prev_a * (1.0 - self._blend) + a * self._blend
            self._blend = min(1.0, self._blend + 1.0 / max(self.blend_steps, 1))
        self._prev_a = a
        return a

    def targets(self, action: np.ndarray) -> np.ndarray:
        return self.map.action_to_target(action)

    def command(self, q_target: np.ndarray) -> LegCommand:
        kp = float(self.cfg["control"]["kp"])
        kd = float(self.cfg["control"]["kd"])
        return LegCommand(q=q_target, kp=np.full(12, kp), kd=np.full(12, kd))

    def step(self, st: LegState, cmd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = self.act(st, cmd)
        return a, self.targets(a)
