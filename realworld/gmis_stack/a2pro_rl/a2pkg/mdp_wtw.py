"""walk-these-ways-style gait clock + gait-shaping rewards for the B2 velocity env.

Ported from walk-these-ways (go1_gym/envs/base/legged_robot.py `_step_contact_targets`
and go1_gym/envs/rewards/corl_rewards.py), adapted to IsaacLab's manager-based API:

  - trot only, FIXED gait parameters from the env cfg (no gait commands): diagonal pairs
    (FL,RR) at phase t, (FR,RL) at t+0.5, stance duration 0.5
  - a per-env phase state lives on the env object and advances exactly once per control
    step (guarded by env.common_step_counter — obs and reward terms may call in any order)
  - desired contact states use the WTW normal-CDF smoothing (kappa)
  - clock obs = sin(2*pi*foot_index) per foot, 4 dims, appended AFTER the stock 45 obs
  - gait rewards are gated on |command| > 0.1 so the 2% standing envs (and zero command
    at deployment) fall back to the stock stand-still behavior instead of trotting in place
    (deviation from WTW, which conditions the gait always)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# clock order (WTW convention): [FL, FR, RL, RR]; trot = diagonal pairs half a cycle apart
FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
PHASE_OFFSETS = [0.0, 0.5, 0.5, 0.0]


class _GaitState:
    def __init__(self, env: ManagerBasedRLEnv):
        robot = env.scene["robot"]
        sensor = env.scene.sensors["contact_forces"]
        self.foot_ids = [robot.find_bodies(n)[0][0] for n in FOOT_NAMES]
        self.sensor_foot_ids = [sensor.find_bodies(n)[0][0] for n in FOOT_NAMES]
        self.offsets = torch.tensor(PHASE_OFFSETS, device=env.device)
        self.gait_indices = torch.rand(env.num_envs, device=env.device)  # cycle phase [0,1)
        self.foot_indices = torch.zeros(env.num_envs, 4, device=env.device)
        self.clock = torch.zeros(env.num_envs, 4, device=env.device)
        self.desired_contact = torch.zeros(env.num_envs, 4, device=env.device)
        self.prev2_action = torch.zeros_like(env.action_manager.action)
        self.smooth2 = torch.zeros(env.num_envs, device=env.device)
        self.last_step = -1
        self._advance(env)  # fill buffers for the pre-step reset obs
        print(f"[WTW] gait state: feet {FOOT_NAMES} -> body ids {self.foot_ids}, "
              f"freq {env.cfg.gait_frequency} Hz, duration {env.cfg.gait_stance_duration}", flush=True)

    def _advance(self, env: ManagerBasedRLEnv):
        cfg = env.cfg
        dt = env.step_dt
        duration = float(cfg.gait_stance_duration)  # stance fraction of the cycle
        kappa = float(cfg.gait_kappa)

        self.gait_indices = torch.remainder(self.gait_indices + dt * float(cfg.gait_frequency), 1.0)
        raw = torch.remainder(self.gait_indices.unsqueeze(1) + self.offsets.unsqueeze(0), 1.0)  # (N,4)

        # WTW duration warp: stance maps to [0,0.5), swing to [0.5,1) regardless of duration
        warped = torch.where(
            raw < duration,
            raw * (0.5 / duration),
            0.5 + (raw - duration) * (0.5 / (1.0 - duration)),
        )
        self.foot_indices = warped
        self.clock = torch.sin(2.0 * torch.pi * warped)

        # WTW desired-contact smoothing (normal CDF, kappa)
        cdf = torch.distributions.normal.Normal(0.0, kappa).cdf
        x = torch.remainder(warped, 1.0)
        self.desired_contact = cdf(x) * (1.0 - cdf(x - 0.5)) + cdf(x - 1.0) * (1.0 - cdf(x - 1.5))

        # 2nd-order action difference (a_t - 2*a_{t-1} + a_{t-2}), masked on fresh episodes
        am = env.action_manager
        diff = am.action - 2.0 * am.prev_action + self.prev2_action
        self.smooth2 = torch.sum(torch.square(diff), dim=1) * (env.episode_length_buf >= 3)
        self.prev2_action = am.prev_action.clone()


def _state(env: ManagerBasedRLEnv) -> _GaitState:
    st = getattr(env, "_wtw_gait", None)
    if st is None:
        st = _GaitState(env)
        env._wtw_gait = st
    if st.last_step != env.common_step_counter:
        st.last_step = env.common_step_counter
        st._advance(env)
    return st


def _cmd_gate(env: ManagerBasedRLEnv) -> torch.Tensor:
    cmd = env.command_manager.get_command("base_velocity")
    return (torch.norm(cmd[:, :3], dim=1) > 0.1).float()


# ---------------- observation ----------------
def gait_clock(env: ManagerBasedRLEnv) -> torch.Tensor:
    """sin clock per foot, order [FL, FR, RL, RR] (4)."""
    return _state(env).clock


# ---------------- events ----------------
def reset_gait_phase(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """mode="reset": random initial cycle phase; clear the smoothness history."""
    st = _state(env)
    st.gait_indices[env_ids] = torch.rand(len(env_ids), device=env.device)
    st.prev2_action[env_ids] = 0.0


# ---------------- rewards (WTW corl_rewards ports) ----------------
def gait_contact_shaped_force(env: ManagerBasedRLEnv, sigma: float = 500.0) -> torch.Tensor:
    """Penalize foot contact FORCE during the desired SWING phase. In [-1, 0]."""
    st = _state(env)
    sensor = env.scene.sensors["contact_forces"]
    forces = sensor.data.net_forces_w_history[:, :, st.sensor_foot_ids, :].norm(dim=-1).max(dim=1)[0]  # (N,4)
    rew = -((1.0 - st.desired_contact) * (1.0 - torch.exp(-forces.square() / sigma))).sum(dim=1) / 4.0
    return rew * _cmd_gate(env)


def gait_contact_shaped_vel(env: ManagerBasedRLEnv, sigma: float = 0.5) -> torch.Tensor:
    """Penalize foot VELOCITY during the desired STANCE phase. In [-1, 0]."""
    st = _state(env)
    robot = env.scene["robot"]
    foot_vel = robot.data.body_lin_vel_w[:, st.foot_ids, :].norm(dim=-1)  # (N,4)
    rew = -(st.desired_contact * (1.0 - torch.exp(-foot_vel.square() / sigma))).sum(dim=1) / 4.0
    return rew * _cmd_gate(env)


def feet_clearance_target(env: ManagerBasedRLEnv, target_height: float = 0.12, foot_radius: float = 0.03) -> torch.Tensor:
    """WTW feet_clearance_cmd_linear with a FIXED footswing height: squared error between
    the swing-phase-shaped target height and the actual foot height, swing feet only."""
    st = _state(env)
    robot = env.scene["robot"]
    # tent profile over the swing half of the warped cycle: 0 at lift-off/touch-down, 1 mid-swing
    phases = 1.0 - torch.abs(1.0 - torch.clip(st.foot_indices * 2.0 - 1.0, 0.0, 1.0) * 2.0)
    foot_z = robot.data.body_pos_w[:, st.foot_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    target = target_height * phases + foot_radius
    rew = (torch.square(target - foot_z) * (1.0 - st.desired_contact)).sum(dim=1)
    return rew * _cmd_gate(env)


def action_smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """2nd-order action difference (WTW action_smoothness_2); complements action_rate_l2."""
    return _state(env).smooth2
