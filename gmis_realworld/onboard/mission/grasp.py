"""One grasp attempt: bank start pose, policy loop, abort gates, closure verdict.

Ported from the simulator's honest branch.  The structure is not negotiable -- it is the
interface a frozen checkpoint was trained through -- but three parts of it are worth calling
out because they are what the measurements bought:

**The abort gates fire at exact step indices, not on trends.**  At step 40 a target still
0.20 m away has never recovered; at 220 a reach over 0.20 m is a stall; at 150 a hand that
has hovered without ever closing near is not going to.  Each gate is one measurement, and
together they are the difference between five attempts in the time of one.

**The tactile probe steps the TARGET, not the robot.**  When the hand arrives and the fingers
find nothing, the cheap conclusion is that the approach was wrong and the robot should walk
again -- 4,000 steps.  The likely truth is that the target is a few centimetres off, so the
frozen estimate is stepped through a 45 mm box pattern instead: 45 mm is the measured median
detection-vs-article error.

**A load is a jammed finger.**  Width in band is not enough; the gripper must be commanded
shut AND the width must have stopped changing, or every closing sweep reads as a catch for
the few steps it spends crossing the band.  See `control.arm.HoldDetector`.
"""
from __future__ import annotations

import numpy as np
import torch

from onboard.cfg import onboard_path
from onboard.control.arm import HoldDetector
from onboard.mission.graspobs import ACT_DIM, GraspObs, PregraspBank, quat_from_mat
from onboard.oio.iface import ArmCommand

# The 45 mm box the target is walked through when the fingers find nothing.
PROBE_PATTERN = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1))


class GraspPolicy:
    """42 -> [256, 128, 64] ELU -> 7.  No observation normaliser, no adaptation module:
    unlike the walk policy the only state is `last_action`."""

    def __init__(self, path, device: str = "cpu") -> None:
        import torch.nn as nn
        net = nn.Sequential(
            nn.Linear(42, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 7),
        )
        ck = torch.load(onboard_path(path), map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
        actor = {k.replace("actor.", ""): v for k, v in sd.items() if k.startswith("actor.")}
        missing, unexpected = net.load_state_dict(actor, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"grasp actor load mismatch: {missing} {unexpected}")
        self.net = net.to(device).eval().requires_grad_(False)
        self.iter = int(ck.get("iter", -1))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        # The MEAN, not a sample.  Deterministic evaluation measured 15-21 % against 0.5 %
        # for sampled rollouts -- the exploration std is for training, not for the venue.
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(obs, np.float32)).unsqueeze(0)
            return self.net(t)[0].numpy().astype(np.float64)


class GraspEpisode:
    """Drives one attempt to a verdict.  The caller ticks it at the control rate."""

    def __init__(self, cfg, piper_cfg, policy: GraspPolicy, bank: PregraspBank,
                 arm_controller, arm_bus) -> None:
        g = cfg["grasp"]
        self.cfg = g
        self.policy = policy
        self.bank = bank
        self.arm = arm_controller
        self.bus = arm_bus
        self.obs = GraspObs(cfg, piper_cfg)
        self.hold = HoldDetector(piper_cfg, arm_controller)

        self.max_steps = int(g["steps"])
        self.stall_at, self.stall_reach = int(g["stall_at"]), float(g["stall_reach_m"])
        self.diverge_at, self.diverge = int(g["diverge_at"]), float(g["diverge_m"])
        self.hover_at, self.hover = int(g["hover_at"]), float(g["hover_m"])
        self.max_probes = int(g["probes"])
        self.probe_d = float(g["probe_d_m"])
        self.pos_cap = float(g.get("pos_cap_m", 0.0))
        self.rot_cap = float(g.get("rot_cap_rad", 0.0))
        self.open_stroke = float(piper_cfg["gripper"]["max_stroke_m"])

        self.reset(np.zeros(3), 0.0, np.zeros(3))

    # ---- lifecycle ---------------------------------------------------------------------
    def reset(self, target_b, target_width: float, obb, dim_err: float = 1.0) -> None:
        self.target_b = np.asarray(target_b, float).copy()
        self._target0 = self.target_b.copy()
        self.obs.reset(target_width, obb, dim_err)
        self.hold.reset()
        self.k = 0
        self.best_reach = 9.9
        self.closed_near = 0
        self.probes_used = 0
        self._probe_due = self.stall_at
        self.abort = ""
        self.done = False
        self.held = False
        self.gw_cmd = self.open_stroke

    def start_pose(self) -> np.ndarray:
        """Bank configuration to ramp onto before the policy takes over."""
        return self.bank.q[self.bank.pick(self.target_b, self.cfg.get("bank_pick", "far"),
                                          float(self.cfg["bank_pick_min_sep_m"]))]

    def drift(self) -> float:
        return self.bank.drift(self.target_b)

    # ---- one control step ----------------------------------------------------------------
    def step(self, arm_state, base_quat, base_ang_vel_b, base_lin_acc_w) -> bool:
        """-> True when the episode is over (`self.held` says how it ended)."""
        if self.done:
            return True

        q6 = np.asarray(arm_state.q, float)
        p_ee, R_ee = self.arm.fk(q6)
        # `ee_pose` in the observation is the gripper_base, not the TCP.
        ee_pos_b = p_ee - R_ee @ np.array([0.0, 0.0, self.arm.tcp_offset])
        ee_quat_b = quat_from_mat(R_ee)

        obs = self.obs.build(ee_pos_b, ee_quat_b, self.target_b,
                             arm_state.gripper_stroke, base_quat,
                             base_ang_vel_b, base_lin_acc_w)
        a7 = self.policy(obs)
        self.obs.last_action = a7.copy()

        reach = float(np.linalg.norm(self.target_b - p_ee))
        self.best_reach = min(self.best_reach, reach)
        shut = bool(a7[6] < 0)
        if reach < 0.10 and shut:
            self.closed_near += 1

        self.gw_cmd = 0.0 if shut else self.open_stroke
        if self.hold.update(arm_state, shut, float(p_ee[2])):
            self.held = True
            self.done = True
            return True

        if self._check_aborts(obs, arm_state, shut):
            return True

        # Command the arm.  The caps are off by default (they have never been measured) but
        # exposed so the venue can A/B the 4x train/deploy gap in commanded pose delta.
        cmd = a7[:6].copy()
        if self.pos_cap > 0.0:
            n = np.linalg.norm(cmd[:3]) * float(self.cfg["action_scale"])
            if n > self.pos_cap:
                cmd[:3] *= self.pos_cap / max(n, 1e-9)
        if self.rot_cap > 0.0:
            n = np.linalg.norm(cmd[3:6]) * float(self.cfg["action_scale"])
            if n > self.rot_cap:
                cmd[3:6] *= self.rot_cap / max(n, 1e-9)
        q_target = self.arm.ik_step(q6, cmd)
        self.bus.write(ArmCommand(q=q_target, gripper_stroke=self.gw_cmd))

        self.k += 1
        if self.k >= self.max_steps:
            self.abort = "timeout"
            self.done = True
        return self.done

    def _check_aborts(self, obs, arm_state, shut: bool) -> bool:
        k = self.k
        if k == self.diverge_at and self.best_reach > 0.15 \
                and float(np.linalg.norm(obs[7:10])) > self.diverge:
            self.abort = "diverge"
        elif k == self.stall_at and self.best_reach > self.stall_reach:
            self.abort = "stall"
        elif k == self.hover_at and self.best_reach > self.hover and self.closed_near == 0:
            self.abort = "hover"
        elif (k >= self._probe_due and self.best_reach <= 0.20
              and self.hold.loaded_n == 0 and arm_state.gripper_stroke > 0.005):
            # Arrived, closed, felt nothing.  Step the TARGET rather than walk again.
            if self.probes_used >= self.max_probes:
                self.abort = "probes"
            else:
                dx, dy = PROBE_PATTERN[self.probes_used % len(PROBE_PATTERN)]
                self.target_b = self.target_b + np.array([dx * self.probe_d,
                                                          dy * self.probe_d, 0.0])
                self.probes_used += 1
                self._probe_due = k + max(60, self.stall_at // 2)
                self.best_reach = 9.9
        if self.abort:
            self.done = True
        return self.done

    # ---- reporting ------------------------------------------------------------------------
    def fill(self, attempt) -> None:
        attempt.path = "rl"
        attempt.grasp_steps = self.k
        attempt.tcp_residual_m = round(self.best_reach, 4)
        attempt.abort = self.abort
        attempt.probes_used = self.probes_used
        attempt.gw_cmd_m = self.gw_cmd
        attempt.loaded_steps = self.hold.loaded_n
        attempt.lift_m = round(self.hold.sensed_lift, 4)
        attempt.bank_drift_m = round(self.bank.drift(self._target0), 4)

    def why(self) -> str:
        """The honest failure classification -- the same three buckets the simulator used."""
        if self.held:
            return "held"
        if self.best_reach > 0.12:
            return "never reached its target"
        if self.hold.loaded_n == 0:
            return "arrived on its target and closed on nothing"
        return "closed on it but it slipped"
