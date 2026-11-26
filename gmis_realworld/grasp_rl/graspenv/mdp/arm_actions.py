"""Arm IK action with the REAL robot's command path: per-step rate limit + 1-step latency.

WHY. The deployed grasp policies were trained through IsaacLab's differential-IK action,
whose joint target jumps as far as the solver asks every 20 ms. On hardware neither is
true: the PiPER driver enforces 10 rad/s (so the target may move at most 0.2 rad/step or
the driver faults), and every command takes one control period to reach the motors. The
mission measured what evaluating a policy under those constraints costs when it was NOT
trained under them: 7/17 baseline -> ~17 % with any two constraints stacked. This term
puts the same constraints INSIDE training, so the policy adapts to the plant it will
actually drive.

Both knobs are env vars and default OFF, so existing configs train byte-identically:

    GRASP_ARM_RATE    rad per control step of allowed joint-target motion (0.2 = 10 rad/s)
    GRASP_ARM_DELAY   control steps of command latency (1 = the n150 deploy model)

Mirrors mission.ArmDriver._push_arm, which applies the identical treatment at deployment.
"""

from __future__ import annotations

import os

import torch

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass

ARM_RATE = float(os.environ.get("GRASP_ARM_RATE", "0"))
ARM_DELAY = int(os.environ.get("GRASP_ARM_DELAY", "0"))


class RateLimitedIKAction(DifferentialInverseKinematicsAction):
    """Differential IK action whose joint target obeys a slew rate and a latency."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._pend: list[torch.Tensor] = []
        if ARM_RATE > 0.0 or ARM_DELAY > 0:
            print(f"[GRASP] arm action realism: rate {ARM_RATE or 'off'} rad/step, "
                  f"delay {ARM_DELAY} step(s)", flush=True)

    def apply_actions(self):
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(
                ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        if ARM_RATE > 0.0:
            step = joint_pos_des - joint_pos
            joint_pos_des = joint_pos + step.clamp(-ARM_RATE, ARM_RATE)
        if ARM_DELAY > 0:
            self._pend.append(joint_pos_des.clone())
            if len(self._pend) > ARM_DELAY:
                joint_pos_des = self._pend.pop(0)
            else:
                joint_pos_des = joint_pos.clone()
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        # a queued command from before the reset belongs to another episode; simplest
        # correct handling at full-batch resets is to drop the queue, and per-env resets
        # zero those envs' queued entries so they hold still for one step instead
        if env_ids is None:
            self._pend.clear()
        else:
            for t in self._pend:
                t[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]


@configclass
class RateLimitedIKActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type = RateLimitedIKAction
