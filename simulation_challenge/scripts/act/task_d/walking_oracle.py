"""Walking oracle for Task D: wraps the pretrained B2 locomotion policy.

This reproduces the obs construction and action mapping of demo/solution_rl.py,
but reads state directly from the env's articulation (not from the packed
`obs["proprio"]` vector) so the collector can also record clean joint targets.

The policy is a TorchScript module taking a 48-D leg observation and returning a
12-D leg action (training scale). We hold the arm at its default pose.
"""

from __future__ import annotations

import torch

from .config import (
    LEG_ACTION_DIM,
    FULL_ACTION_DIM,
    TRAIN_TO_ENV_ACTION_SCALE,
    ENV_TO_TRAIN_ACTION_SCALE,
    FORWARD_VELOCITY_COMMAND,
)

# Env action / proprio joint order for B2 + Piper: legs first, then arm
# (matches UNITREE_B2_PIPER_CFG.joint_names). The locomotion policy consumes
# and produces the 12 leg joints in exactly this order.
LEG_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
ARM_JOINT_NAMES = [
    "arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4",
    "arm_joint5", "arm_joint6", "arm_joint7", "arm_joint8",
]


class WalkingOracle:
    def __init__(self, policy_path: str, robot, device: str):
        self.device = device
        self.robot = robot

        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()

        # Resolve articulation-internal indices for legs/arm. robot.data.* is in
        # USD/articulation order, which is NOT guaranteed to be leg-first, so we
        # map explicitly by name.
        self.leg_ids, _ = robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)
        self.arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
        self.leg_ids = list(self.leg_ids)
        self.arm_ids = list(self.arm_ids)

        self._train_to_env = torch.tensor(
            TRAIN_TO_ENV_ACTION_SCALE, device=device, dtype=torch.float32
        ).view(1, -1)
        self._env_to_train = torch.tensor(
            ENV_TO_TRAIN_ACTION_SCALE, device=device, dtype=torch.float32
        ).view(1, -1)
        self._vel_cmd = torch.tensor(
            FORWARD_VELOCITY_COMMAND, device=device, dtype=torch.float32
        ).view(1, 3)

        # Last env-side leg action, needed as part of the policy observation.
        self._last_leg_action_env = torch.zeros(
            (1, LEG_ACTION_DIM), device=device, dtype=torch.float32
        )

        # Optional CLIMB override: a (1, 12) env-side leg action (offset from the
        # default leg pose, same convention as the policy output) that, when set,
        # REPLACES the walking-policy legs in compute_action(). Used to hand-script a
        # box-climbing motion at the CROSS stage. None -> normal walking.
        self._climb_action_env = None

        # Optional ARM override: a (1, 8) env-side arm action (offset from the default
        # arm pose, same (target-default)/ACTION_SCALE convention as the legs) that,
        # when set, drives the arm in compute_action() instead of holding it at default.
        # Used to reach the arm FORWARD onto the box during the climb (5th support point
        # to pull the body over the 0.4 m step). None -> arm held at default (zeros).
        self._arm_action_env = None

    def reset(self) -> None:
        # Reassign fresh tensors rather than mutating in place: when cameras are
        # enabled the policy-inference path can leave these holding inference-mode
        # tensors, and an in-place .zero_()/[:]= on those raises "Inplace update to
        # inference tensor outside InferenceMode is not allowed".
        self._last_leg_action_env = torch.zeros(
            (1, LEG_ACTION_DIM), device=self.device, dtype=torch.float32
        )
        self._vel_cmd = torch.tensor(
            FORWARD_VELOCITY_COMMAND, device=self.device, dtype=torch.float32
        ).view(1, 3)
        self._climb_action_env = None
        self._arm_action_env = None

    def set_arm_action(self, arm_targets) -> None:
        """Drive the ARM with a hand-scripted action (overlaid on the walking legs).

        `arm_targets`: a length-8 sequence of env-side arm-joint offsets (from the
        default arm pose), in ARM_JOINT_NAMES order: arm_joint1..arm_joint8. Same
        convention as the leg action — the env applies it as
        joint_target = default + ACTION_SCALE * action. Pass None to release the arm
        back to its default (held) pose. Used to reach the arm forward onto the box.
        """
        if arm_targets is None:
            self._arm_action_env = None
            return
        from task_d.config import ARM_ACTION_DIM
        self._arm_action_env = torch.tensor(
            arm_targets, device=self.device, dtype=torch.float32
        ).view(1, ARM_ACTION_DIM)

    def set_climb_action(self, leg_targets) -> None:
        """Override the walking legs with a hand-scripted CLIMB leg action.

        `leg_targets`: a length-12 sequence of env-side leg-joint offsets (from the
        default pose), in LEG_JOINT_NAMES order: per leg (hip, thigh, calf) for
        FR, FL, RR, RL. Pass None to return to normal walking. The values are the same
        thing the walking policy outputs (an offset the JointPositionAction applies on
        top of the default joint pose, scaled by the action scale).
        """
        if leg_targets is None:
            self._climb_action_env = None
            return
        self._climb_action_env = torch.tensor(
            leg_targets, device=self.device, dtype=torch.float32
        ).view(1, LEG_ACTION_DIM)

    def set_velocity_command(self, vx: float, vy: float, wz: float) -> None:
        """Update the base velocity command [vx, vy, wz] fed to the walking policy.

        This lets the collector steer the robot (e.g. sidestep toward the box,
        nudge it into the pit, then drive forward) instead of always going +x.
        """
        self._vel_cmd[0, 0] = float(vx)
        self._vel_cmd[0, 1] = float(vy)
        self._vel_cmd[0, 2] = float(wz)

    def _build_policy_obs(self) -> torch.Tensor:
        """Assemble the 48-D leg observation the same way solution_rl.py does."""
        data = self.robot.data

        base_ang_vel = data.root_ang_vel_b[:, :3]
        projected_gravity = data.projected_gravity_b[:, :3]

        joint_pos_rel = data.joint_pos - data.default_joint_pos
        joint_vel = data.joint_vel

        # robot.data is in articulation order -> gather legs by resolved ids,
        # which yields them in LEG_JOINT_NAMES order (the policy's order).
        joint_pos_leg = joint_pos_rel[:, self.leg_ids]
        joint_vel_leg = joint_vel[:, self.leg_ids]

        actions_train_leg = self._last_leg_action_env * self._env_to_train

        policy_obs = torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                self._vel_cmd,
                joint_pos_leg,
                joint_vel_leg * 0.05,
                actions_train_leg,
            ],
            dim=-1,
        )
        return policy_obs

    @torch.inference_mode()
    def compute_action(self) -> torch.Tensor:
        """Return a (1, FULL_ACTION_DIM) env action: walking legs + default arm."""
        policy_obs = self._build_policy_obs()
        action_train = self.policy(policy_obs)
        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        action_train = action_train.to(self.device, torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        leg_action_env = action_train * self._train_to_env
        # CLIMB OVERRIDE: if a hand-scripted climb action is set, use it instead of the
        # walking-policy legs (the box-climbing motion is authored directly, not learned).
        if self._climb_action_env is not None:
            leg_action_env = self._climb_action_env
        # .clone() (not .detach()): this runs under @torch.inference_mode, so the
        # result would be an inference tensor that can't be mutated in place later
        # (reset/set_velocity_command). clone() yields a normal, mutable tensor.
        self._last_leg_action_env = leg_action_env.clone()

        # The env action vector is leg-first (joint_leg term, 12 dims) then arm
        # (joint_arm term, 8 dims), matching UNITREE_B2_PIPER_CFG.joint_names.
        action_env = torch.zeros((1, FULL_ACTION_DIM), device=self.device, dtype=torch.float32)
        action_env[:, :LEG_ACTION_DIM] = leg_action_env
        # Arm joints: hold at default (zeros) unless a hand-scripted arm action is set
        # (e.g. reach forward onto the box during the climb to help pull the body over).
        if self._arm_action_env is not None:
            action_env[:, LEG_ACTION_DIM:] = self._arm_action_env
        else:
            action_env[:, LEG_ACTION_DIM:] = 0.0
        return action_env
