"""Unitree B2 (with Piper arm) binding for the Task D RL environment.

Uses the SAME robot as the evaluation env (tasks/task_d/env_cfg.py:TaskDEnvB2Cfg):
UNITREE_B2_PIPER_CFG, a 20-DoF B2 + Piper arm spawned at z=0.8. This removes any
robot-level sim2sim gap between training and scoring.

To keep walking learnable, the policy ACTION/OBSERVATION space is restricted to
the 12 leg joints (`leg_joint_names`); the 8 arm joints are held at their default
pose by the arm actuator (stiffness/damping in UNITREE_B2_PIPER_CFG), exactly the
"freeze the arm while walking" behavior we agreed on. Walking-quality rewards and
the Task D objective terms (forward progress + goal) are inherited unchanged.
"""

from isaaclab.utils import configclass

from atec_rl_lab.train.locomotion.taskd.taskd_env_cfg import TaskDRoughEnvCfg
from atec_rl_lab.assets.robots import UNITREE_B2_PIPER_CFG


@configclass
class UnitreeB2TaskDEnvCfg(TaskDRoughEnvCfg):
    base_link_name = "base_link"
    foot_link_name = ".*_foot"
    # Action/observation joints: legs only (arm is held at default by its actuator).
    # fmt: off
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------ Scene -------------------------------
        # Same robot as the evaluation env (TaskDEnvB2Cfg): B2 + Piper arm,
        # spawned at Task D's start (x=-3, y=0) with the eval env's z=0.8.
        self.scene.robot = UNITREE_B2_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_B2_PIPER_CFG.init_state.replace(pos=(-3.0, 0.0, 0.8)),
        )
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------ Observations ------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------ Actions -----------------------------
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------ Events ------------------------------
        # Keep reset randomization modest so the robot reliably starts upright at
        # the Task D spawn (full yaw/roll randomization would face it backwards).
        self.events.randomize_reset_base.params = {
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.3, 0.3)},
            "velocity_range": {
                "x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-10.0, 10.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-5.0, 5.0)

        # ------------------------------ Rewards -----------------------------
        # Walking-quality rewards (same as B2 rough), so the robot stays upright
        # and walks cleanly while the Task D progress/goal terms pull it forward.
        self.rewards.is_terminated.weight = 0
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.base_height_l2.params["target_height"] = 0.53
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.joint_torques_l2.weight = -1e-5
        self.rewards.joint_acc_l2.weight = -1e-7
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_power.weight = -1e-5
        self.rewards.stand_still.weight = -2.0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]
        self.rewards.action_rate_l2.weight = -0.01

        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Keep a mild velocity-tracking prior (forward bias from commands).
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5

        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -5.0
        self.rewards.feet_height_body.params["target_height"] = -0.4
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.upward.weight = 3.0

        # ------------------------------ Terminations ------------------------
        # Base-only illegal contact (box/foot contacts are fine).
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # Drop zero-weight rewards last (also clears the unused reward attrs).
        if self.__class__.__name__ == "UnitreeB2TaskDEnvCfg":
            self.disable_zero_weight_rewards()
