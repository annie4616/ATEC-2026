"""ATEC-Crouch-B2-v0 environment config.

A 12-DOF leg-only COMMANDED stand<->crouch task for the Unitree B2, built by subclassing
the flat velocity-locomotion env and changing ONLY rewards / commands / events /
terminations / episode length in ``__post_init__``.

A per-env crouch command c in {0,1} (0=stand 0.53 m, 1=crouch 0.30 m) is injected into
the velocity-command obs slot and re-sampled mid-episode, so the policy learns to descend
to a deep crouch AND to recover back to a stable stand on command, with every transition
kept fault-free by the illegal_contact / bad_orientation / base_too_low terminations.

CRITICAL — warm-start shape lock: the policy is warm-started from the flat-locomotion
checkpoint (actor 45->512->256->128->12, critic in=48). We must NOT change the observation
DIMENSIONS, the command term count, or the network arch. We only swap what the (still
3-dim) command obs slot CONTAINS and re-weight rewards/events/terminations, which preserves
policy-obs=45 / critic-obs=48 / action=12, so ``runner.load`` transfers the donor MLP.

Goal: on command, descend FAST to a deep crouch (0.30 m) and hold it stably, and on the
reverse command return to a stable upright stand (0.53 m) — all without base/hip/thigh
ground contact (the deployment's illegal_contact failure at 1.0 N).
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab.managers import EventTermCfg as EventTerm
import atec_rl_lab.train.locomotion.velocity.mdp as mdp
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.flat_env_cfg import (
    UnitreeB2FlatEnvCfg,
)

from . import mdp_crouch

# ── commanded posture targets ──
STAND_HEIGHT = 0.53           # base world-Z [m] when commanded to stand (c=0)
CROUCH_HEIGHT = 0.30          # base world-Z [m] when commanded to crouch (c=1)


@configclass
class B2CrouchEnvCfg(UnitreeB2FlatEnvCfg):
    def __post_init__(self):
        # inherits flat: plane terrain, base_height_l2.sensor_cfg=None (world-frame height),
        # policy obs 45 (no base_lin_vel / height_scan), critic obs 48, action 12.
        super().__post_init__()

        R, E, C, T = self.rewards, self.events, self.commands, self.terminations

        # ───────────────────────── Episode ─────────────────────────
        # long enough for several stand<->crouch phases per episode.
        self.episode_length_s = 12.0

        # ──────────────── Command: crouch level in the (repurposed) cmd obs slot ────────
        # Inject c in {0,1} (0=stand, 1=crouch) into the velocity-command obs slot. Obs
        # width is unchanged (still a 3-vector) so the warm-start still loads. The base
        # velocity command itself stays zeroed (so track_lin_vel_xy_exp still anchors XY
        # position to "stay put"), but the obs slot now carries the crouch command.
        C.base_velocity.rel_standing_envs = 1.0
        C.base_velocity.rel_heading_envs = 0.0
        C.base_velocity.heading_command = False
        C.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        C.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        C.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        C.base_velocity.ranges.heading = (0.0, 0.0)
        # replace the velocity_commands obs (policy AND critic) with the crouch command
        self.observations.policy.velocity_commands.func = mdp_crouch.crouch_cmd_obs
        self.observations.policy.velocity_commands.params = {}
        self.observations.critic.velocity_commands.func = mdp_crouch.crouch_cmd_obs
        self.observations.critic.velocity_commands.params = {}
        # sample the command at reset and re-sample mid-episode (~every 3 s) so BOTH
        # stand->crouch and crouch->stand transitions are trained -> stable recovery.
        E.set_crouch_cmd = EventTerm(
            func=mdp_crouch.set_crouch_cmd, mode="reset", params={"p_crouch": 0.5})
        E.resample_crouch_cmd = EventTerm(
            func=mdp_crouch.set_crouch_cmd, mode="interval",
            interval_range_s=(2.5, 4.0), params={"p_crouch": 0.5})

        # ──────────────────── Reset: near-upright, mild ────────────────────
        # The inherited reset uses roll/pitch +/-3.14 (spawns upside-down half the time) —
        # fatal for "descend from upright" and wastes the warm-start. Keep it near upright.
        E.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0.0, 0.05),
                "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0),
                "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1),
            },
        }
        # gentler pushes while the policy is still learning to hold a crouch
        E.randomize_push_robot.params["velocity_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}

        # ───────────────────────── Rewards ─────────────────────────
        # Primary: track the COMMANDED base height (stand 0.53 / crouch 0.30). Because the
        # target follows the command, moving toward it is rewarded in BOTH directions —
        # this is what trains the crouch AND the stable recovery back to standing.
        R.base_height_l2.weight = 0.0          # disable the inherited fixed-target term
        R.commanded_height_l2 = RewTerm(
            func=mdp_crouch.commanded_height_l2,
            weight=-30.0,
            params={"stand_h": STAND_HEIGHT, "crouch_h": CROUCH_HEIGHT,
                    "asset_cfg": SceneEntityCfg("robot")},
        )
        R.crouch_height = RewTerm(
            func=mdp_crouch.commanded_height,
            weight=5.0,
            params={"stand_h": STAND_HEIGHT, "crouch_h": CROUCH_HEIGHT, "std": 0.06,
                    "asset_cfg": SceneEntityCfg("robot")},
        )

        # Posture / stability
        R.flat_orientation_l2.weight = -2.0          # torso level
        R.upward.weight = 1.0                         # anti-flip (NOT gated; positive=upright)
        R.lin_vel_z_l2.weight = -0.5                 # landing damping, kept LOW so descent stays fast
        R.ang_vel_xy_l2.weight = -0.1                # anti-wobble

        # Contacts
        R.undesired_contacts.weight = -2.0           # base/hip/thigh/calf ground contact (deployment failure)
        R.contact_forces.weight = -1.5e-4            # soft foot landing
        R.feet_contact_without_cmd.weight = 0.5      # reward all-4-feet planted while standing
        R.feet_slide.weight = -0.25                  # no foot skating / drift
        R.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        R.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]

        # Position anchor (reward holding zero XY velocity -> stay put during the hold)
        R.track_lin_vel_xy_exp.weight = 1.0
        R.track_ang_vel_z_exp.weight = 0.0

        # Regularizers (smoothness / effort)
        R.joint_mirror.weight = -0.05                # symmetric crouch
        R.joint_pos_limits.weight = -5.0             # keep deep calf within limits
        R.joint_acc_l2.weight = -2.5e-7
        R.joint_torques_l2.weight = -1e-5
        R.joint_power.weight = -1e-5
        R.action_rate_l2.weight = -0.01              # do not raise (would slow descent)

        # Disable the antagonists that pull the legs back to the STANDING default pose.
        R.stand_still.weight = 0.0                   # penalizes deviation from standing (fights crouch)
        R.joint_pos_penalty.weight = 0.0             # same, 5x amplified when standing
        R.feet_height_body.weight = 0.0              # wrong target; dead when cmd=0

        # ───────────────────────── Terminations ─────────────────────────
        # Re-enable illegal_contact = the EXACT deployment failure (base/hip/thigh @ 1.0 N).
        T.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[self.base_link_name, ".*_hip", ".*_thigh"],
                ),
                "threshold": 1.0,
            },
        )
        # Flip / severe tilt — closes the uprightness-gate "fall onto back" local optimum.
        T.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 0.8, "asset_cfg": SceneEntityCfg("robot")},
        )
        # Belly-flop / collapse (world-frame height on flat); below the 0.30 target so it
        # does not fire during a healthy crouch.
        T.base_too_low = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.18, "asset_cfg": SceneEntityCfg("robot")},
        )

        # Prune zero-weight reward terms (parent's auto-prune is gated on the flat class
        # name and will NOT run for this subclass, so call it explicitly).
        self.disable_zero_weight_rewards()
