# Heading-hardened variant of the fast B2 walker.
#
# WHY: the fast policy developed a persistent LEFTWARD YAW BIAS — with heading
# feedback OFF (vy=0, kyaw=0) the robot spins ~90 deg within ~550 steps and walks
# off the course sideways; with heading feedback ON, the external P-controller has
# to inject a constant counter-yaw every step, which makes the gait look asymmetric
# ("limping"). Root cause is a net turning drift baked in during training.
#
# WHY NOT joint-symmetry rewards: on ramps / rough terrain the legs *must* move
# asymmetrically (one foot high, one low) to adapt — hard mirror penalties would
# fight that legitimate asymmetry and hurt exactly the sloped section where the
# robot is already falling. The right objective is not "move symmetrically" but
# "hold the commanded heading, however you step." So we strengthen the HEADING
# objective and leave the gait free to adapt to terrain:
#
#   * heading_control_stiffness 0.5 -> 1.0  (env converts heading error to a yaw-rate
#       command twice as assertively, so a drift is corrected sooner)
#   * track_ang_vel_z_exp  weight 1.5 -> 3.0 (reward following that yaw-rate command
#       harder — directly opposes an unbidden turn)
#   * ang_vel_xy_l2 penalty -0.05 -> -0.1   (mildly damp roll/pitch spin too)
#   * joint_mirror kept at the rough default (-0.05) — a light symmetry prior that
#       does NOT dominate terrain adaptation.
#
# The command already runs with heading_command=True, so the policy is trained to
# null out heading error every step — this just makes that objective decisive.
# Warm-starts from the continued fast run, so it keeps ~2.6 m/s and only re-shapes
# the turning behavior, which is fast.

from isaaclab.utils import configclass

from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.fast_env_cfg import (
    UnitreeB2FastEnvCfg,
)


@configclass
class UnitreeB2FastSymEnvCfg(UnitreeB2FastEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --------------------------- Heading holding --------------------------
        # Make the env correct heading error into a yaw-rate command more assertively.
        self.commands.base_velocity.heading_control_stiffness = 1.0   # was 0.5

        # Reward following the (heading-derived) yaw-rate command much harder, so an
        # unbidden turn is penalized directly. This is the primary fix.
        self.rewards.track_ang_vel_z_exp.weight = 3.0                 # was 1.5

        # Mildly damp off-axis (roll/pitch) angular velocity too — keeps the base
        # from yawing via body sway. Gentle so it doesn't fight terrain.
        self.rewards.ang_vel_xy_l2.weight = -0.1                      # was -0.05

        # Keep joint_mirror only as a LIGHT prior (rough default), so it does not
        # penalize the legitimate left/right asymmetry needed on ramps/rough.
        self.rewards.joint_mirror.weight = -0.05

        # Re-run zero-weight pruning for THIS leaf class (parent pruned as a leaf).
        if self.__class__.__name__ == "UnitreeB2FastSymEnvCfg":
            self.disable_zero_weight_rewards()
