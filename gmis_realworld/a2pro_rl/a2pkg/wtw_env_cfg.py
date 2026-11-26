"""A2 Pro WTW variant — the same gait-clock + gait-shaping additions the B2 wtw1 run
used (obs 45->49, trot 2 Hz), on top of the A2 flat cfg. Contact-force sigma scaled to
the ~2/3 body mass (500 -> 350); footswing clearance target scaled to the shorter legs."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import mdp_wtw
from .flat_env_cfg import UnitreeA2FlatEnvCfg


@configclass
class UnitreeA2FlatWTWEnvCfg(UnitreeA2FlatEnvCfg):
    gait_frequency: float = 2.2        # slightly faster than B2's 2.0 (lighter platform)
    gait_stance_duration: float = 0.5
    gait_kappa: float = 0.07

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.gait_clock = ObsTerm(func=mdp_wtw.gait_clock)
        self.observations.critic.gait_clock = ObsTerm(func=mdp_wtw.gait_clock)
        self.events.reset_gait_phase = EventTerm(func=mdp_wtw.reset_gait_phase, mode="reset")
        self.rewards.gait_contact_force = RewTerm(
            func=mdp_wtw.gait_contact_shaped_force, weight=4.0, params={"sigma": 350.0}
        )
        self.rewards.gait_contact_vel = RewTerm(
            func=mdp_wtw.gait_contact_shaped_vel, weight=4.0, params={"sigma": 0.5}
        )
        self.rewards.feet_clearance_wtw = RewTerm(
            func=mdp_wtw.feet_clearance_target,
            weight=-20.0,
            params={"target_height": 0.10, "foot_radius": 0.025},
        )
        self.rewards.action_smoothness_2 = RewTerm(func=mdp_wtw.action_smoothness_2, weight=-0.02)
        self.rewards.feet_slide.weight = -0.04
        self.disable_zero_weight_rewards()
