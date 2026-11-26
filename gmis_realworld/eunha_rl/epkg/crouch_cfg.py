"""A2 crouch env for the Eunha lineage: crouch task cfg + A2 robot + RMA latent obs.

Policy obs 53 = [45 stock (velocity_commands slot repurposed to the crouch command
[c,0,0] by the base cfg) | 8-d frozen-adaptation latent]. NO gait clock — the Eunha
walk policy is clock-free. Critic gets the latent too (fresh critic anyway)."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from a2pkg.a2 import UNITREE_A2_CFG
from a2crouchpkg.crouch_env_cfg import B2CrouchEnvCfg
from epkg import rma

A2_STAND = 0.43


@configclass
class A2CrouchEunhaEnvCfg(B2CrouchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_A2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        for term in (self.rewards.commanded_height_l2, self.rewards.crouch_height):
            if "stand_h" in term.params:
                term.params["stand_h"] = A2_STAND
        self.observations.policy.rma_latent = ObsTerm(func=rma.rma_latent)
        self.observations.critic.rma_latent = ObsTerm(func=rma.rma_latent)
