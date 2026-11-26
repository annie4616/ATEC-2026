"""A2 flat velocity env + RMA latent obs (53-d policy) — walk-validation env for the
Eunha checkpoints. Identical to the training env recovered from from_eunha env.yaml,
with the frozen-adaptation latent appended as the last policy/critic obs term."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from a2pkg.flat_env_cfg import UnitreeA2FlatEnvCfg
from epkg import rma


@configclass
class A2FlatEunhaEnvCfg(UnitreeA2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.rma_latent = ObsTerm(func=rma.rma_latent_velocity)
        self.observations.critic.rma_latent = ObsTerm(func=rma.rma_latent_velocity)
        self.disable_zero_weight_rewards()
