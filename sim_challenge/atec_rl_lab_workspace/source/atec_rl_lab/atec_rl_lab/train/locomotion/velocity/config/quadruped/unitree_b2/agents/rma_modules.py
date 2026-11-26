# Rapid Motor Adaptation (RMA) phase-1 network for the B2 + Piper policy.
#
# Design (grounded in the installed rsl_rl 3.0.1 obs-dict API):
#   * The runner builds the policy via `eval(class_name)(obs, obs_groups, num_actions, **policy_cfg)`
#     (rsl_rl/runners/on_policy_runner.py:418). `eval` resolves names against that module's globals,
#     which only import ActorCritic/ActorCriticRecurrent -- so a custom class must be *injected* into
#     that namespace. `_register()` at the bottom does exactly that (no library edits).
#   * Observations arrive as a dict keyed by group name. We put the environment factors e_t in their
#     own group (default "factors") and list it in obs_groups["policy"]. The base ActorCritic would
#     just concatenate every policy group; instead we override get_actor_obs to route the factor group
#     through an encoder MLP mu -> latent z_t and feed the actor concat[proprio, z_t].
#   * act / act_inference / update_distribution all funnel through get_actor_obs, so overriding
#     __init__ (to build mu and resize the actor) and get_actor_obs is sufficient. The critic is
#     untouched (still reads the privileged "critic" group).
#
# Phase 2 (not here) freezes this encoder and trains an adaptation module to regress z_t from a
# proprioceptive history window; nothing in this class needs to change for that.

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP


class ActorCriticRMA(ActorCritic):
    """Asymmetric actor-critic with an environment-factor encoder on the actor side."""

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        factor_group: str = "factors",
        latent_dim: int = 8,
        encoder_hidden_dims=(128, 128),
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        **kwargs,
    ):
        if actor_obs_normalization:
            # the base normalizer would be sized proprio+factor, but the actor now consumes
            # proprio+latent -- the dimensions would not match.
            raise ValueError("ActorCriticRMA does not support actor_obs_normalization.")

        # Let the base class build the critic, the action-noise std and the (Identity) normalizers.
        # Its actor is built for concat[proprio, e_t]; we rebuild it below for concat[proprio, z_t].
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )

        assert factor_group in obs, (
            f"ActorCriticRMA: factor group '{factor_group}' missing from observations {list(obs.keys())}"
        )
        assert factor_group in obs_groups["policy"], (
            f"ActorCriticRMA: '{factor_group}' must be listed in obs_groups['policy'] (got {obs_groups['policy']})"
        )

        self.factor_group = factor_group
        self.proprio_groups = [g for g in obs_groups["policy"] if g != factor_group]

        factor_dim = obs[factor_group].shape[-1]
        proprio_dim = sum(obs[g].shape[-1] for g in self.proprio_groups)

        # environment-factor encoder mu:  e_t -> z_t
        self.factor_encoder = MLP(factor_dim, latent_dim, list(encoder_hidden_dims), activation)
        # rebuild the actor to consume concat[proprio, z_t]
        self.actor = MLP(proprio_dim + latent_dim, num_actions, list(actor_hidden_dims), activation)
        self.latent_dim = latent_dim

        print(f"RMA factor encoder (e_t[{factor_dim}] -> z[{latent_dim}]): {self.factor_encoder}")
        print(f"RMA actor (proprio[{proprio_dim}] + z[{latent_dim}]): {self.actor}")

    def encode_factors(self, obs):
        """z_t = mu(e_t). Exposed so phase-2 can use it as the regression target."""
        return self.factor_encoder(obs[self.factor_group])

    def get_actor_obs(self, obs):
        proprio = torch.cat([obs[group] for group in self.proprio_groups], dim=-1)
        latent = self.encode_factors(obs)
        return torch.cat([proprio, latent], dim=-1)


class AdaptationModule(nn.Module):
    """RMA phase-2 adaptation module phi.

    Regresses the latent z_t (produced by the frozen factor encoder mu in phase 1) from a window of
    the most recent proprioceptive observations. At deployment the environment factors e_t are
    unknown, so z_hat = phi(history) replaces z_t = mu(e_t) as the actor's latent input.

    Architecture follows the original RMA: a per-timestep linear embedding, a small 1-D temporal
    CNN over the history, then a linear head to the latent dimension. The flattened conv size is
    resolved with a dummy forward so it works for any history length.
    """

    def __init__(
        self,
        feat_dim: int,
        history_len: int,
        latent_dim: int,
        channels: int = 32,
        activation=nn.ELU,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.history_len = history_len
        self.latent_dim = latent_dim

        self.embed = nn.Sequential(nn.Linear(feat_dim, channels), activation())
        # temporal convolution over the history axis; kernels/strides mirror the RMA paper but are
        # clamped so they never exceed the current sequence length.
        k1, s1 = min(8, history_len), max(1, min(4, history_len // 4))
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=k1, stride=s1), activation(),
            nn.Conv1d(channels, channels, kernel_size=min(5, max(1, history_len // 2)), stride=1), activation(),
            nn.Conv1d(channels, channels, kernel_size=min(5, max(1, history_len // 4)), stride=1), activation(),
        )
        # resolve the flattened conv output dimension
        with torch.no_grad():
            dummy = torch.zeros(1, history_len, feat_dim)
            flat = self._conv_forward(dummy).shape[-1]
        self.head = nn.Linear(flat, latent_dim)

    def _conv_forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: (N, history_len, feat_dim)
        x = self.embed(history)           # (N, T, channels)
        x = x.transpose(1, 2)             # (N, channels, T)
        x = self.conv(x)                  # (N, channels, T')
        return x.flatten(1)               # (N, channels * T')

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.head(self._conv_forward(history))


def _register():
    """Bind ActorCriticRMA where the rsl_rl runner's `eval(class_name)` can see it.

    The runner resolves the policy class against the *globals of its own module*, not against
    rsl_rl.modules, so adding to rsl_rl.modules alone is not enough (though we do that too, for
    discoverability). Importing this module performs the registration; the RMA agent-cfg imports it.
    """
    import rsl_rl.modules as _mods
    import rsl_rl.runners.on_policy_runner as _opr

    if not hasattr(_mods, "ActorCriticRMA"):
        _mods.ActorCriticRMA = ActorCriticRMA
    if not hasattr(_opr, "ActorCriticRMA"):
        _opr.ActorCriticRMA = ActorCriticRMA


_register()
