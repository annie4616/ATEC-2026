# Reference: https://github.com/fan-ziqi/robot_lab

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.velocity.mdp as mdp

from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    This command generator automatically detects "pits" terrain and applies restrictions:
    - For pit terrains: only allow forward movement (no lateral or rotational movement)
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        # Track which robots were on pit terrain in the previous step
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands with threshold."""
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _update_command(self):
        """Update commands and apply terrain-aware restrictions in real-time.

        This function:
        1. Calls parent's update to handle heading and standing envs
        2. Checks which robots are currently on pit terrain
        3. For robots leaving pits: resamples their commands
        4. For robots on pits: restricts to forward-only movement and sets heading to 0
        """
        # First, call parent's update command
        super()._update_command()

        # Check which robots are currently on pit terrain (real-time check every step)
        on_pits = is_robot_on_terrain(self._env, "pits")

        # Find robots that just left pit terrain (need to resample)
        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            left_pit_env_ids = torch.where(left_pit_mask)[0]
            # Resample commands for robots that left pits
            self._resample_command(left_pit_env_ids)

        # For robots currently on pits: restrict to forward-only movement with min/max speed
        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            # Force forward-only movement with min and max speed limits
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0  # no lateral movement
            self.vel_command_b[pit_env_ids, 2] = 0.0  # no yaw rotation
            # Set heading to 0 for pit robots
            if self.cfg.heading_command:
                self.heading_target[pit_env_ids] = 0.0

        # Update tracking state
        self.was_on_pit = on_pits


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


class GridVelocityCommand(UniformThresholdVelocityCommand):
    """SE(2) velocity command with a grid-adaptive velocity curriculum.

    Port of the grid curriculum from "Rapid Locomotion via Reinforcement Learning"
    (Margolis et al., MIT / Improbable-AI, arXiv:2205.02824). The problem it fixes:
    a single global (vx) range that only widens when the *mean* tracking reward over
    ALL envs clears a threshold stalls, because envs handed the hardest (fastest)
    commands drag the mean down and freeze the whole curriculum — exactly the vx=2.63
    ceiling we hit.

    Instead we discretize (vx, wz) into a grid of bins, each with a sampling weight
    in [0, 1]. Commands are sampled ONLY from active (weight>0) bins, per-env. After
    each episode, any bin whose env tracked well (lin & ang reward above thresholds)
    has ITS weight and its immediate NEIGHBORS' weights bumped by +0.2 (clipped to 1).
    So mastery of a speed spreads outward to the next-faster bins on its own, and a
    slow bin never blocks a fast one. The robot walks its own dynamic limit out.

    Only vx/wz are gridded; vy and heading keep the base uniform sampling.
    """

    cfg: GridVelocityCommandCfg  # type: ignore

    def __init__(self, cfg: "GridVelocityCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Bin centers along vx and wz.
        vx_lo, vx_hi = cfg.grid_vx_range
        wz_lo, wz_hi = cfg.grid_wz_range
        self.vx_bins = torch.linspace(vx_lo, vx_hi, cfg.grid_vx_bins, device=self.device)
        self.wz_bins = torch.linspace(wz_lo, wz_hi, cfg.grid_wz_bins, device=self.device)
        self.n_vx = cfg.grid_vx_bins
        self.n_wz = cfg.grid_wz_bins
        # half bin widths for uniform jitter inside a bin
        self.vx_half = (vx_hi - vx_lo) / max(cfg.grid_vx_bins - 1, 1) / 2.0
        self.wz_half = (wz_hi - wz_lo) / max(cfg.grid_wz_bins - 1, 1) / 2.0
        # Weights over the flattened (n_vx * n_wz) grid. Seed strategy: activate every
        # vx bin at or below `grid_seed_vx` (all wz), so a warm-start policy that
        # already walks up to that speed starts with its WHOLE known range open and
        # the grid only has to grow ABOVE it. Seeding only the slowest column instead
        # made the fast warm-start policy struggle to also track very slow commands,
        # inflating mean xy_err. If grid_seed_vx <= 0, fall back to the slowest
        # `grid_seed_frac` fraction of bins (original easy-start behavior).
        self.weights = torch.zeros(self.n_vx * self.n_wz, device=self.device)
        if cfg.grid_seed_vx and cfg.grid_seed_vx > 0:
            seed_vx = int(
                torch.searchsorted(
                    self.vx_bins, torch.tensor(cfg.grid_seed_vx, device=self.device)
                ).item()
            ) + 1
            seed_vx = max(1, min(seed_vx, self.n_vx))
        else:
            seed_vx = max(1, int(round(cfg.grid_seed_frac * self.n_vx)))
        for i in range(seed_vx):
            for j in range(self.n_wz):
                self.weights[i * self.n_wz + j] = 1.0
        # Per-env assigned bin index (flattened), for the curriculum update.
        self.env_bin = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _sample_bins(self, n: int) -> torch.Tensor:
        """Weighted-sample n flattened bin indices from active bins."""
        w = self.weights
        total = w.sum()
        if total <= 0:
            # degenerate: fall back to uniform over the seed column
            probs = torch.zeros_like(w)
            probs[: self.n_wz] = 1.0
            probs = probs / probs.sum()
        else:
            probs = w / total
        return torch.multinomial(probs, n, replacement=True)

    def _resample_command(self, env_ids: Sequence[int]):
        # Base handles vy, heading, standing flags and the small-command threshold.
        super()._resample_command(env_ids)
        n = len(env_ids)
        if n == 0:
            return
        # Draw grid bins for these envs and set vx / wz from bin center + jitter.
        bins = self._sample_bins(n)
        self.env_bin[env_ids] = bins
        vx_idx = bins // self.n_wz
        wz_idx = bins % self.n_wz
        r = torch.empty(n, device=self.device)
        vx = self.vx_bins[vx_idx] + r.uniform_(-1.0, 1.0) * self.vx_half
        r2 = torch.empty(n, device=self.device)
        wz = self.wz_bins[wz_idx] + r2.uniform_(-1.0, 1.0) * self.wz_half
        self.vel_command_b[env_ids, 0] = vx
        self.vel_command_b[env_ids, 2] = wz
        # Re-apply the base small-command zeroing on the (possibly tiny) new vx.
        self.vel_command_b[env_ids, :2] *= (
            torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2
        ).unsqueeze(1)

    def _neighbor_bins(self, flat_idx: torch.Tensor) -> torch.Tensor:
        """Flattened indices of the ONE-STEP-FASTER vx neighbor of each bin.

        We only spread ALONG vx (to the next-faster bin at the same wz), and only
        upward. Spreading sideways in wz too (the naive 4-neighborhood) made the grid
        open explosively — a single easy low-speed success cascaded the whole grid to
        max in ~20 iters, which just removes the wall instead of climbing it. Growing
        one vx step at a time keeps expansion paced to what the policy can actually
        track."""
        vx_i = flat_idx // self.n_wz
        wz_i = flat_idx % self.n_wz
        nvx = torch.clamp(vx_i + 1, 0, self.n_vx - 1)
        return torch.unique(nvx * self.n_wz + wz_i)

    def update_curriculum(
        self, env_ids: torch.Tensor, lin_rewards: torch.Tensor, ang_rewards: torch.Tensor
    ) -> None:
        """MIT grid update: bins whose env tracked WELL grow toward the next vx bin.

        Called every step from the curriculum term with the per-env mean lin/ang
        tracking reward for the episode that just finished on env_ids. A bin only
        counts as mastered when its tracking fraction clears a HIGH threshold (so a
        warm-start policy coasting through easy low speeds doesn't instantly unlock
        everything), and mastery adds a small increment (+0.1) that only opens the
        next-faster vx bin — so the reachable speed grows gradually, gated on the
        robot actually keeping up.
        """
        is_success = (lin_rewards > self.cfg.grid_lin_threshold) & (
            ang_rewards > self.cfg.grid_ang_threshold
        )
        if not torch.any(is_success):
            return
        succ_bins = torch.unique(self.env_bin[env_ids[is_success]])
        # Reinforce mastered bins (keep them firmly active).
        self.weights[succ_bins] = torch.clamp(self.weights[succ_bins] + 0.1, 0.0, 1.0)
        # Crack open only the next-faster vx bin, and only a little.
        nxt = self._neighbor_bins(succ_bins)
        self.weights[nxt] = torch.clamp(self.weights[nxt] + 0.1, 0.0, 1.0)

    @property
    def max_active_vx(self) -> float:
        """Highest vx-bin center that is currently active (weight>0) — for logging."""
        active = self.weights.view(self.n_vx, self.n_wz).sum(dim=1) > 0
        if not torch.any(active):
            return float(self.vx_bins[0])
        return float(self.vx_bins[torch.nonzero(active).max()])


@configclass
class GridVelocityCommandCfg(UniformThresholdVelocityCommandCfg):
    """Configuration for the grid-adaptive velocity command generator."""

    class_type: type = GridVelocityCommand

    grid_vx_range: tuple[float, float] = (0.0, 4.0)
    grid_vx_bins: int = 17          # 0.25 m/s resolution over [0, 4]
    grid_wz_range: tuple[float, float] = (-1.5, 1.5)
    grid_wz_bins: int = 7           # 0.5 rad/s resolution
    grid_seed_frac: float = 0.15    # start with the slowest ~15% of vx bins active
    grid_seed_vx: float = 0.0       # if >0: activate ALL vx bins up to this speed at
    #                                 start (overrides grid_seed_frac). Set to the
    #                                 warm-start policy's top speed so its whole known
    #                                 range is open and the grid only grows above it.
    grid_lin_threshold: float = 0.6  # per-step lin-tracking reward frac to count success
    grid_ang_threshold: float = 0.5  # per-step ang-tracking reward frac to count success


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """
