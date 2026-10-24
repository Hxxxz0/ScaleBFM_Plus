"""GPU-only selection of global or pre-failure reference starts.

The simulator captures ResetContext before resetting managers. This module owns
only the local retry policy and accumulated statistics, not physical state resets.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import NamedTuple


def normalized_returns_to_go(
    rewards: torch.Tensor, lengths: torch.Tensor, horizons: torch.Tensor, gamma: float, max_step_reward: float
) -> torch.Tensor:
    """Discounted return-to-go of every logged step relative to the most attainable before the episode ends.

    rewards, horizons: (n, T) per-step reward and number of transitions the episode could still run from that step
    without failing (to the end of the motion or the time limit); lengths: (n,) logged steps per episode.
    Episodes that end at their motion end or time limit with perfect tracking score 1 at every step, failures less.
    Entries beyond ``lengths`` are 0.
    """
    T = rewards.shape[1]
    steps = torch.arange(T, device=rewards.device)
    valid = steps[None, :] < lengths[:, None]
    gamma_pow = torch.full((T,), gamma, dtype=torch.float64, device=rewards.device) ** steps
    discounted = rewards.double() * valid * gamma_pow
    rtg = torch.flip(torch.cumsum(torch.flip(discounted, [1]), 1), [1]) / gamma_pow
    attainable = max_step_reward * (1.0 - gamma ** horizons.clamp_min(1).double()) / (1.0 - gamma)
    return (rtg / attainable * valid).float()


@dataclass
class FailureResetCfg:
    enabled: bool = False
    local_probability: float = 0.5
    lookback_frames: int = 8
    jitter_frames: int = 4
    min_episode_steps: int = 8
    max_consecutive_local_resets: int = 3
    activate_after_world_model_warmup: bool = True

    def validate(self) -> None:
        if not 0.0 <= self.local_probability <= 1.0:
            raise ValueError("failure_reset.local_probability must be in [0, 1]")
        for name in ("lookback_frames", "min_episode_steps", "max_consecutive_local_resets"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"failure_reset.{name} must be a positive integer")
        if not isinstance(self.jitter_frames, int) or not 0 <= self.jitter_frames < self.lookback_frames:
            raise ValueError("failure_reset.jitter_frames must be an integer in [0, lookback_frames)")


class ResetContext(NamedTuple):
    motion_ids: torch.Tensor
    frames: torch.Tensor
    episode_steps: torch.Tensor
    failed: torch.Tensor
    truncated: torch.Tensor


class HybridResetSampler:
    """Select starts in one vectorized operation and keep rank-local retry state."""

    STAT_NAMES = (
        "total",
        "failed",
        "eligible",
        "local",
        "global",
        "timeout",
        "early_failure",
        "retry_limit",
        "local_episode_steps",
        "global_episode_steps",
    )

    def __init__(self, cfg: FailureResetCfg, num_envs: int, library) -> None:
        cfg.validate()
        self.cfg = cfg
        self.library = library
        self.consecutive_local = torch.zeros(num_envs, dtype=torch.long, device=library.device)
        self.group_names = library.group_names
        self.statistics = torch.zeros(len(self.STAT_NAMES) + len(self.group_names), device=library.device)

    def clear(self, env_ids=None) -> None:
        if env_ids is None:
            self.consecutive_local.zero_()
        else:
            self.consecutive_local[env_ids] = 0

    def select(
        self, global_frames: torch.Tensor, env_ids: torch.Tensor, context: ResetContext | None, active: bool
    ) -> torch.Tensor:
        cfg, lib = self.cfg, self.library
        local = torch.zeros_like(global_frames, dtype=torch.bool)
        failed, truncated, eligible, early, limited = (torch.zeros_like(local) for _ in range(5))
        steps = torch.zeros_like(global_frames)
        result = global_frames
        if context is not None:
            steps = context.episode_steps
            truncated = context.truncated
            failed = context.failed & ~truncated
            early = failed & (steps < cfg.min_episode_steps)
            limited = failed & (self.consecutive_local[env_ids] >= cfg.max_consecutive_local_resets)
            eligible = failed & ~early & ~limited & (lib.lengths[context.motion_ids] >= 3) & active & cfg.enabled
            if cfg.enabled and active and cfg.local_probability > 0.0:
                local = eligible & (torch.rand(global_frames.shape, device=lib.device) < cfg.local_probability)
                jitter = torch.randint(
                    -cfg.jitter_frames, cfg.jitter_frames + 1, global_frames.shape, device=lib.device
                )
                frame = context.frames - cfg.lookback_frames + jitter
                last = (lib.lengths[context.motion_ids] - 3).clamp_min(0)
                frame = torch.minimum(frame.clamp_min(0), last)
                local_frames = lib.starts[context.motion_ids] + frame
                result = torch.where(local, local_frames, global_frames)
        self.consecutive_local[env_ids] = torch.where(local, self.consecutive_local[env_ids] + 1, 0)
        self.statistics[: len(self.STAT_NAMES)] += torch.stack(
            (
                torch.ones_like(local).sum(),
                failed.sum(),
                eligible.sum(),
                local.sum(),
                (~local).sum(),
                truncated.sum(),
                early.sum(),
                limited.sum(),
                (steps * local).sum(),
                (steps * ~local).sum(),
            )
        )
        groups = lib.motion_group[lib.frame_motion[result]]
        # bincount inspects the maximum CUDA index on the host even with minlength.
        # Fixed-size scatter avoids that synchronization in every reset.
        self.statistics[len(self.STAT_NAMES) :].scatter_add_(
            0, groups, torch.ones_like(groups, dtype=self.statistics.dtype)
        )
        return result

    def drain_statistics(self) -> torch.Tensor:
        values = self.statistics.clone()
        self.statistics.zero_()
        return values
