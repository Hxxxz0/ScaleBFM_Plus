"""Explicit update/time budgets and the learning-rate schedule over them."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingBudget:
    mode: str
    max_updates: int | None
    hours: float | None

    def __post_init__(self) -> None:
        if self.mode not in ("hours", "updates"):
            raise ValueError("supermimic.budget_mode must be 'hours' or 'updates'")
        if self.mode == "updates" and (
            isinstance(self.max_updates, bool) or not isinstance(self.max_updates, int) or self.max_updates < 1
        ):
            raise ValueError("update budget requires a positive max_iterations")
        if self.mode == "hours" and (self.hours is None or not math.isfinite(self.hours) or self.hours <= 0):
            raise ValueError("hour budget requires positive finite max_train_hours")

    @property
    def seconds(self) -> float:
        return self.hours * 3600.0 if self.mode == "hours" else math.inf

    def reached(self, completed_updates: int, elapsed_seconds: float) -> bool:
        return (self.max_updates is not None and completed_updates >= self.max_updates) or (
            self.mode == "hours" and elapsed_seconds > self.seconds
        )

    def policy_progress(
        self, completed_updates: int, first_policy_update: int, elapsed_seconds: float, policy_start_seconds: float | None
    ) -> float:
        """Completed fraction of the policy phase, which starts after the world-model-only updates.

        Measured in updates for an update budget and in wall-clock time for an hour budget, so that the schedule
        reaches its end with the budget whatever the (unknown in advance) cost of a policy update.
        """
        if self.mode == "updates":
            done, total = completed_updates - first_policy_update, self.max_updates - first_policy_update
        elif policy_start_seconds is None:
            return 0.0
        else:
            done, total = elapsed_seconds - policy_start_seconds, self.seconds - policy_start_seconds
        return min(max(done / max(total, 1e-9), 0.0), 1.0)


def cosine_lr_ratio(progress: float, final_ratio: float, decay_fraction: float = 1.0) -> float:
    """Cosine decay from 1 to ``final_ratio`` over the first ``decay_fraction`` of training, then constant."""
    x = min(max(progress, 0.0) / decay_fraction, 1.0)
    return final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * x))
