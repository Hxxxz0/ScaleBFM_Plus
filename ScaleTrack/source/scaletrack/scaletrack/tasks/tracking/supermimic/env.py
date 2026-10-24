"""Tracking environment that keeps the pre-reset state of every episode for the replay buffer."""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv

from .reset_sampling import ResetContext
from .state import STATE_DIM, read_state


class SuperMimicTrackingEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv whose resets first record the final state ``s_{T+1}`` of the ending episodes.

    The world model is fit on transitions, and the last transition of an episode ends in the state that triggered
    the reset; IsaacLab overwrites it inside ``step``. ``terminal_state[env_ids]`` holds it after the step.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self.terminal_state = None
        self._motion_reset_context = None
        self._explicit_reset = False
        super().__init__(cfg, render_mode, **kwargs)
        self.terminal_state = torch.zeros(self.num_envs, STATE_DIM, device=self.device)

    def make_imagination_task(self, **kwargs):
        """Imagination interface used by my_rsl_rl's SuperMimicRunner."""
        from .task import TrackingImaginationTask

        return TrackingImaginationTask(self, **kwargs)

    def _reset_idx(self, env_ids: Sequence[int]):
        if self.terminal_state is not None:
            self.terminal_state[env_ids] = read_state(self.scene["robot"], self.scene.env_origins, env_ids)
            command = self.command_manager.get_term("motion")
            if command.reset_sampler is not None and not self._explicit_reset and not command.is_evaluating:
                self._motion_reset_context = ResetContext(
                    command.motion_ids[env_ids].clone(),
                    command.time_steps[env_ids].clone(),
                    self.episode_length_buf[env_ids].clone(),
                    self.termination_manager.get_term("body_pos")[env_ids].clone(),
                    (
                        self.termination_manager.get_term("time_out")[env_ids]
                        | self.termination_manager.get_term("motion_time_out")[env_ids]
                    ).clone(),
                )
        try:
            super()._reset_idx(env_ids)
        finally:
            self._motion_reset_context = None

    def reset(self, seed=None, env_ids=None, options=None):
        """Explicit resets must not reuse terminal flags from the previous step."""
        self._explicit_reset = True
        try:
            return super().reset(seed=seed, env_ids=env_ids, options=options)
        finally:
            self._explicit_reset = False
