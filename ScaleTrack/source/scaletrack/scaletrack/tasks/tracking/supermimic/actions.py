"""Joint position action that executes the policy action with additive Gaussian exploration noise.

The clean action stays the action of record (``action_manager.action``), so the last-action observation and the
action-rate penalty see the deterministic policy output, while the simulator executes ``a + eps``. The executed
action is exposed for the replay buffer, since it is what the world model has to predict the effect of.
"""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass


class NoisyJointPositionAction(JointPositionAction):
    cfg: NoisyJointPositionActionCfg

    def __init__(self, cfg: NoisyJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self.noise_std = cfg.noise_std
        self.executed_raw = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        if self.noise_std > 0:
            self.executed_raw[:] = actions + self.noise_std * torch.randn_like(actions)
        else:
            self.executed_raw[:] = actions
        self._processed_actions = self.executed_raw * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1])


@configclass
class NoisyJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = NoisyJointPositionAction

    noise_std: float = 0.1
