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


class NoisyMimicJointPositionAction(NoisyJointPositionAction):
    """NoisyJointPositionAction over the driven joints; every mimic joint follows ``multiplier * driver target``.

    The action (and the executed action stored for the world model) covers the driven joints only.
    """

    cfg: NoisyMimicJointPositionActionCfg

    def __init__(self, cfg: NoisyMimicJointPositionActionCfg, env):
        super().__init__(cfg, env)
        names = list(cfg.mimic.keys())
        mimic_ids = self._asset.find_joints(names, preserve_order=True)[0]
        self._all_joint_ids = list(self._joint_ids) + list(mimic_ids)
        driver_names = [cfg.mimic[n][0] for n in names]
        self._mimic_driver = torch.tensor([self._joint_names.index(n) for n in driver_names], device=self.device)
        self._mimic_scale = torch.tensor([cfg.mimic[n][1] for n in names], device=self.device)

    def apply_actions(self):
        targets = self._processed_actions[:, self._mimic_driver] * self._mimic_scale
        self._asset.set_joint_position_target(
            torch.cat([self._processed_actions, targets], dim=1), joint_ids=self._all_joint_ids
        )


@configclass
class NoisyMimicJointPositionActionCfg(NoisyJointPositionActionCfg):
    class_type: type = NoisyMimicJointPositionAction

    # mimic joint name -> (driver joint name, multiplier)
    mimic: dict = {}
