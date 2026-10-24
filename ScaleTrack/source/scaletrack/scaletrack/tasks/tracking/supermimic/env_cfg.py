"""Environment configuration of the G1 tracking task for SuperMimic-style training.

Identical MDP to ``G1BFMTrackingEnvCfg`` (observations seen by the policy, rewards, terminations, start-up
randomisation, reset noise) with the following changes:

* the policy action is executed with additive Gaussian noise (exploration and world-model coverage),
* no interval pushes: a deterministic world model cannot predict them (paper §3 / Appendix A),
* the privileged critic observation groups are not computed (the critic works on imagined states),
* motions come from a packed GPU motion library, resets use return-based bin sampling.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from scaletrack.tasks.tracking.config.g1_29dof.flat_env_cfg import G1BFMTrackingEnvCfg

from .actions import NoisyJointPositionActionCfg
from .command import SuperMimicMotionCommandCfg
from .reset_sampling import FailureResetCfg


@configclass
class G1BFMTrackingSuperMimicEnvCfg(G1BFMTrackingEnvCfg):
    action_noise_std: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096

        joint_pos = self.actions.joint_pos
        self.actions.joint_pos = NoisyJointPositionActionCfg(
            asset_name=joint_pos.asset_name,
            joint_names=joint_pos.joint_names,
            use_default_offset=joint_pos.use_default_offset,
            scale=joint_pos.scale,
            noise_std=self.action_noise_std,
        )

        self.events.push_robot = None
        self.observations.critic = None
        self.observations.critic_task = None

        motion = self.commands.motion
        self.commands.motion = SuperMimicMotionCommandCfg(
            asset_name=motion.asset_name,
            resampling_time_range=motion.resampling_time_range,
            debug_vis=False,
            pose_range=motion.pose_range,
            velocity_range=motion.velocity_range,
            joint_position_range=motion.joint_position_range,
            anchor_body_name=motion.anchor_body_name,
            body_names=motion.body_names,
            mode_candidates=motion.mode_candidates,
            ref_frame_buffer_size=motion.ref_frame_buffer_size,
            rand_timestep_range=motion.rand_timestep_range,
            enable_reset_disturbance=motion.enable_reset_disturbance,
        )


@configclass
class G1BFMTrackingPPOBenchmarkEnvCfg(G1BFMTrackingSuperMimicEnvCfg):
    """Same packed motions, actor observations, rewards and reset sampler; PPO gets its privileged critic inputs.

    PPO samples exploration actions itself, so it does not add the external action noise used by SuperMimic.
    """

    action_noise_std: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        original = G1BFMTrackingEnvCfg()
        self.observations.critic = original.observations.critic
        self.observations.critic_task = original.observations.critic_task


@configclass
class G1BFMTrackingSuperMimicHybridResetEnvCfg(G1BFMTrackingSuperMimicEnvCfg):
    """SuperMimic with bounded retries of pre-failure reference segments."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.failure_reset = FailureResetCfg(enabled=True)
