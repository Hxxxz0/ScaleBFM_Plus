"""SuperMimic-style training of the BFM humanoid transformer (world model + BPTT, see my_rsl_rl.algorithms.supermimic).

The policy configuration is the one of PPO so that checkpoints are interchangeable with PPO runs.
"""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import G1BFMTransformerPPORunnerCfg


@configclass
class G1BFMTransformerSuperMimicRunnerCfg(G1BFMTransformerPPORunnerCfg):
    class_name = "SuperMimicRunner"
    experiment_name = "g1_bfm_supermimic_exp"
    save_interval = 1000
    eval_metric_keys = ["error_body_pos_g", "error_body_pos", "error_body_rot", "error_joint_pos"]
    supermimic = {
        # world model (Table A.4 / A.5)
        "wm_hidden_dims": [2048, 2048, 2048],
        "wm_learning_rate": 2.0e-4,
        "wm_window": 8,
        "wm_batches_per_update": 10,
        "wm_batch_size": 2048,
        "wm_max_grad_norm": 1.0,
        # imagination and actor
        "horizon": 32,
        "imagination_batch_size": 2048,
        "actor_learning_rate": 1.0e-4,
        "actor_betas": [0.9, 0.95],
        "actor_max_grad_norm": 1.0,
        "actor_max_grad_value": 0.1,
        # memory: process imagined rollouts in micro-batches (0 = whole batch) and/or recompute actor activations
        "actor_micro_batch_size": 0,
        "actor_checkpointing": False,
        # replay the imagination (forward and backward) as one CUDA graph: removes the kernel-launch overhead
        "cuda_graph": False,
        "gamma": 0.99,
        "lam": 0.95,
        # critic
        "critic_hidden_dims": [256, 128],
        "critic_learning_rate": 2.0e-4,
        "critic_max_grad_norm": 1.0,
        "critic_epochs": 5,
        "critic_mini_batches": 4,
        # data and schedule
        "replay_capacity": 1000,
        "history_length": 3,
        "update_to_data_ratio": 0.4,
        "warmup_env_steps": 64,
        "wm_only_updates": 300,
        # cosine decay over the first lr_decay_fraction of the policy phase, then constant (Table A.5: 250k / 320k)
        "lr_final_ratio": 0.03,
        "lr_decay_fraction": 0.78,
        "lr_scale_with_world_size": True,
        "max_train_hours": 2.6,
        "budget_mode": "hours",
        "eval_interval_updates": 0,
        "eval_interval_minutes": 45.0,
        "log_interval": 50,
        # shapes of the PPO critic inputs (the transformer critic is kept untrained for checkpoint compatibility)
        "ppo_critic_obs_shapes": {"critic": [3, 266], "critic_task": [7, 337]},
    }


@configclass
class G1BFMTransformerSuperMimicHybridRunnerCfg(G1BFMTransformerSuperMimicRunnerCfg):
    """M policy with a fixed 10k-update budget and verified throughput options."""

    max_iterations = 10000
    resume = False

    def __post_init__(self):
        self.supermimic = dict(self.supermimic)
        self.supermimic.update(
            budget_mode="updates",
            max_train_hours=None,
            eval_interval_updates=1000,
            actor_checkpointing=True,
        )
