"""Original ScaleTrack PPO hyperparameters, under the SuperMimic comparison budget."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import G1BFMTransformerPPORunnerCfg


@configclass
class G1BFMTransformerPPOBenchmarkRunnerCfg(G1BFMTransformerPPORunnerCfg):
    class_name = "PPOBenchmarkRunner"
    experiment_name = "g1_bfm_ppo_benchmark_exp"
    max_train_hours: float = 2.6
    eval_interval_minutes: float = 45.0
    mixed_precision: bool = False
    eval_metric_keys = ["error_body_pos_g", "error_body_pos", "error_body_rot", "error_joint_pos"]
