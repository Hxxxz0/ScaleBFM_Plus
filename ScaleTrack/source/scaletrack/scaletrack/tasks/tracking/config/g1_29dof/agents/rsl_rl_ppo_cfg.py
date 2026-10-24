from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class CompleteRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg): # weishuai: This is to use the DoubleHead term for policy
    state_dependent_std: bool = False
    std_clamp_range: list[float] | tuple[float] = [0.001, 1.0]

@configclass
class CompleteRslRlPpoActorCriticTransformerCfg(CompleteRslRlPpoActorCriticCfg):
    use_transformer_critic: bool = True
    embedding_dim: int = 256
    num_heads: int = 4
    ff_dim: int = 256
    num_layers: int = 4
    task_embedder_hidden_dims: list[int] = []

@configclass
class CompleteRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    forward_micro_batch_size: int = 0
    actor_learning_rate: float = 2.0e-5
    critic_learning_rate: float = 1.0e-3

@configclass
class G1BFMTransformerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 50000
    save_interval = 1000
    experiment_name = "g1_bfm_tracking_exp"
    empirical_normalization = False
    eval_during_training=True
    eval_interval=200
    eval_metric_keys=[
        "error_anchor_height","error_anchor_rot", "error_anchor_pos", 
        "error_anchor_lin_vel", "error_anchor_ang_vel", "error_body_pos_g", 
        "error_body_pos", "error_body_pos_relative", 'error_body_rot', "error_body_rot_relative", 'error_joint_pos', "error_joint_vel"
    ]
    eval_max_steps=1000
    success_metric_dict={
        "error_body_pos_g": 0.5
    }
    policy = CompleteRslRlPpoActorCriticTransformerCfg(
        class_name="ActorCriticHumanoidTransformer",
        init_noise_std=0.8,
        state_dependent_std=False,
        critic_hidden_dims=[2048, 2048, 1024, 1024, 512, 512], # weishuai: This defaults to non-activated as use_transformer_critic is activated!
        activation="elu",
        
    )
    algorithm = CompleteRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=2,
        num_mini_batches=32,
        actor_learning_rate=2.0e-5,
        critic_learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class G1BFMMlpPPORunnerCfg(G1BFMTransformerPPORunnerCfg):
    policy = CompleteRslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[2048, 2048, 1024, 1024, 512, 512], 
        critic_hidden_dims=[2048, 2048, 1024, 1024, 512, 512],
        activation="elu",
        state_dependent_std=False, # weishuai: DoubleHead
    )
