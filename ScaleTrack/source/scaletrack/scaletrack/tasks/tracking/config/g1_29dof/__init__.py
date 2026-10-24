import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="G1-BFM-Transformer-Tracking",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1BFMTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1BFMTransformerPPORunnerCfg",
    },
)

gym.register(
    id="G1-BFM-Mlp-Tracking",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1BFMTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1BFMMlpPPORunnerCfg",
    },
)
gym.register(
    id="G1-BFM-Transformer-Tracking-SuperMimic",
    entry_point="scaletrack.tasks.tracking.supermimic.env:SuperMimicTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "scaletrack.tasks.tracking.supermimic.env_cfg:G1BFMTrackingSuperMimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_supermimic_cfg:G1BFMTransformerSuperMimicRunnerCfg",
    },
)

gym.register(
    id="G1-BFM-Transformer-Tracking-PPO-Benchmark",
    entry_point="scaletrack.tasks.tracking.supermimic.env:SuperMimicTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "scaletrack.tasks.tracking.supermimic.env_cfg:G1BFMTrackingPPOBenchmarkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_benchmark_cfg:G1BFMTransformerPPOBenchmarkRunnerCfg",
    },
)

gym.register(
    id="G1-BFM-Transformer-Tracking-SuperMimic-HybridReset",
    entry_point="scaletrack.tasks.tracking.supermimic.env:SuperMimicTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "scaletrack.tasks.tracking.supermimic.env_cfg:G1BFMTrackingSuperMimicHybridResetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_supermimic_cfg:G1BFMTransformerSuperMimicHybridRunnerCfg",
    },
)
