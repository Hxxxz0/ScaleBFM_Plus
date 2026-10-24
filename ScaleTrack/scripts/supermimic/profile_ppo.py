"""Validate a full-size PPO minibatch under a target CUDA memory limit."""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--motion_lib", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--memory_ballast_gb", type=float, default=12.0)
parser.add_argument("--memory_limit_gb", type=float, default=27.0)
parser.add_argument("--microbatch", type=int, default=2048)
parser.add_argument("--mixed_precision", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import scaletrack.tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from my_rsl_rl.runners.ppo_benchmark_runner import PPOBenchmarkRunner

task = "G1-BFM-Transformer-Tracking-PPO-Benchmark"
cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
cfg.commands.motion.motion_lib_dir = args.motion_lib
cfg.commands.motion.test_motion_lib_dir = args.motion_lib
total = torch.cuda.get_device_properties(cfg.sim.device).total_memory
torch.cuda.set_per_process_memory_fraction(args.memory_limit_gb * 2**30 / total, cfg.sim.device)
ballast = torch.empty(int(args.memory_ballast_gb * 2**30), dtype=torch.uint8, device=cfg.sim.device)
agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
agent.num_steps_per_env = 64
agent.algorithm.num_mini_batches = 4
agent.algorithm.num_learning_epochs = 1
agent.eval_max_steps = 20
agent.mixed_precision = args.mixed_precision
agent.algorithm.forward_micro_batch_size = args.microbatch
env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg))
runner = PPOBenchmarkRunner(env, agent.to_dict(), device=env.unwrapped.device)
obs = env.get_observations()
for iteration in range(2):
    with torch.inference_mode():
        for _ in range(agent.num_steps_per_env):
            actions = runner.alg.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            runner.alg.process_env_step(obs, rewards, dones, extras)
        runner.alg.compute_returns(obs)
    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.mixed_precision):
        losses = runner.alg.update()
    torch.cuda.synchronize()
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values()), losses
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in runner.policy.parameters())
    print(f"[profile] PPO BF16={args.mixed_precision} iteration {iteration+1}, minibatch=8192 microbatch={args.microbatch}, "
          f"update={time.perf_counter()-t0:.3f}s peak={torch.cuda.max_memory_allocated()/2**30:.2f}GB "
          f"including ballast={args.memory_ballast_gb}GB losses={losses}", flush=True)
runner._checkpoint_and_eval(final=True)
print("[profile] PPO evaluation passed", flush=True)
env.close()
app.close()
