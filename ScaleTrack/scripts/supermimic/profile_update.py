"""Profile one SuperMimic update (world model fit, imagination + BPTT, critic) on a small live environment.

Example (an RTX 5090 budget on a larger GPU, with the motion-library shard of an 8-GPU run as ballast):
    python scripts/supermimic/profile_update.py --motion_lib /dev/shm/sm_lib/train_lafan --batch 2048 \
        --microbatch 1024 --memory_limit_gb 31 --memory_ballast_gb 7 --headless
"""

import argparse
import math
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="G1-BFM-Transformer-Tracking-SuperMimic")
parser.add_argument("--motion_lib", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--batch", type=int, default=2048, help="imagined rollouts per update")
parser.add_argument("--microbatch", type=int, default=0, help="actor micro-batch size (0 = whole batch)")
parser.add_argument("--checkpointing", action="store_true", help="recompute actor activations in backward")
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--memory_limit_gb", type=float, default=0.0, help="limit the CUDA allocator (smaller GPU)")
parser.add_argument("--memory_ballast_gb", type=float, default=0.0, help="reserve memory (larger motion library)")
parser.add_argument("--cuda_graph", action="store_true", help="replay the actor pass as one CUDA graph")
parser.add_argument("--trace", action="store_true", help="print a torch.profiler table of one policy update")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import scaletrack.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

from my_rsl_rl.runners import SuperMimicRunner

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
if args.memory_limit_gb:
    total = torch.cuda.get_device_properties(cfg.sim.device).total_memory
    torch.cuda.set_per_process_memory_fraction(args.memory_limit_gb * 2**30 / total, cfg.sim.device)
ballast = torch.empty(int(args.memory_ballast_gb * 2**30), dtype=torch.uint8, device=cfg.sim.device)
cfg.commands.motion.motion_lib_dir = args.motion_lib
agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
agent.supermimic.update(
    imagination_batch_size=args.batch,
    actor_micro_batch_size=args.microbatch,
    actor_checkpointing=args.checkpointing,
    horizon=args.horizon,
    cuda_graph=args.cuda_graph,
)
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
runner = SuperMimicRunner(env, agent.to_dict(), log_dir=None, device=env.unwrapped.device)
alg, task = runner.alg, runner.task

obs = env.get_observations()
state, info = task.read_state(), task.step_info()
ep_start = torch.ones(env.num_envs, dtype=torch.bool, device=env.unwrapped.device)
for _ in range(64):
    a = alg.act(obs)
    obs, _, dones, _ = env.step(a)
    alg.buffer.add(state, a, task.executed_actions(), info, ep_start)
    ids = dones.nonzero().squeeze(-1)
    alg.buffer.add_terminal(ids, task.terminal_state(ids))
    state, info, ep_start = task.read_state(), task.step_info(), dones.bool()


def timed(name, fn, n=10):
    for _ in range(5):  # compilation and CUDA-graph capture
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(n):
        metrics = fn()
        assert all(math.isfinite(float(v)) for v in metrics.values()), metrics
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"[profile] {name:<16s} {(time.time() - t) / n:.3f} s   peak mem {peak:.1f} GB", flush=True)


alg.buffer.prepare(alg.wm_window, alg.history_length)
timed("world model", alg.update_world_model)
timed("policy+critic", alg.update_policy)
if args.trace:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        alg.update_policy()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))
# SimulationApp.close() can hang after headless runs; the numbers are already printed.
os._exit(0)
