"""Simulator integration gate for failure context, replay terminals and update budgets.

Uses a small packed fixture and reduced optimizer batches only in this check.
Production training parameters come from the experiment YAML.
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion_file", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

from my_rsl_rl.runners.supermimic_runner import SuperMimicRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import scaletrack.tasks  # noqa: F401
from scaletrack.tasks.tracking.supermimic.reset_sampling import FailureResetCfg, HybridResetSampler, ResetContext
from scaletrack.tasks.tracking.supermimic.state import read_state

TASK = "G1-BFM-Transformer-Tracking-SuperMimic-HybridReset"
cfg = parse_env_cfg(TASK, device=args.device, num_envs=32)
cfg.commands.motion.motion_lib_dir = args.motion_file
cfg.commands.motion.test_motion_lib_dir = args.motion_file
cfg.commands.motion.failure_reset.local_probability = 1.0
env = gym.make(TASK, cfg=cfg).unwrapped
env.reset()
command = env.command_manager.get_term("motion")
ids = torch.arange(env.num_envs, device=env.device)


def inject_failure():
    command.motion_ids.zero_()
    command.time_steps.fill_(30)
    command._clear_caches()
    env.episode_length_buf.fill_(20)
    env.termination_manager.get_term("body_pos").fill_(True)
    env.termination_manager.get_term("time_out").zero_()
    env.termination_manager.get_term("motion_time_out").zero_()


command.set_failure_reset_active(False)
inject_failure()
env._reset_idx(ids)
assert (command.reset_sampler.consecutive_local == 0).all(), "warmup must use global resets"
command.set_failure_reset_active(True)
inject_failure()
before = read_state(env.scene["robot"], env.scene.env_origins).clone()
env._reset_idx(ids)
torch.testing.assert_close(env.terminal_state, before)
assert (command.motion_ids == 0).all()
assert ((command.time_steps >= 18) & (command.time_steps <= 26)).all()
assert (command.reset_sampler.consecutive_local == 1).all()
assert env._motion_reset_context is None
inject_failure()
env.termination_manager.get_term("time_out").fill_(True)
env._reset_idx(ids)
assert (command.reset_sampler.consecutive_local == 0).all(), "timeout has priority"
inject_failure()
env._reset_idx(ids)
env.reset()
assert (command.reset_sampler.consecutive_local == 0).all(), "manual reset must be global"
command.reset_sampler.consecutive_local.fill_(2)
command.switch_motion_set(True)
assert (command.reset_sampler.consecutive_local == 0).all(), "evaluation must clear retries"
command.is_evaluating = True
env.reset()
assert (command.time_steps == 0).all(), "evaluation must start at frame zero"
command.switch_motion_set(False)
command.is_evaluating = False
env.reset()
print("[check] simulator failure context, real terminal state, timeout/manual/eval transitions passed", flush=True)

# Same reset workload and input shapes; CUDA events include device work without per-reset host synchronization.
n = 4096
fixture_ids = torch.arange(n, device=env.device)
global_frames = torch.zeros(n, dtype=torch.long, device=env.device)
context = ResetContext(
    torch.zeros_like(global_frames),
    torch.full_like(global_frames, 30),
    torch.full_like(global_frames, 20),
    torch.ones(n, device=env.device, dtype=torch.bool),
    torch.zeros(n, device=env.device, dtype=torch.bool),
)
for probability in (0.0, 0.5):
    sampler = HybridResetSampler(FailureResetCfg(enabled=True, local_probability=probability), n, command.train_lib)
    for _ in range(10):
        sampler.select(global_frames, fixture_ids, context, True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        sampler.select(global_frames, fixture_ids, context, True)
    torch.cuda.synchronize()
    print(f"[bench] reset_batch={n} probability={probability} ms={(time.perf_counter() - t0) * 5:.3f}", flush=True)

agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
assert agent_cfg.max_iterations == 10000
assert agent_cfg.supermimic["budget_mode"] == "updates"
assert agent_cfg.supermimic["max_train_hours"] is None
assert agent_cfg.policy.embedding_dim == 256 and agent_cfg.policy.num_layers == 4
agent = agent_cfg.to_dict()
agent["supermimic"].update(
    wm_hidden_dims=[64, 64],
    wm_batch_size=32,
    wm_batches_per_update=1,
    imagination_batch_size=32,
    horizon=4,
    replay_capacity=64,
    warmup_env_steps=8,
    wm_only_updates=1,
    log_interval=1,
    eval_interval_updates=0,
    compile=False,
    actor_checkpointing=False,
)
agent["eval_max_steps"] = 4
log = tempfile.mkdtemp(prefix="sm_hybrid_check_")
runner = SuperMimicRunner(RslRlVecEnvWrapper(env), agent, log_dir=log, device=args.device)
runner.learn(num_learning_iterations=5)
assert runner.num_updates == 5
assert Path(log, "model_5.pt").is_file()
assert command._failure_reset_active
# Last update (index 4) is 3/4 of the 4-update policy phase: cosine decay over its first 78%, then 0.03x.
expected = runner._lr_ratio(3 / 4)
assert abs(runner.alg.actor_optimizer.param_groups[0]["lr"] - runner.alg.base_lrs["actor"] * expected) < 1e-12
assert abs(runner._lr_ratio(1.0) - 0.03) < 1e-12
print(
    f"[check] M config serialization, warmup gate, exact update stop and final checkpoint passed; log={log}", flush=True
)
# Headless SimulationApp shutdown can hang; all assertions and outputs above have completed.
os._exit(0)
