"""PPO on the packed tracking task, with SuperMimic's wall-clock budget and test protocol."""

from __future__ import annotations

import json
import os
import time
from collections import deque

import torch
import torch.distributed as dist

from .on_policy_runner import OnPolicyRunner
from .supermimic_runner import SuperMimicRunner
from my_rsl_rl.utils import store_code_state


class PPOBenchmarkRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        self.policy = self.alg.policy
        self.max_train_hours = self.cfg["max_train_hours"]
        self.eval_interval_minutes = self.cfg["eval_interval_minutes"]
        if self.max_train_hours <= 0 or self.eval_interval_minutes <= 0:
            raise ValueError("training budget and evaluation interval must be positive")
        for module in self.policy.modules():
            if hasattr(module, "need_weights"):
                module.need_weights = False

    # Reuse the exact test rollout and per-dataset aggregation of SuperMimic, without constructing its algorithm.
    evaluate_test_set = SuperMimicRunner.evaluate_test_set
    _allreduce_sum = SuperMimicRunner._allreduce_sum
    _broadcast_scalar = SuperMimicRunner._broadcast_scalar

    @torch.inference_mode()
    def _checkpoint_and_eval(self, final=False):
        if not self.disable_logs and self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        metrics = self.evaluate_test_set()
        if not self.disable_logs and self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f"Eval_Test/{key}", value, self.current_learning_iteration)
            print("[eval] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
            self.writer.flush()
        if not final:
            self.env.reset()
            self.env.unwrapped.command_manager.get_term(self.command_name).ep_len.zero_()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self._prepare_logging_writer()
        if not self.disable_logs and self.log_dir is not None:
            store_code_state(self.log_dir, self.git_status_repos)
            with open(os.path.join(self.log_dir, "benchmark_config.json"), "w") as f:
                json.dump(self.cfg, f, indent=2, default=str)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        self.train_mode()
        if self.is_distributed:
            self.alg.broadcast_parameters()
        obs = self.env.get_observations().to(self.device)
        started = time.time()
        last_eval = started
        budget_s = self.max_train_hours * 3600.0
        command = self.env.unwrapped.command_manager.get_term(self.command_name)
        max_step_reward = command.cfg.max_step_reward
        returns, lengths = deque(maxlen=1000), deque(maxlen=1000)
        ep_return = torch.zeros(self.env.num_envs, device=self.device)
        ep_length = torch.zeros_like(ep_return)
        steps = 0
        first = self.current_learning_iteration
        for it in range(first, first + num_learning_iterations):
            t0 = time.perf_counter()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    ep_return += rewards
                    ep_length += 1
                    ids = dones.nonzero().squeeze(-1)
                    if ids.numel():
                        returns.extend((ep_return[ids] / (max_step_reward * ep_length[ids])).tolist())
                        lengths.extend(ep_length[ids].tolist())
                        ep_return[ids] = 0
                        ep_length[ids] = 0
                    steps += 1
                self.alg.compute_returns(obs)
            collection_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.get("mixed_precision", False)):
                losses = self.alg.update()
            torch.cuda.synchronize(self.device)
            learning_time = time.perf_counter() - t0
            self.current_learning_iteration = it + 1
            elapsed = time.time() - started
            stats = self._allreduce_sum(torch.tensor(
                [sum(returns), len(returns), sum(lengths)], dtype=torch.float64, device=self.device
            ))
            if not self.disable_logs and self.writer is not None:
                norm_return = float(stats[0] / stats[1]) if stats[1] else float("nan")
                mean_length = float(stats[2] / stats[1]) if stats[1] else float("nan")
                self.writer.add_scalar("Train/episode_norm_return", norm_return, it + 1)
                self.writer.add_scalar("Train/mean_episode_length", mean_length, it + 1)
                self.writer.add_scalar("Perf/hours", elapsed / 3600.0, it + 1)
                self.writer.add_scalar("Perf/env_steps", steps, it + 1)
                self.writer.add_scalar("Perf/total_timesteps", steps * self.env.num_envs * self.gpu_world_size, it + 1)
                self.writer.add_scalar("Perf/collection_time", collection_time, it + 1)
                self.writer.add_scalar("Perf/learning_time", learning_time, it + 1)
                for key, value in losses.items():
                    self.writer.add_scalar(f"Loss/{key}", value, it + 1)
                self.writer.add_scalar("Loss/actor_learning_rate", self.alg.actor_learning_rate, it + 1)
                self.writer.add_scalar("Loss/critic_learning_rate", self.alg.critic_learning_rate, it + 1)
                self.writer.add_scalar("Policy/mean_noise_std", self.policy.action_std.mean().item(), it + 1)
                print(
                    f"ppo iteration {it + 1} | {elapsed / 3600:.3f} h | env steps {steps} | "
                    f"train ret {norm_return:.3f} len {mean_length:.0f} | "
                    f"t sim/pi {collection_time:.1f}/{learning_time:.1f}s | "
                    + " ".join(f"{k}={v:.4g}" for k, v in losses.items()), flush=True,
                )
            flags = self._broadcast_scalar(
                float(elapsed >= budget_s or it + 1 == first + num_learning_iterations)
                + 2.0 * float(time.time() - last_eval >= self.eval_interval_minutes * 60.0)
            )
            if flags % 2 == 1:
                break
            if flags >= 2:
                self._checkpoint_and_eval()
                last_eval = time.time()
                obs = self.env.get_observations().to(self.device)
                ep_return.zero_()
                ep_length.zero_()
            elif (it + 1) % self.save_interval == 0 and not self.disable_logs and self.log_dir is not None:
                self.save(os.path.join(self.log_dir, f"model_{it + 1}.pt"))
        self._checkpoint_and_eval(final=True)
        if self.is_distributed:
            dist.barrier()
