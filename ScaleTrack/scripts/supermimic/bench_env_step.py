"""Benchmark the raw env.step cost of the tracking task (Phase-0 gate B0.1)."""

import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="G1-BFM-Transformer-Tracking")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--motion_file", type=str, required=True)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--no_critic_obs", action="store_true")
parser.add_argument("--no_push", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import scaletrack.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.commands.motion.motion_file = args.motion_file
if args.no_critic_obs:
    cfg.observations.critic = None
    cfg.observations.critic_task = None
if args.no_push:
    cfg.events.push_robot = None
cfg.commands.motion.debug_vis = False
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)


def timed(fn, n):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n


def physics_only():
    env.action_manager.process_action(actions)
    for _ in range(env.cfg.decimation):
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)


t_step = timed(lambda: env.step(actions), args.steps)
t_phys = timed(physics_only, args.steps)
t_obs = timed(lambda: env.observation_manager.compute(), args.steps)
t_rew = timed(lambda: env.reward_manager.compute(dt=env.step_dt), args.steps)
print(f"[bench] num_envs={env.num_envs} env.step={t_step*1e3:.1f}ms physics(4 substeps)={t_phys*1e3:.1f}ms "
      f"obs={t_obs*1e3:.1f}ms reward={t_rew*1e3:.1f}ms -> {env.num_envs/t_step:.0f} env-steps/s")
# SimulationApp.close() can hang after headless runs; the numbers are already printed.
os._exit(0)
