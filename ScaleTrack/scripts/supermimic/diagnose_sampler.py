"""Diagnose the return-based reset sampler with a trained policy (no learning).

Runs the training-time collection (action noise, return-bin resets) for a while and reports, grouped by the
position of each 1 s bin relative to the end of its motion: the bin return estimates, the share of the
"hard" softmax mass and the share of actual resets; plus why episodes end and how far reset frames are from the
end of their motion.

Example:
    python scripts/supermimic/diagnose_sampler.py --checkpoint <model.pt> --motion_lib /dev/shm/sm_lib/eval --headless
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="G1-BFM-Transformer-Tracking-SuperMimic")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--motion_lib", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--steps", type=int, default=2000)
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
cfg.commands.motion.motion_lib_dir = args.motion_lib
agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
agent.supermimic["compile"] = False
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
runner = SuperMimicRunner(env, agent.to_dict(), log_dir=None, device=env.unwrapped.device)
runner.load(args.checkpoint, load_optimizer=False)
command = env.unwrapped.command_manager.get_term("motion")
tm = env.unwrapped.termination_manager
lib = command.train_lib
dev = lib.device

# Bin position from the end of its motion: 0 = last bin, 1 = second to last, 2 = all others.
first_bin_of_next = torch.zeros(lib.num_motions, dtype=torch.long, device=dev)
bins_per_motion = torch.bincount(lib.bin_motion, minlength=lib.num_motions)
last_bin = torch.cumsum(bins_per_motion, 0) - 1
pos = (last_bin[lib.bin_motion] - torch.arange(lib.num_bins, device=dev)).clamp_max(2)
names = ["last bin", "second-to-last bin", "other bins"]

resets_by_pos = torch.zeros(3, device=dev)
remaining_at_reset = []
orig_sample = command._sample_start_frames


def sample_and_record(n):
    f = orig_sample(n)
    resets_by_pos.index_add_(0, pos[lib.frame_bin[f]], torch.ones(n, device=dev))
    remaining_at_reset.append((lib.frame_last[f] - f).float())
    return f


command._sample_start_frames = sample_and_record
ends = torch.zeros(3, device=dev)  # failed, motion_time_out, time_out
obs = env.get_observations()
with torch.no_grad():
    for step in range(args.steps):
        obs, _, dones, _ = env.step(runner.alg.act(obs))
        ends[0] += tm.get_term("body_pos").sum()
        ends[1] += tm.get_term("motion_time_out").sum()
        ends[2] += (tm.get_term("time_out") & ~tm.get_term("motion_time_out")).sum()

touched = command.bin_return < 1.0 - 1e-6
hard = torch.softmax(-command.cfg.softmax_temp * command.bin_return, dim=0)
print(f"[diag] {lib.num_motions} motions, {lib.num_bins} bins, {args.steps} steps x {args.num_envs} envs")
for i, name in enumerate(names):
    sel = pos == i
    t = sel & touched
    print(
        f"[diag] {name:<20s} bins={int(sel.sum()):6d} ({float(sel.float().mean()):.1%})  "
        f"mean return ratio (updated bins)={float(command.bin_return[t].mean()) if t.any() else float('nan'):.2f}  "
        f"hard-mass share={float(hard[sel].sum()):.1%}  reset share={float(resets_by_pos[i] / resets_by_pos.sum()):.1%}"
    )
total = ends.sum()
print(f"[diag] episode ends: failed={float(ends[0] / total):.1%} motion_end={float(ends[1] / total):.1%} timeout={float(ends[2] / total):.1%}")
rem = torch.cat(remaining_at_reset)
print(f"[diag] reset frames with < 32 frames left in the motion: {float((rem < 32).float().mean()):.1%}; median frames left {float(rem.median()):.0f}")
os._exit(0)
