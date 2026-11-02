"""Evaluate a SuperMimic checkpoint on a packed library with the training-time test protocol.

    python scripts/supermimic/eval_checkpoint.py --task G1-BFM-Transformer-Tracking-SuperMimic-HOI \
        --checkpoint <model.pt> --train_lib <lib used to build the env> --eval_lib <lib to evaluate> --headless
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="G1-BFM-Transformer-Tracking-SuperMimic-HOI")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--train_lib", required=True)
parser.add_argument("--eval_lib", required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--output", default="")
parser.add_argument("--init_only", action="store_true", help="load the checkpoint as init_policy_from (robot-only)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym

import scaletrack.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

from my_rsl_rl.runners.supermimic_runner import SuperMimicRunner

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env_cfg.commands.motion.motion_lib_dir = args.train_lib
env_cfg.commands.motion.test_motion_lib_dir = args.eval_lib
agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
agent_cfg.device = args.device
if args.init_only:
    agent_cfg.supermimic["init_policy_from"] = args.checkpoint
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
runner = SuperMimicRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
if not args.init_only:
    runner.load(args.checkpoint, load_optimizer=False)
result = runner.evaluate_test_set()
for k in sorted(result):
    print(f"[eval] {k} = {result[k]:.4f}")
if args.output:
    with open(args.output, "w") as f:
        json.dump(dict(checkpoint=args.checkpoint, eval_lib=args.eval_lib, result=result), f, indent=1)
os._exit(0)
