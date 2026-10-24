"""Motion command for SuperMimic-style training.

Same observations, rewards and terminations as :class:`MotionCommand`, but

* the motions come from a packed library kept on the GPU (optionally sharded across data-parallel ranks),
* index tensors live on the GPU and current-frame references are cached once per step,
* resets sample the start frame with SuperMimic's return-based bin sampling (paper Appendix D.1),
* metrics are only computed while evaluating.
"""

from __future__ import annotations

import math
import os
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform

from scaletrack.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg

from .motion_lib import MotionLibrary
from .reset_sampling import FailureResetCfg, HybridResetSampler, normalized_returns_to_go

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_CACHED = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


class SuperMimicMotionCommand(MotionCommand):
    cfg: SuperMimicMotionCommandCfg

    def __init__(self, cfg: SuperMimicMotionCommandCfg, env: ManagerBasedRLEnv):
        CommandTerm.__init__(self, cfg, env)

        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0")) if self.is_distributed else 0
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        self.robot = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self._future_manual_cache: dict = {}
        self._frame_cache: dict[str, torch.Tensor] = {}
        rank, world = (self.gpu_global_rank, self.gpu_world_size) if cfg.shard_across_ranks else (0, 1)
        print(f"Loading packed motion library {cfg.motion_lib_dir} (shard {rank}/{world})")
        self.train_lib = MotionLibrary(
            cfg.motion_lib_dir,
            cfg.body_names,
            self.robot.joint_names,
            self.device,
            rank,
            world,
            cfg.bin_frames,
            cfg.shard_seed,
        )
        self._set_library_attrs("train", self.train_lib)
        self.has_test_set = bool(cfg.test_motion_lib_dir)
        if self.has_test_set:
            self.test_lib = MotionLibrary(cfg.test_motion_lib_dir, cfg.body_names, self.robot.joint_names, self.device)
            self._set_library_attrs("test", self.test_lib)
        self.lib = self.train_lib
        self.switch_motion_set(train_to_test=False)
        print(
            f"Motion library: {self.num_motion} motions, {self.train_lib.num_frames} frames, {self.train_lib.num_bins} bins"
        )

        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_sampling_prob = torch.ones(self.num_motion_train, dtype=torch.float, device=self.device)

        for k in (
            "error_anchor_height", "error_anchor_pos", "error_anchor_rot", "error_anchor_lin_vel", "error_anchor_ang_vel",
            "error_body_pos_g", "error_body_pos", "error_body_pos_relative", "error_body_rot", "error_body_rot_relative",
            "error_joint_pos", "error_joint_vel", "error_body_lin_vel", "error_body_ang_vel",
        ):  # fmt: skip
            self.metrics[k] = torch.zeros(self.num_envs, device=self.device)
        self.is_evaluating = False
        self.randomize_next_resampling = True

        self._mode_table = torch.zeros(len(cfg.mode_candidates), len(cfg.body_names), device=self.device)
        for mode_idx, link_names in enumerate(cfg.mode_candidates.values()):
            for name in link_names:
                self._mode_table[mode_idx, cfg.body_names.index(name)] = 1
        self.mode_ids = torch.randint(0, self._mode_table.shape[0], (self.num_envs,), device=self.device)
        self._mode = self._mode_table[self.mode_ids].clone()
        assert cfg.rand_timestep_range[1] <= cfg.ref_frame_buffer_size
        self._rand_timestep = torch.randint(*cfg.rand_timestep_range, (self.num_envs, 1), device=self.device)

        # Return-based reset sampling over 1 s bins of the training library. Each bin keeps the running mean of
        # the discounted return-to-go of the steps tracked in it, relative to the most that could have been earned
        # until the episode would end anyway (end of motion or episode time limit): 1 = tracked perfectly.
        lib = self.train_lib
        self.max_episode_steps = int(env.max_episode_length) + 1
        self.bin_return = torch.ones(lib.num_bins, device=self.device)
        self.bin_memory = torch.full((lib.num_bins,), 1.0 / lib.num_bins, device=self.device)
        self.ep_rewards = torch.zeros(self.num_envs, self.max_episode_steps, device=self.device)
        self.ep_bins = torch.zeros(self.num_envs, self.max_episode_steps, dtype=torch.long, device=self.device)
        self.ep_horizon = torch.zeros(self.num_envs, self.max_episode_steps, dtype=torch.long, device=self.device)
        self.ep_len = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        cfg.failure_reset.validate()
        self.reset_sampler = (
            HybridResetSampler(cfg.failure_reset, self.num_envs, lib) if cfg.failure_reset.enabled else None
        )
        self._failure_reset_active = not cfg.failure_reset.activate_after_world_model_warmup

    # ------------------------------------------------------------------ library

    def _set_library_attrs(self, set_name: str, lib: MotionLibrary):
        setattr(self, f"motion_names_{set_name}", lib.motion_names)
        setattr(self, f"num_motion_{set_name}", lib.num_motions)
        setattr(self, f"time_totals_{set_name}", lib.lengths)
        setattr(self, f"time_offsets_{set_name}", lib.starts)
        for name, field in (
            ("joint_pos", "joint_pos"), ("joint_vel", "joint_vel"), ("body_pos_w", "body_pos_w"),
            ("body_quat_w", "body_quat_w"), ("body_lin_vel_w", "body_lin_vel_w"), ("body_ang_vel_w", "body_ang_vel_w"),
        ):  # fmt: skip
            setattr(self, f"cat_{name}_{set_name}", getattr(lib, field))

    def switch_motion_set(self, train_to_test=True):
        super().switch_motion_set(train_to_test)
        self.lib = self.test_lib if train_to_test else self.train_lib
        if hasattr(self, "motion_ids"):
            # Motion ids index the active set: drop the ones of the other set and the unfinished episode logs.
            self.motion_ids.zero_()
            self.time_steps.zero_()
            self.ep_len.zero_()
        self._clear_caches()

        if getattr(self, "reset_sampler", None) is not None:
            self.reset_sampler.clear()

    def resample_motions(self):
        pass

    @property
    def global_frame(self) -> torch.Tensor:
        """Global library frame tracked by every env (the frame the next reward is scored against)."""
        return self._global_time_index()

    # ------------------------------------------------------------------ cached references

    def _clear_caches(self):
        self._future_manual_cache.clear()
        self._frame_cache.clear()

    def _current(self, name: str) -> torch.Tensor:
        if name not in self._frame_cache:
            value = getattr(self, f"cat_{name}")[self._global_time_index()]
            if name == "body_pos_w":
                value = value + self._env.scene.env_origins[:, None, :]
            self._frame_cache[name] = value
        return self._frame_cache[name]

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._current("joint_pos")

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._current("joint_vel")

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._current("body_pos_w")

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._current("body_quat_w")

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._current("body_lin_vel_w")

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._current("body_ang_vel_w")

    def _future_frame_offsets_manual(self, future_idx: Sequence[int]) -> torch.Tensor:
        idx = torch.as_tensor(future_idx, dtype=torch.long, device=self.device)
        idx_expand = idx.unsqueeze(0).expand(self.num_envs, -1)
        if torch.any(idx < 0):
            return torch.where(idx.unsqueeze(0) < 0, self._rand_timestep.expand(-1, idx.shape[0]), idx_expand)
        return idx_expand

    def _update_metrics(self):
        if self.is_evaluating:
            super()._update_metrics()

    # ------------------------------------------------------------------ return-based reset sampling

    def _log_step_rewards(self, env_ids: torch.Tensor):
        """Record this step's reward and the bin of the frame it was scored against."""
        if env_ids.numel() == 0 or not hasattr(self._env, "reward_buf"):
            return
        pos = self.ep_len[env_ids].clamp_max(self.max_episode_steps - 1)
        frame = self.global_frame[env_ids]
        self.ep_rewards[env_ids, pos] = self._env.reward_buf[env_ids]
        self.ep_bins[env_ids, pos] = self.train_lib.frame_bin[frame]
        # Transitions left until the episode ends without failure: the motion is truncated on its last frame,
        # the episode at the time limit (episode_length_buf already counts this step).
        to_motion_end = self.train_lib.frame_last[frame] - frame + 1
        to_time_limit = int(self._env.max_episode_length) - self._env.episode_length_buf[env_ids] + 1
        self.ep_horizon[env_ids, pos] = torch.minimum(to_motion_end, to_time_limit).clamp_min(1)
        self.ep_len[env_ids] = (self.ep_len[env_ids] + 1).clamp_max(self.max_episode_steps)

    def _finalize_episodes(self, env_ids: torch.Tensor):
        """Fold the normalised discounted returns-to-go of finished episodes into the per-bin estimates.

        Normalising by the attainable return makes episodes that reach the end of their motion or the time limit
        score 1 when tracked perfectly, wherever they started; only failures lower the estimate of a bin.
        """
        cfg = self.cfg
        lens = self.ep_len[env_ids]
        T = int(lens.max().item()) if env_ids.numel() else 0
        if T == 0:
            return
        valid = torch.arange(T, device=self.device)[None, :] < lens[:, None]
        ratios = normalized_returns_to_go(
            self.ep_rewards[env_ids, :T], lens, self.ep_horizon[env_ids, :T], cfg.gamma, cfg.max_step_reward
        )
        bins = self.ep_bins[env_ids, :T][valid]
        ratio = ratios[valid]
        nb = self.train_lib.num_bins
        sums = torch.zeros(nb, device=self.device).index_add_(0, bins, ratio)
        counts = torch.zeros(nb, device=self.device).index_add_(0, bins, torch.ones_like(ratio))
        touched = counts > 0
        mean = sums[touched] / counts[touched]
        self.bin_return[touched] = (1 - cfg.bin_ema) * self.bin_return[touched] + cfg.bin_ema * mean
        self.ep_len[env_ids] = 0

    def _sample_start_frames(self, n: int) -> torch.Tensor:
        cfg, lib = self.cfg, self.train_lib
        hard = torch.softmax(-cfg.softmax_temp * self.bin_return, dim=0)
        self.bin_memory = (1 - cfg.memory_ema) * self.bin_memory + cfg.memory_ema * hard
        p = (
            cfg.uniform_ratio / lib.num_bins
            + cfg.hard_ratio * hard
            + cfg.memory_ratio * self.bin_memory / self.bin_memory.sum()
        )
        b = torch.multinomial(p, n, replacement=True)
        f = lib.bin_start[b] + (torch.rand(n, device=self.device) * (lib.bin_end[b] - lib.bin_start[b])).long()
        # Leave at least two steps before the end of the motion.
        f = torch.maximum(torch.minimum(f, lib.frame_last[f] - 2), lib.starts[lib.frame_motion[f]])
        self.metrics["sampling_entropy"][:] = -(p * (p + 1e-12).log()).sum() / math.log(max(lib.num_bins, 2))
        return f

    # ------------------------------------------------------------------ command interface

    def set_failure_reset_active(self, active: bool) -> None:
        self._failure_reset_active = active

    def drain_reset_statistics(self):
        if self.reset_sampler is None:
            return None
        return self.reset_sampler.group_names, self.reset_sampler.drain_statistics()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device)
        if self.is_evaluating:
            self.time_steps[env_ids] = 0
        else:
            if getattr(self._env, "_explicit_reset", False):
                # env.reset() (start, after an evaluation): no episode ended, drop the partial logs.
                self.ep_len[env_ids] = 0
            else:
                # The reward of the final transition was computed before this reset.
                self._log_step_rewards(env_ids)
                self._finalize_episodes(env_ids)
            frames = self._sample_start_frames(len(env_ids))
            if self.reset_sampler is not None:
                frames = self.reset_sampler.select(
                    frames, env_ids, getattr(self._env, "_motion_reset_context", None), self._failure_reset_active
                )
            motion = self.train_lib.frame_motion[frames]
            self.motion_ids[env_ids] = motion
            self.time_steps[env_ids] = frames - self.train_lib.starts[motion]
        self._clear_caches()

        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_ori = self.body_quat_w[env_ids, 0].clone()
        root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
        root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()
        joint_pos = self.joint_pos[env_ids].clone()
        joint_vel = self.joint_vel[env_ids].clone()
        if self.cfg.enable_reset_disturbance:
            ranges = torch.tensor(
                [self.cfg.pose_range.get(k, (0.0, 0.0)) for k in "x y z roll pitch yaw".split()], device=self.device
            )
            rand = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
            root_pos += rand[:, 0:3]
            root_ori = quat_mul(quat_from_euler_xyz(rand[:, 3], rand[:, 4], rand[:, 5]), root_ori)
            ranges = torch.tensor(
                [self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in "x y z roll pitch yaw".split()], device=self.device
            )
            rand = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
            root_lin_vel += rand[:, :3]
            root_ang_vel += rand[:, 3:]
            joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
            limits = self.robot.data.soft_joint_pos_limits[env_ids]
            joint_pos = torch.clip(joint_pos, limits[:, :, 0], limits[:, :, 1])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1), env_ids=env_ids
        )

        self.mode_ids[env_ids] = torch.randint(0, self._mode_table.shape[0], (len(env_ids),), device=self.device)
        self._mode[env_ids] = self._mode_table[self.mode_ids[env_ids]]
        self._rand_timestep[env_ids] = torch.randint(
            *self.cfg.rand_timestep_range, (len(env_ids), 1), device=self.device
        )

    def _update_command(self):
        if not self.is_evaluating:
            self._log_step_rewards(torch.nonzero(~self._env.reset_buf).squeeze(-1))
        self.time_steps += 1
        self._clear_caches()


@configclass
class SuperMimicMotionCommandCfg(MotionCommandCfg):
    class_type: type = SuperMimicMotionCommand

    motion_file: str = ""
    motion_lib_dir: str = ""
    test_motion_lib_dir: str = ""
    shard_across_ranks: bool = True
    shard_seed: int = 0
    bin_frames: int = 50
    gamma: float = 0.99
    max_step_reward: float = 0.11
    uniform_ratio: float = 0.5
    hard_ratio: float = 0.4
    memory_ratio: float = 0.1
    softmax_temp: float = 25.0
    bin_ema: float = 0.05
    memory_ema: float = 0.01
    failure_reset: FailureResetCfg = FailureResetCfg()
