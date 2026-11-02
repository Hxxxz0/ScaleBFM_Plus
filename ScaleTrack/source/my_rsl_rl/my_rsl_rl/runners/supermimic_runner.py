"""Runner for SuperMimic-style training (world model + BPTT) of the humanoid transformer policy.

Training alternates vectorised simulator steps (collected into the replay buffer) with updates of the world model,
actor and critic at a fixed update-to-data ratio. Checkpoints keep the format of :class:`OnPolicyRunner`, so the
existing play / export scripts load them unchanged; the SuperMimic state is stored under an extra key.
"""

from __future__ import annotations

import inspect
import os
import time
import torch
import torch.distributed as dist
from collections import deque
from pathlib import Path

import my_rsl_rl
from my_rsl_rl.algorithms.supermimic import SuperMimic
from my_rsl_rl.modules import ActorCriticHumanoidTransformer
from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner
from my_rsl_rl.runners.training_budget import TrainingBudget, cosine_lr_ratio
from my_rsl_rl.utils import resolve_obs_groups, store_code_state
from my_rsl_rl.utils.compile_cache import CompileCache


class SuperMimicRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.sm_cfg = dict(train_cfg["supermimic"])
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self._configure_multi_gpu()
        self.command_name = self.cfg.get("command_name", "motion")
        self.save_interval = self.cfg["save_interval"]
        self.eval_metric_keys = self.cfg.get("eval_metric_keys", [])
        self.eval_max_steps = self.cfg.get("eval_max_steps", None)
        self.success_metric_dict = self.cfg.get("success_metric_dict", {})

        self.task = env.unwrapped.make_imagination_task()
        obs = self.env.get_observations()
        # The critic of the PPO policy is not trained here but kept so that checkpoints load into PPO runners.
        for name, shape in self.sm_cfg.pop("ppo_critic_obs_shapes").items():
            obs[name] = torch.zeros(obs.batch_size[0], *shape, device=obs.device)
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], ["critic"])
        policy_cfg = dict(self.policy_cfg)
        policy_cfg.pop("class_name")
        self.policy = ActorCriticHumanoidTransformer(
            obs, self.cfg["obs_groups"], self.env.num_actions, **policy_cfg
        ).to(device)
        for m in self.policy.modules():
            if hasattr(m, "need_weights"):
                m.need_weights = False
        self.utd = self.sm_cfg.pop("update_to_data_ratio")
        self.warmup_env_steps = self.sm_cfg.pop("warmup_env_steps")
        self.wm_only_updates = self.sm_cfg.pop("wm_only_updates")
        self.lr_final_ratio = self.sm_cfg.pop("lr_final_ratio")
        self.lr_decay_fraction = self.sm_cfg.pop("lr_decay_fraction", 1.0)
        self.max_train_hours = self.sm_cfg.pop("max_train_hours")
        self.budget_mode = self.sm_cfg.pop("budget_mode", "hours")
        self.eval_interval_updates = self.sm_cfg.pop("eval_interval_updates", 0)
        self.eval_interval_minutes = self.sm_cfg.pop("eval_interval_minutes")
        self.log_interval = self.sm_cfg.pop("log_interval")
        init_policy_from = self.sm_cfg.pop("init_policy_from", "")
        self.init_policy_names = self.sm_cfg.pop("init_policy_names", "")
        if self.log_interval < 1 or self.eval_interval_updates < 0:
            raise ValueError("log_interval must be positive and eval_interval_updates nonnegative")
        if self.eval_interval_updates and self.eval_interval_updates % self.log_interval:
            raise ValueError("eval_interval_updates must be a multiple of log_interval")
        self.alg = SuperMimic(self.policy, self.task, self.env.num_envs, device, self.env.num_actions, **self.sm_cfg)
        alg = self.alg
        if init_policy_from:
            self._init_policy(init_policy_from)
        self.compile_cache = CompileCache(
            os.environ.get("SCALETRACK_COMPILE_CACHE_DIR"),
            dict(
                policy=policy_cfg,
                shapes=(alg.imagination_batch_size, alg.actor_micro_batch_size, alg.horizon, alg.wm_batch_size,
                        alg.wm_window, alg.history_length, alg.actor_checkpointing, self.task.state_dim,
                        self.task.xi_dim, self.task.critic_dim, self.env.num_actions),
            ),
            sources=(Path(__file__).resolve().parents[1], Path(inspect.getfile(type(self.task))).parent),
            root=Path(__file__).resolve().parents[4],
        )  # fmt: skip

        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        self.log_dir = log_dir
        self.writer = None
        self.num_updates = 0
        self.env_steps = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [my_rsl_rl.__file__]
        if not self.disable_logs:
            actor_params = sum(p.numel() for p in self.policy.actor.parameters())
            actor_params += sum(p.numel() for p in self.policy.actor_task_embedder.parameters())
            if hasattr(self.policy, "_actor_encoders"):
                actor_params += sum(p.numel() for e in self.policy._actor_encoders() for p in e.parameters())
            print(
                f"[model] deployment_actor_parameters={actor_params} world_size={self.gpu_world_size} "
                f"imagination_global_batch={self.alg.imagination_batch_size * self.gpu_world_size} "
                f"learning_rates={self.alg.base_lrs}",
                flush=True,
            )

    def _init_policy(self, path: str) -> None:
        """Initialise the policy from a checkpoint.

        Parameters missing from the checkpoint keep their initialisation (e.g. the object encoders, near zero).
        When the robot differs (``init_policy_names``: index.json of a library of the checkpoint's robot, holding its
        joint and body names), the actor's joint / action / body dimensions are remapped by name: new inputs get zero
        weights and new outputs zero rows, so the checkpoint's joints behave as before and new joints hold their
        default. The (untrained) transformer critic is left initialised. World model and critic are not loaded. Every
        rank loads the same file, so the replicas stay identical."""
        state = torch.load(path, map_location=self.device, weights_only=False)["model_state_dict"]
        own = self.policy.state_dict()
        unexpected = [k for k in state if k not in own]
        mismatched = [k for k in state if k in own and own[k].shape != state[k].shape]
        if unexpected:
            raise ValueError(f"init_policy_from {path}: unexpected {unexpected}")
        if mismatched:
            if not self.init_policy_names:
                raise ValueError(f"init_policy_from {path}: shape mismatch {mismatched}; set init_policy_names to remap")
            state = self._remap_actor(state, own, mismatched)
        self.policy.load_state_dict(state, strict=False)
        if int(os.getenv("RANK", "0")) == 0:
            print(f"[init] policy from {path}; remapped {mismatched}; new parameters: {sorted(set(own) - set(state))}", flush=True)

    def _remap_actor(self, state: dict, own: dict, keys: list[str]) -> dict:
        import json

        old = json.loads(Path(self.init_policy_names).read_text())
        env = self.env.unwrapped
        command = env.command_manager.get_term(self.command_name)
        new_joints = list(command.robot.joint_names)
        new_actions = list(env.action_manager.get_term("joint_pos")._joint_names)
        new_bodies = list(command.cfg.body_names)
        old_joints, old_bodies = old["joint_names"], old["body_names"]

        def index(names, new):
            return [new.index(n) for n in names]

        jo, ja, bo = index(old_joints, new_joints), index(old_joints, new_actions), index(old_bodies, new_bodies)
        Jo, Jn, Bo, Bn = len(old_joints), len(new_joints), len(old_bodies), len(new_bodies)
        # Columns of the prop frame [gravity 3, ang vel 3, joint pos J, joint vel J].
        prop_cols = list(range(6)) + [6 + j for j in jo] + [6 + Jn + j for j in jo]
        # Columns of a task frame [pos 3B, pos_rel 3B, rot 6B, rot_rel 6B, time 1] followed by the mode (B).
        task_cols, off_n = [], 0
        for d in (3, 3, 6, 6):
            task_cols += [off_n + b * d + i for b in bo for i in range(d)]
            off_n += Bn * d
        task_cols += [off_n] + [off_n + 1 + b for b in bo]
        out = dict(state)
        for k in keys:
            old_t, new_t = state[k], torch.zeros_like(own[k])
            if k.startswith(("critic.", "critic_task_embedder.")):
                del out[k]  # untrained in SuperMimic: keep the fresh initialisation
                continue
            if k == "actor.prop_projection.weight":
                new_t[:, prop_cols] = old_t
            elif k == "actor.action_projection.weight":
                new_t[:, ja] = old_t
            elif k in ("actor.projection_head.weight", "actor.projection_head.bias"):
                new_t[ja] = old_t
            elif k in ("std", "log_std"):
                new_t = own[k].clone()
                new_t[ja] = old_t
            elif k.startswith("actor_task_embedder.") and k.endswith("weight") and old_t.dim() == 2:
                new_t[:, task_cols] = old_t
            else:
                raise ValueError(f"cannot remap {k}: {tuple(old_t.shape)} -> {tuple(new_t.shape)}")
            out[k] = new_t
        return out

    # ------------------------------------------------------------------ distributed helpers

    def _broadcast_values(self, *values: float) -> list[float]:
        """Rank 0's values on every rank (decisions taken from rank 0's wall clock)."""
        if not self.is_distributed:
            return list(values)
        t = torch.tensor(values, dtype=torch.float64, device=self.device)
        dist.broadcast(t, src=0)
        return t.tolist()

    def _broadcast_scalar(self, x: float) -> float:
        return self._broadcast_values(x)[0]

    def _allreduce_sum(self, t: torch.Tensor) -> torch.Tensor:
        if self.is_distributed:
            dist.all_reduce(t)
        return t

    # ------------------------------------------------------------------ training

    def learn(self, num_learning_iterations: int = 0, init_at_random_ep_len: bool = False) -> None:
        """Train for ``max_train_hours`` of wall-clock time (or ``num_learning_iterations`` updates if > 0)."""
        self._prepare_logging_writer()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        if self.log_dir is not None and not self.disable_logs:
            store_code_state(self.log_dir, self.git_status_repos)
        task, alg, buf = self.task, self.alg, self.alg.buffer
        max_updates = num_learning_iterations if num_learning_iterations > 0 else None
        self.budget = TrainingBudget(self.budget_mode, max_updates, self.max_train_hours)
        budget_s = self.budget.seconds
        t_start = time.time()
        last_eval = t_start
        # Wall-clock time of the first policy update (hour budgets schedule the LR over time); a resumed run past the
        # world-model-only updates schedules over its own budget.
        policy_start = 0.0 if self.num_updates > self.wm_only_updates else None
        run_steps = 0  # environment steps of this run: a resumed run refills its (empty) replay buffer first
        cache_pending = True  # save the compiled artifacts after the first policy update of this run
        lr_ratio = 1.0
        command = self.env.unwrapped.command_manager.get_term(self.command_name)
        if not self.disable_logs:
            print(
                f"[budget] mode={self.budget_mode} target_updates={max_updates} hours={self.max_train_hours} "
                f"eval_interval_updates={self.eval_interval_updates}",
                flush=True,
            )
        if self.budget.reached(self.num_updates, 0.0):
            self._checkpoint_and_eval(final=True)
            return
        if self.compile_cache.load() and not self.disable_logs:
            print(f"[compile] loaded cached artifacts {self.compile_cache.loaded}", flush=True)
        utd_acc = 0.0
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
        agg: dict[str, float] = {}
        agg_n = 0
        ret_buf, len_buf = deque(maxlen=1000), deque(maxlen=1000)
        ep_ret = torch.zeros(self.env.num_envs, device=self.device)
        ep_len = torch.zeros(self.env.num_envs, device=self.device)
        max_step_reward = task.max_step_reward
        # Per-step reward terms of the simulator, comparable with the imagined ones (SuperMimic/imag_reward/*).
        reward_manager = self.env.unwrapped.reward_manager
        term_names = list(reward_manager.active_terms)
        sim_terms = torch.zeros(len(term_names), device=self.device)
        sim_terms_n = 0

        obs = self.env.get_observations().to(self.device)
        state, info = task.read_state(), task.step_info()
        ep_start = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        stop = False
        while not stop:
            if hasattr(command, "set_failure_reset_active"):
                command.set_failure_reset_active(
                    not command.cfg.failure_reset.activate_after_world_model_warmup
                    or (self.num_updates >= self.wm_only_updates and run_steps >= self.warmup_env_steps)
                )
            # ---- collect one vectorised step
            t0 = time.perf_counter()
            actions = alg.act(obs)
            obs, rewards, dones, _ = self.env.step(actions)
            obs = obs.to(self.device)
            buf.add(state, actions, task.executed_actions(), info, ep_start)
            reset_ids = dones.nonzero().squeeze(-1)
            buf.add_terminal(reset_ids, task.terminal_state(reset_ids))
            state, info, ep_start = task.read_state(), task.step_info(), dones.bool()
            ep_ret += rewards
            ep_len += 1
            sim_terms += reward_manager._step_reward.mean(0) * self.env.unwrapped.step_dt
            sim_terms_n += 1
            if reset_ids.numel() > 0:
                ret_buf.extend((ep_ret[reset_ids] / (max_step_reward * ep_len[reset_ids])).tolist())
                len_buf.extend(ep_len[reset_ids].tolist())
                ep_ret[reset_ids] = 0
                ep_len[reset_ids] = 0
            self.env_steps += 1
            run_steps += 1
            timers["sim"] += time.perf_counter() - t0
            if run_steps < self.warmup_env_steps:
                continue

            # ---- updates at the configured update-to-data ratio
            utd_acc += self.utd
            while utd_acc >= 1.0:
                utd_acc -= 1.0
                if self.num_updates == self.wm_only_updates:
                    policy_start = time.time() - t_start
                if self.budget_mode == "updates":
                    lr_ratio = self._lr_ratio(self.budget.policy_progress(self.num_updates, self.wm_only_updates, 0.0, None))
                alg.set_lr_ratio(lr_ratio)
                buf.prepare(alg.wm_window, alg.history_length)
                t0 = time.perf_counter()
                m = alg.update_world_model()
                torch.cuda.synchronize()
                timers["wm"] += time.perf_counter() - t0
                if self.num_updates >= self.wm_only_updates:
                    t0 = time.perf_counter()
                    m.update(alg.update_policy())
                    torch.cuda.synchronize()
                    timers["policy"] += time.perf_counter() - t0
                m["lr_ratio"] = lr_ratio
                for k, v in m.items():
                    agg[k] = agg.get(k, 0.0) + v
                agg_n += 1
                self.num_updates += 1
                self.current_learning_iteration = self.num_updates
                if cache_pending and self.num_updates > self.wm_only_updates:
                    cache_pending = False
                    if self.compile_cache.save() and not self.disable_logs:
                        print(f"[compile] saved artifacts {self.compile_cache.path}", flush=True)
                stop = self.budget_mode == "updates" and self.budget.reached(self.num_updates, 0.0)

                if self.num_updates % self.log_interval == 0 or stop:
                    elapsed = time.time() - t_start
                    for name, value in zip(term_names, (sim_terms / max(sim_terms_n, 1)).tolist()):
                        agg[f"sim_reward/{name}"] = value * agg_n
                    for name, value in alg.drain_communication_ms().items():
                        agg[name] = value  # total over the logging window; _log divides by its update count
                    sim_terms.zero_()
                    sim_terms_n = 0
                    self._log(agg, agg_n, timers, ret_buf, len_buf, elapsed)
                    agg, agg_n = {}, 0
                    timers = {k: 0.0 for k in timers}
                    # Every eval_interval_updates updates when set (also with an hour budget), else every eval_interval_minutes.
                    by_updates = bool(self.eval_interval_updates and self.num_updates % self.eval_interval_updates == 0)
                    if self.budget_mode == "hours":
                        out_of_time, eval_due, progress = self._broadcast_values(
                            float(elapsed > budget_s),
                            float((time.time() - last_eval) / 60.0 > self.eval_interval_minutes),
                            self.budget.policy_progress(self.num_updates, self.wm_only_updates, elapsed, policy_start),
                        )
                        lr_ratio = self._lr_ratio(progress)
                        stop = bool(out_of_time) or self.budget.reached(self.num_updates, 0.0)
                        evaluate = by_updates if self.eval_interval_updates else bool(eval_due)
                    else:
                        evaluate = by_updates
                    if evaluate and not stop:
                        self._checkpoint_and_eval()
                        last_eval = time.time()
                        obs = self.env.get_observations().to(self.device)
                        state, info = task.read_state(), task.step_info()
                        ep_start[:] = True
                        ep_ret[:] = 0
                        ep_len[:] = 0
                if stop:
                    break
        self._checkpoint_and_eval(final=True)

    def _lr_ratio(self, progress: float) -> float:
        return cosine_lr_ratio(progress, self.lr_final_ratio, self.lr_decay_fraction)

    def _checkpoint_and_eval(self, final: bool = False) -> None:
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.num_updates}.pt"))
        eval_dict = self.evaluate_test_set()
        if self.writer is not None and not self.disable_logs:
            for k, v in eval_dict.items():
                self.writer.add_scalar(f"Eval_Test/{k}", v, self.num_updates)
            print("[eval] " + " ".join(f"{k}={v:.4f}" for k, v in eval_dict.items()), flush=True)
        if not final:
            self.env.reset()
            self.alg.buffer.end_all_episodes()
            self.env.unwrapped.command_manager.get_term(self.command_name).ep_len.zero_()

    def _log(self, agg, agg_n, timers, ret_buf, len_buf, elapsed) -> None:
        stats = torch.tensor([sum(ret_buf), float(len(ret_buf)), sum(len_buf)], dtype=torch.float64, device=self.device)
        stats = self._allreduce_sum(stats)
        command = self.env.unwrapped.command_manager.get_term(self.command_name)
        reset_stats = command.drain_reset_statistics() if hasattr(command, "drain_reset_statistics") else None
        if reset_stats is not None:
            group_names, values = reset_stats
            values = self._allreduce_sum(values)
        if self.disable_logs or self.writer is None:
            return
        it = self.num_updates
        if reset_stats is not None:
            count = dict(zip(command.reset_sampler.STAT_NAMES, values[:10].tolist()))
            total = max(count["total"], 1.0)
            for key in ("total", "failed", "eligible", "local", "global", "timeout", "early_failure", "retry_limit"):
                self.writer.add_scalar(f"Reset/{key}", count[key], it)
            self.writer.add_scalar("Reset/local_fraction", count["local"] / total, it)
            self.writer.add_scalar("Reset/local_fraction_eligible", count["local"] / max(count["eligible"], 1.0), it)
            for source in ("local", "global"):
                self.writer.add_scalar(
                    f"Reset/{source}_episode_steps", count[f"{source}_episode_steps"] / max(count[source], 1.0), it
                )
            for name, n in zip(group_names, values[10:].tolist()):
                self.writer.add_scalar(f"Sampling/{name}_share", n / total, it)
            print(
                f"[reset] update={it} local={count['local']:.0f}/{count['total']:.0f} "
                f"eligible={count['eligible']:.0f} retry_limit={count['retry_limit']:.0f}",
                flush=True,
            )
        for k, v in agg.items():
            self.writer.add_scalar(f"SuperMimic/{k}", v / agg_n, it)
        for k, v in timers.items():
            self.writer.add_scalar(f"Perf/{k}_s", v, it)
        self.writer.add_scalar("Perf/hours", elapsed / 3600.0, it)
        self.writer.add_scalar("Perf/env_steps", self.env_steps, it)
        train_ret = float(stats[0] / stats[1]) if stats[1] > 0 else float("nan")
        mean_len = float(stats[2] / stats[1]) if stats[1] > 0 else float("nan")
        self.writer.add_scalar("Train/episode_norm_return", train_ret, it)
        self.writer.add_scalar("Train/mean_episode_length", mean_len, it)
        g = lambda k: agg.get(k, float("nan")) / agg_n  # noqa: E731
        print(
            f"update {it} | {elapsed / 3600:.2f} h | env steps {self.env_steps} | train ret {train_ret:.3f} len {mean_len:.0f} | "
            f"wm nse1 {g('wm_nse_1step'):.4f} nse8 {g('wm_nse_kstep'):.4f} | imag ret {g('imag_return'):.3f} "
            f"surv {g('imag_survival_H'):.3f} | critic {g('critic_loss'):.2e} | lr x{g('lr_ratio'):.2f} | "
            f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s | "
            f"allreduce ms/update wm/actor/critic {g('allreduce_ms/wm'):.0f}/{g('allreduce_ms/actor'):.0f}/{g('allreduce_ms/critic'):.0f}",
            flush=True,
        )
        if "wm_object_nse_kstep" in agg:
            print(
                f"  object | wm nse8 {g('wm_object_nse_kstep'):.4f} | imag motion {g('imag_reward/object_motion'):.2e} "
                f"relation {g('imag_reward/object_relation'):.2e}",
                flush=True,
            )

    # ------------------------------------------------------------------ evaluation

    @torch.no_grad()
    def evaluate_test_set(self) -> dict[str, float]:
        """Track every test motion once from its first frame without noise or early termination.

        Same protocol and success criterion as ``OnPolicyRunner.evaluate_policy`` (inference-mode policy, metrics
        averaged over each motion, a motion fails if a metric of ``success_metric_dict`` exceeds its threshold),
        with the test motions split across ranks.
        """
        env = self.env.unwrapped
        command = env.command_manager.get_term(self.command_name)
        if not command.has_test_set:
            return {}
        self.policy.eval()
        self._set_env_is_evaluating(test_set=True)
        action_term = env.action_manager.get_term("joint_pos")
        noise_std, action_term.noise_std = action_term.noise_std, 0.0

        # Motions may start their evaluation after frame 0 (HOI: the audited start of the interaction).
        starts = command.lib.eval_start if getattr(command.lib, "has_objects", False) else torch.zeros_like(command.time_totals)
        lengths = command.time_totals - starts
        motion_range = torch.argsort(lengths)[self.gpu_global_rank :: self.gpu_world_size]
        keys = self.eval_metric_keys
        object_eval = hasattr(command, "object_evaluation")
        object_failed = torch.zeros(len(motion_range), dtype=torch.bool, device=self.device)
        object_keys = (
            "object_max_pos_error", "object_max_rot_error", "object_motion_coverage", "object_fail_pos",
            "object_fail_rot", "object_fail_coverage", "object_fail_lift", "robot_fail",
        )  # fmt: skip
        object_sums = torch.zeros(len(motion_range), len(object_keys), device=self.device)
        sums = torch.zeros(len(motion_range), len(keys), device=self.device)
        maxs = torch.zeros(len(motion_range), len(keys), device=self.device)
        steps = torch.clamp(lengths[motion_range], max=self.eval_max_steps or 10**9)
        if hasattr(command, "evaluation_chunks"):
            # Environments hold different objects: a motion runs where its object is.
            chunks = command.evaluation_chunks(motion_range)
        else:
            chunks = [
                (torch.arange(i, min(i + self.env.num_envs, len(motion_range)), device=self.device),
                 torch.arange(min(self.env.num_envs, len(motion_range) - i), device=self.device))
                for i in range(0, len(motion_range), self.env.num_envs)
            ]  # fmt: skip
        for pos, slots in chunks:
            pos = pos.to(self.device)
            ids = motion_range[pos]
            command.motion_ids[slots] = ids
            if hasattr(command, "fill_evaluation"):
                command.fill_evaluation(slots)
            else:
                command.motion_ids[len(ids) :] = ids[0]
            obs, _ = self.env.reset()
            limit = steps[pos]
            if object_eval:
                command.reset_object_evaluation()
            for t in range(int(limit.max())):
                if object_eval:
                    command.eval_active[slots] = t < limit
                obs, _, _, _ = self.env.step(self.policy.act_inference(obs.to(self.device)))
                active = (t < limit)[:, None]
                vals = torch.stack([command.metrics[key][slots] for key in keys], dim=-1)
                sums[pos] += vals * active
                maxs[pos] = torch.where(active, torch.maximum(maxs[pos], vals), maxs[pos])
            if object_eval:
                f, m = command.object_evaluation(slots)
                object_failed[pos] = f
                object_sums[pos, :-1] = torch.stack([m[key] for key in object_keys[:-1]], -1)
        means = sums / steps[:, None]
        failed = torch.zeros(len(motion_range), dtype=torch.bool, device=self.device)
        for key, thr in self.success_metric_dict.items():
            failed |= maxs[:, keys.index(key)] > thr
        if object_eval:
            object_sums[:, object_keys.index("robot_fail")] = failed.float()
        failed |= object_failed
        has_object = (
            command.lib.object_valid[motion_range] if object_eval else torch.zeros_like(failed)
        )

        # Aggregate over ranks, overall and per dataset (motion-name prefix before '/').
        names = command.motion_names
        groups = sorted({names[j].split("/")[0] for j in range(len(names))})
        group_of = torch.tensor(
            [groups.index(names[j].split("/")[0]) for j in motion_range.tolist()], device=self.device
        )
        n_obj = len(object_keys) if object_eval else 0
        table = torch.zeros(len(groups) + 1, 3 + len(keys) + n_obj, dtype=torch.float64, device=self.device)
        for gi in range(len(groups) + 1):
            sel = torch.ones_like(failed) if gi == len(groups) else group_of == gi
            table[gi, 0] = sel.sum()
            table[gi, 1] = (~failed & sel).sum()
            table[gi, 2 : 2 + len(keys)] = means[sel].sum(0)
            if n_obj:
                table[gi, 2 + len(keys)] = (has_object & sel).sum()
                table[gi, 3 + len(keys) :] = object_sums[has_object & sel].sum(0)
        table = self._allreduce_sum(table)
        out = {}
        for gi, gname in enumerate(groups + ["all"]):
            n = float(table[gi, 0])
            if n == 0:
                continue
            prefix = "" if gname == "all" else f"{gname}/"
            out[f"{prefix}success_rate"] = float(table[gi, 1]) / n
            for j, key in enumerate(keys):
                out[f"{prefix}{key}"] = float(table[gi, 2 + j]) / n
            n_with_object = float(table[gi, 2 + len(keys)]) if n_obj else 0.0
            for j, key in enumerate(object_keys[:n_obj]):
                if n_with_object > 0:
                    out[f"{prefix}{key}"] = float(table[gi, 3 + len(keys) + j]) / n_with_object

        action_term.noise_std = noise_std
        self._set_env_no_evaluating(test_set=True)
        self.policy.train()
        return out

    # ------------------------------------------------------------------ checkpoints

    def save(self, path: str, infos: dict | None = None) -> None:
        torch.save(
            {
                "model_state_dict": self.policy.state_dict(),
                "actor_optimizer_state_dict": self.alg.actor_optimizer.state_dict(),
                # A PPO critic optimizer state over the (untrained) transformer critic, for OnPolicyRunner.load.
                "critic_optimizer_state_dict": torch.optim.Adam(self.policy.critic_parameters).state_dict(),
                "iter": self.num_updates,
                "infos": infos,
                "supermimic": self.alg.state_dict() | {"env_steps": self.env_steps},
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded = torch.load(path, weights_only=False, map_location=map_location)
        self.policy.load_state_dict(loaded["model_state_dict"])
        if "supermimic" in loaded:
            self.alg.load_state_dict(loaded["supermimic"], load_optimizer)
            self.env_steps = loaded["supermimic"].get("env_steps", 0)
        self.num_updates = loaded["iter"]
        self.current_learning_iteration = self.num_updates
        return loaded["infos"]

    def get_inference_policy(self, device: str | None = None):
        self.policy.eval()
        if device is not None:
            self.policy.to(device)
        return self.policy.act_inference

    def train_mode(self) -> None:
        self.policy.train()

    def eval_mode(self) -> None:
        self.policy.eval()
