"""SuperMimic: policy learning through a learned differentiable world model (Liao et al., arXiv 2610.28480).

One update consists of
  1. fitting the world model on k-step autoregressive windows of logged simulator transitions (Eq. 1),
  2. back-propagating the return of H-step imagined rollouts into the actor, with a soft survival weight and a
     critic bootstrap beyond the horizon (Eq. 2),
  3. fitting the critic to lambda-returns of the imagined transitions (Eq. 3).

The algorithm is task agnostic. The task object supplies the state parameterisation, the analytic observation,
reward and termination heads and the per-environment simulation context (see
scaletrack.tasks.tracking.supermimic.task.TrackingImaginationTask for the interface).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from my_rsl_rl.networks.world_model import DynamicsModel, ValueMLP
from my_rsl_rl.storage.replay_buffer import ReplayBuffer, StartBatch
from my_rsl_rl.utils.cuda_graph import GraphedFunction


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


@torch.no_grad()
def broadcast_module(module: torch.nn.Module) -> None:
    if _world_size() == 1:
        return
    for t in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(t.data, src=0)


class GradientAverager:
    """Averages the gradients of a parameter group across data-parallel ranks with one flat all-reduce.

    The time spent in the all-reduce is measured with CUDA events (no host synchronisation until it is read).
    """

    def __init__(self, params):
        self.params = list(params)
        self._flat = None
        self._events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    @torch.no_grad()
    def __call__(self) -> None:
        if _world_size() == 1:
            return
        grads = [p.grad for p in self.params if p.grad is not None]
        if not grads:
            return
        n = sum(g.numel() for g in grads)
        if self._flat is None or self._flat.numel() != n:
            self._flat = grads[0].new_empty(n)
        torch.cat([g.reshape(-1) for g in grads], out=self._flat)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        dist.all_reduce(self._flat, op=dist.ReduceOp.SUM)
        self._flat.div_(_world_size())
        end.record()
        self._events.append((start, end))
        views, offset = [], 0
        for g in grads:
            views.append(self._flat[offset : offset + g.numel()].view_as(g))
            offset += g.numel()
        torch._foreach_copy_(grads, views)

    def drain_ms(self) -> float:
        """Total all-reduce time (ms) since the last call."""
        if self._events:
            self._events[-1][1].synchronize()
        total = sum(start.elapsed_time(end) for start, end in self._events)
        self._events.clear()
        return total


def actor_objective(r, d, tr, v, v_final, gamma):
    """Eq. 2 for imagined rollouts of H steps (tensors (H, n)); returns the loss and the survival / alive weights.

    c_t is the soft survival weight (1 - d), m_t is 1 until the reference motion ends (truncation, where the
    detached value ``v`` bootstraps), and the differentiable ``v_final`` bootstraps beyond the horizon.
    """
    H = r.shape[0]
    ones = torch.ones_like(r[:1])
    c = torch.cumprod(torch.cat((ones, 1.0 - d)), dim=0)
    m = torch.cumprod(torch.cat((ones, 1.0 - tr)), dim=0)
    discount = gamma ** torch.arange(H, device=r.device, dtype=r.dtype)
    J = (discount[:, None] * m[:-1] * (c[:-1] * r + gamma * c[1:] * tr * v[1:])).sum(0)
    J = J + gamma**H * c[-1] * m[-1] * v_final
    return -J.mean() / H, J, c, m


def lambda_targets(r, d, tr, v, gamma, lam):
    """Eq. 3: lambda-returns of the imagined transitions, cut at truncation (tensors (H, n), values (H + 1, n))."""
    ret = v[-1]
    targets = []
    for t in range(r.shape[0] - 1, -1, -1):
        nxt = tr[t] * v[t + 1] + (1.0 - tr[t]) * ((1.0 - lam) * v[t + 1] + lam * ret)
        ret = r[t] + gamma * (1.0 - d[t]) * nxt
        targets.append(ret)
    return torch.stack(targets[::-1])


class SuperMimic:
    def __init__(
        self,
        policy,
        task,
        num_envs: int,
        device: str,
        num_actions: int,
        # world model
        wm_hidden_dims: tuple[int, ...] = (2048, 2048, 2048),
        wm_learning_rate: float = 2e-4,
        wm_window: int = 8,
        wm_batches_per_update: int = 10,
        wm_batch_size: int = 2048,
        wm_max_grad_norm: float = 1.0,
        # imagination / actor
        horizon: int = 32,
        imagination_batch_size: int = 4096,
        actor_micro_batch_size: int = 0,
        actor_checkpointing: bool = False,
        actor_learning_rate: float = 1e-4,
        actor_betas: tuple[float, float] = (0.9, 0.95),
        actor_max_grad_norm: float = 1.0,
        actor_max_grad_value: float = 0.1,
        gamma: float = 0.99,
        lam: float = 0.95,
        # critic
        critic_hidden_dims: tuple[int, ...] = (256, 128),
        critic_learning_rate: float = 2e-4,
        critic_max_grad_norm: float = 1.0,
        critic_epochs: int = 5,
        critic_mini_batches: int = 4,
        # data
        replay_capacity: int = 1000,
        history_length: int = 3,
        lr_scale_with_world_size: bool = True,
        compile: bool = True,
        cuda_graph: bool = False,
        **kwargs,
    ):
        if kwargs:
            print("SuperMimic.__init__ got unexpected arguments, which will be ignored: " + str(list(kwargs)))
        self.policy = policy
        self.task = task
        self.device = device
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.wm_window = wm_window
        self.wm_batches_per_update = wm_batches_per_update
        self.wm_batch_size = wm_batch_size
        self.wm_max_grad_norm = wm_max_grad_norm
        self.imagination_batch_size = imagination_batch_size
        # Imagined rollouts are processed in micro-batches whose actor gradients accumulate into one update.
        self.actor_micro_batch_size = actor_micro_batch_size or imagination_batch_size
        if not 0 < self.actor_micro_batch_size <= imagination_batch_size:
            raise ValueError("actor_micro_batch_size must be between 1 and imagination_batch_size")
        if (imagination_batch_size * horizon) % critic_mini_batches:
            raise ValueError("imagined transitions must divide evenly into critic minibatches")
        self.actor_checkpointing = actor_checkpointing
        self.actor_max_grad_norm = actor_max_grad_norm
        self.actor_max_grad_value = actor_max_grad_value
        self.critic_max_grad_norm = critic_max_grad_norm
        self.critic_epochs = critic_epochs
        self.critic_mini_batches = critic_mini_batches
        self.history_length = history_length

        spec = task.spec
        self.world_model = DynamicsModel(
            spec, spec.in_scale.numel(), num_actions, task.xi_dim, spec.out_scale.numel(), wm_hidden_dims
        ).to(device)
        self.critic = ValueMLP(task.critic_dim, critic_hidden_dims).to(device)
        for m in (self.world_model, self.critic, self.policy):
            broadcast_module(m)

        # Paper: learning rates grow with sqrt(number of GPUs), every GPU keeping its own environments and buffer.
        scale = _world_size() ** 0.5 if lr_scale_with_world_size else 1.0
        self.base_lrs = {"wm": wm_learning_rate * scale, "actor": actor_learning_rate * scale, "critic": critic_learning_rate * scale}
        fused = {"fused": True} if torch.device(device).type == "cuda" else {}
        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=self.base_lrs["wm"], **fused)
        self.actor_optimizer = torch.optim.Adam(policy.actor_parameters, lr=self.base_lrs["actor"], betas=tuple(actor_betas), **fused)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.base_lrs["critic"], **fused)
        self.actor_params = [p for p in policy.actor_parameters if p.requires_grad]
        self._average = {
            "wm": GradientAverager(self.world_model.parameters()),
            "actor": GradientAverager(self.actor_params),
            "critic": GradientAverager(self.critic.parameters()),
        }

        self.buffer = ReplayBuffer(num_envs, replay_capacity, task.state_dim, num_actions, ["frame", "mode", "roff"], device)

        # Hundreds of small kernels per imagined step (forward kinematics, observation and reward heads): fuse them.
        def maybe_compile(fn):
            return torch.compile(fn, dynamic=False) if compile else fn

        self._wm_step = maybe_compile(self._wm_step_impl)
        self._model_step = maybe_compile(self._model_step_impl)
        self._critic_input = maybe_compile(self._critic_input_impl)
        self._actor_step = maybe_compile(self._actor_step_impl)
        self._critic_loss = maybe_compile(self._critic_loss_impl)
        self._objective = maybe_compile(actor_objective)
        self._targets = maybe_compile(lambda_targets)
        # Even fused, the imagination is launch bound (thousands of kernels per update): replay it as one CUDA graph.
        self._actor_pass = GraphedFunction(self._actor_pass_impl, enabled=cuda_graph and torch.device(device).type == "cuda")

    # ------------------------------------------------------------------ helpers

    def drain_communication_ms(self) -> dict[str, float]:
        return {f"allreduce_ms/{k}": avg.drain_ms() for k, avg in self._average.items()}

    def set_lr_ratio(self, ratio: float) -> None:
        for opt, key in ((self.wm_optimizer, "wm"), (self.actor_optimizer, "actor"), (self.critic_optimizer, "critic")):
            for g in opt.param_groups:
                g["lr"] = self.base_lrs[key] * ratio

    def _actor(self, prop, task_obs, act, mode, mapping):
        return self.policy.actor_mean_from_parts(prop, task_obs, act, mode, mapping)

    def _actor_bf16(self, prop, task_obs, act, mode, mapping):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=prop.is_cuda):
            return self._actor(prop, task_obs, act, mode, mapping).float()

    def _actor_step_impl(self, prop, task_obs, act, mode, mapping):
        if self.actor_checkpointing:
            return checkpoint(self._actor_bf16, prop, task_obs, act, mode, mapping, use_reentrant=False)
        return self._actor_bf16(prop, task_obs, act, mode, mapping)

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        """Deterministic training-mode action from an environment observation dict."""
        return self._actor(obs["policy"], obs["policy_task"], obs["action"], obs["mode"], obs["mode_mapping"])

    # ------------------------------------------------------------------ world model

    def _wm_step_impl(self, s, a, xi_norm, s_true):
        s_next = self.world_model(s, a, xi_norm)
        err = self.task.spec.error(s_next, s_true)
        return s_next, (err * err).sum(-1).mean()

    def update_world_model(self) -> dict[str, float]:
        task = self.task
        stats = torch.zeros(4, device=self.device)
        for _ in range(self.wm_batches_per_update):
            w = self.buffer.sample_windows(self.wm_batch_size, self.wm_window)
            xi = task.xi_norm[w.env_ids]
            s = w.states[0]
            loss = 0.0
            for j in range(self.wm_window):
                s, sq = self._wm_step(s, w.actions_exec[j], xi, w.states[j + 1])
                loss = loss + sq
                if j == 0:
                    stats[1] += sq.detach() / task.spec.err_scale.numel()
            stats[2] += sq.detach() / task.spec.err_scale.numel()
            loss = loss / self.wm_window
            self.wm_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._average["wm"]()
            stats[3] += torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.wm_max_grad_norm, foreach=True)
            self.wm_optimizer.step()
            stats[0] += loss.detach()
        stats = (stats / self.wm_batches_per_update).tolist()
        return {"wm_loss": stats[0], "wm_nse_1step": stats[1], "wm_nse_kstep": stats[2], "wm_grad_norm": stats[3]}

    # ------------------------------------------------------------------ imagination

    def _model_step_impl(self, s, a, prev_a, frame, roff, mode_ids, xi_norm, joint_offset, com_pos_b):
        """One imagined step after the actor: dynamics, reward / termination and the next observations.

        The critic input of the next state is computed without gradient: only the value beyond the horizon is
        differentiated (``_critic_input`` of the final state), bootstraps at truncation are constants.
        """
        task = self.task
        ctx = {"xi_norm": xi_norm, "joint_offset": joint_offset, "com_pos_b": com_pos_b}
        s_next = self.world_model(s, a, xi_norm)
        bodies = task.bodies(s_next, ctx)
        tr = task.transition(s_next, bodies, a, prev_a, frame)
        frame_next = task.next_frame(frame)
        with torch.no_grad():
            critic_in = task.critic_features(s_next, bodies, frame_next, mode_ids, ctx)
        out = {
            "reward": tr["reward"],
            "done": tr["done"],
            "truncated": tr["truncated"],
            "terms": torch.stack([tr["terms"][k] for k in task.reward_term_names]),
            "prop": task.prop_obs(s_next, ctx, noise=True),
            "task_obs": task.policy_task_obs(bodies, frame_next, roff, noise=True),
            "critic_in": critic_in,
        }
        return s_next, frame_next, out

    def _critic_input_impl(self, s, frame, mode_ids, xi_norm, joint_offset, com_pos_b):
        ctx = {"xi_norm": xi_norm, "joint_offset": joint_offset, "com_pos_b": com_pos_b}
        return self.task.critic_features(s, self.task.bodies(s, ctx), frame, mode_ids, ctx)

    def imagine(self, start: StartBatch) -> dict:
        """Roll the policy through the world model; returns per-step tensors with gradients to the actor."""
        task, H = self.task, self.horizon
        ctx = task.context(start.env_ids)
        args = (ctx["xi_norm"], ctx["joint_offset"], ctx["com_pos_b"])
        frame, mode_ids, roff = start.info["frame"], start.info["mode"], start.info["roff"]
        mode, mapping = task.mode_inputs(mode_ids)
        prop_hist = torch.stack([task.prop_obs(start.hist_states[i], ctx, noise=True) for i in range(self.history_length)], 1)
        act_hist = start.hist_actions.transpose(0, 1)

        s = start.state
        bodies = task.bodies(s, ctx)
        task_obs = task.policy_task_obs(bodies, frame, roff, noise=True)
        with torch.no_grad():
            critic_in = [task.critic_features(s, bodies, frame, mode_ids, ctx)]
        rewards, dones, truncs, actions, terms = [], [], [], [], []
        for _ in range(H):
            a = self._actor_step(prop_hist, task_obs, act_hist, mode, mapping)
            s, frame, out = self._model_step(s, a, act_hist[:, -1], frame, roff, mode_ids, *args)
            rewards.append(out["reward"])
            dones.append(out["done"])
            truncs.append(out["truncated"])
            actions.append(a)
            terms.append(out["terms"].detach())
            prop_hist = torch.cat([prop_hist[:, 1:], out["prop"][:, None]], dim=1)
            act_hist = torch.cat([act_hist[:, 1:], a[:, None]], dim=1)
            task_obs = out["task_obs"]
            critic_in.append(out["critic_in"])
        return {
            "rewards": torch.stack(rewards),  # (H, n) r_{t+1}
            "dones": torch.stack(dones),  # (H, n) d_{t+1}
            "truncated": torch.stack(truncs).float(),  # (H, n) transition t truncates the episode
            "actions": torch.stack(actions),
            "critic_in": torch.stack(critic_in),  # (H + 1, n, C), no gradient
            "critic_in_final": self._critic_input(s, frame, mode_ids, *args),  # (n, C), differentiable
            "terms": torch.stack(terms).mean(dim=2),  # (H, num_terms)
        }

    # ------------------------------------------------------------------ policy and critic

    def _critic_loss_impl(self, x, y, w):
        err = self.critic(x) - y
        return (w * err * err).sum() / w.sum().clamp_min(1.0)

    def _actor_pass_impl(self, state, env_ids, info, hist_states, hist_actions) -> dict:
        """Actor gradient of the imagined return through the frozen world model and critic, plus what the critic
        fit needs. Micro-batches accumulate their share of the batch-mean gradient (one full-batch step)."""
        start = StartBatch(state, env_ids, info, hist_states, hist_actions)
        n = state.shape[0]
        self.actor_optimizer.zero_grad(set_to_none=False)
        chunks = {k: [] for k in ("critic_in", "rewards", "dones", "truncated", "alive")}
        stats = torch.zeros(6, device=self.device)
        terms = 0.0
        for lo in range(0, n, self.actor_micro_batch_size):
            hi = min(n, lo + self.actor_micro_batch_size)
            ro = self.imagine(start.slice(lo, hi))
            loss, J, c, m = self._objective(
                ro["rewards"], ro["dones"], ro["truncated"], self.critic(ro["critic_in"]),
                self.critic(ro["critic_in_final"]), self.gamma,
            )  # fmt: skip
            share = (hi - lo) / n
            (loss * share).backward()
            with torch.no_grad():
                stats += share * torch.stack(
                    [J.mean(), (c[:-1] * ro["rewards"]).sum(0).mean(), c[-1].mean(), ro["dones"].mean(),
                     ro["actions"].abs().mean(), 1.0 - m[-1].mean()]
                )  # fmt: skip
                terms = terms + share * ro["terms"].mean(0)
            for k in ("critic_in", "rewards", "dones", "truncated"):
                chunks[k].append(ro[k].detach())
            chunks["alive"].append(m[:-1].detach())
            del ro, loss, J, c, m
        out = {k: torch.cat(v, dim=1) for k, v in chunks.items()}
        out.update(stats=stats, terms=terms)
        return out

    def update_policy(self) -> dict[str, float]:
        H, n = self.horizon, self.imagination_batch_size
        start = self.buffer.sample_starts(n, self.history_length)

        # ---- actor
        self.world_model.requires_grad_(False)
        self.critic.requires_grad_(False)
        out = self._actor_pass(start.state, start.env_ids, start.info, start.hist_states, start.hist_actions)
        self._average["actor"]()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_params, self.actor_max_grad_norm, foreach=True)
        torch.nn.utils.clip_grad_value_(self.actor_params, self.actor_max_grad_value, foreach=True)
        self.actor_optimizer.step()
        self.world_model.requires_grad_(True)
        self.critic.requires_grad_(True)

        # ---- critic: lambda-returns of the whole imagined batch (targets fixed while fitting)
        with torch.no_grad():
            x, r, d, tr, alive = (out[k] for k in ("critic_in", "rewards", "dones", "truncated", "alive"))
            self.critic.norm.update(x[:-1])
            values = self.critic(x)
            targets = self._targets(r, d, tr, values, self.gamma, self.lam)
            X, Y, W = x[:-1].reshape(H * n, -1), targets.reshape(-1), alive.reshape(-1)
        N = Y.shape[0]
        mb = N // self.critic_mini_batches
        for _ in range(self.critic_epochs):
            perm = torch.randperm(N, device=self.device)
            for i in range(self.critic_mini_batches):
                ids = perm[i * mb : (i + 1) * mb]
                critic_loss = self._critic_loss(X[ids], Y[ids], W[ids])
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                self._average["critic"]()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_max_grad_norm, foreach=True)
                self.critic_optimizer.step()

        # One host transfer for all diagnostics.
        keys = ("imag_objective", "imag_return", "imag_survival_H", "imag_done_mean", "action_abs_mean",
                "imag_truncated_frac", "actor_grad_norm", "critic_loss", "critic_value_mean", "critic_target_mean")  # fmt: skip
        scalars = torch.cat(
            [out["stats"], torch.stack([grad_norm, critic_loss.detach(), values[:-1].mean(), targets.mean()]), out["terms"]]
        ).tolist()
        metrics = dict(zip(keys, scalars))
        metrics.update({f"imag_reward/{k}": v for k, v in zip(self.task.reward_term_names, scalars[len(keys) :])})
        return metrics

    # ------------------------------------------------------------------ checkpoint

    def state_dict(self) -> dict:
        return {
            "world_model": self.world_model.state_dict(),
            "critic": self.critic.state_dict(),
            "wm_optimizer": self.wm_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict, load_optimizer: bool = True) -> None:
        self.world_model.load_state_dict(state["world_model"])
        self.critic.load_state_dict(state["critic"])
        if load_optimizer:
            self.wm_optimizer.load_state_dict(state["wm_optimizer"])
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
