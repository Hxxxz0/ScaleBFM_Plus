"""GPU replay buffer of simulator transitions for world-model fitting and imagination starts.

Laid out as [env, time]: every environment writes into its own ring of ``capacity`` rows. A row holds the state
``s_t`` at the start of a control step, the clean and executed actions taken in it and the task's integer
bookkeeping of the step (e.g. the tracked reference frame). Its successor ``s_{t+1}`` is the next row of the same
episode. When an episode ends, an extra *terminal* row stores the final state (it has no successor), so that the
last transition of every episode can be learnt as well.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WindowBatch:
    states: torch.Tensor  # (k + 1, n, S)
    actions_exec: torch.Tensor  # (k, n, A)
    env_ids: torch.Tensor  # (n,)


@dataclass
class StartBatch:
    state: torch.Tensor  # (n, S)
    env_ids: torch.Tensor  # (n,)
    info: dict[str, torch.Tensor]  # integer bookkeeping of the start step, (n,) each
    hist_states: torch.Tensor  # (hist, n, S) states s_{t-hist+1} .. s_t (back-filled at episode start)
    hist_actions: torch.Tensor  # (hist, n, A) clean actions a_{t-hist} .. a_{t-1} (zero before episode start)

    def slice(self, lo: int, hi: int) -> StartBatch:
        return StartBatch(
            self.state[lo:hi],
            self.env_ids[lo:hi],
            {k: v[lo:hi] for k, v in self.info.items()},
            self.hist_states[:, lo:hi],
            self.hist_actions[:, lo:hi],
        )


class ReplayBuffer:
    def __init__(self, num_envs: int, capacity: int, state_dim: int, action_dim: int, info_keys: list[str], device):
        E, T = num_envs, capacity
        self.E, self.T = E, T
        self.device = torch.device(device)
        self.states = torch.zeros(E, T, state_dim, device=self.device)
        self.actions = torch.zeros(E, T, action_dim, device=self.device)
        self.actions_exec = torch.zeros(E, T, action_dim, device=self.device)
        self.info = {k: torch.zeros(E, T, dtype=torch.long, device=self.device) for k in info_keys}
        self.ep_id = torch.full((E, T), -1, dtype=torch.long, device=self.device)
        self.valid = torch.zeros(E, T, dtype=torch.bool, device=self.device)
        self.has_next = torch.zeros(E, T, dtype=torch.bool, device=self.device)
        self.ep_start = torch.zeros(E, T, dtype=torch.bool, device=self.device)
        self.ptr = torch.zeros(E, dtype=torch.long, device=self.device)
        self.ep_counter = torch.zeros(E, dtype=torch.long, device=self.device)
        self.rows_written = 0
        self._all_envs = torch.arange(E, device=self.device)
        self._window_candidates = None
        self._start_candidates = None

    # ------------------------------------------------------------------ writes

    @torch.no_grad()
    def add(self, states, actions, actions_exec, info: dict[str, torch.Tensor], ep_start: torch.Tensor) -> None:
        """Append one transition row for every env."""
        e, p = self._all_envs, self.ptr
        self.states[e, p] = states
        self.actions[e, p] = actions
        self.actions_exec[e, p] = actions_exec
        for k, v in info.items():
            self.info[k][e, p] = v
        self.ep_id[e, p] = self.ep_counter
        self.valid[e, p] = True
        self.has_next[e, p] = True
        self.ep_start[e, p] = ep_start
        self.ptr = (p + 1) % self.T
        self.rows_written += self.E

    @torch.no_grad()
    def add_terminal(self, env_ids: torch.Tensor, states: torch.Tensor) -> None:
        """Append the final state of the episodes ending in ``env_ids`` and start new episodes there."""
        if env_ids.numel() == 0:
            return
        p = self.ptr[env_ids]
        self.states[env_ids, p] = states
        self.actions[env_ids, p] = 0.0
        self.actions_exec[env_ids, p] = 0.0
        self.ep_id[env_ids, p] = self.ep_counter[env_ids]
        self.valid[env_ids, p] = True
        self.has_next[env_ids, p] = False
        self.ep_start[env_ids, p] = False
        self.ptr[env_ids] = (p + 1) % self.T
        self.ep_counter[env_ids] += 1

    @torch.no_grad()
    def end_all_episodes(self) -> None:
        """Cut every running episode (e.g. after an evaluation reset all environments) without a terminal row."""
        last = (self.ptr - 1) % self.T
        self.has_next[self._all_envs, last] = False
        self.ep_counter += 1

    # ------------------------------------------------------------------ sampling

    def _rolled(self, x: torch.Tensor, j: int) -> torch.Tensor:
        return torch.roll(x, shifts=-j, dims=1) if j else x

    def _age(self) -> torch.Tensor:
        """Number of rows written after each row (0 = newest)."""
        t = torch.arange(self.T, device=self.device)[None, :]
        return (self.ptr[:, None] - t - 1) % self.T

    @torch.no_grad()
    def prepare(self, window: int, history: int) -> None:
        """Index the rows that can start a k-step window / an imagined rollout (one host sync per update)."""
        age = self._age()
        ok = self.valid & self.has_next & (age >= window)
        for j in range(1, window + 1):
            same = self._rolled(self.valid, j) & (self._rolled(self.ep_id, j) == self.ep_id)
            if j < window:
                same &= self._rolled(self.has_next, j)
            ok &= same
        self._window_candidates = ok.reshape(-1).nonzero().squeeze(-1)
        starts = self.valid & self.has_next & (age < self.T - history)
        self._start_candidates = starts.reshape(-1).nonzero().squeeze(-1)

    def num_window_candidates(self) -> int:
        return 0 if self._window_candidates is None else int(self._window_candidates.numel())

    def _pick(self, candidates: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = candidates[torch.randint(0, candidates.numel(), (n,), device=self.device)]
        return idx // self.T, idx % self.T

    @torch.no_grad()
    def sample_windows(self, n: int, window: int) -> WindowBatch:
        e, t = self._pick(self._window_candidates, n)
        tt = (t[None, :] + torch.arange(window + 1, device=self.device)[:, None]) % self.T
        ee = e[None, :].expand_as(tt)
        return WindowBatch(states=self.states[ee, tt], actions_exec=self.actions_exec[ee[:-1], tt[:-1]], env_ids=e)

    @torch.no_grad()
    def sample_starts(self, n: int, history: int) -> StartBatch:
        e, t = self._pick(self._start_candidates, n)
        tt = (t[None, :] + torch.arange(history, device=self.device)[:, None] - (history - 1)) % self.T
        ee = e[None, :].expand_as(tt)
        same = (self.ep_id[ee, tt] == self.ep_id[e, t][None, :]) & self.valid[ee, tt]
        # Rows before the episode start are back-filled with the episode's first row (IsaacLab history buffers
        # are filled with the first observation after a reset).
        first = same.float().argmax(0)
        tt = torch.where(same, tt, tt[first, torch.arange(n, device=self.device)][None, :])
        prev = (tt - 1) % self.T
        has_prev = same & ~self.ep_start[ee, tt]
        hist_actions = torch.where(has_prev[..., None], self.actions[ee, prev], torch.zeros_like(self.actions[ee, prev]))
        return StartBatch(
            state=self.states[e, t],
            env_ids=e,
            info={k: v[e, t] for k, v in self.info.items()},
            hist_states=self.states[ee, tt],
            hist_actions=hist_actions,
        )
