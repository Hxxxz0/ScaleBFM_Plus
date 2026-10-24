"""Networks of SuperMimic-style training: learned dynamics, critic and a running input normaliser."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


def mlp(in_dim: int, hidden: tuple[int, ...] | list[int], out_dim: int, activation: type[nn.Module]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(in_dim, h), activation()]
        in_dim = h
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class DynamicsModel(nn.Module):
    """Deterministic world model s_{t+1} = integrate(s_t, f(features(s_t), a_t, xi)).

    The state parameterisation (``features`` / ``integrate`` / ``error``) is supplied by the task; the network output
    layer starts at zero, i.e. at constant-velocity extrapolation.
    """

    def __init__(self, spec, feature_dim: int, action_dim: int, xi_dim: int, delta_dim: int, hidden=(2048, 2048, 2048)):
        super().__init__()
        self.spec = spec
        self.net = mlp(feature_dim + action_dim + xi_dim, tuple(hidden), delta_dim, nn.SiLU)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s: torch.Tensor, action: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.spec.features(s), action, xi_norm], dim=-1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=s.is_cuda):
            delta = self.net(x)
        return self.spec.integrate(s, delta.float())


class RunningNorm(nn.Module):
    """Running mean / variance normaliser, synchronised across data-parallel ranks."""

    def __init__(self, dim: int, eps: float = 1e-2, max_count: float = 1e9):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.eps = eps
        self.max_count = max_count

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if self.count >= self.max_count:
            return
        x = x.reshape(-1, x.shape[-1]).float()
        stats = torch.cat([x.sum(0), (x * x).sum(0), x.new_tensor([x.shape[0]])])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats)
        d = x.shape[-1]
        n = stats[-1]
        mean_b = stats[:d] / n
        var_b = (stats[d : 2 * d] / n - mean_b * mean_b).clamp_min(0.0)
        total = self.count + n
        delta = mean_b - self.mean
        # In place: CUDA graphs captured on the imagination read these buffers.
        self.mean += delta * n / total
        self.var.copy_((self.var * self.count + var_b * n + delta * delta * self.count * n / total) / total)
        self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps)


class ValueMLP(nn.Module):
    def __init__(self, input_dim: int, hidden=(256, 128)):
        super().__init__()
        self.norm = RunningNorm(input_dim)
        self.net = mlp(input_dim, tuple(hidden), 1, nn.ELU)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x)).squeeze(-1)
