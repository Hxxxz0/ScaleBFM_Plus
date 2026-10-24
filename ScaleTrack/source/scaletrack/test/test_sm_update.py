"""SuperMimic actor objective, lambda-returns and micro-batched updates (no simulator required)."""

import types
import unittest

import torch

from my_rsl_rl.algorithms.supermimic import GradientAverager, SuperMimic, actor_objective, lambda_targets
from my_rsl_rl.networks.world_model import ValueMLP
from my_rsl_rl.storage.replay_buffer import StartBatch


def loop_objective(r, d, tr, v, v_final, gamma):
    """Eq. 2 written step by step: survival c, truncation mask m, bootstrap at truncation and beyond H."""
    c, m = torch.ones_like(r[0]), torch.ones_like(r[0])
    J, discount = torch.zeros_like(c), 1.0
    for t in range(len(r)):
        J = J + discount * c * m * r[t]
        c = c * (1 - d[t])
        J = J + discount * gamma * c * m * tr[t] * v[t + 1]
        m = m * (1 - tr[t])
        discount *= gamma
    J = J + discount * c * m * v_final
    return -J.mean() / len(r)


def imagine(self, start):
    """Toy differentiable rollout with terminations and clip ends, standing in for the world model."""
    s = start.state
    states, rewards, dones, truncs, actions = [s], [], [], [], []
    for t in range(self.horizon):
        a = self.policy(s)
        s = torch.tanh(self.world_model(torch.cat((s, a), -1)))
        states.append(s.detach())
        actions.append(a)
        rewards.append(torch.exp(-s.square().sum(-1)))
        dones.append(torch.sigmoid(s[:, 0] - 1.5))
        truncs.append((start.info["clip_end"] == t).float())
    r = torch.stack(rewards)
    return dict(critic_in=torch.stack(states), critic_in_final=s, rewards=r, dones=torch.stack(dones),
                truncated=torch.stack(truncs), actions=torch.stack(actions), terms=r.detach()[..., None].mean(1))


def toy_algorithm(micro_batch):
    torch.manual_seed(3)
    a = SuperMimic.__new__(SuperMimic)
    a.device = "cpu"
    a.horizon, a.gamma, a.lam = 8, 0.99, 0.95
    a.imagination_batch_size, a.actor_micro_batch_size, a.history_length = 13, micro_batch, 3
    a.actor_max_grad_norm, a.actor_max_grad_value, a.critic_max_grad_norm = 1.0, 0.1, 1.0
    a.critic_epochs, a.critic_mini_batches = 2, 4
    a.policy = torch.nn.Linear(6, 3)
    a.world_model = torch.nn.Linear(9, 6)
    a.critic = ValueMLP(6, (12, 8))
    a.actor_params = list(a.policy.parameters())
    a.actor_optimizer = torch.optim.Adam(a.actor_params, lr=1e-4)
    a.critic_optimizer = torch.optim.Adam(a.critic.parameters(), lr=2e-4)
    a._average = {k: GradientAverager([]) for k in ("wm", "actor", "critic")}
    a._objective, a._targets = actor_objective, lambda_targets
    a._critic_loss = types.MethodType(SuperMimic._critic_loss_impl, a)
    a._actor_pass = types.MethodType(SuperMimic._actor_pass_impl, a)
    a.task = types.SimpleNamespace(reward_term_names=("tracking",))
    start = StartBatch(torch.randn(13, 6), torch.arange(13), {"clip_end": torch.arange(13) % 10},
                       torch.randn(3, 13, 6), torch.randn(3, 13, 3))
    a.buffer = types.SimpleNamespace(sample_starts=lambda n, h: start)
    a.imagine = types.MethodType(imagine, a)
    return a


class UpdateTests(unittest.TestCase):
    def test_objective_and_gradients_match_the_stepwise_definition(self):
        torch.manual_seed(7)
        r = torch.randn(8, 11, requires_grad=True)
        d = torch.rand(8, 11)
        d[2, :3] = 1.0
        d.requires_grad_()
        tr = (torch.rand(8, 11) < 0.15).float()
        v = torch.randn(9, 11)
        v_final = torch.randn(11, requires_grad=True)
        expected = loop_objective(r, d, tr, v, v_final, 0.99)
        actual = actor_objective(r, d, tr, v, v_final, 0.99)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for a, b in zip(torch.autograd.grad(actual, (r, d, v_final)), torch.autograd.grad(expected, (r, d, v_final))):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_lambda_returns(self):
        torch.manual_seed(1)
        r, d, v = torch.randn(8, 5), torch.rand(8, 5), torch.randn(9, 5)
        tr = (torch.rand(8, 5) < 0.2).float()
        expected, ret = torch.empty_like(r), v[-1]
        for t in range(7, -1, -1):
            nxt = tr[t] * v[t + 1] + (1 - tr[t]) * (0.05 * v[t + 1] + 0.95 * ret)
            ret = r[t] + 0.99 * (1 - d[t]) * nxt
            expected[t] = ret
        torch.testing.assert_close(lambda_targets(r, d, tr, v, 0.99, 0.95), expected)

    def test_uneven_micro_batches_equal_one_full_batch_update(self):
        full, micro = toy_algorithm(13), toy_algorithm(5)
        torch.manual_seed(99)
        m_full = full.update_policy()
        torch.manual_seed(99)
        m_micro = micro.update_policy()
        for name in ("policy", "critic"):
            for a, b in zip(getattr(full, name).state_dict().values(), getattr(micro, name).state_dict().values()):
                torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-6)
        for k in m_full:
            self.assertAlmostEqual(m_full[k], m_micro[k], delta=2e-5, msg=k)


if __name__ == "__main__":
    unittest.main()
