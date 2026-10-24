"""Compare full PPO minibatches and FP32 gradient accumulation, including Adam state and adaptive LR."""

import copy
import unittest

import torch
from tensordict import TensorDict

from my_rsl_rl.algorithms.ppo import PPO


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Sequential(torch.nn.Linear(5, 12), torch.nn.Tanh(), torch.nn.Linear(12, 3))
        self.critic = torch.nn.Sequential(torch.nn.Linear(5, 12), torch.nn.Tanh(), torch.nn.Linear(12, 1))
        self.std = torch.nn.Parameter(torch.full((3,), 0.8))

    @property
    def actor_parameters(self):
        return list(self.actor.parameters()) + [self.std]

    @property
    def critic_parameters(self):
        return list(self.critic.parameters())

    def act(self, obs):
        self.action_mean = self.actor(obs["policy"])
        self.action_std = self.std.expand_as(self.action_mean)
        self.distribution = torch.distributions.Normal(self.action_mean, self.action_std)
        return self.distribution.sample()

    def evaluate(self, obs):
        return self.critic(obs["policy"])

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(-1)

    @property
    def entropy(self):
        return self.distribution.entropy().sum(-1)


class Storage:
    def __init__(self, batch):
        self.batch = batch
        self.cleared = False

    def mini_batch_generator(self, num_batches, num_epochs):
        for _ in range(num_epochs):
            for _ in range(num_batches):
                yield self.batch

    def clear(self):
        self.cleared = True


class MicrobatchTests(unittest.TestCase):
    def test_updates_and_adaptive_lr(self):
        for clipped in (True, False):
            for old_mu_shift in (0.0, 0.5):
                with self.subTest(clipped=clipped, old_mu_shift=old_mu_shift):
                    torch.manual_seed(41)
                    p = Policy()
                    obs = TensorDict({"policy": torch.randn(17, 5)}, batch_size=[17])
                    with torch.no_grad():
                        actions = p.act(obs)
                        old_log_prob = p.get_actions_log_prob(actions).unsqueeze(-1)
                        values = p.evaluate(obs)
                        mu, sigma = p.action_mean.clone() + old_mu_shift, p.action_std.clone()
                    batch = (obs, actions, values, torch.randn(17, 1), torch.randn(17, 1),
                             old_log_prob, mu, sigma)
                    full = PPO(copy.deepcopy(p), num_learning_epochs=2, num_mini_batches=3,
                               use_clipped_value_loss=clipped, max_grad_norm=0.1,
                               actor_learning_rate=1e-4, critic_learning_rate=2e-4)
                    micro = PPO(copy.deepcopy(p), num_learning_epochs=2, num_mini_batches=3,
                                use_clipped_value_loss=clipped, max_grad_norm=0.1,
                                actor_learning_rate=1e-4, critic_learning_rate=2e-4,
                                forward_micro_batch_size=5)
                    full.storage, micro.storage = Storage(batch), Storage(batch)
                    before, after = full.update(), micro.update()
                    for a, b in zip(full.policy.parameters(), micro.policy.parameters()):
                        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
                    for opt in ("actor_optimizer", "critic_optimizer"):
                        a, b = getattr(full, opt).state_dict(), getattr(micro, opt).state_dict()
                        self.assertEqual(a["param_groups"], b["param_groups"])
                        for key in a["state"]:
                            for attr, value in a["state"][key].items():
                                torch.testing.assert_close(value, b["state"][key][attr], rtol=1e-5, atol=1e-7)
                            self.assertEqual(a["state"][key]["step"].item(), 6)
                    for key in before:
                        self.assertAlmostEqual(before[key], after[key], delta=1e-6)
                    self.assertTrue(full.storage.cleared and micro.storage.cleared)


if __name__ == "__main__":
    unittest.main()
