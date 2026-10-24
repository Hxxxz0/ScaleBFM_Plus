"""Failure retry safety, sampling distribution and training budget regressions."""

import importlib.util
import sys
import torch
import types
import unittest
from pathlib import Path

from sm_offline import load

reset = load("reset_sampling")
budget_path = Path(__file__).resolve().parents[3] / "source/my_rsl_rl/my_rsl_rl/runners/training_budget.py"
spec = importlib.util.spec_from_file_location("sm_budget", budget_path)
budget_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = budget_module
spec.loader.exec_module(budget_module)
TrainingBudget = budget_module.TrainingBudget
cosine_lr_ratio = budget_module.cosine_lr_ratio


def library(device="cpu"):
    lengths = torch.tensor([80, 50, 2], device=device)
    return types.SimpleNamespace(
        device=device,
        lengths=lengths,
        starts=torch.tensor([0, 80, 130], device=device),
        frame_motion=torch.repeat_interleave(torch.arange(3, device=device), lengths),
        motion_group=torch.tensor([0, 1, 1], device=device),
        group_names=["bones", "lafan"],
    )


def context(n, device="cpu", **overrides):
    values = dict(
        motion_ids=torch.ones(n, dtype=torch.long, device=device),
        frames=torch.full((n,), 30, device=device),
        episode_steps=torch.full((n,), 20, device=device),
        failed=torch.ones(n, dtype=torch.bool, device=device),
        truncated=torch.zeros(n, dtype=torch.bool, device=device),
    )
    values.update(overrides)
    return reset.ResetContext(**values)


class HybridResetTests(unittest.TestCase):
    def test_disabled_zero_probability_and_warmup_preserve_global_rng(self):
        for cfg, active in [
            (reset.FailureResetCfg(), True),
            (reset.FailureResetCfg(enabled=True, local_probability=0), True),
            (reset.FailureResetCfg(enabled=True), False),
        ]:
            sampler = reset.HybridResetSampler(cfg, 32, library())
            global_frames = torch.arange(32)
            rng = torch.random.get_rng_state().clone()
            actual = sampler.select(global_frames, torch.arange(32), context(32), active)
            self.assertTrue(torch.equal(actual, global_frames))
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_distribution_and_motion_boundaries(self):
        torch.manual_seed(41)
        n = 20000
        sampler = reset.HybridResetSampler(reset.FailureResetCfg(enabled=True), n, library())
        global_frames = torch.zeros(n, dtype=torch.long)
        frames = sampler.select(global_frames, torch.arange(n), context(n), True)
        local = frames != 0
        self.assertAlmostEqual(float(local.float().mean()), 0.5, delta=0.015)
        self.assertTrue(((frames[local] >= 98) & (frames[local] <= 106)).all())
        self.assertTrue((sampler.library.frame_motion[frames[local]] == 1).all())
        cfg = reset.FailureResetCfg(enabled=True, local_probability=1)
        sampler = reset.HybridResetSampler(cfg, 3, library())
        c = context(3, frames=torch.tensor([0, 500, 1]), motion_ids=torch.tensor([0, 1, 2]))
        actual = sampler.select(torch.zeros(3, dtype=torch.long), torch.arange(3), c, True)
        self.assertEqual(actual[0], 0)
        self.assertEqual(actual[1], 127)  # last start with two subsequent frames
        self.assertEqual(actual[2], 0)  # too short for two subsequent frames

    def test_timeout_early_failures_retry_cap_and_explicit_reset(self):
        cfg = reset.FailureResetCfg(enabled=True, local_probability=1)
        sampler = reset.HybridResetSampler(cfg, 4, library())
        c = context(
            4,
            truncated=torch.tensor([False, True, False, False]),
            episode_steps=torch.tensor([20, 20, 7, 20]),
            failed=torch.tensor([True, True, True, False]),
        )
        ids, global_frames = torch.arange(4), torch.zeros(4, dtype=torch.long)
        for _ in range(3):
            actual = sampler.select(global_frames, ids, c, True)
            self.assertTrue(actual[0] > 80)
            self.assertTrue(torch.equal(actual[1:], global_frames[1:]))
        self.assertTrue(torch.equal(sampler.select(global_frames, ids, c, True), global_frames))
        self.assertEqual(sampler.consecutive_local[0], 0)
        sampler.select(global_frames, ids, c, True)
        sampler.select(global_frames, ids, None, True)
        self.assertTrue((sampler.consecutive_local == 0).all())
        sampler.consecutive_local[:] = 2
        sampler.clear(torch.tensor([1, 3]))
        self.assertTrue(torch.equal(sampler.consecutive_local, torch.tensor([2, 0, 2, 0])))
        stats = sampler.drain_statistics()
        self.assertEqual(stats[0], 24)
        self.assertEqual(stats[3] + stats[4], stats[0])
        self.assertEqual(stats[10:].sum(), stats[0])
        self.assertTrue((sampler.drain_statistics() == 0).all())


class ReturnNormalizationTests(unittest.TestCase):
    R, GAMMA = 0.11, 0.99

    def ratios(self, rewards, horizons):
        r = torch.tensor([rewards], dtype=torch.float32)
        h = torch.tensor([horizons])
        return reset.normalized_returns_to_go(r, torch.tensor([len(rewards)]), h, self.GAMMA, self.R)[0]

    def test_perfect_tracking_scores_one_at_motion_end_and_time_limit(self):
        # Ends at the motion end after 7 steps, or at the time limit after 40 steps: no bias toward either.
        for length in (7, 40):
            ratio = self.ratios([self.R] * length, list(range(length, 0, -1)))
            torch.testing.assert_close(ratio, torch.ones(length))

    def test_failures_score_below_one_and_lowest_near_the_failure(self):
        ratio = self.ratios([self.R] * 10, list(range(100, 90, -1)))  # fails after 10 of 100 possible steps
        self.assertTrue((ratio < 0.5).all())
        self.assertTrue((ratio[1:] < ratio[:-1]).all())

    def test_padding_beyond_episode_length_is_zero(self):
        r = torch.full((2, 5), self.R)
        out = reset.normalized_returns_to_go(r, torch.tensor([5, 2]), torch.tensor([[5, 4, 3, 2, 1]] * 2), self.GAMMA, self.R)
        self.assertTrue((out[1, 2:] == 0).all())


class TrainingBudgetTests(unittest.TestCase):
    def test_exact_fixed_budget(self):
        b = TrainingBudget("updates", 10000, None)
        self.assertFalse(b.reached(9999, 1e12))
        self.assertTrue(b.reached(10000, 0))
        completed = 0
        for _ in range(40000):
            completed += 1
            if b.reached(completed, 0):
                break
        self.assertEqual(completed, 10000)

    def test_progress_covers_only_the_policy_phase(self):
        b = TrainingBudget("updates", 10300, None)
        self.assertEqual(b.policy_progress(0, 300, 0.0, None), 0.0)
        self.assertEqual(b.policy_progress(300, 300, 0.0, None), 0.0)
        self.assertAlmostEqual(b.policy_progress(5300, 300, 0.0, None), 0.5)
        self.assertEqual(b.policy_progress(10300, 300, 0.0, None), 1.0)
        h = TrainingBudget("hours", None, 2.0)
        self.assertEqual(h.policy_progress(50, 300, 100.0, None), 0.0)
        self.assertAlmostEqual(h.policy_progress(5000, 300, 600.0 + 3300.0, 600.0), 0.5)
        self.assertEqual(h.policy_progress(9000, 300, 9000.0, 600.0), 1.0)

    def test_cosine_schedule_holds_after_decay_fraction(self):
        self.assertEqual(cosine_lr_ratio(0.0, 0.03, 0.78), 1.0)
        self.assertAlmostEqual(cosine_lr_ratio(0.39, 0.03, 0.78), 0.515)
        self.assertAlmostEqual(cosine_lr_ratio(0.78, 0.03, 0.78), 0.03)
        self.assertAlmostEqual(cosine_lr_ratio(1.0, 0.03, 0.78), 0.03)
        ratios = [cosine_lr_ratio(p / 100, 0.03, 0.78) for p in range(101)]
        self.assertTrue(all(a >= b for a, b in zip(ratios, ratios[1:])))

    def test_invalid_budgets(self):
        self.assertFalse(TrainingBudget("hours", None, 2.6).reached(50000, 100))
        self.assertTrue(TrainingBudget("hours", None, 2.6).reached(1, 9361))
        for values in [
            ("updates", 0, None),
            ("updates", True, None),
            ("hours", None, None),
            ("hours", None, float("nan")),
            ("bad", 100, None),
        ]:
            with self.assertRaises(ValueError):
                TrainingBudget(*values)


if __name__ == "__main__":
    unittest.main()
