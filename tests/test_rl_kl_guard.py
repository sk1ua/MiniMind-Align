import copy
import unittest
from pathlib import Path

import torch

from align.rl_rules import (
    aggregate_reference_kl_values,
    finite_diagnostics,
    kl_backoff_multipliers,
    reference_kl_stats,
)
from trainer.train_grpo_lite import restore_training_state, snapshot_training_state


class RLKLGuardTest(unittest.TestCase):
    def test_guard_is_optional_by_default(self):
        source = (Path(__file__).resolve().parents[1] / "trainer" / "train_grpo_lite.py").read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--post-step-kl-target", type=float)', source)
        self.assertIn('kl_guard_enabled = args.post_step_kl_target is not None', source)
        self.assertIn('else:\n            optimizer.step()', source)

    def test_rejected_step_cannot_write_checkpoint(self):
        source = (Path(__file__).resolve().parents[1] / "trainer" / "train_grpo_lite.py").read_text(encoding="utf-8")
        self.assertLess(source.index('if optimizer_step_rejected:'), source.index('should_checkpoint ='))

    def test_guard_telemetry_schema_is_declared(self):
        source = (Path(__file__).resolve().parents[1] / "trainer" / "train_grpo_lite.py").read_text(encoding="utf-8")
        for field in (
            '"post_step_kl_target"',
            '"post_step_kl_mean"',
            '"post_step_kl_p95"',
            '"post_step_kl_max"',
            '"kl_guard_attempts"',
            '"kl_guard_backoff_count"',
            '"kl_guard_accepted_lr_multiplier"',
            '"optimizer_step_applied"',
            '"optimizer_step_rejected"',
        ):
            self.assertIn(field, source)

    def test_masked_reference_kl_stats_ignore_padding(self):
        new = torch.zeros(1, 3)
        ref = torch.tensor([[0.0, 0.0, 0.69314718056]])
        mask = torch.tensor([[1, 1, 0]], dtype=torch.float32)
        stats = reference_kl_stats(new, ref, mask)
        self.assertEqual(stats["token_count"], 2.0)
        self.assertAlmostEqual(stats["reference_kl_mean"], 0.0, places=6)

    def test_all_microbatches_are_aggregated(self):
        first = torch.tensor([0.0, 1.0])
        second = torch.tensor([3.0])
        stats = aggregate_reference_kl_values([first, second])
        self.assertEqual(stats["token_count"], 3.0)
        self.assertAlmostEqual(stats["reference_kl_mean"], 4.0 / 3.0, places=6)
        self.assertEqual(stats["reference_kl_max"], 3.0)

    def test_backoff_schedule_is_bounded(self):
        self.assertEqual(kl_backoff_multipliers(0.5, 3), [1.0, 0.5, 0.25, 0.125])
        with self.assertRaises(ValueError):
            kl_backoff_multipliers(1.0, 3)

    def test_policy_and_optimizer_state_rollback(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-7)
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        optimizer.step()
        snapshot = snapshot_training_state(model, optimizer)
        expected_model = {key: value.clone() for key, value in model.state_dict().items()}
        expected_optimizer = copy.deepcopy(optimizer.state_dict())

        optimizer.zero_grad(set_to_none=True)
        model.weight.data.add_(1.0)
        model.bias.data.sub_(1.0)
        restore_training_state(model, optimizer, snapshot)

        for key, value in expected_model.items():
            self.assertTrue(torch.equal(model.state_dict()[key], value))
        self.assertEqual(optimizer.state_dict()["param_groups"], expected_optimizer["param_groups"])

    def test_diagnostics_finite_check(self):
        self.assertTrue(finite_diagnostics({"mean": 0.005, "max": 0.9}))
        self.assertFalse(finite_diagnostics({"mean": float("nan")}))


if __name__ == "__main__":
    unittest.main()
