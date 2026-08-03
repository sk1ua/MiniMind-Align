import argparse
import copy
import tempfile
import unittest
from pathlib import Path

import torch

from align.rl_rules import kl_backoff_multipliers
from evaluation.audit_rl_kl_guard_telemetry import audit, dtype_gap_threshold
from trainer.train_grpo_lite import (
    optimizer_state_digest,
    parameter_delta_metrics,
    policy_state_digest,
    restore_training_state,
    snapshot_training_state,
)


class RLKLGuardTelemetryTest(unittest.TestCase):
    def test_four_backoff_multipliers_are_auditable(self):
        self.assertEqual(kl_backoff_multipliers(0.5, 3), [1.0, 0.5, 0.25, 0.125])

    def test_bfloat16_and_float32_use_same_rollout_source(self):
        source = (Path(__file__).resolve().parents[1] / "trainer" / "train_grpo_lite.py").read_text(encoding="utf-8")
        self.assertIn("post_update_kl_stats(\n                        policy,\n                        guard_rollouts", source)
        self.assertIn("torch.float32,\n                        False", source)
        self.assertIn('"post_step_kl_bfloat16"', source)
        self.assertIn('"post_step_kl_float32"', source)

    def test_dtype_gap_threshold_is_fixed(self):
        self.assertAlmostEqual(dtype_gap_threshold(0.01, 0.005), 0.001)
        self.assertAlmostEqual(dtype_gap_threshold(0.0001, 0.005), 0.0005)

    def test_parameter_delta_is_zero_before_and_nonzero_after_update(self):
        model = torch.nn.Linear(2, 1)
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        zero = parameter_delta_metrics(model, before)
        self.assertEqual(zero["parameter_delta_l2"], 0.0)
        model.weight.data.add_(0.25)
        changed = parameter_delta_metrics(model, before)
        self.assertGreater(changed["parameter_delta_l2"], 0.0)
        self.assertGreater(changed["parameter_delta_max_abs"], 0.0)

    def test_policy_and_optimizer_digests_match_after_rollback(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-7)
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        snapshot = snapshot_training_state(model, optimizer)
        policy_before = policy_state_digest(model)
        optimizer_before = optimizer_state_digest(optimizer)
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        optimizer.step()
        post_step_digest = optimizer_state_digest(optimizer)
        self.assertNotEqual(post_step_digest, optimizer_before)
        model.weight.data.add_(1.0)
        restore_training_state(model, optimizer, snapshot)
        self.assertEqual(policy_state_digest(model), policy_before)
        self.assertEqual(optimizer_state_digest(optimizer), optimizer_before)
        self.assertEqual(optimizer.state_dict()["param_groups"], expected_optimizer["param_groups"])

    def test_rejected_step_precedes_checkpoint_and_default_path_remains(self):
        source = (Path(__file__).resolve().parents[1] / "trainer" / "train_grpo_lite.py").read_text(encoding="utf-8")
        self.assertLess(source.index('if optimizer_step_rejected:'), source.index('should_checkpoint ='))
        self.assertIn('kl_guard_enabled = args.post_step_kl_target is not None', source)
        self.assertIn('else:\n            optimizer.step()', source)

    def test_attempt_log_and_audit_output_refuse_overwrite(self):
        source = (Path(__file__).resolve().parents[1] / "trainer" / "train_grpo_lite.py").read_text(encoding="utf-8")
        audit_source = (Path(__file__).resolve().parents[1] / "evaluation" / "audit_rl_kl_guard_telemetry.py").read_text(encoding="utf-8")
        self.assertIn("kl_guard_attempt_log_path", source)
        self.assertIn("refusing to overwrite non-empty audit directory", audit_source)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit"
            output.mkdir()
            (output / "existing.json").write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                experiment_root=str(Path(directory) / "root"),
                output_dir=str(output),
                run_name="run",
            )
            with self.assertRaises(FileExistsError):
                audit(args)


if __name__ == "__main__":
    unittest.main()
