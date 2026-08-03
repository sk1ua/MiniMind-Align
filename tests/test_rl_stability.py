import json
import tempfile
import unittest
from pathlib import Path

import torch

from align.rl_rules import policy_diagnostics
from evaluation.audit_rl_stability import audit_run, compare_run_to_control


class RLStabilityTests(unittest.TestCase):
    def test_policy_diagnostics_are_masked_and_finite(self) -> None:
        new = torch.tensor([[0.1, -0.1, 4.0]])
        old = torch.zeros_like(new)
        ref = torch.zeros_like(new)
        mask = torch.tensor([[1.0, 1.0, 0.0]])

        metrics = policy_diagnostics(new, old, ref, mask)

        self.assertEqual(metrics["token_count"], 2.0)
        self.assertAlmostEqual(metrics["ratio_mean"], (torch.exp(torch.tensor(0.1)) + torch.exp(torch.tensor(-0.1))).item() / 2, places=5)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

    def test_stability_comparison_requires_four_step_delay(self) -> None:
        control = {
            "run": "grpo_control_seed42",
            "mode": "grpo",
            "condition": "control",
            "first_kl_trigger_step": 12,
            "max_kl_p95": 0.02,
            "max_kl": 0.04,
            "max_grad_norm_pre_clip": 2.0,
            "validation": {
                "selected_safety_pass": 4,
                "selected_termination_pass": 4,
            },
        }
        variant = {
            "run": "grpo_low_lr_seed42",
            "mode": "grpo",
            "condition": "low_lr",
            "first_kl_trigger_step": 16,
            "max_kl_p95": 0.01,
            "max_kl": 0.02,
            "max_grad_norm_pre_clip": 1.5,
            "validation": {
                "selected_safety_pass": 4,
                "selected_termination_pass": 4,
            },
        }

        comparison = compare_run_to_control(variant, control)

        self.assertTrue(comparison["trigger_improved"])
        self.assertTrue(comparison["stability_improved"])

    def test_audit_run_records_missing_telemetry_as_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "grpo_control_seed42"
            run_dir.mkdir()
            (run_dir / "selection.json").write_text(json.dumps({"status": "baseline_retained"}), encoding="utf-8")
            (run_dir / "baseline_validation.json").write_text(
                json.dumps({"metrics": {"validator_pass": 1, "safety_pass": 1, "termination_pass": 1}}),
                encoding="utf-8",
            )
            (run_dir / "step_summaries.jsonl").write_text(
                json.dumps({"step": 1, "kl_mean": 0.0}) + "\n", encoding="utf-8"
            )
            (run_dir / "validation_history.jsonl").write_text("", encoding="utf-8")

            report = audit_run(run_dir)

            self.assertEqual(report["status"], "FAILED")
            self.assertIn("missing_stability_telemetry", {item["code"] for item in report["warnings"]})


if __name__ == "__main__":
    unittest.main()
