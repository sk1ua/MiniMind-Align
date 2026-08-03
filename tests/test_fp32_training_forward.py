import unittest
from pathlib import Path

import torch

from align.rl_rules import (
    clipped_policy_diagnostic_terms,
    clipped_policy_loss,
    training_forward_uses_autocast,
)
from evaluation.audit_fp32_training_forward import _active_precision_checks


class Fp32TrainingForwardTest(unittest.TestCase):
    def test_default_mode_keeps_legacy_autocast_selection(self):
        self.assertTrue(training_forward_uses_autocast("legacy_bfloat16_autocast", True))
        self.assertFalse(training_forward_uses_autocast("fp32_no_autocast", True))
        self.assertFalse(training_forward_uses_autocast("legacy_bfloat16_autocast", False))

    def test_nonzero_advantage_contract_matches_diagnostic_terms(self):
        new = torch.tensor([[-1.2, -0.8], [-1.1, -0.9]], requires_grad=True)
        old = torch.full_like(new, -1.0)
        reference = torch.full_like(new, -1.05)
        advantages = torch.tensor([1.0, -0.75])
        mask = torch.ones_like(new)
        for mode in ("grpo", "cispo"):
            loss, kl = clipped_policy_loss(
                new, old, reference, advantages, mask,
                mode=mode, beta=0.02, epsilon=0.2, epsilon_high=5.0,
            )
            terms = clipped_policy_diagnostic_terms(
                new, old, reference, advantages,
                mode=mode, beta=0.02, epsilon=0.2, epsilon_high=5.0,
            )
            self.assertGreater(float(advantages.abs().max()), 0.0)
            self.assertAlmostEqual(float(loss.detach()), float(terms["token_loss"].mean()), places=6)
            self.assertAlmostEqual(float(kl.detach()), float(terms["reference_kl"].mean()), places=6)

    def test_audit_active_mode_requires_selected_loss_and_gradient(self):
        precision = {
            "training_forward_mode": "fp32_no_autocast",
            "active_variant": "fp32_no_autocast",
            "fp32_no_autocast_loss": 0.25,
            "fp32_no_autocast_kl": 0.001,
            "active_gradient_matches_selected_variant": True,
        }
        micro = [{
            "step": 1,
            "micro_index": 0,
            "training_forward_mode": "fp32_no_autocast",
            "loss": 0.25,
            "kl": 0.001,
            "pre_step_loss_precision": precision,
        }]
        step = [{
            "step": 1,
            "training_forward_mode": "fp32_no_autocast",
            "pre_step_loss_precision": {
                "training_forward_mode": "fp32_no_autocast",
                "active_variant": "fp32_no_autocast",
                "active_loss_mean": 0.25,
                "fp32_no_autocast_loss_mean": 0.25,
                "active_kl_mean": 0.001,
                "fp32_no_autocast_kl_mean": 0.001,
                "active_gradient_matches_selected_variant": True,
            },
        }]
        checks = _active_precision_checks(micro, step, "fp32_no_autocast")
        self.assertTrue(checks["mode_ok"])
        self.assertTrue(checks["active_loss_ok"])
        self.assertTrue(checks["active_kl_ok"])
        self.assertTrue(checks["active_gradient_ok"])

    def test_trainer_has_explicit_opt_in_and_active_telemetry(self):
        source = (Path(__file__).resolve().parents[1] / "trainer" / "train_grpo_lite.py").read_text(encoding="utf-8")
        self.assertIn('"--training-forward-mode"', source)
        self.assertIn("training_use_autocast = training_forward_uses_autocast", source)
        self.assertIn('"active_loss_precision": args.training_forward_mode', source)
        self.assertIn('"active_gradient_matches_selected_variant"', source)

    def test_wrapper_is_single_grpo_two_step_isolated_smoke(self):
        source = (Path(__file__).resolve().parents[1] / "scripts" / "run_fp32_training_forward_smoke.sh").read_text(encoding="utf-8")
        self.assertIn("rl_fp32_training_forward_smoke_20260802", source)
        self.assertIn("--training-forward-mode fp32_no_autocast", source)
        self.assertIn("--max-steps 2", source)
        self.assertNotIn("--formal", source)
        self.assertNotIn("cispo", source.lower())


if __name__ == "__main__":
    unittest.main()
