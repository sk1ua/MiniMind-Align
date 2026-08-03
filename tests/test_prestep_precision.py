import tempfile
import unittest
from pathlib import Path

import torch

from align.rl_rules import clipped_policy_diagnostic_terms, clipped_policy_loss
from evaluation.audit_prestep_precision import (
    determine_status,
    ensure_empty_output_dir,
    json_sha256,
    validate_prestep_replay_row,
)
from trainer.train_grpo_lite import gradient_sequence_norm, masked_diagnostic_means


def replay_row():
    generated = [[10, 11, 12], [10, 13, 14]]
    mask_values = [[1.0, 1.0], [1.0, 1.0]]
    old_values = [[-1.0, -1.0], [-1.0, -1.0]]
    ref_values = [[-1.0, -1.0], [-1.0, -1.0]]
    advantage_values = [0.0, 0.0]
    legacy_values = [[-1.2, -1.1], [-1.15, -1.05]]
    fp32_values = old_values
    mask = torch.tensor(mask_values)
    old = torch.tensor(old_values)
    reference = torch.tensor(ref_values)
    advantages = torch.tensor(advantage_values)
    variants = {
        "bfloat16_autocast": torch.tensor(legacy_values),
        "fp32_no_autocast": torch.tensor(fp32_values),
    }
    row = {
        "schema_version": 1,
        "record_type": "pre_step_loss_replay",
        "run_id": "test",
        "step": 2,
        "micro_index": 0,
        "prompt_id": "prompt",
        "category": "conciseness",
        "sample_keys": ["2:0:0", "2:0:1"],
        "mode": "grpo",
        "beta": 0.02,
        "epsilon": 0.2,
        "epsilon_high": 5.0,
        "generated_ids": generated,
        "completion_mask": mask_values,
        "old_log_probs": old_values,
        "ref_log_probs": ref_values,
        "advantages": advantage_values,
        "bfloat16_autocast_new_log_probs": legacy_values,
        "fp32_no_autocast_new_log_probs": fp32_values,
        "fp32_shadow_gradient_norm_scaled": 0.0,
        "fp32_shadow_gradient_norm_unscaled": 0.0,
        "fp32_shadow_grad_isolation_ok": True,
        "source_sha256": {
            "generated": json_sha256(generated),
            "completion_mask": json_sha256(mask_values),
            "old_log_probs": json_sha256(old_values),
            "ref_log_probs": json_sha256(ref_values),
            "advantages": json_sha256(advantage_values),
        },
    }
    for name, new_log_probs in variants.items():
        terms = clipped_policy_diagnostic_terms(
            new_log_probs,
            old,
            reference,
            advantages,
            mode="grpo",
            beta=0.02,
            epsilon=0.2,
            epsilon_high=5.0,
        )
        means = masked_diagnostic_means(terms, mask)
        row[f"{name}_terms"] = {key: value.tolist() for key, value in terms.items()}
        row[f"{name}_means"] = means
        row[f"{name}_loss"] = means["token_loss"]
        row[f"{name}_kl"] = means["reference_kl"]
    return row


class PrestepPrecisionTest(unittest.TestCase):
    def test_diagnostic_terms_match_production_grpo_loss(self):
        new = torch.tensor([[-1.2, -0.8], [-1.1, -0.9]], requires_grad=True)
        old = torch.full_like(new, -1.0)
        reference = torch.full_like(new, -1.0)
        advantages = torch.tensor([1.0, -1.0])
        mask = torch.ones_like(new)
        loss, kl = clipped_policy_loss(
            new, old, reference, advantages, mask, mode="grpo", beta=0.02, epsilon=0.2
        )
        means = masked_diagnostic_means(
            clipped_policy_diagnostic_terms(
                new, old, reference, advantages, mode="grpo", beta=0.02, epsilon=0.2
            ),
            mask,
        )
        self.assertAlmostEqual(float(loss.detach()), means["token_loss"], places=6)
        self.assertAlmostEqual(float(kl.detach()), means["reference_kl"], places=6)

    def test_diagnostic_terms_match_production_cispo_loss(self):
        new = torch.tensor([[-1.2, -0.8]], requires_grad=True)
        old = torch.full_like(new, -1.0)
        reference = torch.full_like(new, -1.0)
        advantages = torch.tensor([0.5])
        mask = torch.ones_like(new)
        loss, _ = clipped_policy_loss(
            new, old, reference, advantages, mask, mode="cispo", beta=0.02, epsilon_high=5.0
        )
        means = masked_diagnostic_means(
            clipped_policy_diagnostic_terms(
                new, old, reference, advantages, mode="cispo", beta=0.02, epsilon_high=5.0
            ),
            mask,
        )
        self.assertAlmostEqual(float(loss.detach()), means["token_loss"], places=6)

    def test_shadow_autograd_does_not_write_parameter_grad(self):
        model = torch.nn.Linear(2, 1, bias=False)
        loss = model(torch.ones(1, 2)).sum()
        gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
        self.assertGreater(gradient_sequence_norm(gradients), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_same_token_replay_row_validates(self):
        result = validate_prestep_replay_row(replay_row())
        self.assertTrue(result["valid"], result)

    def test_replay_detects_changed_term(self):
        row = replay_row()
        row["bfloat16_autocast_terms"]["token_loss"][0][0] += 0.1
        self.assertFalse(validate_prestep_replay_row(row)["valid"])

    def test_sensitive_status_requires_integrity_and_gradient_evidence(self):
        step = [{"loss_disagreement": True, "kl_disagreement": True}]
        micros = [{"gradient_disagreement": True}, {"gradient_disagreement": True}]
        self.assertEqual(
            determine_status(integrity_ok=True, step_precision=step, micro_precision=micros),
            "TRAINING_AUTOCAST_PRECISION_SENSITIVE",
        )
        self.assertEqual(
            determine_status(integrity_ok=False, step_precision=step, micro_precision=micros),
            "TELEMETRY_INCOMPLETE",
        )

    def test_kl_only_and_consistent_statuses(self):
        self.assertEqual(
            determine_status(
                integrity_ok=True,
                step_precision=[{"loss_disagreement": False, "kl_disagreement": True}],
                micro_precision=[{"gradient_disagreement": False}],
            ),
            "PRESTEP_KL_ONLY_PRECISION_SENSITIVE",
        )
        self.assertEqual(
            determine_status(
                integrity_ok=True,
                step_precision=[{"loss_disagreement": False, "kl_disagreement": False}],
                micro_precision=[{"gradient_disagreement": False}],
            ),
            "PRESTEP_PRECISION_CONSISTENT",
        )

    def test_trainer_path_is_opt_in_and_production_loss_call_remains(self):
        source = (Path(__file__).resolve().parents[1] / "trainer" / "train_grpo_lite.py").read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--pre-step-loss-diagnostic-fp32", action="store_true")', source)
        self.assertIn("(loss / args.accumulation_steps).backward()", source)
        self.assertIn("torch.autograd.grad(", source)
        self.assertIn('"same_rollout_mask_old_ref_advantages": True', source)

    def test_wrapper_is_narrow_and_isolated(self):
        source = (Path(__file__).resolve().parents[1] / "scripts" / "run_prestep_precision_smoke.sh").read_text(encoding="utf-8")
        self.assertIn("rl_prestep_precision_smoke_20260802", source)
        self.assertIn("--max-steps 2", source)
        self.assertIn("--pre-step-loss-diagnostic-fp32", source)
        self.assertNotIn("--formal", source)
        self.assertNotIn("cispo", source.lower())

    def test_audit_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit"
            output.mkdir()
            (output / "existing.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_empty_output_dir(output)


if __name__ == "__main__":
    unittest.main()
