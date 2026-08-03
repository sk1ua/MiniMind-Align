import tempfile
import unittest
from pathlib import Path

from align.rl_rules import select_reference_kl_gate_stats
from evaluation.audit_corrected_kl_gate import (
    determine_status,
    ensure_empty_output_dir,
    validate_attempts,
)


def stats(mean):
    return {
        "reference_kl_mean": mean,
        "reference_kl_p95": mean * 2,
        "reference_kl_max": mean * 3,
        "token_count": 8.0,
    }


def accepted_attempt(step, pre_policy, pre_optimizer, post_policy, post_optimizer, attempt_index=0):
    return {
        "step": step,
        "attempt_index": attempt_index,
        "post_step_kl_gate_mode": "fp32_no_autocast",
        "post_step_kl_bfloat16": stats(0.27),
        "post_step_kl_float32": stats(0.0002),
        "post_step_kl_gate": stats(0.0002),
        "post_step_kl_gate_passed": True,
        "legacy_bfloat16_gate_passed": False,
        "fp32_no_autocast_gate_passed": True,
        "shadow_gate_disagreement": True,
        "production_gate_passed": True,
        "accepted": True,
        "parameter_delta": {"parameter_delta_l2": 0.01},
        "pre_attempt_policy_state_digest": pre_policy,
        "pre_attempt_optimizer_state_digest": pre_optimizer,
        "post_attempt_policy_state_digest": post_policy,
        "post_attempt_optimizer_state_digest": post_optimizer,
        "rollback_after_attempt": None,
    }


class CorrectedKLGateTest(unittest.TestCase):
    def test_gate_helper_defaults_can_select_legacy(self):
        legacy = stats(0.27)
        fp32 = stats(0.0002)
        self.assertEqual(
            select_reference_kl_gate_stats("legacy_bfloat16_autocast", legacy, fp32),
            legacy,
        )

    def test_gate_helper_selects_fp32_without_mutating_inputs(self):
        legacy = stats(0.27)
        fp32 = stats(0.0002)
        selected = select_reference_kl_gate_stats("fp32_no_autocast", legacy, fp32)
        self.assertEqual(selected, fp32)
        self.assertIsNot(selected, fp32)

    def test_fp32_gate_requires_fp32_stats(self):
        with self.assertRaises(ValueError):
            select_reference_kl_gate_stats("fp32_no_autocast", stats(0.27), None)

    def test_shadow_over_threshold_does_not_reject_selected_gate(self):
        row = accepted_attempt(1, "p0", "o0", "p1", "o1")
        result = validate_attempts([row], 0.005)
        self.assertTrue(result["gate_consistent"])
        self.assertEqual(result["accepted_steps"], [1])

    def test_accepted_state_is_nonzero_and_not_rolled_back(self):
        row = accepted_attempt(1, "p0", "o0", "p1", "o1")
        result = validate_attempts([row], 0.005)
        self.assertTrue(result["accepted_state_ok"])

    def test_rejected_attempt_requires_exact_zero_delta_rollback(self):
        row = accepted_attempt(1, "p0", "o0", "p1", "o1")
        row.update({
            "post_step_kl_float32": stats(0.02),
            "post_step_kl_gate": stats(0.02),
            "post_step_kl_gate_passed": False,
            "production_gate_passed": False,
            "fp32_no_autocast_gate_passed": False,
            "accepted": False,
            "rollback_after_attempt": {
                "exact_match": True,
                "parameter_delta": {
                    "parameter_delta_l2": 0.0,
                    "parameter_delta_max_abs": 0.0,
                },
            },
        })
        result = validate_attempts([row], 0.005)
        self.assertTrue(result["gate_consistent"])
        self.assertTrue(result["rejected_rollback_ok"])

    def test_two_accepted_steps_have_digest_continuity(self):
        rows = [
            accepted_attempt(1, "p0", "o0", "p1", "o1"),
            accepted_attempt(2, "p1", "o1", "p2", "o2"),
        ]
        result = validate_attempts(rows, 0.005)
        self.assertTrue(result["state_continuity_ok"])
        self.assertEqual(result["accepted_steps"], [1, 2])

    def test_status_distinguishes_clean_and_backoff_acceptance(self):
        clean = [accepted_attempt(1, "p0", "o0", "p1", "o1"), accepted_attempt(2, "p1", "o1", "p2", "o2")]
        self.assertEqual(
            determine_status(integrity_ok=True, accepted_steps=[1, 2], accepted_attempts=clean, optimizer_rejected=False),
            "CORRECTED_GATE_ACCEPTED_2_STEPS_DIAGNOSTIC",
        )
        clean[1]["attempt_index"] = 1
        self.assertEqual(
            determine_status(integrity_ok=True, accepted_steps=[1, 2], accepted_attempts=clean, optimizer_rejected=False),
            "CORRECTED_GATE_BACKOFF_REQUIRED_DIAGNOSTIC",
        )

    def test_status_distinguishes_unresolved_and_incomplete(self):
        self.assertEqual(
            determine_status(integrity_ok=True, accepted_steps=[], accepted_attempts=[], optimizer_rejected=True),
            "CORRECTED_GATE_UNRESOLVED_BASELINE_RETAINED",
        )
        self.assertEqual(
            determine_status(integrity_ok=False, accepted_steps=[1, 2], accepted_attempts=[], optimizer_rejected=False),
            "TELEMETRY_INCOMPLETE",
        )

    def test_trainer_default_and_checkpoint_order_are_preserved(self):
        source = (Path(__file__).resolve().parents[1] / "trainer" / "train_grpo_lite.py").read_text(encoding="utf-8")
        self.assertIn('default="legacy_bfloat16_autocast"', source)
        self.assertIn('args.post_step_kl_gate_mode == "fp32_no_autocast"', source)
        self.assertIn('production_gate_passed = post_step_kl_gate_passed', source)
        self.assertLess(source.index('if optimizer_step_rejected:'), source.index('should_checkpoint ='))
        self.assertIn('evaluation_source = "reloaded_checkpoint"', source)

    def test_new_telemetry_schema_is_present(self):
        source = (Path(__file__).resolve().parents[1] / "trainer" / "train_grpo_lite.py").read_text(encoding="utf-8")
        for field in (
            '"post_step_kl_gate_mode"',
            '"post_step_kl_gate_mean"',
            '"post_step_kl_gate_passed"',
            '"legacy_bfloat16_gate_passed"',
            '"fp32_no_autocast_gate_passed"',
            '"shadow_gate_disagreement"',
            '"pre_step_kl_fp32_no_autocast_mean"',
        ):
            self.assertIn(field, source)

    def test_output_directory_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit"
            output.mkdir()
            (output / "existing.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_empty_output_dir(output)


if __name__ == "__main__":
    unittest.main()
