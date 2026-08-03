import unittest

from evaluation.audit_kl_measurement_precision import classify_attempt


def kl(mean: float) -> dict[str, float]:
    return {"reference_kl_mean": mean, "reference_kl_p95": mean, "reference_kl_max": mean, "token_count": 4.0}


class KLMeasurementPrecisionTest(unittest.TestCase):
    def test_full_float32_variant_is_distinct_from_no_autocast_variant(self):
        attempt = {
            "post_step_kl_bfloat16": kl(0.27),
            "post_step_kl_float32": kl(0.00002),
            "post_step_kl_full_float32": kl(0.000021),
            "parameter_delta": {"parameter_delta_l2": 0.001},
            "rollback_after_attempt": {"exact_match": True},
        }
        result = classify_attempt(attempt, 0.005)
        self.assertFalse(result["bfloat16_gate_passed"])
        self.assertTrue(result["no_autocast_gate_passed"])
        self.assertTrue(result["full_float32_gate_passed"])
        self.assertTrue(result["no_autocast_matches_full_fp32"])
        self.assertTrue(result["bfloat16_sensitive"])

    def test_full_float32_over_target_is_not_called_bfloat16_only(self):
        attempt = {
            "post_step_kl_bfloat16": kl(0.27),
            "post_step_kl_float32": kl(0.02),
            "post_step_kl_full_float32": kl(0.021),
            "parameter_delta": {"parameter_delta_l2": 0.001},
            "rollback_after_attempt": {"exact_match": True},
        }
        result = classify_attempt(attempt, 0.005)
        self.assertFalse(result["full_float32_gate_passed"])
        self.assertFalse(result["no_autocast_gate_passed"])

    def test_trainer_exposes_full_fp32_diagnostic_without_changing_gate(self):
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "trainer" / "train_grpo_lite.py").read_text(encoding="utf-8")
        self.assertIn("--post-step-kl-diagnostic-full-fp32", source)
        self.assertIn("build_full_fp32_measurement_model", source)
        self.assertIn('"post_step_kl_full_float32"', source)
        self.assertIn('post_step_kl["reference_kl_mean"] <= args.post_step_kl_target', source)


if __name__ == "__main__":
    unittest.main()
