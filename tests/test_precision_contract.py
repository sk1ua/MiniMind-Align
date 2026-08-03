import json
import tempfile
import unittest
from pathlib import Path

from align.rl_rules import precision_contract_spec, require_precision_contract
from evaluation.audit_precision_contract import (
    CONTRACT_FIELDS,
    dtype_gap,
    ensure_empty_output_dir,
    precision_contract_check,
)


def valid_kwargs():
    return {
        "use_autocast": True,
        "post_step_kl_target_enabled": True,
        "post_step_kl_diagnostic_fp32": True,
        "post_step_kl_diagnostic_full_fp32": True,
        "pre_step_kl_diagnostic_fp32": True,
        "pre_step_loss_diagnostic_fp32": True,
        "pre_step_loss_replay_enabled": True,
        "microbatch_telemetry_enabled": True,
        "microbatch_gradient_telemetry_enabled": True,
    }


class PrecisionContractTests(unittest.TestCase):
    def test_legacy_compat_is_valid_and_keeps_legacy_variants(self):
        spec = precision_contract_spec(
            "legacy_compat",
            "legacy_bfloat16_autocast",
            "legacy_bfloat16_autocast",
            "torch.float32",
            "torch.float32",
            use_autocast=True,
        )
        self.assertTrue(spec["precision_contract_valid"])
        self.assertEqual(spec["active_loss_variant"], "legacy_bfloat16_autocast")
        self.assertEqual(spec["active_gate_source"], "post_step_kl_bfloat16")
        self.assertTrue(spec["active_loss_autocast_enabled"])

    def test_opt_in_contract_requires_all_diagnostics(self):
        spec = precision_contract_spec(
            "no_autocast_v1",
            "fp32_no_autocast",
            "fp32_no_autocast",
            "torch.float32",
            "torch.float32",
            use_autocast=True,
        )
        self.assertFalse(spec["precision_contract_valid"])
        with self.assertRaises(ValueError):
            require_precision_contract(spec)
        self.assertIn("post_step_kl_target_required", spec["precision_contract_errors"])

    def test_opt_in_contract_selects_same_active_variant_and_shadows(self):
        spec = precision_contract_spec(
            "no_autocast_v1",
            "fp32_no_autocast",
            "fp32_no_autocast",
            "torch.bfloat16",
            "torch.bfloat16",
            **valid_kwargs(),
        )
        self.assertTrue(spec["precision_contract_valid"])
        self.assertEqual(spec["active_loss_variant"], "policy_bfloat16_no_autocast")
        self.assertEqual(spec["active_loss_variant"], spec["active_gate_variant"])
        self.assertEqual(spec["active_gate_source"], "post_step_kl_float32")
        self.assertFalse(spec["active_loss_autocast_enabled"])
        self.assertFalse(spec["active_gate_autocast_enabled"])
        self.assertTrue(spec["full_float32_shadow_only"])

    def test_audit_contract_rejects_active_shadow(self):
        spec = precision_contract_spec(
            "no_autocast_v1",
            "fp32_no_autocast",
            "fp32_no_autocast",
            "float32",
            "float32",
            **valid_kwargs(),
        )
        row = {"precision_contract_mode": "no_autocast_v1", "precision_contract": spec}
        self.assertTrue(precision_contract_check(row)[0])
        bad = dict(spec)
        bad["active_gate_variant"] = bad["legacy_bfloat16_shadow_variant"]
        self.assertFalse(precision_contract_check({"precision_contract_mode": "no_autocast_v1", "precision_contract": bad})[0])

    def test_dtype_gap_threshold(self):
        gap, threshold, within = dtype_gap(0.0056, 0.005, 0.005)
        self.assertAlmostEqual(gap, 0.0006, places=7)
        self.assertAlmostEqual(threshold, 0.0005, places=7)
        self.assertFalse(within)

    def test_schema_fields_and_non_overwrite(self):
        spec = precision_contract_spec(
            "no_autocast_v1",
            "fp32_no_autocast",
            "fp32_no_autocast",
            "float32",
            "float32",
            **valid_kwargs(),
        )
        self.assertTrue(set(CONTRACT_FIELDS).issubset(spec))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "audit"
            ensure_empty_output_dir(root)
            (root / "existing.json").write_text(json.dumps({"keep": True}), encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_empty_output_dir(root)


if __name__ == "__main__":
    unittest.main()
