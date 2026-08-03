import json
import tempfile
import unittest
from pathlib import Path

from evaluation.audit_rl_prompt_reward_components import (
    analyze_run_records,
    analyze_run_records_v2,
    run_audit,
    run_audit_v2,
)


def _sample(step: int, micro: int, prompt: str, category: str, *, bad: bool = False) -> dict:
    return {
        "step": step,
        "micro_index": micro,
        "sample_key": f"{step}:{micro}:0",
        "prompt_id": prompt,
        "category": category,
        "candidate_index": 0,
        "reward": -0.2 if bad else 0.2,
        "components": {
            "validator_reward": 0.0 if bad else 0.5,
            "parse_reward": 0.0,
            "format_reward": 0.0 if bad else 0.1,
            "termination_reward": 0.0 if bad else 0.1,
            "repetition_penalty": 0.2 if bad else 0.0,
        },
        "generated_tokens": 16 if bad else 8,
        "finished_naturally": not bad,
        "empty_response": False,
        "max_length_hit": bad,
    }


def _micro(step: int, micro: int, prompt: str, category: str, *, bad: bool = False) -> dict:
    return {
        "schema_version": 2,
        "step": step,
        "micro_index": micro,
        "prompt_id": prompt,
        "category": category,
        "loss": 0.1,
        "reward_mean": -0.2 if bad else 0.2,
        "policy_diagnostics": {
            "reference_kl_mean": 0.01 if bad else 0.001,
            "reference_kl_p95": 0.2 if bad else 0.002,
            "reference_kl_max": 0.4 if bad else 0.01,
            "ratio_p95": 1.5,
            "ratio_max": 2.0,
        },
        "micro_grad_norm_scaled": 0.8 if bad else 0.1,
        "micro_grad_norm_unscaled": 4.0 if bad else 0.5,
        "accumulated_grad_norm": 1.0,
        "sample_keys": [f"{step}:{micro}:0"],
    }


class RLPromptRewardAuditTest(unittest.TestCase):
    def test_prompt_recurrence_and_component_correlation_are_deterministic(self):
        rows_a = [_micro(1, 0, "p_repeat", "format", bad=True), _micro(1, 1, "p_other", "termination")]
        samples_a = [
            _sample(1, 0, "p_repeat", "format", bad=True),
            _sample(1, 1, "p_other", "termination"),
        ]
        rows_b = [_micro(1, 0, "p_repeat", "format", bad=True), _micro(1, 1, "p_other", "termination")]
        samples_b = [
            _sample(1, 0, "p_repeat", "format", bad=True),
            _sample(1, 1, "p_other", "termination"),
        ]
        report_a = analyze_run_records("control", rows_a, samples_a, top_k=1)
        report_b = analyze_run_records("low_lr", rows_b, samples_b, top_k=1)
        self.assertEqual(report_a["status"], "COMPLETE_DIAGNOSTIC")
        self.assertEqual(report_a["top_k"]["reference_kl_max"][0]["prompt_id"], "p_repeat")
        self.assertIsNotNone(
            report_a["reward_component_correlations"]["repetition_penalty"]["correlations"]["reference_kl_max"]["pearson_r"]
        )
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            output = Path(temp) / "audit"
            for name, report in (("grpo_control_seed42", report_a), ("grpo_low_lr_seed42", report_b)):
                run = source / name
                run.mkdir(parents=True)
                (run / "microbatch_summaries.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in (rows_a if name.endswith("control_seed42") else rows_b)) + "\n",
                    encoding="utf-8",
                )
                (run / "samples.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in (samples_a if name.endswith("control_seed42") else samples_b)) + "\n",
                    encoding="utf-8",
                )
                (run / "step_summaries.jsonl").write_text("{}\n", encoding="utf-8")
                (run / "validation_history.jsonl").write_text("{}\n", encoding="utf-8")
                (run / "selection.json").write_text(json.dumps({"checkpoints": []}), encoding="utf-8")
                (run / "baseline_validation.json").write_text(json.dumps({"metrics": {}}), encoding="utf-8")
            # The public audit requires the three production runs; add a third
            # identical condition to make recurrence across >=2 conditions explicit.
            third = source / "grpo_clip_half_seed42"
            third.mkdir(parents=True)
            for filename in (
                "microbatch_summaries.jsonl",
                "samples.jsonl",
                "step_summaries.jsonl",
                "validation_history.jsonl",
                "selection.json",
                "baseline_validation.json",
            ):
                source_file = source / "grpo_control_seed42" / filename
                (third / filename).write_text(source_file.read_text(encoding="utf-8"), encoding="utf-8")
            summary = run_audit(source, output, top_k=1)
            self.assertEqual(summary["status"], "RECURRING_PROMPT_DIAGNOSTIC")
            self.assertEqual(summary["recurring_prompts"][0]["key"], "p_repeat")
            self.assertTrue((output / "prompt_summary.jsonl").is_file())
            with self.assertRaises(FileExistsError):
                run_audit(source, output, top_k=1)

    def test_missing_link_is_incomplete_without_changing_gate(self):
        report = analyze_run_records(
            "control",
            [_micro(1, 0, "p", "format", bad=True)],
            [],
            top_k=1,
        )
        self.assertEqual(report["status"], "AUDIT_INCOMPLETE")
        self.assertFalse(report["linkage"]["all_links_complete"])

    def test_category_summary_contains_reward_components(self):
        report = analyze_run_records(
            "control",
            [_micro(1, 0, "p", "termination")],
            [_sample(1, 0, "p", "termination")],
        )
        category = report["category_summary"]["termination"]
        self.assertIn("reward_components", category)
        self.assertAlmostEqual(category["reward_components"]["termination_reward"], 0.1)
        self.assertEqual(report["linkage"]["all_links_complete"], True)

    def test_v2_separates_within_run_and_cross_condition_occurrence(self):
        source_row = {
            "id": "p_v2",
            "category": "conciseness",
            "family": "family_v2",
            "difficulty": "basic",
            "prompt": "解释变量，不超过50字。",
            "chosen": "变量是保存值的名称。",
            "metadata": {"max_chars": 50, "required_terms": ["变量", "保存", "值"]},
        }
        sample = {
            "step": 1,
            "micro_index": 0,
            "sample_key": "1:0:0",
            "prompt_id": "p_v2",
            "category": "conciseness",
            "candidate_index": 0,
            "reward": 1.1,
            "components": {"validator_reward": 1.0, "termination_reward": 0.1},
            "response": "变量是保存值的名称。",
            "generated_tokens": 8,
            "termination_reason": "eos",
            "eos_seen": True,
            "finished_naturally": True,
            "empty_response": False,
            "max_length_hit": False,
        }
        micro = {
            "schema_version": 3,
            "step": 1,
            "micro_index": 0,
            "prompt_id": "p_v2",
            "category": "conciseness",
            "loss": 0.1,
            "reward_mean": 1.1,
            "policy_diagnostics": {
                "reference_kl_max": 0.01,
                "reference_kl_p95": 0.01,
                "reference_kl_mean": 0.01,
                "ratio_p95": 1.0,
                "ratio_max": 1.0,
            },
            "micro_grad_norm_unscaled": 0.5,
            "termination_reason_counts": {"eos": 1, "max_new_tokens": 0, "no_eos_short_generation": 0},
            "sample_keys": ["1:0:0"],
        }
        first = analyze_run_records_v2("control", [micro], [sample], top_k=1)
        second = analyze_run_records_v2("low_lr", [micro], [sample], top_k=1)
        first_group = first["prompt_summary"]["p_v2"]
        self.assertEqual(first_group["within_run_occurrence_count"], {"control": 1})
        self.assertEqual(first_group["cross_condition_condition_count"], 1)
        self.assertEqual(second["prompt_summary"]["p_v2"]["cross_condition_condition_count"], 1)
        self.assertEqual(first["quality_telemetry_status"], "CORRECTED_QUALITY_TELEMETRY")

    def test_v2_replay_exposure_and_output_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            output = root / "audit"
            manifest = root / "train.jsonl"
            manifest.write_text(
                json.dumps({
                    "id": "p_v2",
                    "category": "conciseness",
                    "family": "family_v2",
                    "difficulty": "basic",
                    "prompt": "解释变量，不超过50字。",
                    "chosen": "变量是保存值的名称。",
                    "metadata": {"max_chars": 50, "required_terms": ["变量", "保存", "值"]},
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            sample = {
                "step": 1, "micro_index": 0, "sample_key": "1:0:0", "prompt_id": "p_v2",
                "category": "conciseness", "candidate_index": 0, "reward": 1.1,
                "components": {"validator_reward": 1.0, "termination_reward": 0.1},
                "response": "变量是保存值的名称。", "generated_tokens": 8,
                "termination_reason": "eos", "eos_seen": True,
                "finished_naturally": True, "empty_response": False, "max_length_hit": False,
            }
            micro = {
                "schema_version": 3, "step": 1, "micro_index": 0, "prompt_id": "p_v2",
                "category": "conciseness", "loss": 0.1, "reward_mean": 1.1,
                "policy_diagnostics": {"reference_kl_max": 0.01, "reference_kl_p95": 0.01,
                                       "reference_kl_mean": 0.01, "ratio_p95": 1.0, "ratio_max": 1.0},
                "micro_grad_norm_unscaled": 0.5,
                "termination_reason_counts": {"eos": 1, "max_new_tokens": 0, "no_eos_short_generation": 0},
                "sample_keys": ["1:0:0"],
            }
            for name in ("grpo_control_seed42", "grpo_low_lr_seed42", "grpo_clip_half_seed42"):
                run = source / name
                run.mkdir(parents=True)
                (run / "microbatch_summaries.jsonl").write_text(json.dumps(micro) + "\n", encoding="utf-8")
                (run / "samples.jsonl").write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
                (run / "step_summaries.jsonl").write_text("{}\n", encoding="utf-8")
                (run / "validation_history.jsonl").write_text("{}\n", encoding="utf-8")
                (run / "selection.json").write_text(json.dumps({"checkpoints": []}), encoding="utf-8")
                (run / "baseline_validation.json").write_text(json.dumps({"metrics": {}}), encoding="utf-8")
            summary = run_audit_v2(source, manifest, output, top_k=1)
            self.assertEqual(summary["validator_replay"]["persisted_replay_exact"], True)
            self.assertEqual(summary["validator_replay"]["persisted_replay_match_count"], 3)
            self.assertEqual(summary["source_manifest"]["metadata_missing_count"], 0)
            self.assertEqual(summary["cross_condition_prompt_count"], 1)
            self.assertEqual(summary["exposure_denominators"]["category"]["conciseness"]["source_prompt_count"], 1)
            self.assertEqual(summary["exposure_denominators"]["family"]["family_v2"]["source_prompt_count"], 1)
            with self.assertRaises(FileExistsError):
                run_audit_v2(source, manifest, output, top_k=1)

    def test_v2_legacy_quality_is_explicitly_untrusted(self):
        report = analyze_run_records_v2(
            "control",
            [_micro(1, 0, "p", "format", bad=True)],
            [_sample(1, 0, "p", "format", bad=True)],
            top_k=1,
        )
        self.assertEqual(report["quality_telemetry_status"], "LEGACY_QUALITY_TELEMETRY_UNTRUSTED")
        self.assertEqual(report["top_metrics_used"], ["reference_kl_max", "micro_grad_norm_unscaled"])


if __name__ == "__main__":
    unittest.main()
