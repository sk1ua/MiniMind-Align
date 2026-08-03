import json
import tempfile
import unittest
from pathlib import Path

from evaluation.audit_rl_spike_sources import (
    _sample_stats,
    analyze_run_data,
    discover_runs,
    run_audit,
)


def _sample(step: int, category: str, reward: float, validator: float, max_length: bool) -> dict:
    return {
        "step": step,
        "micro_index": 0,
        "sample_key": f"{step}:0:0",
        "prompt_id": f"{category}_{step}",
        "category": category,
        "candidate_index": 0,
        "reward": reward,
        "components": {"validator_reward": validator, "repetition_penalty": 0.2 if max_length else 0.0},
        "generated_tokens": 16 if max_length else 8,
        "finished_naturally": not max_length,
        "empty_response": False,
        "max_length_hit": max_length,
    }


def _step(step: int, reward: float, validator: float, max_length: float, repeat: float) -> dict:
    return {
        "step": step,
        "reward_mean": reward,
        "kl_mean": 0.006 if step == 2 else 0.001,
        "kl_p95": 0.02 if step == 2 else 0.004,
        "kl_max": 0.3 if step == 2 else 0.02,
        "grad_norm_pre_clip": 2.0 if step == 2 else 1.2,
        "ratio_max": 2.5 if step == 2 else 1.2,
        "train_validator_pass_rate": validator,
        "train_max_length_hit_rate": max_length,
        "train_repetition_penalty_mean": repeat,
        "train_empty_response_rate": 0.0,
    }


def _micro(step: int, micro: int, category: str, prompt_id: str, kl: float, grad: float, quality: bool) -> dict:
    return {
        "schema_version": 2,
        "step": step,
        "micro_index": micro,
        "prompt_id": prompt_id,
        "category": category,
        "sample_keys": [f"{step}:{micro}:0"],
        "policy_diagnostics": {
            "reference_kl_mean": kl / 20,
            "reference_kl_p95": kl / 2,
            "reference_kl_max": kl,
            "ratio_p95": 1.1,
            "ratio_max": 1.2,
        },
        "micro_grad_norm_scaled": grad / 8,
        "micro_grad_norm_unscaled": grad,
        "accumulated_grad_norm": grad,
        "reward_mean": 0.8 if quality else 0.1,
        "train_validator_pass_rate": 1.0 if quality else 0.0,
        "train_max_length_hit_rate": 1.0 if quality else 0.0,
        "train_natural_end_rate": 0.0 if quality else 1.0,
        "train_repetition_penalty_mean": 0.2 if quality else 0.0,
        "train_empty_response_rate": 0.0,
    }


class RLSpikeSourcesTest(unittest.TestCase):
    def test_sample_stats_are_category_aware(self):
        rows = [_sample(1, "format", 0.1, 0.0, False), _sample(1, "format", 0.8, 1.0, True)]
        stats = _sample_stats(rows)[1]
        self.assertEqual(stats["sample_count"], 2)
        self.assertAlmostEqual(stats["validator_pass_rate"], 0.5)
        self.assertAlmostEqual(stats["max_length_hit_rate"], 0.5)
        self.assertEqual(stats["category_stats"]["format"]["count"], 2)

    def test_spike_and_reward_transfer_gap_are_flagged(self):
        report = analyze_run_data(
            "grpo_control_seed42",
            [_step(1, 0.1, 0.1, 0.1, 0.0), _step(2, 0.5, 0.8, 0.7, 0.2)],
            [_sample(1, "format", 0.1, 0.0, False), _sample(2, "format", 0.8, 1.0, True)],
            {"validator_pass": 1, "validator_pass_rate": 0.25, "safety_pass": 1, "termination_pass": 1},
            [{"step": 2, "metrics": {"validator_pass": 1, "safety_pass": 1, "termination_pass": 1}}],
            {"validator_pass": 1, "safety_pass": 1, "termination_pass": 1, "natural_end": 1},
        )
        self.assertIn("kl_max_tail_concentration", report["signal_counts"])
        self.assertIn("gradient_clipping_active", report["signal_counts"])
        self.assertIn("train_validator_gain_without_validation_gain", report["warnings"])
        self.assertIn("reward_gain_without_validation_gain", report["warnings"])
        self.assertEqual(report["selected_metrics"]["validator_pass"], 1)

    def test_run_audit_writes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "root"
            run = root / "grpo_control_seed42"
            run.mkdir(parents=True)
            (run / "selection.json").write_text(
                json.dumps({"selected_step": 1, "checkpoints": [{"step": 1, "metrics": {"validator_pass": 1, "safety_pass": 1, "termination_pass": 1, "natural_end": 1}}]}),
                encoding="utf-8",
            )
            (run / "baseline_validation.json").write_text(
                json.dumps({"metrics": {"validator_pass": 1, "validator_pass_rate": 0.25, "safety_pass": 1, "termination_pass": 1}}),
                encoding="utf-8",
            )
            (run / "step_summaries.jsonl").write_text(json.dumps(_step(1, 0.1, 0.1, 0.1, 0.0)) + "\n", encoding="utf-8")
            (run / "samples.jsonl").write_text(json.dumps(_sample(1, "format", 0.1, 0.0, False)) + "\n", encoding="utf-8")
            (run / "validation_history.jsonl").write_text(json.dumps({"step": 1, "metrics": {"validator_pass": 1}}) + "\n", encoding="utf-8")
            output = root / "spike_source_audit"
            summary = run_audit(root, output)
            self.assertEqual(summary["run_count"], 1)
            self.assertTrue((output / "summary.json").is_file())
            with self.assertRaises(FileExistsError):
                run_audit(root, output)

    def test_discover_runs_excludes_smoke_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "grpo_control_seed42").mkdir()
            (root / "grpo_smoke_control_seed42").mkdir()
            self.assertEqual([item.name for item in discover_runs(root)], ["grpo_control_seed42"])
            self.assertEqual(len(discover_runs(root, include_smoke=True)), 2)

    def test_microbatch_attribution_localizes_kl_gradient_and_quality(self):
        micro_rows = [
            _micro(1, 0, "format", "format_a", 0.5, 8.0, True),
            _micro(1, 1, "format", "format_b", 0.4, 7.0, True),
            _micro(2, 0, "format", "format_a", 0.3, 6.0, True),
            _micro(2, 1, "reasoning", "reasoning_a", 0.02, 1.0, False),
        ]
        samples = [
            {
                "step": row["step"],
                "micro_index": row["micro_index"],
                "sample_key": row["sample_keys"][0],
                "prompt_id": row["prompt_id"],
                "category": row["category"],
                "candidate_index": 0,
                "reward": row["reward_mean"],
                "components": {
                    "validator_reward": row["train_validator_pass_rate"],
                    "repetition_penalty": row["train_repetition_penalty_mean"],
                },
                "generated_tokens": 16 if row["train_max_length_hit_rate"] else 8,
                "finished_naturally": not row["train_max_length_hit_rate"],
                "empty_response": False,
                "max_length_hit": bool(row["train_max_length_hit_rate"]),
            }
            for row in micro_rows
        ]
        report = analyze_run_data(
            "grpo_control_seed42",
            [_step(1, 0.1, 0.1, 0.1, 0.0), _step(2, 0.5, 0.8, 0.7, 0.2)],
            samples,
            {"validator_pass": 1, "validator_pass_rate": 0.25, "safety_pass": 1, "termination_pass": 1},
            [{"step": 2, "metrics": {"validator_pass": 1, "safety_pass": 1, "termination_pass": 1}}],
            {"validator_pass": 1, "safety_pass": 1, "termination_pass": 1, "natural_end": 1},
            micro_rows=micro_rows,
            microbatch_available=True,
        )
        attribution = report["microbatch_attribution"]
        self.assertEqual(attribution["status"], "SOURCE_LOCALIZED_DIAGNOSTIC")
        self.assertEqual(attribution["category_summary"]["format"]["microbatch_count"], 3)
        self.assertEqual(len(attribution["top_kl"]["top_rows"]), 4)
        self.assertTrue(attribution["overlap"]["top3_kl_gradient_overlap"])

    def test_required_missing_microbatch_is_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "root"
            run = root / "grpo_control_seed42"
            run.mkdir(parents=True)
            (run / "selection.json").write_text(
                json.dumps({"selected_step": 1, "checkpoints": [{"step": 1, "metrics": {"validator_pass": 1}}]}),
                encoding="utf-8",
            )
            (run / "baseline_validation.json").write_text(
                json.dumps({"metrics": {"validator_pass": 1, "validator_pass_rate": 0.25}}),
                encoding="utf-8",
            )
            (run / "step_summaries.jsonl").write_text(json.dumps(_step(1, 0.1, 0.1, 0.1, 0.0)) + "\n", encoding="utf-8")
            (run / "samples.jsonl").write_text(json.dumps(_sample(1, "format", 0.1, 0.0, False)) + "\n", encoding="utf-8")
            (run / "validation_history.jsonl").write_text(json.dumps({"step": 1, "metrics": {"validator_pass": 1}}) + "\n", encoding="utf-8")
            summary = run_audit(root, root / "audit", require_microbatch=True)
            self.assertEqual(summary["status"], "TELEMETRY_INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
