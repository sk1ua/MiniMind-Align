import tempfile
import unittest
from pathlib import Path

from evaluation.audit_natural_reward_diversity import (
    _group_id,
    _component_summary,
    audit_paths,
    source_label,
)


class NaturalRewardDiversityTests(unittest.TestCase):
    def test_source_labels_keep_current_and_legacy_separate(self):
        self.assertEqual(
            source_label(Path("results/experiments/rl_natural_rule_reward_smoke_20260802/run/samples.jsonl")),
            "e027_natural_rule_reward",
        )
        self.assertEqual(
            source_label(Path("results/experiments/rl_data_isolation_reload_fixed_20260801/grpo_seed42/samples.jsonl")),
            "e010_v2_reload_fixed_legacy_schema",
        )
        self.assertEqual(
            source_label(Path("results/experiments/rl_method_upgrade_20260801/grpo_seed42/samples.jsonl")),
            "e009_v1_legacy_schema",
        )

    def test_component_summary_tracks_coverage_and_nonzero(self):
        rows = [
            {"components": {"validator_reward": 1.0}},
            {"components": {"validator_reward": 0.0}},
            {"components": {}},
        ]
        summary = _component_summary(rows, "validator_reward")
        self.assertEqual(summary["observed_count"], 2)
        self.assertEqual(summary["missing_count"], 1)
        self.assertEqual(summary["nonzero_count"], 1)
        self.assertAlmostEqual(summary["coverage_rate"], 2 / 3)

    def test_group_id_distinguishes_microbatch(self):
        path = Path("results/experiments/rl_natural_rule_reward_smoke_20260802/run/samples.jsonl")
        first = _group_id(path, {"step": 1, "prompt_id": "p", "micro_index": 0})
        second = _group_id(path, {"step": 1, "prompt_id": "p", "micro_index": 1})
        self.assertNotEqual(first, second)

    def test_audit_detects_current_reward_collapse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            source = root / "results/experiments/rl_natural_rule_reward_smoke_20260802/run"
            source.mkdir(parents=True)
            rows = []
            for candidate in range(2):
                rows.append({
                    "step": 1,
                    "micro_index": 0,
                    "prompt_id": "p0",
                    "candidate_index": candidate,
                    "reward": 0.1,
                    "reward_source": "rule_reward",
                    "components": {"validator_reward": 0.0, "termination_reward": 0.1, "repetition_penalty": 0.0},
                })
            (source / "samples.jsonl").write_text(
                "\n".join(__import__("json").dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            output = Path(tmp) / "audit"
            result = audit_paths([root], output)
            self.assertEqual(result["status"], "NATURAL_REWARD_DIVERSITY_AUDIT_COLLAPSE_CONFIRMED")
            self.assertIn("NATURAL_REWARD_GROUP_COLLAPSE", result["warnings"])

    def test_audit_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            output = Path(tmp) / "audit"
            output.mkdir()
            (output / "existing.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                audit_paths([root], output)


if __name__ == "__main__":
    unittest.main()
