import json
import tempfile
import unittest
from pathlib import Path

from align.rl_rules import rule_reward
from evaluation.audit_balanced_output_quality import audit_paths


def _manifest_row(prompt_id: str, category: str, family: str, chosen: str, metadata: dict) -> dict:
    return {
        "id": prompt_id,
        "category": category,
        "family": family,
        "difficulty": "basic",
        "prompt": "test prompt",
        "chosen": chosen,
        "metadata": metadata,
    }


def _sample(manifest: dict, response: str, step: int, candidate: int, reason: str) -> dict:
    reward, components = rule_reward(
        manifest["category"], manifest["prompt"], response, manifest["metadata"]
    )
    return {
        "step": step,
        "micro_index": 0,
        "sample_key": f"{step}:0:{candidate}",
        "prompt_id": manifest["id"],
        "candidate_index": candidate,
        "response": response,
        "reward": reward,
        "rule_reward": reward,
        "components": components,
        "generated_tokens": min(16, max(1, len(response))),
        "termination_reason": reason,
        "eos_seen": reason == "eos",
        "finished_naturally": reason == "eos",
        "max_length_hit": reason == "max_new_tokens",
        "empty_response": not response.strip(),
    }


class BalancedOutputQualityAuditTests(unittest.TestCase):
    def test_replay_category_aggregation_and_quality_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            conciseness = _manifest_row(
                "p1", "conciseness", "family_a", "变量是程序中保存可变化值的名称。",
                {"max_chars": 50, "required_terms": ["变量", "保存", "值"]},
            )
            safety = _manifest_row(
                "p2", "safety", "family_b", "风险：存在不确定性。建议：先停止操作并核实。",
                {"required_markers": ["风险", "建议"]},
            )
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in (conciseness, safety)) + "\n",
                encoding="utf-8",
            )
            samples = root / "samples"
            samples.mkdir()
            rows = [
                _sample(conciseness, conciseness["chosen"], 1, 0, "eos"),
                _sample(conciseness, "错误输出", 1, 1, "max_new_tokens"),
                _sample(safety, safety["chosen"], 1, 0, "eos"),
                _sample(safety, "风险：请注意。", 1, 1, "eos"),
            ]
            (samples / "samples.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            result = audit_paths(manifest, samples, root / "audit")
            current = result["current_generation"]
            self.assertEqual(result["status"], "OUTPUT_QUALITY_FAILURE_CONCENTRATED_DIAGNOSTIC")
            self.assertEqual(current["category_coverage_count"], 2)
            self.assertEqual(current["validator_pass_count"], 2)
            self.assertEqual(current["replay_reward_mismatch_count"], 0)
            self.assertEqual(current["replay_component_mismatch_count"], 0)
            self.assertIn("conciseness", current["category_summary"])
            self.assertIn("repeat_3gram_ratio", current["category_summary"]["safety"])
            self.assertTrue((root / "audit" / "sample_diagnostics.jsonl").exists())

    def test_sparse_validator_signal_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_row = _manifest_row(
                "p1", "conciseness", "family_a", "变量是程序中保存可变化值的名称。",
                {"max_chars": 50, "required_terms": ["变量", "保存", "值"]},
            )
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(manifest_row, ensure_ascii=False) + "\n", encoding="utf-8")
            samples = root / "samples"
            samples.mkdir()
            rows = [_sample(manifest_row, "", 1, index, "eos") for index in range(20)]
            (samples / "samples.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            result = audit_paths(manifest, samples, root / "audit")
            self.assertEqual(result["status"], "OUTPUT_QUALITY_SIGNAL_SPARSE_DIAGNOSTIC")
            self.assertIn("VALIDATOR_SIGNAL_SPARSE", result["warnings"])

    def test_audit_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            row = _manifest_row(
                "p1", "conciseness", "family_a", "变量是程序中保存可变化值的名称。",
                {"max_chars": 50, "required_terms": ["变量", "保存", "值"]},
            )
            manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            samples = root / "samples"
            samples.mkdir()
            (samples / "samples.jsonl").write_text("", encoding="utf-8")
            output = root / "audit"
            output.mkdir()
            (output / "existing.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                audit_paths(manifest, samples, output)


if __name__ == "__main__":
    unittest.main()
