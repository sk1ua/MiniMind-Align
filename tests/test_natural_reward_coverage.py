import json
import tempfile
import unittest
from pathlib import Path

from evaluation.audit_natural_reward_coverage import _probe_summary, audit_paths


def _manifest_row(
    prompt_id: str,
    category: str = "conciseness",
    family: str = "family_a",
    chosen: str = "变量是程序中保存可变化值的名称。",
) -> dict:
    metadata = {
        "max_chars": 50,
        "required_terms": ["变量", "保存", "值"],
    }
    if category == "safety":
        metadata = {"required_markers": ["风险", "建议"]}
        chosen = "风险：存在不确定性。建议：先停止操作并核实。"
    return {
        "id": prompt_id,
        "category": category,
        "family": family,
        "difficulty": "basic",
        "prompt": "test prompt",
        "chosen": chosen,
        "metadata": metadata,
    }


def _sample(prompt_id: str, response: str, step: int, candidate: int, reason: str) -> dict:
    if response == "变量是程序中保存可变化值的名称。":
        components = {
            "validator_reward": 1.0,
            "parse_reward": 0.0,
            "field_reward": 0.0,
            "item_count_reward": 0.0,
            "arithmetic_reward": 0.0,
            "format_reward": 0.0,
            "termination_reward": 0.1,
            "repetition_penalty": 0.0,
        }
        reward = 1.1
    else:
        components = {
            "validator_reward": 0.0,
            "parse_reward": 0.0,
            "field_reward": 0.0,
            "item_count_reward": 0.0,
            "arithmetic_reward": 0.0,
            "format_reward": 0.0,
            "termination_reward": 0.1,
            "repetition_penalty": 0.0,
        }
        reward = 0.1
    return {
        "step": step,
        "micro_index": 0,
        "sample_key": f"{step}:0:{candidate}",
        "prompt_id": prompt_id,
        "response": response,
        "reward": reward,
        "rule_reward": reward,
        "components": components,
        "generated_tokens": 8,
        "termination_reason": reason,
        "finished_naturally": reason == "eos",
        "max_length_hit": reason == "max_new_tokens",
        "empty_response": False,
    }


class NaturalRewardCoverageTests(unittest.TestCase):
    def test_rule_reward_probes_exercise_validator_and_termination_variation(self):
        result = _probe_summary([_manifest_row("p1")])
        self.assertTrue(result["probe_signal_variation_present"])
        self.assertEqual(result["variant_summaries"]["chosen"]["validator_pass_count"], 1)
        self.assertEqual(result["variant_summaries"]["empty"]["validator_pass_count"], 0)

    def test_replay_and_component_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "train.jsonl"
            manifest.write_text(
                "\n".join(json.dumps(_manifest_row("p1"), ensure_ascii=False) for _ in range(1)) + "\n",
                encoding="utf-8",
            )
            samples = root / "samples"
            samples.mkdir()
            (samples / "samples.jsonl").write_text(
                "\n".join([
                    json.dumps(_sample("p1", "变量是程序中保存可变化值的名称。", 1, 0, "eos"), ensure_ascii=False),
                    json.dumps(_sample("p1", "错误输出", 1, 1, "max_new_tokens"), ensure_ascii=False),
                ]) + "\n",
                encoding="utf-8",
            )
            result = audit_paths(manifest, [samples], root / "audit")
            current = result["current_generation"]
            self.assertEqual(current["persisted_reward_mismatch_count"], 0)
            self.assertEqual(current["persisted_component_mismatch_count"], 0)
            self.assertEqual(current["validator_pass_count"], 1)
            self.assertEqual(current["nonzero_reward_spread_group_count"], 1)

    def test_category_coverage_and_termination_reason_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_rows = [_manifest_row("p1"), _manifest_row("p2", category="safety", family="family_b")]
            manifest = root / "train.jsonl"
            manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in manifest_rows) + "\n", encoding="utf-8")
            samples = root / "samples"
            samples.mkdir()
            (samples / "samples.jsonl").write_text(
                json.dumps(_sample("p1", "错误输出", 1, 0, "eos"), ensure_ascii=False) + "\n"
                + json.dumps(_sample("p1", "错误输出", 1, 1, "max_new_tokens"), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            result = audit_paths(manifest, [samples], root / "audit")
            current = result["current_generation"]
            self.assertEqual(current["category_coverage_count"], 1)
            self.assertEqual(current["category_coverage_total"], 2)
            self.assertIn("CURRENT_CATEGORY_COVERAGE_INCOMPLETE", result["warnings"])
            self.assertIn("TERMINATION_REWARD_NOT_SENSITIVE_TO_EOS_REASON", result["warnings"])

    def test_audit_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(_manifest_row("p1")) + "\n", encoding="utf-8")
            samples = root / "samples"
            samples.mkdir()
            (samples / "samples.jsonl").write_text("", encoding="utf-8")
            output = root / "audit"
            output.mkdir()
            (output / "existing.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                audit_paths(manifest, [samples], output)


if __name__ == "__main__":
    unittest.main()
