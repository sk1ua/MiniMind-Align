import json
import tempfile
import unittest
from pathlib import Path

from evaluation.audit_quality_signal_repair import (
    ALL_CATEGORIES,
    TARGET_CATEGORIES,
    compare_evaluations,
    ensure_empty_output_dir,
    select_repair_rows,
    termination_reason,
)


def _row(index: int, category: str) -> dict:
    metadata = {
        "max_chars": 80,
        "required_terms": ["变量", "保存", "值"],
    }
    chosen = "变量是程序中保存可变化值的名称。"
    if category == "format":
        metadata = {"format_type": "csv", "expected": [["name"], ["x"]]}
        chosen = "name\nx"
    if category == "instruction":
        metadata = {"count": 2, "separator": "，", "allowed_words": ["数组", "链表"]}
        chosen = "数组，链表"
    if category == "reasoning":
        metadata = {"answer": 3}
        chosen = "1 + 2 = 3"
    if category == "repetition":
        metadata = {"count": 2}
        chosen = "甲：第一项\n乙：第二项"
    if category == "safety":
        metadata = {"required_markers": ["风险", "建议"]}
        chosen = "风险：需要注意。建议：先停止并核实。"
    if category == "termination":
        metadata = {"max_chars": 80}
        chosen = "请先确认信息。"
    if category == "uncertainty":
        metadata = {"required_markers": ["无法确认", "实时信息"]}
        chosen = "无法确认，因为缺少实时信息。"
    return {
        "id": f"v2_{category}_{index:04d}",
        "category": category,
        "family": f"{category}_family_{index:04d}",
        "prompt": f"prompt {category} {index}",
        "chosen": chosen,
        "metadata": metadata,
        "source": "alignment_v2_programmatic_v1",
    }


class QualitySignalRepairTests(unittest.TestCase):
    def test_selection_is_deterministic_and_balanced(self):
        rows = [_row(index, category) for category in ALL_CATEGORIES for index in range(120)]
        first, first_meta = select_repair_rows(rows)
        second, second_meta = select_repair_rows(rows)
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertEqual(first_meta, second_meta)
        self.assertEqual(len(first), 576)
        self.assertEqual(first_meta["category_counts"][TARGET_CATEGORIES[0]], 96)
        self.assertEqual(first_meta["category_counts"]["safety"], 32)

    def test_selection_fails_closed_when_category_is_short(self):
        rows = [_row(index, category) for category in ALL_CATEGORIES for index in range(32)]
        with self.assertRaises(ValueError):
            select_repair_rows(rows)

    def test_termination_reason_uses_eos_and_length(self):
        self.assertEqual(termination_reason([1, 2, 9], 9, 128), "eos")
        self.assertEqual(termination_reason([1] * 128, 9, 128), "max_new_tokens")
        self.assertEqual(termination_reason([1] * 4, 9, 128), "no_eos_short_generation")

    def test_compare_applies_release_gain_and_quality_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_dir = root / "baseline"
            candidate_dir = root / "candidate"
            baseline_dir.mkdir()
            candidate_dir.mkdir()

            def write_eval(path: Path, pass_count: int, safety: int, termination: int, repeat: float) -> None:
                full_details = []
                release_details = []
                for index in range(160):
                    passed = index < pass_count
                    row = {
                        "id": f"p{index}",
                        "category": "safety" if index < 12 else "termination" if index < 22 else "conciseness",
                        "validator_pass": passed,
                    }
                    full_details.append(row)
                    if index < 32:
                        release_details.append(row)
                full = {"count": 160, "validator_pass": pass_count, "validator_pass_rate": pass_count / 160, "safety_pass_rate": safety / 12, "termination_pass_rate": termination / 10, "max_length_hit_rate": 0.01, "natural_end_rate": 0.99, "average_repeat_3gram": repeat, "categories": {category: {"validator_pass": 0} for category in ALL_CATEGORIES}}
                release = {"count": 32, "validator_pass": 13 if path == baseline_dir else 16, "validator_pass_rate": (13 if path == baseline_dir else 16) / 32, "safety_pass_rate": safety / 12, "termination_pass_rate": termination / 10, "max_length_hit_rate": 0.01, "natural_end_rate": 0.99, "average_repeat_3gram": repeat, "categories": {category: {"validator_pass": 0} for category in ALL_CATEGORIES}}
                release["categories"]["conciseness"]["validator_pass"] = 3 if path != baseline_dir else 0
                release["categories"]["format"]["validator_pass"] = 2
                payload = {"checkpoint_reload_ok": True, "row_linkage_ok": True, "full": full, "release": release}
                (path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
                (path / "full_generation.jsonl").write_text("\n".join(json.dumps(row) for row in full_details) + "\n", encoding="utf-8")

            write_eval(baseline_dir, 65, 12, 10, 0.10)
            write_eval(candidate_dir, 70, 12, 10, 0.11)
            result = compare_evaluations(baseline_dir, candidate_dir, root / "decision")
            self.assertEqual(result["status"], "QUALITY_REPAIR_PASS_DIAGNOSTIC")
            self.assertEqual(result["delta"]["release_validator_pass"], 3)

    def test_nonempty_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            output.mkdir()
            (output / "existing.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_empty_output_dir(output)


if __name__ == "__main__":
    unittest.main()
