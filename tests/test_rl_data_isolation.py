import unittest

from evaluation.audit_rl_reward_hacking import detect_warnings
from scripts.prepare_rl_data_isolation import (
    EXPECTED_CATEGORIES,
    select_native_train,
    select_validation_slice,
    validate_selection,
)


def row(category: str, index: int, *, source: str = "alignment_v2_programmatic_v1", metadata: bool = True) -> dict:
    return {
        "id": f"{category}_{index:04d}",
        "category": category,
        "family": f"{category}_family_{index:02d}",
        "source": source,
        "metadata": {"seed": index} if metadata else {},
    }


class RLDataIsolationTests(unittest.TestCase):
    def test_native_train_selection_is_balanced_and_seeded(self):
        rows = [
            row(category, index)
            for category in EXPECTED_CATEGORIES
            for index in range(5)
        ]
        rows.extend([row("format", 99, source="dataset/alignment_v1/final/sft_train.jsonl")])
        first = select_native_train(rows, per_category=2, seed=42)
        second = select_native_train(rows, per_category=2, seed=42)
        self.assertEqual([item["id"] for item in first], [item["id"] for item in second])
        self.assertEqual(len(first), 16)
        self.assertEqual({item["category"] for item in first}, set(EXPECTED_CATEGORIES))
        self.assertTrue(all(item["metadata"] for item in first))
        self.assertNotIn("format_0099", {item["id"] for item in first})

    def test_validation_slice_uses_offset_and_checks_overlap(self):
        rows = [
            row(category, index)
            for category in EXPECTED_CATEGORIES
            for index in range(6)
        ]
        selected = select_validation_slice(rows, per_category=2, offset=2)
        self.assertEqual(len(selected), 16)
        self.assertTrue(all(item["id"].endswith(("02", "03")) for item in selected))
        existing = [item for item in rows if item["id"].endswith(("00", "01"))]
        report = validate_selection(selected, [], existing)
        self.assertEqual(report["overlap"]["validation_existing_ids"], [])

    def test_reward_hacking_audit_emits_diagnostic_only_flags(self):
        steps = [
            {
                "reward_mean": 0.10,
                "train_validator_pass_rate": 0.10,
                "train_empty_response_rate": 0.00,
                "train_max_length_hit_rate": 0.00,
                "train_repetition_penalty_mean": 0.01,
            },
            {
                "reward_mean": 0.30,
                "train_validator_pass_rate": 0.30,
                "train_empty_response_rate": 0.20,
                "train_max_length_hit_rate": 0.20,
                "train_repetition_penalty_mean": 0.04,
            },
        ]
        warnings = detect_warnings(
            steps,
            {"validator_pass": 5, "natural_end_rate": 0.9, "average_repeat_3gram": 0.01},
            {"validator_pass": 5, "natural_end_rate": 0.9, "average_repeat_3gram": 0.01},
        )
        codes = {item["code"] for item in warnings}
        self.assertIn("reward_gain_without_validation_gain", codes)
        self.assertIn("train_validator_gain_without_validation_gain", codes)
        self.assertIn("empty_response_increase", codes)
        self.assertIn("max_length_hit_increase", codes)
        self.assertIn("repetition_penalty_increase", codes)


if __name__ == "__main__":
    unittest.main()
