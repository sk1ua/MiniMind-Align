from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluation.audit_rl_category_weighting import (
    CATEGORIES,
    ensure_empty_output_dir,
    group_advantages,
    summarize_group,
)


class CategoryWeightingAuditTest(unittest.TestCase):
    def test_group_advantages_use_population_std(self) -> None:
        advantages = group_advantages([0.0, 1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(advantages), 0.0, places=6)
        self.assertGreater(max(advantages), 0.0)
        self.assertLess(min(advantages), 0.0)

    def test_collapsed_group_has_zero_advantage_signal(self) -> None:
        rows = [
            {"step": 1, "prompt_id": "p", "micro_index": 0, "category": "reasoning", "reward": 0.5}
            for _ in range(8)
        ]
        summary = summarize_group(rows)
        self.assertEqual(summary["sample_count"], 8)
        self.assertTrue(summary["collapsed"])
        self.assertEqual(summary["advantage_nonzero_count"], 0)

    def test_category_registry_is_complete(self) -> None:
        self.assertEqual(len(CATEGORIES), 8)
        self.assertEqual(len(set(CATEGORIES)), 8)

    def test_non_empty_output_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "audit"
            output.mkdir()
            (output / "existing.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_empty_output_dir(output)


if __name__ == "__main__":
    unittest.main()
