import json
import tempfile
import unittest
from pathlib import Path

from evaluation.prepare_balanced_coverage_manifests import (
    CATEGORIES,
    prepare_balanced_manifests,
    select_one_per_category,
)


def _row(category: str, index: int, split: str) -> dict:
    return {
        "id": f"{split}_{category}_{index}",
        "split": split,
        "category": category,
        "family": f"{category}_{split}_family_{index}",
        "difficulty": "basic",
        "prompt": f"prompt {category} {index}",
        "chosen": f"chosen {category} {index}",
        "metadata": {"marker": category},
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class BalancedCoverageManifestTests(unittest.TestCase):
    def test_selects_one_deterministic_row_per_category(self):
        rows = [_row(category, index, "train") for category in CATEGORIES for index in range(4)]
        first = select_one_per_category(rows, CATEGORIES, 42)
        second = select_one_per_category(rows, CATEGORIES, 42)
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertEqual([row["category"] for row in first], list(CATEGORIES))
        self.assertTrue(all(row["metadata"] for row in first))

    def test_writes_balanced_outputs_and_overlap_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            _write(train, [_row(category, index, "train") for category in CATEGORIES for index in range(2)])
            _write(validation, [_row(category, index, "validation") for category in CATEGORIES for index in range(2)])
            result = prepare_balanced_manifests(train, validation, root / "out")
            self.assertEqual(result["train_count"], 8)
            self.assertEqual(result["validation_count"], 8)
            self.assertEqual(result["train_validation_id_overlap"], 0)
            self.assertEqual(result["train_validation_family_overlap"], 0)
            self.assertTrue((root / "out/balanced_train_manifest.jsonl").exists())
            self.assertTrue((root / "out/balanced_validation_manifest.jsonl").exists())
            self.assertTrue((root / "out/selection.json").exists())

    def test_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            _write(train, [_row(category, 0, "train") for category in CATEGORIES])
            _write(validation, [_row(category, 0, "validation") for category in CATEGORIES])
            output = root / "out"
            output.mkdir()
            (output / "existing.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_balanced_manifests(train, validation, output)


if __name__ == "__main__":
    unittest.main()
