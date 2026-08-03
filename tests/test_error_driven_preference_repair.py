import json
import tempfile
import unittest
from pathlib import Path

from evaluation.prepare_error_driven_preference_repair import (
    conversation,
    ensure_empty,
    stable_key,
)


class ErrorDrivenPreferenceRepairTests(unittest.TestCase):
    def test_stable_selection_key(self):
        self.assertEqual(stable_key(42, "x"), stable_key(42, "x"))
        self.assertNotEqual(stable_key(42, "x"), stable_key(43, "x"))

    def test_conversation_shape(self):
        self.assertEqual(
            conversation("p", "a"),
            [
                {"role": "user", "content": "p"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_output_directory_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out"
            path.mkdir()
            (path / "existing.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_empty(path)

    def test_preference_rows_are_distinct_and_finite(self):
        rows = [
            {"id": "a", "chosen": conversation("p", "good"), "rejected": conversation("p", "bad")},
            {"id": "b", "chosen": conversation("q", "good"), "rejected": conversation("q", "bad")},
        ]
        self.assertEqual(len({row["id"] for row in rows}), 2)
        self.assertTrue(all(row["chosen"] != row["rejected"] for row in rows))
        json.dumps(rows, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
