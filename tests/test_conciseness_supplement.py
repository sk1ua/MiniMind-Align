import unittest

from dataset.alignment_v2.prepare_conciseness_supplement_20260803 import NATIVE_SOURCE, build_rows
from dataset.alignment_v2.validators import validate_record


class ConcisenessSupplementTests(unittest.TestCase):
    def test_exactly_sixteen_native_rows_and_validator_pass(self):
        rows = build_rows()
        self.assertEqual(len(rows), 16)
        self.assertEqual({row["source"] for row in rows}, {NATIVE_SOURCE})
        self.assertEqual({row["category"] for row in rows}, {"conciseness"})
        for row in rows:
            record = {"conversations": [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["chosen"]}]}
            self.assertEqual(validate_record(record, row), (True, ""))

    def test_ids_families_prompts_are_unique(self):
        rows = build_rows()
        for key in ("id", "family", "prompt"):
            self.assertEqual(len(rows), len({row[key] for row in rows}))

    def test_generation_is_deterministic(self):
        self.assertEqual(build_rows(), build_rows())


if __name__ == "__main__":
    unittest.main()
