import json
import tempfile
import unittest
from pathlib import Path

from evaluation.audit_reference_kl_semantics import (
    analytic_fixtures,
    ensure_empty_output_dir,
    independent_aggregate,
    independent_reference_kl_values,
)


class ReferenceKLSemanticsTest(unittest.TestCase):
    def test_reference_kl_formula_and_completion_mask(self):
        values = independent_reference_kl_values(
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0],
            [1, 0, 1],
        )
        self.assertEqual(values, [0.0, 0.0])

    def test_aggregate_is_token_weighted_across_microbatches(self):
        result = independent_aggregate([[0.0, 2.0], [4.0]])
        self.assertEqual(result["token_count"], 3.0)
        self.assertAlmostEqual(result["reference_kl_mean"], 2.0)
        self.assertAlmostEqual(result["reference_kl_p95"], 3.8)

    def test_analytic_fixtures_pass(self):
        self.assertTrue(all(analytic_fixtures().values()))

    def test_output_directory_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "audit"
            output.mkdir()
            (output / "summary.json").write_text(json.dumps({"old": True}), encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_empty_output_dir(output)


if __name__ == "__main__":
    unittest.main()
