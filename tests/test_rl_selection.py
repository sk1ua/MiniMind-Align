import unittest

from evaluation.rl_selection import quality_triggered, select_best_checkpoint


class RLSelectionTests(unittest.TestCase):
    def test_quality_gate_uses_percentage_points(self):
        baseline = {"safety_pass_rate": 1.0, "termination_pass_rate": 0.8}
        self.assertFalse(quality_triggered({"safety_pass_rate": 0.91, "termination_pass_rate": 0.71}, baseline)["triggered"])
        self.assertTrue(quality_triggered({"safety_pass_rate": 0.899, "termination_pass_rate": 0.8}, baseline)["triggered"])

    def test_selection_uses_tie_breaks_and_excludes_triggered(self):
        records = [
            {"step": 4, "checkpoint": "a", "metrics": {"validator_pass": 10, "safety_pass": 2, "termination_pass": 2, "natural_end": 3, "average_repeat_3gram": 0.1}},
            {"step": 8, "checkpoint": "b", "metrics": {"validator_pass": 10, "safety_pass": 2, "termination_pass": 2, "natural_end": 3, "average_repeat_3gram": 0.05}},
            {"step": 12, "checkpoint": "c", "kl_triggered": True, "metrics": {"validator_pass": 99}},
        ]
        self.assertEqual(select_best_checkpoint(records)["checkpoint"], "b")

    def test_selection_returns_none_when_all_are_gated(self):
        self.assertIsNone(select_best_checkpoint([{"checkpoint": "a", "quality_triggered": True, "metrics": {"validator_pass": 1}}]))


if __name__ == "__main__":
    unittest.main()
