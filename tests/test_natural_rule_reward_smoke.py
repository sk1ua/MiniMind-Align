import tempfile
import unittest
from pathlib import Path

from evaluation.audit_natural_rule_reward_smoke import (
    _rule_reward_source_checks,
    audit_run,
)


class NaturalRuleRewardSmokeTests(unittest.TestCase):
    def test_rule_reward_source_rejects_controlled_override(self):
        good = {
            "reward_source": "rule_reward",
            "controlled_reward_pattern": None,
            "controlled_reward_values": None,
            "rule_reward_values": [0.1, 0.1],
        }
        bad = {**good, "reward_source": "controlled_reward_pattern"}
        good_sample = {
            "reward_source": "rule_reward",
            "controlled_reward": None,
            "controlled_reward_delta": 0.0,
            "rule_reward": 0.1,
        }
        good_step = {
            "reward_source": "rule_reward",
            "controlled_reward_override_enabled": False,
            "controlled_reward_pattern": None,
        }
        checks = _rule_reward_source_checks([good], [good_sample], [good], [good_step])
        self.assertTrue(all(checks.values()))
        checks = _rule_reward_source_checks([bad], [good_sample], [good], [good_step])
        self.assertFalse(checks["micro_rule_reward_ok"])

    def test_audit_missing_artifacts_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            output = Path(tmp) / "audit"
            result = audit_run(root, "run", output)
            self.assertEqual(result["status"], "TELEMETRY_INCOMPLETE")
            self.assertTrue((output / "summary.json").exists())

    def test_audit_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            output = Path(tmp) / "audit"
            output.mkdir()
            (output / "existing.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                audit_run(root, "run", output)

    def test_wrapper_is_natural_reward_and_isolated(self):
        wrapper = Path(__file__).resolve().parents[1] / "scripts" / "run_natural_rule_reward_smoke.sh"
        text = wrapper.read_text(encoding="utf-8")
        self.assertIn("rl_natural_rule_reward_smoke_20260802", text)
        self.assertIn("REWARD_SOURCE=rule_reward", text)
        self.assertIn("--training-forward-mode fp32_no_autocast", text)
        self.assertNotIn("--controlled-reward-pattern", text)
        self.assertIn("evaluation/audit_natural_rule_reward_smoke.py", text)


if __name__ == "__main__":
    unittest.main()
