import json
import tempfile
import unittest
from pathlib import Path

from align.rl_rules import group_advantages, parse_controlled_reward_pattern
from evaluation.audit_controlled_nonzero_advantage import audit_run


class ControlledNonzeroAdvantageSmokeTest(unittest.TestCase):
    def test_pattern_is_strict_and_produces_nonzero_advantages(self):
        pattern = parse_controlled_reward_pattern("1.0,0.0", 2)
        self.assertEqual(pattern, [1.0, 0.0])
        advantages = group_advantages(__import__("torch").tensor(pattern), 2)
        self.assertGreater(float(advantages.abs().max()), 0.0)

    def test_pattern_rejects_equal_wrong_length_and_nonfinite_values(self):
        with self.assertRaises(ValueError):
            parse_controlled_reward_pattern("1.0,1.0", 2)
        with self.assertRaises(ValueError):
            parse_controlled_reward_pattern("1.0", 2)
        with self.assertRaises(ValueError):
            parse_controlled_reward_pattern("nan,0.0", 2)

    def test_audit_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "experiment"
            output = Path(temp) / "audit"
            output.mkdir()
            (output / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                audit_run(root, "run", output, [1.0, 0.0])

    def test_audit_requires_active_artifact_set(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "experiment"
            run = root / "run"
            run.mkdir(parents=True)
            output = root / "audit"
            result = audit_run(root, "run", output, [1.0, 0.0])
            self.assertEqual(result["status"], "TELEMETRY_INCOMPLETE")
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "TELEMETRY_INCOMPLETE")

    def test_trainer_and_wrapper_expose_opt_in_only_control(self):
        project = Path(__file__).resolve().parents[1]
        trainer = (project / "trainer" / "train_grpo_lite.py").read_text(encoding="utf-8")
        wrapper = (project / "scripts" / "run_controlled_nonzero_advantage_smoke.sh").read_text(encoding="utf-8")
        self.assertIn('"--controlled-reward-pattern"', trainer)
        self.assertIn('reward_source = "controlled_reward_pattern"', trainer)
        self.assertIn("rl_controlled_nonzero_advantage_smoke_20260802", wrapper)
        self.assertIn("--controlled-reward-pattern", wrapper)
        self.assertNotIn("--mode cispo", wrapper.lower())
        self.assertNotIn("--mode=cispo", wrapper.lower())


if __name__ == "__main__":
    unittest.main()
