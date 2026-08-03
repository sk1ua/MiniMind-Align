import unittest
from pathlib import Path


class UpdateScaleWrapperContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.script = Path(__file__).resolve().parents[1] / "scripts" / "run_rl_update_scale_diagnostic.sh"
        self.text = self.script.read_text(encoding="utf-8")

    def test_matrix_isolated_and_conditions_are_explicit(self) -> None:
        self.assertIn("rl_update_scale_diagnostic_20260801", self.text)
        self.assertIn('"control": {"learning_rate": 3e-7, "accumulation_steps": 8, "max_grad_norm": 1.0}', self.text)
        self.assertIn('"low_lr": {"learning_rate": 1e-7, "accumulation_steps": 8, "max_grad_norm": 1.0}', self.text)
        self.assertIn('"clip_half": {"learning_rate": 3e-7, "accumulation_steps": 8, "max_grad_norm": 0.5}', self.text)

    def test_telemetry_and_resource_guards_are_required(self) -> None:
        for flag in (
            "--microbatch-log-path",
            "--microbatch-gradient-norm",
            "--interleave-categories",
            "--kl-threshold",
            "--kl-patience",
            "--quality-drop-points",
            "sleep 60",
            "timeout --signal=TERM",
        ):
            self.assertIn(flag, self.text)

    def test_audits_are_diagnostic_only_and_do_not_reuse_runs(self) -> None:
        self.assertIn("refusing to reuse existing experiment directory", self.text)
        self.assertIn("evaluation/audit_rl_stability.py", self.text)
        self.assertIn("evaluation/audit_rl_spike_sources.py", self.text)
        self.assertIn("--require-microbatch", self.text)


if __name__ == "__main__":
    unittest.main()
