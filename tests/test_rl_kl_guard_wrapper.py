import unittest
from pathlib import Path


class RLKLGuardWrapperContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.script = Path(__file__).resolve().parents[1] / "scripts" / "run_rl_kl_guard_diagnostic.sh"
        self.text = self.script.read_text(encoding="utf-8")

    def test_guard_scope_and_backoff_are_explicit(self) -> None:
        for flag in (
            "--post-step-kl-target 0.005",
            "--kl-backoff-factor 0.5",
            "--kl-max-backoffs 3",
            "--interleave-categories",
            "--microbatch-log-path",
            "--microbatch-gradient-norm",
        ):
            self.assertIn(flag, self.text)

    def test_resource_and_directory_guards_are_present(self) -> None:
        for text in (
            "refusing to reuse existing experiment directory",
            "refusing to reuse existing tmux session",
            "sleep 60",
            "timeout --signal=TERM",
            "RL_GUARD_HARD_LIMIT_SECONDS",
        ):
            self.assertIn(text, self.text)

    def test_audit_and_isolated_root_are_explicit(self) -> None:
        self.assertIn("rl_kl_guard_diagnostic_20260802", self.text)
        self.assertIn("evaluation/audit_rl_kl_guard.py", self.text)
        self.assertIn("rl_corrected_balanced_diagnostic_20260802", self.text)


if __name__ == "__main__":
    unittest.main()
