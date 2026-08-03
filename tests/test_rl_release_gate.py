import unittest
from pathlib import Path

from evaluation.audit_rl_release_gate import quality_gate


def _step(step: int, previous: str | None, current: str) -> dict:
    return {
        "step": step,
        "optimizer_step_applied": True,
        "optimizer_step_rejected": False,
        "post_step_kl_gate_mode": "fp32_no_autocast",
        "post_step_kl_gate_passed": True,
        "post_step_kl_gate_mean": 0.0001,
        "pre_step_policy_state_digest": previous,
        "pre_step_optimizer_state_digest": f"opt-{previous}",
        "kl_guard_attempt_history": [{
            "accepted": True,
            "post_attempt_policy_state_digest": current,
            "post_attempt_optimizer_state_digest": f"opt-{current}",
        }],
    }


class RLReleaseGateTests(unittest.TestCase):
    def test_balanced_preflight_passes_only_with_all_quality_evidence(self):
        steps = [_step(1, "start", "after1"), _step(2, "after1", "after2")]
        result = quality_gate(
            quality_generation={
                "max_length_hit_rate": 0.20,
                "natural_end_rate": 0.60,
            },
            validation_metrics={
                "validator_pass": 4,
                "safety_pass_rate": 0.5,
                "termination_pass_rate": 0.5,
            },
            baseline_metrics={
                "safety_pass_rate": 0.5,
                "termination_pass_rate": 0.5,
            },
            step_rows=steps,
            expected_steps=2,
            checkpoint_ok=True,
            replay_ok=True,
            continuity_ok=True,
            finite_ok=True,
        )
        self.assertEqual(result["status"], "PREFLIGHT_PASS")
        self.assertTrue(all(result["criteria"].values()))

    def test_max_length_or_natural_end_blocks_formal_rl(self):
        steps = [_step(1, "start", "after1")]
        result = quality_gate(
            quality_generation={
                "max_length_hit_rate": 0.30,
                "natural_end_rate": 0.40,
            },
            validation_metrics={"validator_pass": 4},
            baseline_metrics={},
            step_rows=steps,
            expected_steps=1,
            checkpoint_ok=True,
            replay_ok=True,
            continuity_ok=True,
            finite_ok=True,
        )
        self.assertEqual(result["status"], "PREFLIGHT_BLOCKED_NO_FORMAL_RL")
        self.assertFalse(result["criteria"]["max_length_quality"])
        self.assertFalse(result["criteria"]["natural_end_quality"])

    def test_active_gate_must_be_fp32_and_under_target(self):
        row = _step(1, "start", "after1")
        row["post_step_kl_gate_mode"] = "legacy_bfloat16_autocast"
        row["post_step_kl_gate_mean"] = 0.1
        result = quality_gate(
            quality_generation={"max_length_hit_rate": 0.1, "natural_end_rate": 0.8},
            validation_metrics={"validator_pass": 4},
            baseline_metrics={},
            step_rows=[row],
            expected_steps=1,
            checkpoint_ok=True,
            replay_ok=True,
            continuity_ok=True,
            finite_ok=True,
        )
        self.assertFalse(result["criteria"]["active_fp32_gate_within_budget"])
        self.assertEqual(result["status"], "PREFLIGHT_BLOCKED_NO_FORMAL_RL")

    def test_wrapper_has_phase_and_safety_contracts(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_rl_release_gate.sh"
        text = script.read_text(encoding="utf-8")
        for token in (
            "--dry-run",
            "--preflight",
            "--formal",
            "PREFLIGHT_PASS",
            "refusing to reuse non-empty experiment root",
            "sleep 60",
            "timeout --signal=TERM",
            "RL_RELEASE_GATE_HARD_LIMIT_SECONDS",
            "fp32_no_autocast",
            "--max-gen-len 128",
            "--microbatch-log-path",
            "--kl-guard-token-replay-path",
        ):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
