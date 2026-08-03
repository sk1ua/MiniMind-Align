import json
import tempfile
import unittest
from pathlib import Path

from evaluation.audit_nonzero_advantage_contract import evaluate_fixture
from evaluation.audit_corrected_kl_gate import ensure_empty_output_dir


class NonzeroAdvantageContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_path = Path(__file__).resolve().parents[1] / "results" / "inputs" / "rl_nonzero_advantage_contract_fixture_20260802.json"
        cls.fixture = json.loads(cls.fixture_path.read_text(encoding="utf-8"))

    def test_fixture_has_replayed_nonzero_group_advantages(self):
        result = evaluate_fixture(self.fixture)
        self.assertEqual(result["status"], "NONZERO_ADVANTAGE_CONTRACT_PASS")
        self.assertTrue(result["advantage_replay_ok"])
        self.assertTrue(all(value != 0.0 for value in self.fixture["advantages"]))

    def test_grpo_and_cispo_clipping_and_gradients_are_observed(self):
        result = evaluate_fixture(self.fixture)
        records = {row["mode"]: row for row in result["records"]}
        self.assertEqual(records["grpo"]["clipped_token_count"], 4)
        self.assertEqual(records["cispo"]["clipped_token_count"], 2)
        for record in records.values():
            self.assertTrue(record["objective_matches_clipped_formula"])
            self.assertTrue(record["loss_match"])
            self.assertTrue(record["kl_match"])
            self.assertTrue(record["fp32_gradient_nonzero"])
            self.assertTrue(record["bfloat16_quantized_gradient_nonzero"])

    def test_audit_output_refuses_nonempty_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit"
            output.mkdir()
            (output / "existing.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_empty_output_dir(output)

    def test_wrapper_is_offline_and_isolated(self):
        source = (Path(__file__).resolve().parents[1] / "scripts" / "run_nonzero_advantage_contract.sh").read_text(encoding="utf-8")
        self.assertIn("rl_nonzero_advantage_contract_20260802", source)
        self.assertIn("CUDA_VISIBLE_DEVICES=", source)
        self.assertNotIn("trainer/train_grpo_lite.py", source)
        self.assertNotIn("--smoke", source)


if __name__ == "__main__":
    unittest.main()
