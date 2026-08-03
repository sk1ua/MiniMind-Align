import json
import math
import tempfile
import unittest
from pathlib import Path

from evaluation.audit_kl_token_replay import (
    aggregate_variant,
    ensure_empty_output_dir,
    replay_row_check,
    replay_variant_groups,
)


def digest(value):
    import hashlib

    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode("utf-8")).hexdigest()


def make_row(variant, new_values):
    generated = [[1, 2, 3]]
    mask = [[1, 0]]
    refs = [[0.0, math.log(2.0)]]
    token_values = [[
        0.0,
        math.exp(refs[0][1] - new_values[0][1]) - (refs[0][1] - new_values[0][1]) - 1.0,
    ]]
    valid_values = [token_values[0][0]]
    return {
        "step": 1,
        "attempt_index": 0,
        "variant": variant,
        "rollout_index": 0,
        "micro_index": 0,
        "prompt_id": "p0",
        "category": "format",
        "sample_keys": ["1:0:0"],
        "generated_ids": generated,
        "completion_mask": mask,
        "ref_log_probs": refs,
        "new_log_probs": new_values,
        "token_kl_values": token_values,
        "valid_token_kl_values": valid_values,
        "generated_sha256": digest(generated),
        "mask_sha256": digest(mask),
        "reference_log_probs_sha256": digest(refs),
        "new_log_probs_sha256": digest(new_values),
    }


class KLTokenReplayTest(unittest.TestCase):
    def test_row_replays_formula_and_mask(self):
        row = make_row("bfloat16_autocast", [[0.0, 0.0]])
        self.assertTrue(replay_row_check(row)["valid"])

    def test_variant_groups_and_aggregate(self):
        rows = [
            make_row("bfloat16_autocast", [[0.0, 0.0]]),
            make_row("bfloat16_no_autocast", [[0.0, 0.0]]),
            make_row("full_float32_no_autocast", [[0.0, 0.0]]),
        ]
        groups = replay_variant_groups(rows)
        self.assertEqual(set(groups[(1, 0, 0)]), {
            "bfloat16_autocast",
            "bfloat16_no_autocast",
            "full_float32_no_autocast",
        })
        self.assertAlmostEqual(aggregate_variant(rows, "bfloat16_autocast")["reference_kl_mean"], 0.0)

    def test_bad_token_value_is_rejected(self):
        row = make_row("bfloat16_autocast", [[0.0, 0.0]])
        row["token_kl_values"][0][0] = 1.0
        self.assertFalse(replay_row_check(row)["valid"])

    def test_output_directory_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "audit"
            output.mkdir()
            (output / "summary.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_empty_output_dir(output)


if __name__ == "__main__":
    unittest.main()
