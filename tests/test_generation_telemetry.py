import unittest

from trainer.train_grpo_lite import classify_generation_outcome


class GenerationOutcomeTelemetryTest(unittest.TestCase):
    def test_eos_before_padded_batch_end_is_natural(self):
        outcome = classify_generation_outcome(
            eos_position=3,
            completion_width=8,
            max_gen_len=8,
        )
        self.assertEqual(outcome["termination_reason"], "eos")
        self.assertTrue(outcome["eos_seen"])
        self.assertTrue(outcome["finished_naturally"])
        self.assertFalse(outcome["max_length_hit"])

    def test_eos_at_padded_batch_end_is_still_natural(self):
        outcome = classify_generation_outcome(
            eos_position=7,
            completion_width=8,
            max_gen_len=8,
        )
        self.assertEqual(outcome["termination_reason"], "eos")
        self.assertTrue(outcome["finished_naturally"])
        self.assertFalse(outcome["max_length_hit"])

    def test_no_eos_at_max_length_is_truncation(self):
        outcome = classify_generation_outcome(
            eos_position=None,
            completion_width=16,
            max_gen_len=16,
        )
        self.assertEqual(outcome["termination_reason"], "max_new_tokens")
        self.assertFalse(outcome["eos_seen"])
        self.assertFalse(outcome["finished_naturally"])
        self.assertTrue(outcome["max_length_hit"])

    def test_no_eos_short_generation_is_unknown(self):
        outcome = classify_generation_outcome(
            eos_position=None,
            completion_width=8,
            max_gen_len=16,
        )
        self.assertEqual(outcome["termination_reason"], "no_eos_short_generation")
        self.assertFalse(outcome["finished_naturally"])
        self.assertFalse(outcome["max_length_hit"])

    def test_padding_width_does_not_change_eos_classification(self):
        outcomes = [
            classify_generation_outcome(eos_position=3, completion_width=4, max_gen_len=16),
            classify_generation_outcome(eos_position=3, completion_width=12, max_gen_len=16),
        ]
        self.assertEqual(outcomes[0], outcomes[1])


if __name__ == "__main__":
    unittest.main()
