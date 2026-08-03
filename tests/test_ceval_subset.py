import unittest

from evaluation.run_ceval_subset import family_for_model, parse_first_choice, sample_indices


class CEvalSubsetTests(unittest.TestCase):
    def test_first_standalone_choice(self):
        self.assertEqual(parse_first_choice("答案是 C。"), "C")
        self.assertEqual(parse_first_choice("先解释，再选 b"), "B")
        self.assertIsNone(parse_first_choice("ABC"))
        self.assertIsNone(parse_first_choice("没有合法选项"))

    def test_sampling_is_seeded(self):
        self.assertEqual(sample_indices(30, 5, 42), sample_indices(30, 5, 42))
        self.assertNotEqual(sample_indices(30, 5, 42), sample_indices(30, 5, 43))

    def test_family_names(self):
        self.assertEqual(family_for_model("grpo_seed42"), "grpo")
        self.assertEqual(family_for_model("cispo_seed44"), "cispo")
        self.assertEqual(family_for_model("align_sft_v2_pilot"), "baseline")


if __name__ == "__main__":
    unittest.main()
