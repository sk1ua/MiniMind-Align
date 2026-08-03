import unittest

import torch

from align.reward_model import (
    build_response_mask,
    last_response_positions,
    pairwise_preference_accuracy,
    pairwise_reward_loss,
)


class RewardModelTest(unittest.TestCase):
    def test_last_response_position_uses_response_then_fallback(self) -> None:
        attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
        response = torch.tensor([[0, 0, 1, 1, 0], [0, 0, 0, 0, 0]])
        positions = last_response_positions(attention, response)
        self.assertEqual(positions.tolist(), [3, 2])

    def test_pairwise_loss_direction(self) -> None:
        good = torch.tensor([2.0, 1.0])
        bad = torch.tensor([0.0, -1.0])
        self.assertGreater(float(pairwise_reward_loss(bad, good)), float(pairwise_reward_loss(good, bad)))
        self.assertEqual(float(pairwise_preference_accuracy(good, bad)), 1.0)

    def test_response_mask_marks_eos_and_handles_missing_marker(self) -> None:
        mask = build_response_mask([9, 1, 2, 3, 4, 0], [1, 1, 1, 1, 1, 0], [1], [4])
        self.assertEqual(mask, [0, 0, 1, 1, 1, 0])
        fallback = build_response_mask([9, 2, 3, 4], [1, 1, 1, 1], [8], [4])
        self.assertEqual(fallback, [0, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()
