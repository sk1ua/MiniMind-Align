import unittest

import torch
import torch.nn.functional as F

from trainer.train_simpo import simpo_loss


class SimPOLossTest(unittest.TestCase):
    def test_manual_formula(self):
        chosen = torch.tensor([0.5, 0.2])
        rejected = torch.tensor([0.1, 0.4])
        beta = 2.0
        gamma = 0.1
        expected = -F.logsigmoid(beta * (chosen - rejected - gamma))
        torch.testing.assert_close(simpo_loss(chosen, rejected, beta, gamma), expected)

    def test_preference_margin_direction(self):
        better = simpo_loss(torch.tensor([0.8]), torch.tensor([0.1]), 2.0, 0.1)
        worse = simpo_loss(torch.tensor([0.2]), torch.tensor([0.1]), 2.0, 0.1)
        self.assertLess(float(better), float(worse))


if __name__ == "__main__":
    unittest.main()
