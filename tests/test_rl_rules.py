import unittest

import torch

from align.rl_rules import clipped_policy_loss, group_advantages, rule_reward


class RLRuleTest(unittest.TestCase):
    def test_group_advantages_zero_mean(self) -> None:
        advantages = group_advantages(torch.tensor([0.0, 1.0, 2.0, 3.0]), 2)
        self.assertAlmostEqual(float(advantages.view(2, 2).mean(dim=1)[0]), 0.0, places=5)
        self.assertAlmostEqual(float(advantages.view(2, 2).mean(dim=1)[1]), 0.0, places=5)

    def test_grpo_and_cispo_are_finite(self) -> None:
        shape = (4, 3)
        new = torch.zeros(shape, requires_grad=True)
        old = torch.zeros(shape)
        ref = torch.zeros(shape)
        advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])
        mask = torch.ones(shape)
        for mode in ("grpo", "cispo"):
            loss, kl = clipped_policy_loss(new, old, ref, advantages, mask, mode=mode)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(kl))

    def test_rule_reward_prefers_valid_termination(self) -> None:
        metadata = {"max_chars": 20}
        good, _ = rule_reward("termination", "一句话。", "这是答案。", metadata)
        bad, _ = rule_reward("termination", "一句话。", "这是答案\n还有内容", metadata)
        self.assertGreater(good, bad)


if __name__ == "__main__":
    unittest.main()
