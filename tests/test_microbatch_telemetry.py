import unittest

import torch

from trainer.train_grpo_lite import (
    gradient_delta_norm,
    gradient_norm,
    gradient_snapshot,
    interleave_rows_by_category,
)


class MicrobatchTelemetryTest(unittest.TestCase):
    def test_gradient_delta_isolated_from_accumulated_gradient(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.data.fill_(1.0)
        model.zero_grad(set_to_none=True)

        first_loss = model(torch.tensor([[1.0, 1.0]])).sum()
        first_loss.backward()
        before_second = gradient_snapshot(model)
        first_norm = gradient_norm(model)

        second_loss = model(torch.tensor([[2.0, 2.0]])).sum()
        second_loss.backward()
        second_delta = gradient_delta_norm(model, before_second)

        self.assertGreater(first_norm, 0.0)
        self.assertGreater(second_delta, 0.0)
        self.assertNotAlmostEqual(second_delta, gradient_norm(model))

    def test_gradient_snapshot_and_delta_are_finite(self):
        model = torch.nn.Linear(2, 1)
        model.zero_grad(set_to_none=True)
        model(torch.ones(1, 2)).sum().backward()
        before = gradient_snapshot(model)
        model(torch.ones(1, 2)).sum().backward()
        delta = gradient_delta_norm(model, before)
        self.assertTrue(torch.isfinite(torch.tensor(delta)))

    def test_category_interleave_covers_each_category_before_repeating(self):
        rows = [
            {"id": "a0", "category": "a"},
            {"id": "a1", "category": "a"},
            {"id": "b0", "category": "b"},
            {"id": "b1", "category": "b"},
            {"id": "c0", "category": "c"},
        ]
        ordered = interleave_rows_by_category(rows, ["a", "b", "c"])
        self.assertEqual([row["id"] for row in ordered], ["a0", "b0", "c0", "a1", "b1"])


if __name__ == "__main__":
    unittest.main()
