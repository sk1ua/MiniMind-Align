"""Unit tests for q_proj/v_proj LoRA attachment and round-trip."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from model.model_lora import apply_lora, load_lora, save_lora


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(5, 3, bias=False)
        self.v_proj = nn.Linear(5, 3, bias=False)
        self.other = nn.Linear(5, 5, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x) + self.v_proj(x)


class LoRATest(unittest.TestCase):
    def test_targets_non_square_v_projection(self) -> None:
        torch.manual_seed(42)
        model = TinyAttention()
        attached = apply_lora(model, rank=2, alpha=4)
        self.assertEqual(set(attached), {"q_proj", "v_proj"})
        self.assertEqual(sum(p.numel() for n, p in model.named_parameters() if "lora" in n), 2 * (5 + 3) + 2 * (5 + 3))

    def test_save_load_round_trip(self) -> None:
        torch.manual_seed(42)
        model = TinyAttention()
        base_state = {key: value.clone() for key, value in model.state_dict().items()}
        apply_lora(model, rank=2, alpha=4)
        model.q_proj.lora.B.weight.data.fill_(0.1)
        model.v_proj.lora.B.weight.data.fill_(0.2)
        x = torch.randn(3, 5)
        expected = model(x)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pth"
            save_lora(model, path)
            restored = TinyAttention()
            restored.load_state_dict(base_state)
            apply_lora(restored, rank=2, alpha=4)
            load_lora(restored, path)
            actual = restored(x)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
