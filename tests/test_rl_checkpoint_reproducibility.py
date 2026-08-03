import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from trainer.train_grpo_lite import load_checkpoint_for_evaluation


class RLCheckpointReproducibilityTests(unittest.TestCase):
    @patch("trainer.train_grpo_lite.torch.load")
    @patch("trainer.train_grpo_lite.MiniMindForCausalLM")
    def test_evaluation_loader_uses_serialized_checkpoint(self, model_cls, torch_load):
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        model_cls.return_value = model
        state = {"weight": torch.zeros(1)}
        torch_load.return_value = state
        config = Mock()
        device = torch.device("cpu")

        result = load_checkpoint_for_evaluation(config, Path("checkpoint.pth"), device)

        self.assertIs(result, model)
        model_cls.assert_called_once_with(config)
        torch_load.assert_called_once_with(Path("checkpoint.pth"), map_location=device)
        model.load_state_dict.assert_called_once_with(state, strict=True)
        model.to.assert_called_once_with(device)
        model.eval.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
