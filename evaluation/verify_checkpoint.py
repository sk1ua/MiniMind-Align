"""Verify a MiniMind checkpoint can load, run finite inference, and generate."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def verify(weight_path: Path, device: str, max_new_tokens: int) -> None:
    """Load a checkpoint and run a deterministic one-prompt smoke."""
    weights = torch.load(weight_path, map_location="cpu")
    nonfinite = [name for name, value in weights.items() if torch.is_tensor(value) and not torch.isfinite(value).all()]
    if nonfinite:
        raise ValueError(f"nonfinite tensors: {nonfinite[:5]}")
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    model = MiniMindForCausalLM(config)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise ValueError(f"state_dict mismatch missing={missing[:3]} unexpected={unexpected[:3]}")
    tokenizer = AutoTokenizer.from_pretrained("model")
    target = torch.device(device)
    model = model.to(target).eval()
    messages = [{"role": "user", "content": "请只回答：检查通过。"}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(target)
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0, top_p=1.0)
    new_tokens = generated.shape[1] - input_ids.shape[1]
    print(f"CHECKPOINT_OK path={weight_path} device={target} input_tokens={input_ids.shape[1]} new_tokens={new_tokens}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    verify(args.weight_path, args.device, args.max_new_tokens)


if __name__ == "__main__":
    main()
