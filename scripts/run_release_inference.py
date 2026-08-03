"""Run greedy inference with a MiniMind-Align weight downloaded from a release.

This entry point intentionally accepts an explicit weight path so that release
assets stay outside the Git history and no private experiment directory is
required for inference.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MiniMind-Align release-weight inference")
    parser.add_argument("--weight", type=Path, required=True, help="Path to a downloaded 768-dim .pth state dict")
    parser.add_argument("--prompt", required=True, help="User prompt")
    parser.add_argument("--model-dir", type=Path, default=Path("model"), help="Tokenizer directory")
    parser.add_argument("--device", default=None, help="torch device; defaults to cuda when available, otherwise cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--do-sample", action="store_true", help="Use sampling instead of the default greedy decode")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # Keep public audit/test environments usable without installing torch.
    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    if not args.weight.is_file():
        raise FileNotFoundError(f"weight not found: {args.weight}")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"tokenizer directory not found: {args.model_dir}")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
    try:
        state = torch.load(args.weight, map_location="cpu", weights_only=True)
    except TypeError:  # compatibility with older torch releases
        state = torch.load(args.weight, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        generated = model.generate(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(
        generated[0, encoded.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
