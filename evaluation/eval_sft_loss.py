"""Evaluate SFT loss and perplexity on a fixed JSONL validation file."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.lm_dataset import SFTDataset
from model.model_lora import apply_lora, load_lora
from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import init_model


def evaluate(weight_name: str, data_path: Path, output_path: Path, device: str, batch_size: int, max_batches: int, tokenizer_path: str, model_dir: str, adapter_path: Path | None, lora_rank: int, lora_alpha: float) -> None:
    """Compute mean language-model loss and perplexity without updating weights."""
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    model, tokenizer = init_model(config, weight_name, tokenizer_path=tokenizer_path, save_dir=model_dir, device=device)
    if adapter_path is not None:
        apply_lora(model, rank=lora_rank, alpha=lora_alpha, target_modules=("q_proj", "v_proj"))
        load_lora(model, adapter_path)
    dataset = SFTDataset(str(data_path), tokenizer, max_length=512)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    losses: list[float] = []
    tokens = 0
    with torch.no_grad():
        for index, (input_ids, labels) in enumerate(loader):
            if max_batches > 0 and index >= max_batches:
                break
            input_ids, labels = input_ids.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                result = model(input_ids, labels=labels)
            losses.append(float(result.loss))
            tokens += int((labels != -100).sum())
    if not losses:
        raise ValueError("no validation batches")
    mean_loss = sum(losses) / len(losses)
    payload = {"weight": weight_name, "data": str(data_path), "batches": len(losses), "supervised_tokens": tokens, "loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20.0))}
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--tokenizer-path", default="model")
    parser.add_argument("--model-dir", default="out")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    args = parser.parse_args()
    evaluate(args.weight, args.data_path, args.output, args.device, args.batch_size, args.max_batches, args.tokenizer_path, args.model_dir, args.adapter_path, args.lora_rank, args.lora_alpha)


if __name__ == "__main__":
    main()
