"""Train the lightweight MiniMind Bradley--Terry reward model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from align.reward_model import (
    MiniMindRewardModel,
    PreferenceRewardDataset,
    pairwise_preference_accuracy,
    pairwise_reward_loss,
)
from model.model_minimind import MiniMindConfig


def _batch_rewards(model: MiniMindRewardModel, batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    chosen = model(
        batch["chosen_input_ids"].to(device),
        batch["chosen_attention_mask"].to(device),
        batch["chosen_response_mask"].to(device),
    )
    rejected = model(
        batch["rejected_input_ids"].to(device),
        batch["rejected_attention_mask"].to(device),
        batch["rejected_response_mask"].to(device),
    )
    return chosen, rejected


@torch.no_grad()
def evaluate(model: MiniMindRewardModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    margins: list[float] = []
    accuracies: list[float] = []
    for batch in loader:
        chosen, rejected = _batch_rewards(model, batch, device)
        losses.append(float(pairwise_reward_loss(chosen, rejected)))
        margins.append(float((chosen - rejected).mean()))
        accuracies.append(float(pairwise_preference_accuracy(chosen, rejected)))
    model.train()
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "margin": sum(margins) / max(len(margins), 1),
        "preference_accuracy": sum(accuracies) / max(len(accuracies), 1),
    }


def save_checkpoint(model: MiniMindRewardModel, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    for name, value in model.state_dict().items():
        state[name] = value.detach().cpu()
    torch.save(
        {
            "state_dict": state,
            "config": {"hidden_size": model.config.hidden_size, "num_hidden_layers": model.config.num_hidden_layers},
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("out"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--from-weight", default="align_sft_v2_pilot")
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--save-weight", default="reward_model_v1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    model = MiniMindRewardModel(config)
    base_path = args.model_dir / f"{args.from_weight}_768.pth"
    model.backbone.load_state_dict(torch.load(base_path, map_location=device), strict=True)
    model.to(device).train()

    train_ds = PreferenceRewardDataset(args.train_data, tokenizer, max_length=args.max_seq_len)
    validation_ds = PreferenceRewardDataset(args.validation_data, tokenizer, max_length=args.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    validation_loader = DataLoader(validation_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    autocast_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    use_autocast = device.type == "cuda" and args.dtype != "float32"
    output_path = args.save_dir / f"{args.save_weight}_768.pth"
    step = 0
    losses: list[float] = []
    while args.max_steps <= 0 or step < args.max_steps:
        for batch in train_loader:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                chosen, rejected = _batch_rewards(model, batch, device)
                loss = pairwise_reward_loss(chosen, rejected)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite reward loss at step {step}: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if step % args.log_interval == 0 or step == 1:
                print(json.dumps({
                    "step": step,
                    "loss": losses[-1],
                    "chosen_reward": float(chosen.mean().detach().cpu()),
                    "rejected_reward": float(rejected.mean().detach().cpu()),
                    "margin": float((chosen - rejected).mean().detach().cpu()),
                    "preference_accuracy": float(pairwise_preference_accuracy(chosen, rejected).detach().cpu()),
                }, ensure_ascii=False))
            if step % args.save_interval == 0:
                save_checkpoint(model, args.save_dir / f"{args.save_weight}_step{step}_768.pth")
            if args.max_steps > 0 and step >= args.max_steps:
                break
        if args.max_steps <= 0 or step >= args.max_steps:
            break

    save_checkpoint(model, output_path)
    validation = evaluate(model, validation_loader, device)
    print(json.dumps({
        "status": "PASS",
        "steps": step,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "validation": validation,
        "output": str(output_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
