"""Train SimPO on on-policy preference pairs."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.lm_dataset import DPODataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def simpo_loss(chosen_avg: torch.Tensor, rejected_avg: torch.Tensor, beta: float, gamma: float) -> torch.Tensor:
    """Return SimPO loss from normalized chosen/rejected response log-probs."""
    return -F.logsigmoid(beta * (chosen_avg - rejected_avg - gamma))


def sequence_average_logprob(model, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    outputs = model(x)
    token_log_probs = F.log_softmax(outputs.logits.float(), dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * mask.float()).sum(dim=-1) / mask.float().sum(dim=-1).clamp_min(1.0)


def save_weight(model: MiniMindForCausalLM, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    torch.save({key: value.detach().half().cpu() for key, value in model.state_dict().items()}, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--save-weight", default="simpo_v1")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--from-weight", default="align_sft_v2_pilot")
    parser.add_argument("--model-dir", type=Path, default=Path("out"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_path = args.save_dir / f"{args.save_weight}_768.pth"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
    model.load_state_dict(torch.load(args.model_dir / f"{args.from_weight}_768.pth", map_location=device), strict=True)
    model.to(device).train()
    dataset = DPODataset(str(args.data_path), tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    autocast_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    use_autocast = device.type == "cuda" and args.dtype != "float32"
    step = 0
    losses = []
    while args.max_steps <= 0 or step < args.max_steps:
        for batch in loader:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                chosen_avg = sequence_average_logprob(model, batch["x_chosen"].to(device), batch["y_chosen"].to(device), batch["mask_chosen"].to(device))
                rejected_avg = sequence_average_logprob(model, batch["x_rejected"].to(device), batch["y_rejected"].to(device), batch["mask_rejected"].to(device))
                loss = simpo_loss(chosen_avg, rejected_avg, args.beta, args.gamma).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))
            if step % args.log_interval == 0 or step == 1:
                print(json.dumps({"step": step, "loss": losses[-1], "chosen_avg": float(chosen_avg.mean()), "rejected_avg": float(rejected_avg.mean()), "margin": float((chosen_avg - rejected_avg).mean()), "beta": args.beta, "gamma": args.gamma}, ensure_ascii=False))
            if step % args.save_interval == 0 or (args.max_steps > 0 and step == args.max_steps):
                save_weight(model, args.save_dir / f"{args.save_weight}_step{step}_768.pth")
            if args.max_steps > 0 and step >= args.max_steps:
                break
        if args.max_steps <= 0 or step >= args.max_steps:
            break
    save_weight(model, output_path)
    print(json.dumps({"status": "PASS", "steps": step, "loss_first": losses[0], "loss_last": losses[-1], "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
