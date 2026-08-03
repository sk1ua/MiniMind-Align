"""Evaluate a lightweight reward model on hard preference pairs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from align.reward_model import (
    PreferenceRewardDataset,
    load_reward_checkpoint,
    pairwise_preference_accuracy,
)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denominator = math.sqrt(var_x * var_y)
    return cov / denominator if denominator else None


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gemini-judgement", type=Path, default=None)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    dataset = PreferenceRewardDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = load_reward_checkpoint(args.weight_path, device=device).to(device).eval()

    rows: list[dict[str, object]] = []
    category_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for batch in loader:
        chosen = model(batch["chosen_input_ids"].to(device), batch["chosen_attention_mask"].to(device), batch["chosen_response_mask"].to(device))
        rejected = model(batch["rejected_input_ids"].to(device), batch["rejected_attention_mask"].to(device), batch["rejected_response_mask"].to(device))
        margins = chosen - rejected
        for i in range(chosen.shape[0]):
            row = {
                "id": batch["id"][i],
                "category": batch["category"][i],
                "family": batch["family"][i],
                "chosen_reward": float(chosen[i].cpu()),
                "rejected_reward": float(rejected[i].cpu()),
                "margin": float(margins[i].cpu()),
                "prefers_chosen": bool(margins[i].item() > 0),
                "chosen_response_tokens": int(batch["chosen_response_mask"][i].sum()),
                "rejected_response_tokens": int(batch["rejected_response_mask"][i].sum()),
            }
            rows.append(row)
            category_rows[str(row["category"])].append(row)

    chosen_rewards = torch.tensor([row["chosen_reward"] for row in rows], dtype=torch.float32)
    rejected_rewards = torch.tensor([row["rejected_reward"] for row in rows], dtype=torch.float32)
    length_deltas = [float(row["chosen_response_tokens"] - row["rejected_response_tokens"]) for row in rows]
    margins = [float(row["margin"]) for row in rows]
    summary = {
        "model": "reward_model_v1",
        "count": len(rows),
        "pairwise_accuracy": float(pairwise_preference_accuracy(chosen_rewards, rejected_rewards)),
        "chosen_reward_mean": float(chosen_rewards.mean()),
        "rejected_reward_mean": float(rejected_rewards.mean()),
        "margin_mean": sum(margins) / max(len(margins), 1),
        "chosen_response_tokens_mean": sum(row["chosen_response_tokens"] for row in rows) / max(len(rows), 1),
        "rejected_response_tokens_mean": sum(row["rejected_response_tokens"] for row in rows) / max(len(rows), 1),
        "margin_vs_length_delta_pearson": _pearson(length_deltas, margins),
        "categories": {},
    }
    if args.gemini_judgement is not None:
        gemini = {}
        with args.gemini_judgement.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    judgement = json.loads(line)
                    gemini[str(judgement["id"])] = str(judgement.get("winner_model", "tie"))
        comparable = []
        for row in rows:
            winner = gemini.get(str(row["id"]))
            if winner not in {"validator_chosen", "hard_rejected"}:
                continue
            reward_choice = "validator_chosen" if row["prefers_chosen"] else "hard_rejected"
            comparable.append(reward_choice == winner)
        summary["gemini_agreement"] = {
            "judgement_count": len(gemini),
            "non_tie_count": len(comparable),
            "agreement": sum(comparable) / len(comparable) if comparable else None,
        }
    for category, items in sorted(category_rows.items()):
        summary["categories"][category] = {
            "count": len(items),
            "pairwise_accuracy": sum(bool(item["prefers_chosen"]) for item in items) / len(items),
            "margin_mean": sum(float(item["margin"]) for item in items) / len(items),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reward_details.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (args.output_dir / "reward_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
