"""Evaluate preference accuracy and normalized margins for a pair dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.lm_dataset import DPODataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def load_model(path: Path, device: torch.device) -> MiniMindForCausalLM:
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
    model.load_state_dict(torch.load(path, map_location=device), strict=True)
    return model.half().eval().to(device)


def logprob(model, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    with torch.inference_mode():
        logits = model(x).logits
        token_logprob = F.log_softmax(logits.float(), dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        summed = (token_logprob * mask).sum(dim=-1)
        normalized = summed / mask.sum(dim=-1).clamp_min(1)
    return float(summed.item()), float(normalized.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    dataset = DPODataset(str(args.data), tokenizer, max_length=args.max_seq_len)
    reference = load_model(args.reference_path, device)
    policy = load_model(args.policy_path, device)
    details = []
    for index in range(len(dataset)):
        batch = {key: value.unsqueeze(0).to(device) for key, value in dataset[index].items()}
        ref_c, ref_c_avg = logprob(reference, batch["x_chosen"], batch["y_chosen"], batch["mask_chosen"])
        ref_r, ref_r_avg = logprob(reference, batch["x_rejected"], batch["y_rejected"], batch["mask_rejected"])
        pol_c, pol_c_avg = logprob(policy, batch["x_chosen"], batch["y_chosen"], batch["mask_chosen"])
        pol_r, pol_r_avg = logprob(policy, batch["x_rejected"], batch["y_rejected"], batch["mask_rejected"])
        details.append({"index": index, "reference_margin": ref_c - ref_r, "policy_margin": pol_c - pol_r, "relative_margin": (pol_c - pol_r) - (ref_c - ref_r), "reference_normalized_margin": ref_c_avg - ref_r_avg, "policy_normalized_margin": pol_c_avg - pol_r_avg, "relative_normalized_margin": (pol_c_avg - pol_r_avg) - (ref_c_avg - ref_r_avg), "policy_preference_correct": pol_c > pol_r})
    summary = {"count": len(details), "policy_preference_accuracy": sum(row["policy_preference_correct"] for row in details) / max(len(details), 1), "relative_margin_mean": sum(row["relative_margin"] for row in details) / max(len(details), 1), "relative_normalized_margin_mean": sum(row["relative_normalized_margin"] for row in details) / max(len(details), 1), "details": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "details"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
