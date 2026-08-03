"""Generate on-policy candidates and validator-ranked hard preference pairs."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.audit_chosen_pilot import check_row
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def repeat_ratio(token_ids: list[int], n: int = 3) -> float:
    grams = [tuple(token_ids[i : i + n]) for i in range(max(0, len(token_ids) - n + 1))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def load_model(weight: str, model_dir: Path, device: torch.device) -> MiniMindForCausalLM:
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    model = MiniMindForCausalLM(config)
    state = torch.load(model_dir / f"{weight}_768.pth", map_location=device)
    model.load_state_dict(state, strict=True)
    return model.half().to(device).eval()


def response_logprob(model, generated: torch.Tensor, prompt_len: int) -> tuple[float, float]:
    attention = torch.ones_like(generated)
    with torch.inference_mode():
        logits = model(generated, attention_mask=attention).logits[:, :-1]
        labels = generated[:, 1:]
        token_log_probs = torch.log_softmax(logits.float(), dim=-1).gather(
            2, labels.unsqueeze(-1)
        ).squeeze(-1)
    response_log_probs = token_log_probs[:, max(prompt_len - 1, 0) :]
    return float(response_log_probs.sum().item()), float(response_log_probs.mean().item())


def candidate_quality(row: dict) -> tuple:
    problems = row["validator_problems"]
    valid = not problems
    return (
        int(valid),
        -len(problems),
        -float(row["repeat_3gram_ratio"]),
        -int(row["generated_tokens"]),
        float(row["normalized_logprob"]),
    )


def choose_pair(candidates: list[dict]) -> tuple[dict, dict, str]:
    ranked = sorted(candidates, key=candidate_quality, reverse=True)
    chosen = ranked[0]
    invalid = [row for row in ranked[1:] if row["validator_problems"]]
    rejected = min(invalid or ranked[1:], key=candidate_quality)
    reason = "validator_pass_vs_hard_invalid" if invalid else "quality_margin_within_candidates"
    return chosen, rejected, reason


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--preference-output", type=Path, required=True)
    parser.add_argument("--weight", default="align_sft_v2_pilot")
    parser.add_argument("--model-dir", type=Path, default=Path("out"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.candidate_output.exists() or args.preference_output.exists():
        raise FileExistsError("refusing to overwrite on-policy outputs")
    if args.num_candidates < 2:
        raise ValueError("num-candidates must be at least 2")

    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = records[: args.limit]
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = load_model(args.weight, args.model_dir, device)
    all_candidates: list[dict] = []
    pairs: list[dict] = []
    start = time.perf_counter()

    for position, sample in enumerate(records):
        messages = [{"role": "user", "content": sample["prompt"]}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, open_thinking=False
        )
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True).to(device)
        prompt_len = inputs["input_ids"].shape[1]
        candidates: list[dict] = []
        for candidate_index in range(args.num_candidates):
            seed = args.seed + position * 100 + candidate_index
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            generated = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_ids = generated[0][prompt_len:].tolist()
            response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            candidate = {
                "prompt_id": sample["id"],
                "category": sample["category"],
                "family": sample.get("family"),
                "prompt": sample["prompt"],
                "candidate_index": candidate_index,
                "seed": seed,
                "response": response,
                "generated_tokens": len(new_ids),
                "finished_naturally": len(new_ids) < args.max_new_tokens,
                "repeat_3gram_ratio": round(repeat_ratio(new_ids), 6),
                "sampling_config": {
                    "do_sample": True,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "max_new_tokens": args.max_new_tokens,
                },
            }
            candidate["validator_problems"] = check_row(
                {"prompt": sample["prompt"], "category": sample["category"], "chosen": response, "rejected": ""}
            )
            candidate["validator_pass"] = not candidate["validator_problems"]
            total_lp, normalized_lp = response_logprob(model, generated, prompt_len)
            candidate["logprob"] = round(total_lp, 6)
            candidate["normalized_logprob"] = round(normalized_lp, 6)
            candidates.append(candidate)
            all_candidates.append(candidate)
        chosen, rejected, reason = choose_pair(candidates)
        pairs.append(
            {
                "id": sample["id"],
                "split": sample.get("split"),
                "category": sample["category"],
                "family": sample.get("family"),
                "prompt": sample["prompt"],
                "chosen": [{"role": "user", "content": sample["prompt"]}, {"role": "assistant", "content": chosen["response"]}],
                "rejected": [{"role": "user", "content": sample["prompt"]}, {"role": "assistant", "content": rejected["response"]}],
                "chosen_candidate_index": chosen["candidate_index"],
                "rejected_candidate_index": rejected["candidate_index"],
                "selection_reason": reason,
                "chosen_validator_pass": chosen["validator_pass"],
                "rejected_validator_problems": rejected["validator_problems"],
                "source": "align_sft_v2_on_policy",
                "seed": args.seed,
            }
        )
        if position % 8 == 0 or position + 1 == len(records):
            print(f"on_policy: {position + 1}/{len(records)}")

    args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
    args.preference_output.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in all_candidates) + "\n", encoding="utf-8"
    )
    args.preference_output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in pairs) + "\n", encoding="utf-8"
    )
    pass_count = sum(row["chosen_validator_pass"] for row in pairs)
    print(json.dumps({"records": len(records), "candidates": len(all_candidates), "pairs": len(pairs), "chosen_validator_pass": pass_count, "elapsed_seconds": round(time.perf_counter() - start, 2)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
