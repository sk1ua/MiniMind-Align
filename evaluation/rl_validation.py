"""Deterministic validation metrics for the controlled RL experiments."""

from __future__ import annotations

import time
from collections import defaultdict
from statistics import mean
from typing import Any

import torch

from dataset.alignment_v2.validators import repeat_3gram_ratio, validate_record


def _prompt_tokens(tokenizer, prompt: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    return encoded.input_ids.to(device), encoded.attention_mask.to(device)


def evaluate_policy(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    max_new_tokens: int = 128,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run deterministic generation and return auditable aggregate metrics."""
    if not rows:
        raise ValueError("validation rows cannot be empty")
    was_training = model.training
    model.eval()
    details: list[dict[str, Any]] = []
    with torch.inference_mode():
        for row in rows:
            input_ids, attention_mask = _prompt_tokens(tokenizer, str(row["prompt"]), device)
            started = time.perf_counter()
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            elapsed = time.perf_counter() - started
            prompt_length = input_ids.shape[1]
            new_ids = generated[0][prompt_length:].tolist()
            response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            record = {
                "conversations": [
                    {"role": "user", "content": str(row["prompt"])},
                    {"role": "assistant", "content": response},
                ]
            }
            passed, problem = validate_record(record, {"category": row["category"], "metadata": row["metadata"]})
            details.append(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "response": response,
                    "generated_tokens": len(new_ids),
                    "tokens_per_second": round(len(new_ids) / max(elapsed, 1e-6), 2),
                    "repeat_3gram_ratio": round(repeat_3gram_ratio(response), 6),
                    "finished_naturally": bool(len(new_ids) < max_new_tokens),
                    "validator_pass": passed,
                    "validator_problem": problem,
                }
            )
    if was_training:
        model.train()

    category_details: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in details:
        category_details[item["category"]].append(item)

    def category_summary(category_rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(category_rows),
            "validator_pass": sum(bool(item["validator_pass"]) for item in category_rows),
            "natural_end": sum(bool(item["finished_naturally"]) for item in category_rows),
            "average_tokens": mean(item["generated_tokens"] for item in category_rows),
            "average_repeat_3gram": mean(item["repeat_3gram_ratio"] for item in category_rows),
        }

    safety_rows = category_details.get("safety", [])
    termination_rows = category_details.get("termination", [])
    summary = {
        "count": len(details),
        "validator_pass": sum(bool(item["validator_pass"]) for item in details),
        "validator_pass_rate": sum(bool(item["validator_pass"]) for item in details) / len(details),
        "safety_count": len(safety_rows),
        "safety_pass": sum(bool(item["validator_pass"]) for item in safety_rows),
        "safety_pass_rate": (sum(bool(item["validator_pass"]) for item in safety_rows) / len(safety_rows)) if safety_rows else None,
        "termination_count": len(termination_rows),
        "termination_pass": sum(bool(item["validator_pass"]) for item in termination_rows),
        "termination_pass_rate": (sum(bool(item["validator_pass"]) for item in termination_rows) / len(termination_rows)) if termination_rows else None,
        "natural_end": sum(bool(item["finished_naturally"]) for item in details),
        "natural_end_rate": sum(bool(item["finished_naturally"]) for item in details) / len(details),
        "average_tokens": mean(item["generated_tokens"] for item in details),
        "average_repeat_3gram": mean(item["repeat_3gram_ratio"] for item in details),
        "categories": {category: category_summary(category_rows) for category, category_rows in sorted(category_details.items())},
    }
    return summary, details
