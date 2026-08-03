"""Evaluate MiniMind weights on a pinned, lightweight C-Eval subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

DEFAULT_SUBJECTS = [
    "high_school_chinese",
    "high_school_mathematics",
    "high_school_physics",
    "computer_network",
    "business_ethics",
]
DATASET_SUBJECT_ALIASES = {"business_ethics": "business_administration"}
CHOICES = ("A", "B", "C", "D")


def parse_first_choice(response: str) -> str | None:
    """Return the first standalone legal option or None for an invalid answer."""
    match = re.search(r"(?<![A-Z])([ABCD])(?![A-Z])", response.upper())
    return match.group(1) if match else None


def sample_indices(length: int, count: int, seed: int) -> list[int]:
    if count < 1 or count > length:
        raise ValueError(f"cannot sample {count} questions from {length}")
    return sorted(random.Random(seed).sample(range(length), count))


def family_for_model(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("grpo"):
        return "grpo"
    if lowered.startswith("cispo"):
        return "cispo"
    return "baseline"


def read_model_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"model spec must be NAME=WEIGHT_PATH: {value}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError(f"model spec must be NAME=WEIGHT_PATH: {value}")
        specs.append((name, Path(path)))
    if not specs:
        raise ValueError("at least one --model NAME=WEIGHT_PATH is required")
    return specs


def load_ceval_rows(subjects: list[str], count: int, seed: int, revision: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import __version__ as datasets_version
    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    source_subjects = []
    for subject in subjects:
        dataset_subject = DATASET_SUBJECT_ALIASES.get(subject, subject)
        datasets_by_split = {"val": load_dataset("ceval/ceval-exam", dataset_subject, split="val", revision=revision)}
        if len(datasets_by_split["val"]) < count:
            datasets_by_split["dev"] = load_dataset("ceval/ceval-exam", dataset_subject, split="dev", revision=revision)
        pool = [(split, index) for split, dataset in datasets_by_split.items() for index in range(len(dataset))]
        if len(pool) < count:
            raise ValueError(f"subject {subject} has only {len(pool)} labelled val/dev rows")
        selected = sorted(random.Random(seed).sample(pool, count))
        source_subjects.append({"subject": subject, "dataset_config": dataset_subject, "splits": {split: len(dataset) for split, dataset in datasets_by_split.items()}, "selected": [{"split": split, "index": index} for split, index in selected]})
        for split, index in selected:
            raw = dict(datasets_by_split[split][index])
            missing = [key for key in ("question", "A", "B", "C", "D", "answer") if key not in raw]
            if missing:
                raise ValueError(f"C-Eval schema missing {missing} for {subject}")
            answer = str(raw["answer"]).strip().upper()
            if answer.isdigit() and answer in {"0", "1", "2", "3"}:
                answer = CHOICES[int(answer)]
            if answer not in CHOICES:
                raise ValueError(f"unexpected C-Eval answer {answer!r} for {subject}")
            rows.append(
                {
                    "id": f"{subject}:{split}:{index}",
                    "subject": subject,
                    "source_split": split,
                    "source_index": index,
                    "question": str(raw["question"]),
                    "options": {choice: str(raw[choice]) for choice in CHOICES},
                    "answer": answer,
                }
            )
    source = {
        "dataset": "ceval/ceval-exam",
        "revision": revision,
        "split_policy": "val; if fewer than 20, supplement from dev",
        "subject_aliases": DATASET_SUBJECT_ALIASES,
        "datasets_version": datasets_version,
        "seed": seed,
        "questions_per_subject": count,
        "subjects": source_subjects,
    }
    return rows, source


def prompt_for(row: dict[str, Any]) -> str:
    options = "\n".join(f"{choice}. {row['options'][choice]}" for choice in CHOICES)
    return (
        "请回答以下选择题，只输出选项字母 A、B、C 或 D，不要输出其他内容。\n"
        f"题目：{row['question']}\n{options}\n答案："
    )


def load_model(weight_path: Path, device: torch.device):
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=True)
    return model.half().to(device).eval()


def evaluate_model(model, tokenizer, rows: list[dict[str, Any]], device: torch.device, name: str, max_new_tokens: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = []
    with torch.inference_mode():
        for row in rows:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_for(row)}],
                tokenize=False,
                add_generation_prompt=True,
                open_thinking=False,
            )
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True).to(device)
            generated = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_length = inputs["input_ids"].shape[1]
            response = tokenizer.decode(generated[0][prompt_length:].tolist(), skip_special_tokens=True).strip()
            parsed = parse_first_choice(response)
            predictions.append(
                {
                    "id": row["id"],
                    "subject": row["subject"],
                    "source_split": row["source_split"],
                    "source_index": row["source_index"],
                    "question": row["question"],
                    "options": row["options"],
                    "gold": row["answer"],
                    "response": response,
                    "parsed_answer": parsed,
                    "valid_answer": parsed is not None,
                    "correct": parsed == row["answer"],
                    "model": name,
                    "decode_config": {"do_sample": False, "max_new_tokens": max_new_tokens},
                }
            )
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        by_subject.setdefault(row["subject"], []).append(row)
    summary = {
        "model": name,
        "family": family_for_model(name),
        "count": len(predictions),
        "correct": sum(bool(row["correct"]) for row in predictions),
        "accuracy": sum(bool(row["correct"]) for row in predictions) / len(predictions),
        "invalid_answers": sum(not row["valid_answer"] for row in predictions),
        "subjects": {
            subject: {
                "count": len(subject_rows),
                "correct": sum(bool(row["correct"]) for row in subject_rows),
                "accuracy": sum(bool(row["correct"]) for row in subject_rows) / len(subject_rows),
                "invalid_answers": sum(not row["valid_answer"] for row in subject_rows),
            }
            for subject, subject_rows in sorted(by_subject.items())
        },
    }
    return predictions, summary


def aggregate_families(model_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, list[dict[str, Any]]] = {}
    for summary in model_summaries:
        families.setdefault(summary["family"], []).append(summary)
    result = {}
    for family, summaries in sorted(families.items()):
        subject_names = sorted({subject for summary in summaries for subject in summary["subjects"]})
        result[family] = {
            "n_models": len(summaries),
            "models": [summary["model"] for summary in summaries],
            "accuracy_mean": statistics.mean(summary["accuracy"] for summary in summaries),
            "accuracy_std": statistics.pstdev(summary["accuracy"] for summary in summaries) if len(summaries) > 1 else 0.0,
            "subjects": {
                subject: {
                    "accuracy_mean": statistics.mean(summary["subjects"][subject]["accuracy"] for summary in summaries if subject in summary["subjects"]),
                    "accuracy_std": statistics.pstdev(summary["subjects"][subject]["accuracy"] for summary in summaries if subject in summary["subjects"]) if sum(subject in summary["subjects"] for summary in summaries) > 1 else 0.0,
                }
                for subject in subject_names
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="NAME=WEIGHT_PATH; repeat for every model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="Pinned Hugging Face dataset revision; floating main is rejected")
    parser.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    parser.add_argument("--questions-per-subject", type=int, default=20)
    parser.add_argument("--smoke", action="store_true", help="Allow the registered 1-subject x 2-question smoke protocol")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.revision in {"main", "master", "", "latest"}:
        raise ValueError("C-Eval revision must be pinned to a commit or immutable tag")
    if not args.smoke and args.questions_per_subject != 20:
        raise ValueError("registered C-Eval protocol requires exactly 20 questions per subject")
    existing = [path for path in args.output_dir.iterdir() if path.name not in {"run.log", "omitted_models.log"}] if args.output_dir.exists() else []
    if existing:
        raise FileExistsError(f"refusing to write into non-empty experiment directory {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = [subject.strip() for subject in args.subjects.split(",") if subject.strip()]
    if (not args.smoke and subjects != DEFAULT_SUBJECTS) or (args.smoke and (len(subjects) != 1 or args.questions_per_subject != 2)):
        raise ValueError(f"registered C-Eval protocol requires subjects {DEFAULT_SUBJECTS}")
    rows, source = load_ceval_rows(subjects, args.questions_per_subject, args.seed, args.revision)
    manifest_path = args.output_dir / "ceval_manifest.jsonl"
    manifest_path.write_text("\n".join(json.dumps({**row, "prompt": prompt_for(row)}, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    source["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (args.output_dir / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model_summaries = []
    for name, weight_path in read_model_specs(args.model):
        model = load_model(weight_path, device)
        predictions, summary = evaluate_model(model, tokenizer, rows, device, name, args.max_new_tokens)
        prediction_path = args.output_dir / f"predictions_{re.sub(r'[^A-Za-z0-9_.-]+', '_', name)}.jsonl"
        prediction_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n", encoding="utf-8")
        summary["predictions_sha256"] = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        model_summaries.append(summary)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps(summary, ensure_ascii=False))
    output = {
        "protocol": {"dataset": "ceval/ceval-exam", "revision": args.revision, "subjects": subjects, "subject_aliases": DATASET_SUBJECT_ALIASES, "questions_per_subject": args.questions_per_subject, "total_questions": len(rows), "seed": args.seed, "split_policy": "val; if fewer than 20, supplement from dev", "decoding": "greedy", "answer_policy": "first_standalone_A_B_C_D"},
        "models": model_summaries,
        "family_aggregates": aggregate_families(model_summaries),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_dir": str(args.output_dir), "models": len(model_summaries), "questions": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
