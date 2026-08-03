"""Audit and materialize on-policy DPO v2 preference pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(rows: list[dict], test_prompts: set[str], split: str) -> dict:
    ids: set[str] = set()
    prompts: set[str] = set()
    for row in rows:
        required = {"id", "category", "family", "prompt", "chosen", "rejected"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"{split} {row.get('id')}: missing {sorted(missing)}")
        if row["id"] in ids or row["prompt"] in prompts:
            raise ValueError(f"{split}: duplicate id or prompt {row['id']}")
        ids.add(row["id"])
        prompts.add(row["prompt"])
        if row["prompt"] in test_prompts:
            raise ValueError(f"{split} {row['id']}: test prompt leakage")
        for field in ("chosen", "rejected"):
            value = row[field]
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"{split} {row['id']}: {field} must be user/assistant list")
            if value[0].get("role") != "user" or value[1].get("role") != "assistant":
                raise ValueError(f"{split} {row['id']}: invalid roles")
            if not str(value[1].get("content", "")).strip():
                raise ValueError(f"{split} {row['id']}: empty {field}")
        if row["chosen"] == row["rejected"]:
            raise ValueError(f"{split} {row['id']}: chosen equals rejected")
    return {"split": split, "count": len(rows), "categories": sorted({row['category'] for row in rows}), "families": len({row['family'] for row in rows})}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument("--validation-pairs", type=Path, required=True)
    parser.add_argument("--test-prompts", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    outputs = (args.train_output, args.validation_output, args.report)
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite DPO v2 artifacts")
    train = load(args.train_pairs)
    validation = load(args.validation_pairs)
    test_prompts = {row["prompt"] for row in load(args.test_prompts)}
    train_info = validate(train, test_prompts, "train")
    val_info = validate(validation, test_prompts, "validation")
    if {row["prompt"] for row in train} & {row["prompt"] for row in validation}:
        raise ValueError("train/validation prompt overlap")
    for path, rows in ((args.train_output, train), (args.validation_output, validation)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    report = {"status": "PASS", "train": train_info, "validation": val_info, "test_prompt_count": len(test_prompts), "source": "align_sft_v2_on_policy", "hard_negative_rule": "validator-ranked candidate; Gemini smoke consistency checked on validation pairs"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
