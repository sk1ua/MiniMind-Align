from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 on-policy chosen/rejected pairs 转成 Gemini A/B judge 输入"
    )
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--baseline-output", required=True)
    parser.add_argument("--candidate-output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_pairs(path: Path, limit: int) -> list[dict]:
    rows = []
    seen_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["id"]
            if sample_id in seen_ids:
                raise RuntimeError(f"重复 pair id：{sample_id}")
            seen_ids.add(sample_id)
            if not row.get("chosen") or not row.get("rejected"):
                raise RuntimeError(f"pair 缺少 chosen/rejected：{sample_id}")
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def assistant_text(messages: list[dict], sample_id: str) -> str:
    assistants = [
        message["content"]
        for message in messages
        if message.get("role") == "assistant"
    ]
    if not assistants or not assistants[-1].strip():
        raise RuntimeError(f"pair assistant response 为空：{sample_id}")
    return assistants[-1]


def convert(row: dict, response: str, model: str) -> dict:
    return {
        "id": row["id"],
        "category": row["category"],
        "family": row.get("family", ""),
        "prompt": row["prompt"],
        "response": response,
        "model": model,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise RuntimeError(f"拒绝覆盖已有文件：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    pairs = load_pairs(Path(args.pairs), args.limit)
    baseline_rows = [
        convert(row, assistant_text(row["rejected"], row["id"]), "hard_rejected")
        for row in pairs
    ]
    candidate_rows = [
        convert(row, assistant_text(row["chosen"], row["id"]), "validator_chosen")
        for row in pairs
    ]
    write_jsonl(Path(args.baseline_output), baseline_rows)
    write_jsonl(Path(args.candidate_output), candidate_rows)
    print(json.dumps({"count": len(pairs), "ids_unique": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
