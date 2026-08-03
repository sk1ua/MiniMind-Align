"""Run the existing rule validator over frozen test generations."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.audit_chosen_pilot import check_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.generation.read_text(encoding="utf-8").splitlines() if line.strip()]
    details = []
    for row in rows:
        problems = check_row({**row, "chosen": row.get("response", ""), "rejected": ""})
        details.append({**row, "validator_pass": not problems, "validator_problems": problems})
    summary = {
        "model": rows[0].get("model") if rows else None,
        "count": len(rows),
        "validator_pass": sum(item["validator_pass"] for item in details),
        "natural_end": sum(bool(item.get("finished_naturally")) for item in details),
        "average_tokens": mean([item.get("generated_tokens", 0) for item in details]) if details else 0.0,
        "average_repeat_3gram": mean([item.get("repeat_3gram_ratio", 0.0) for item in details]) if details else 0.0,
        "categories": {},
    }
    by_category: dict[str, list[dict]] = {}
    for item in details:
        by_category.setdefault(item["category"], []).append(item)
    for category, category_rows in sorted(by_category.items()):
        summary["categories"][category] = {
            "count": len(category_rows),
            "validator_pass": sum(item["validator_pass"] for item in category_rows),
            "natural_end": sum(bool(item.get("finished_naturally")) for item in category_rows),
            "average_tokens": mean([item.get("generated_tokens", 0) for item in category_rows]),
            "average_repeat_3gram": mean([item.get("repeat_3gram_ratio", 0.0) for item in category_rows]),
        }
    (args.output_dir / "validator_details.jsonl").write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in details) + "\n", encoding="utf-8")
    (args.output_dir / "validator_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
