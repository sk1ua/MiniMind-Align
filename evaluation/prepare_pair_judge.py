"""Convert preference pairs into anonymous judge generation rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def response(conversation: list[dict]) -> str:
    return str(conversation[-1]["content"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    if args.baseline.exists() or args.candidate.exists():
        raise FileExistsError("refusing to overwrite judge inputs")
    rows = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line.strip()]
    baseline, candidate = [], []
    for row in rows:
        base = {key: row[key] for key in ("id", "category", "family", "prompt")}
        baseline.append({**base, "response": response(row["rejected"]), "model": "hard_rejected"})
        candidate.append({**base, "response": response(row["chosen"]), "model": "validator_chosen"})
    for path, data in ((args.baseline, baseline), (args.candidate, candidate)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in data) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(rows), "baseline": str(args.baseline), "candidate": str(args.candidate)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
