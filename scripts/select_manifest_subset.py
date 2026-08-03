"""Select a deterministic, category-balanced subset from an alignment manifest."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-category", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    groups: dict[str, list[dict]] = defaultdict(list)
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            groups[row["category"]].append(row)
    selected = []
    for category in sorted(groups):
        if len(groups[category]) < args.per_category:
            raise ValueError(f"{category}: only {len(groups[category])} rows")
        selected.extend(groups[category][: args.per_category])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in selected) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(selected), "per_category": args.per_category, "categories": sorted(groups)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
