"""Prepare independent one-row-per-category manifests for a coverage smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


CATEGORIES = (
    "conciseness",
    "format",
    "instruction",
    "reasoning",
    "repetition",
    "safety",
    "termination",
    "uncertainty",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row is not an object")
        rows.append(row)
    return rows


def select_one_per_category(rows: list[dict], categories: tuple[str, ...], seed: int) -> list[dict]:
    grouped: dict[str, list[dict]] = {category: [] for category in categories}
    for row in rows:
        category = str(row.get("category", ""))
        if category in grouped:
            grouped[category].append(row)
    missing = [category for category in categories if not grouped[category]]
    if missing:
        raise ValueError(f"missing categories: {missing}")
    rng = random.Random(seed)
    selected: list[dict] = []
    for category in categories:
        bucket = sorted(grouped[category], key=lambda row: str(row.get("id", "")))
        rng.shuffle(bucket)
        row = dict(bucket[0])
        if not row.get("metadata"):
            raise ValueError(f"selected row has empty metadata: {row.get('id')}")
        selected.append(row)
    ids = [str(row.get("id")) for row in selected]
    if len(set(ids)) != len(ids):
        raise ValueError("selected ids are not unique")
    return selected


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def prepare_balanced_manifests(
    train_source: Path,
    validation_source: Path,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(train_source)
    validation_rows = read_jsonl(validation_source)
    train_selected = select_one_per_category(train_rows, CATEGORIES, seed)
    validation_selected = select_one_per_category(validation_rows, CATEGORIES, seed)
    train_ids = {str(row.get("id")) for row in train_selected}
    validation_ids = {str(row.get("id")) for row in validation_selected}
    train_families = {str(row.get("family")) for row in train_selected}
    validation_families = {str(row.get("family")) for row in validation_selected}
    train_output = output_dir / "balanced_train_manifest.jsonl"
    validation_output = output_dir / "balanced_validation_manifest.jsonl"
    _write_jsonl(train_output, train_selected)
    _write_jsonl(validation_output, validation_selected)
    selection = {
        "schema_version": 1,
        "seed": seed,
        "selection_policy": f"sort each category by id, shuffle with deterministic seed {seed}, select one row per category",
        "categories": list(CATEGORIES),
        "train_source": str(train_source),
        "validation_source": str(validation_source),
        "train_source_sha256": sha256_file(train_source),
        "validation_source_sha256": sha256_file(validation_source),
        "train_output": str(train_output),
        "validation_output": str(validation_output),
        "train_count": len(train_selected),
        "validation_count": len(validation_selected),
        "train_category_counts": dict(Counter(str(row["category"]) for row in train_selected)),
        "validation_category_counts": dict(Counter(str(row["category"]) for row in validation_selected)),
        "train_ids": [str(row["id"]) for row in train_selected],
        "validation_ids": [str(row["id"]) for row in validation_selected],
        "train_validation_id_overlap": len(train_ids & validation_ids),
        "train_validation_family_overlap": len(train_families & validation_families),
        "train_output_sha256": sha256_file(train_output),
        "validation_output_sha256": sha256_file(validation_output),
        "original_manifests_unchanged": True,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--validation-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare_balanced_manifests(args.train_source, args.validation_source, args.output_dir, args.seed)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
