"""Prepare deterministic native-Alignment-v2 manifests for the RL isolation run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


EXPECTED_CATEGORIES = (
    "conciseness",
    "format",
    "instruction",
    "reasoning",
    "repetition",
    "safety",
    "termination",
    "uncertainty",
)
NATIVE_SOURCE = "alignment_v2_programmatic_v1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_categories(rows: Iterable[dict], expected_per_category: int) -> None:
    counts = Counter(str(row["category"]) for row in rows)
    if tuple(sorted(counts)) != EXPECTED_CATEGORIES:
        raise ValueError(f"unexpected categories: {sorted(counts)}")
    if any(count != expected_per_category for count in counts.values()):
        raise ValueError(f"unbalanced categories: {dict(counts)}")


def select_native_train(rows: list[dict], *, per_category: int = 16, seed: int = 42) -> list[dict]:
    """Select metadata-rich v2 rows using a stable seeded ID ordering."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("source") != NATIVE_SOURCE or not row.get("metadata"):
            continue
        if not isinstance(row.get("metadata"), dict):
            raise ValueError(f"metadata must be an object: {row.get('id')}")
        groups[str(row["category"])].append(row)
    selected: list[dict] = []
    for category in EXPECTED_CATEGORIES:
        if len(groups[category]) < per_category:
            raise ValueError(f"{category}: only {len(groups[category])} native rows")
        ranked = sorted(
            groups[category],
            key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode("utf-8")).hexdigest(),
        )
        selected.extend(ranked[:per_category])
    _validate_categories(selected, per_category)
    return selected


def select_validation_slice(rows: list[dict], *, per_category: int = 4, offset: int = 4) -> list[dict]:
    """Select the second deterministic validation slice after the registered first slice."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["category"])].append(row)
    selected: list[dict] = []
    for category in EXPECTED_CATEGORIES:
        category_rows = groups[category]
        end = offset + per_category
        if len(category_rows) < end:
            raise ValueError(f"{category}: only {len(category_rows)} validation rows")
        selected.extend(category_rows[offset:end])
    _validate_categories(selected, per_category)
    return selected


def validate_selection(train_rows: list[dict], validation_rows: list[dict], existing_validation_rows: list[dict]) -> dict:
    train_ids = {str(row["id"]) for row in train_rows}
    validation_ids = {str(row["id"]) for row in validation_rows}
    existing_ids = {str(row["id"]) for row in existing_validation_rows}
    train_families = {str(row["family"]) for row in train_rows}
    validation_families = {str(row["family"]) for row in validation_rows}
    overlap = {
        "train_validation_ids": sorted(train_ids & validation_ids),
        "train_validation_families": sorted(train_families & validation_families),
        "validation_existing_ids": sorted(validation_ids & existing_ids),
    }
    if any(overlap.values()):
        raise ValueError(f"manifest overlap detected: {overlap}")
    return {
        "counts": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "train_by_category": dict(sorted(Counter(row["category"] for row in train_rows).items())),
            "validation_by_category": dict(sorted(Counter(row["category"] for row in validation_rows).items())),
        },
        "overlap": overlap,
        "train_metadata_nonempty": sum(bool(row.get("metadata")) for row in train_rows),
        "validation_metadata_nonempty": sum(bool(row.get("metadata")) for row in validation_rows),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--validation-input", type=Path, required=True)
    parser.add_argument("--existing-validation", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-category", type=int, default=16)
    parser.add_argument("--validation-per-category", type=int, default=4)
    parser.add_argument("--validation-offset", type=int, default=4)
    args = parser.parse_args()

    train_source = read_jsonl(args.train_input)
    validation_source = read_jsonl(args.validation_input)
    existing_validation = read_jsonl(args.existing_validation)
    train_rows = select_native_train(train_source, per_category=args.train_per_category, seed=args.seed)
    validation_rows = select_validation_slice(
        validation_source,
        per_category=args.validation_per_category,
        offset=args.validation_offset,
    )
    selection = validate_selection(train_rows, validation_rows, existing_validation)
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.validation_output, validation_rows)
    if args.selection_output.exists():
        raise FileExistsError(f"refusing to overwrite {args.selection_output}")
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    selection.update(
        {
            "status": "PASS",
            "seed": args.seed,
            "native_source": NATIVE_SOURCE,
            "train_selection": {
                "source": str(args.train_input),
                "rule": "metadata_nonempty and source==alignment_v2_programmatic_v1; sha256(seed:id) order; first 16 per category",
            },
            "validation_selection": {
                "source": str(args.validation_input),
                "rule": "original manifest order; rows 4:8 per category",
                "offset": args.validation_offset,
            },
            "source_sha256": {
                "train": sha256_file(args.train_input),
                "validation": sha256_file(args.validation_input),
                "existing_validation": sha256_file(args.existing_validation),
            },
            "output_sha256": {
                "train": sha256_file(args.train_output),
                "validation": sha256_file(args.validation_output),
            },
            "outputs": {
                "train": str(args.train_output),
                "validation": str(args.validation_output),
            },
        }
    )
    args.selection_output.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
