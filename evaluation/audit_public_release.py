from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root: Path) -> dict:
    public = root / "results/public"
    summary_path = public / "summary.json"
    metrics_path = public / "metrics.csv"
    limitations_path = public / "limitations.md"
    hashes_path = public / "artifact_hashes.json"
    for path in (summary_path, metrics_path, limitations_path, hashes_path):
        if not path.exists():
            raise AssertionError(f"missing public artifact: {path}")
    summary = _load(summary_path)
    if summary["default_model_changed"] is not False:
        raise AssertionError("default model change must remain false")
    if summary["formal_rl_authorized"] is not False:
        raise AssertionError("formal RL must remain unauthorized")
    if summary["corrected_grpo"]["validator_gain_over_source_sft"] != 0:
        raise AssertionError("public RL conclusion changed")
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5:
        raise AssertionError("expected five public metric rows")
    if not any(row["method"] == "Error-driven SFT" and row["full_validation"] == "79/160" for row in rows):
        raise AssertionError("SFT public metric missing")
    if not any(row["method"] == "corrected-GRPO" and row["release_slice"] == "19/32" for row in rows):
        raise AssertionError("GRPO public metric missing")
    hashes = _load(hashes_path)
    for relative, expected in hashes["public_files"].items():
        if _sha256(root / relative) != expected:
            raise AssertionError(f"hash mismatch: {relative}")
    return {"status": "PUBLIC_AUDIT_PASS", "methods": len(rows), "default_model_changed": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(audit(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
