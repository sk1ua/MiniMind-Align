"""Fail-closed audit for Alignment v2 generated data."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from validators import jaccard_ngrams, normalize_text, validate_record


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CATEGORIES = ("format", "instruction", "reasoning", "safety", "repetition", "conciseness", "termination", "uncertainty")
REQUIRED_MANIFEST_KEYS = ("id", "split", "category", "family", "difficulty", "prompt", "chosen", "generation_method", "validator", "seed", "source", "metadata")
SECRET_RE = re.compile(r"(AIza[0-9A-Za-z_-]{20,}|sk-[0-9A-Za-z_-]{16,}|gh[pousr]_[0-9A-Za-z_-]{16,}|-----BEGIN)", re.I)


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows, errors = [], []
    if not path.exists():
        return rows, [f"missing:{path}"]
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                errors.append(f"{path}:{number}:not_object")
            else:
                rows.append(value)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{number}:{exc}")
    return rows, errors


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sft_dataset(path: Path) -> tuple[bool, str]:
    """Load one real SFTDataset batch with the repository tokenizer."""
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer

        random.seed(42)
        torch.manual_seed(42)
        sys.path.insert(0, str(REPO))
        from dataset.lm_dataset import SFTDataset

        tokenizer = AutoTokenizer.from_pretrained(str(REPO / "model"))
        dataset = SFTDataset(str(path), tokenizer, max_length=512)
        batch = next(iter(DataLoader(dataset, batch_size=min(2, len(dataset)))))
        if batch[0].ndim != 2 or batch[1].ndim != 2 or batch[0].shape != batch[1].shape:
            return False, "unexpected_batch_shape"
        if int((batch[1] != -100).sum()) <= 0:
            return False, "no_supervised_tokens"
        return True, f"size={len(dataset)},batch_shape={tuple(batch[0].shape)}"
    except Exception as exc:  # fail closed; the exact exception is reportable
        return False, f"{type(exc).__name__}:{exc}"


def choose_files(mode: str) -> tuple[Path, Path, Path, Path]:
    generated, manifests, reports = ROOT / "generated", ROOT / "manifests", ROOT / "reports"
    if mode == "smoke":
        return generated / "smoke_train.jsonl", generated / "smoke_validation.jsonl", manifests / "smoke_train_manifest.jsonl", manifests / "smoke_validation_manifest.jsonl"
    return generated / "new_train_pilot.jsonl", generated / "new_validation_pilot.jsonl", manifests / "train_manifest.jsonl", manifests / "validation_manifest.jsonl"


def audit(mode: str) -> tuple[dict[str, Any], str]:
    train_path, validation_path, train_manifest_path, validation_manifest_path = choose_files(mode)
    train, train_errors = read_jsonl(train_path)
    validation, validation_errors = read_jsonl(validation_path)
    train_manifest_all, train_manifest_errors = read_jsonl(train_manifest_path)
    validation_manifest, validation_manifest_errors = read_jsonl(validation_manifest_path)
    generated_train_manifest = [row for row in train_manifest_all if row.get("generation_method") != "existing_alignment_v1"]
    errors = train_errors + validation_errors + train_manifest_errors + validation_manifest_errors
    if mode == "pilot" and len(train_manifest_all) != len(train) + 600:
        errors.append(f"merged_train_manifest_count:{len(train_manifest_all)} expected:{len(train) + 600}")
    if mode == "smoke":
        generated_train_manifest = train_manifest_all
    if len(train) != len(generated_train_manifest):
        errors.append(f"train_manifest_alignment:{len(train)}:{len(generated_train_manifest)}")
    if len(validation) != len(validation_manifest):
        errors.append(f"validation_manifest_alignment:{len(validation)}:{len(validation_manifest)}")

    report: dict[str, Any] = {
        "mode": mode,
        "paths": {key: str(value) for key, value in {"train": train_path, "validation": validation_path, "train_manifest": train_manifest_path, "validation_manifest": validation_manifest_path}.items()},
        "sha256": {str(path): sha256(path) for path in (train_path, validation_path, train_manifest_path, validation_manifest_path) if path.exists()},
        "counts": {"train": len(train), "validation": len(validation)},
        "category_counts": {"train": dict(Counter(row.get("category") for row in generated_train_manifest)), "validation": dict(Counter(row.get("category") for row in validation_manifest))},
        "generation_method_counts": dict(Counter(row.get("generation_method") for row in generated_train_manifest + validation_manifest)),
        "checks": {},
        "errors": errors,
    }

    ids, prompts, categories = set(), set(), Counter()
    for records, manifests, split in ((train, generated_train_manifest, "train"), (validation, validation_manifest, "validation")):
        for record, manifest in zip(records, manifests):
            for key in REQUIRED_MANIFEST_KEYS:
                if key not in manifest:
                    errors.append(f"missing_manifest_key:{split}:{key}")
            if manifest.get("split") != split:
                errors.append(f"split_mismatch:{manifest.get('id')}")
            if manifest.get("id") in ids:
                errors.append(f"duplicate_id:{manifest.get('id')}")
            ids.add(manifest.get("id"))
            prompt = record.get("conversations", [{}])[0].get("content", "")
            chosen = record.get("conversations", [{}, {}])[1].get("content", "")
            if prompt in prompts:
                errors.append(f"duplicate_prompt:{prompt[:80]}")
            prompts.add(prompt)
            categories[manifest.get("category")] += 1
            if prompt != manifest.get("prompt") or chosen != manifest.get("chosen"):
                errors.append(f"manifest_record_mismatch:{manifest.get('id')}")
            passed, reason = validate_record(record, manifest)
            if not passed:
                errors.append(f"validator:{manifest.get('id')}:{reason}")
            if SECRET_RE.search(prompt + "\n" + chosen) or "</think>" in chosen:
                errors.append(f"secret_or_think_marker:{manifest.get('id')}")
            if not prompt or not chosen:
                errors.append(f"empty_prompt_or_chosen:{manifest.get('id')}")
    report["checks"]["jsonl_structure_and_specialized_validators"] = not any(error.startswith(("missing_manifest_key", "split_mismatch", "duplicate_id", "duplicate_prompt", "manifest_record_mismatch", "validator:", "secret_or_think_marker", "empty_prompt_or_chosen")) for error in errors)
    config = json.loads((ROOT / "config" / "pilot_v1.json").read_text(encoding="utf-8"))
    expected_train = {category: (int(config["smoke_count_per_category"]) if mode == "smoke" else int(config["train_counts"][category])) for category in CATEGORIES}
    expected_validation = {category: (int(config["smoke_count_per_category"]) if mode == "smoke" else int(config["validation_counts"][category])) for category in CATEGORIES}
    if dict(Counter(row.get("category") for row in generated_train_manifest)) != expected_train:
        errors.append("train_category_counts_mismatch")
    if dict(Counter(row.get("category") for row in validation_manifest)) != expected_validation:
        errors.append("validation_category_counts_mismatch")
    report["checks"]["exact_category_counts"] = not any(error.endswith("category_counts_mismatch") for error in errors)

    train_prompts = {row.get("prompt") for row in generated_train_manifest}
    val_prompts = {row.get("prompt") for row in validation_manifest}
    if train_prompts & val_prompts:
        errors.append("train_validation_prompt_overlap")
    report["checks"]["train_validation_disjoint"] = not bool(train_prompts & val_prompts)

    test_rows, test_errors = read_jsonl(REPO / "dataset" / "alignment_v1" / "splits" / "prompts_test.jsonl")
    test_prompts = [row.get("prompt", "") for row in test_rows]
    v1_rows, v1_errors = read_jsonl(REPO / "dataset" / "alignment_v1" / "splits" / "prompts_train.jsonl")
    v1_prompts = {row.get("prompt", "") for row in v1_rows}
    leakage: list[dict[str, Any]] = []
    threshold = json.loads((ROOT / "config" / "pilot_v1.json").read_text(encoding="utf-8"))["similarity"]
    for manifest in generated_train_manifest + validation_manifest:
        prompt = manifest.get("prompt", "")
        if prompt in test_prompts or normalize_text(prompt) in {normalize_text(item) for item in test_prompts}:
            leakage.append({"id": manifest.get("id"), "type": "exact_or_normalized_test_prompt"})
            continue
        for test_prompt in test_prompts:
            sequence = difflib.SequenceMatcher(None, normalize_text(prompt), normalize_text(test_prompt)).ratio()
            jaccard = jaccard_ngrams(prompt, test_prompt)
            if sequence >= threshold["sequence_matcher_threshold"] or (len(normalize_text(prompt)) > 20 and jaccard >= threshold["jaccard_3gram_threshold"]):
                leakage.append({"id": manifest.get("id"), "type": "high_similarity_test_prompt", "sequence": round(sequence, 4), "jaccard": round(jaccard, 4), "test_prompt": test_prompt})
                break
        if any(manifest.get("family") == row.get("family") for row in test_rows):
            leakage.append({"id": manifest.get("id"), "type": "test_family_collision"})
    if train_prompts & v1_prompts:
        errors.append("alignment_v1_train_prompt_overlap")
    if test_errors or v1_errors:
        errors.extend(test_errors + v1_errors)
    errors.extend(f"test_leak:{item['id']}:{item['type']}" for item in leakage)
    report["checks"]["no_test_prompt_leakage"] = not leakage
    report["checks"]["no_alignment_v1_prompt_overlap"] = not bool(train_prompts & v1_prompts)
    report["leakage"] = leakage

    if mode == "pilot":
        sft_train_path = ROOT / "generated" / "sft_train_pilot.jsonl"
        sft_val_path = ROOT / "generated" / "sft_validation_pilot.jsonl"
        sft_train, e1 = read_jsonl(sft_train_path)
        sft_val, e2 = read_jsonl(sft_val_path)
        if len(sft_train) != len(train) + 600 or len(sft_val) != len(validation):
            errors.append(f"sft_merge_count:{len(sft_train)}:{len(sft_val)}")
        report["checks"]["alignment_v1_chosen_merge"] = not e1 and not e2 and len(sft_train) == len(train) + 600 and len(sft_val) == len(validation)
        report["sft_merge_counts"] = {"train": len(sft_train), "validation": len(sft_val)}
        sft_smoke_path = sft_train_path
    else:
        report["checks"]["alignment_v1_chosen_merge"] = True
        sft_smoke_path = train_path

    loaded, load_reason = check_sft_dataset(sft_smoke_path)
    report["checks"]["real_sftdataset_batch"] = loaded
    report["sftdataset_batch"] = load_reason
    if not loaded:
        errors.append(f"SFTDataset_load:{load_reason}")

    report["checks"]["all_categories_present"] = set(categories) == set(CATEGORIES)
    if set(categories) != set(CATEGORIES):
        errors.append("missing_category")
    report["errors"] = errors
    report["passed"] = not errors
    report["error_count"] = len(errors)
    return report, "\n".join(errors)


def write_report(mode: str, report: dict[str, Any]) -> None:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    json_path = reports / f"{'smoke' if mode == 'smoke' else 'pilot'}_audit.json"
    md_path = reports / f"{'smoke' if mode == 'smoke' else 'pilot'}_audit.md"
    if json_path.exists() or md_path.exists():
        raise FileExistsError("refusing to overwrite audit report")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [f"# Alignment v2 {mode} audit", "", f"- passed: {report['passed']}", f"- errors: {report['error_count']}", f"- counts: {json.dumps(report['counts'], ensure_ascii=False)}", "", "## Checks", ""]
    lines.extend(f"- {key}: {value}" for key, value in report["checks"].items())
    lines.extend(["", "## Errors", ""] + [f"- {error}" for error in report["errors"]] if report["errors"] else ["- none"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "pilot"), required=True)
    args = parser.parse_args()
    report, _ = audit(args.mode)
    write_report(args.mode, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
