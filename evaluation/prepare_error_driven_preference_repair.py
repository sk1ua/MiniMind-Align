"""Prepare error-driven SFT and validator-backed preference data.

The preparation is deliberately offline. Native Alignment v2 chosen answers
are the only positive targets; failed model generations are used only as
rejected preference examples and as error tags for targeted SFT oversampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.alignment_v2.validators import validate_record  # noqa: E402


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
TARGET_CATEGORIES = {"conciseness", "format", "instruction", "reasoning", "repetition"}
GUARD_CATEGORIES = {"safety", "termination", "uncertainty"}
NATIVE_SOURCE = "alignment_v2_programmatic_v1"


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return True


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def stable_key(seed: int, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).hexdigest()


def validate_chosen(row: dict) -> tuple[bool, str]:
    record = {
        "conversations": [
            {"role": "user", "content": row.get("prompt", "")},
            {"role": "assistant", "content": row.get("chosen", "")},
        ]
    }
    return validate_record(record, {"category": row["category"], "metadata": row["metadata"]})


def conversation(prompt: str, response: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def repeat_ratio(text: str) -> float:
    grams = [text[i : i + 3] for i in range(max(0, len(text) - 2))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def prepare(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    ensure_empty(output_dir)
    train_path = Path(args.train_manifest)
    validation_path = Path(args.validation_manifest)
    release_path = Path(args.release_manifest)
    failure_path = Path(args.failure_samples)

    train_all = read_jsonl(train_path)
    validation = read_jsonl(validation_path)
    release = read_jsonl(release_path)
    samples = read_jsonl(failure_path)

    native_train = [
        row
        for row in train_all
        if row.get("source") == NATIVE_SOURCE and isinstance(row.get("metadata"), dict) and row["metadata"]
    ]
    if len(native_train) != args.expected_native_train_count:
        raise ValueError(
            f"native train count mismatch: expected {args.expected_native_train_count}, got {len(native_train)}"
        )
    if len(validation) != args.expected_validation_count:
        raise ValueError(f"validation count mismatch: expected {args.expected_validation_count}, got {len(validation)}")
    if len(release) != args.expected_release_count:
        raise ValueError(f"release count mismatch: expected {args.expected_release_count}, got {len(release)}")

    by_id = {row["id"]: row for row in native_train}
    if len(by_id) != len(native_train):
        raise ValueError("duplicate native train ids")

    chosen_replay_mismatches: list[dict] = []
    for row in native_train:
        passed, reason = validate_chosen(row)
        if not passed:
            chosen_replay_mismatches.append({"id": row["id"], "reason": reason})
    if chosen_replay_mismatches:
        raise ValueError(f"native chosen validator replay failed: {chosen_replay_mismatches[:3]}")

    train_ids = {row["id"] for row in native_train}
    validation_ids = {row["id"] for row in validation}
    release_ids = {row["id"] for row in release}
    train_families = {row.get("family") for row in native_train}
    validation_families = {row.get("family") for row in validation}
    release_families = {row.get("family") for row in release}
    train_prompts = {row.get("prompt") for row in native_train}
    validation_prompts = {row.get("prompt") for row in validation}
    release_prompts = {row.get("prompt") for row in release}
    leakage = {
        "train_validation_ids": len(train_ids & validation_ids),
        "train_release_ids": len(train_ids & release_ids),
        "train_validation_families": len(train_families & validation_families),
        "train_release_families": len(train_families & release_families),
        "train_validation_prompts": len(train_prompts & validation_prompts),
        "train_release_prompts": len(train_prompts & release_prompts),
        "validation_release_expected_subset_ids": len(validation_ids & release_ids),
    }
    if any(leakage[key] for key in (
        "train_validation_ids",
        "train_release_ids",
        "train_validation_families",
        "train_release_families",
        "train_validation_prompts",
        "train_release_prompts",
    )):
        raise ValueError(f"train/evaluation leakage detected: {leakage}")

    sample_linkage_errors: list[dict] = []
    sample_family_missing = 0
    sample_failures: list[dict] = []
    for sample in samples:
        prompt_id = sample.get("prompt_id")
        manifest = by_id.get(prompt_id)
        if manifest is None:
            sample_linkage_errors.append({"sample_key": sample.get("sample_key"), "prompt_id": prompt_id})
            continue
        if sample.get("category") != manifest.get("category") or (
            sample.get("family") is not None and sample.get("family") != manifest.get("family")
        ):
            sample_linkage_errors.append({
                "sample_key": sample.get("sample_key"),
                "prompt_id": prompt_id,
                "sample_category": sample.get("category"),
                "manifest_category": manifest.get("category"),
                "sample_family": sample.get("family"),
                "manifest_family": manifest.get("family"),
            })
            continue
        if sample.get("family") is None:
            sample_family_missing += 1
        passed, reason = validate_record(
            {
                "conversations": [
                    {"role": "user", "content": manifest["prompt"]},
                    {"role": "assistant", "content": sample.get("response", "")},
                ]
            },
            {"category": manifest["category"], "metadata": manifest["metadata"]},
        )
        if not passed:
            failure = dict(sample)
            failure.update({
                "failure_reason": reason,
                "manifest_id": manifest["id"],
                "manifest_category": manifest["category"],
                "manifest_family": manifest.get("family"),
                "repetition_ratio_replayed": repeat_ratio(str(sample.get("response", ""))),
            })
            sample_failures.append(failure)

    if sample_linkage_errors:
        raise ValueError(f"sample linkage errors: {sample_linkage_errors[:3]}")
    if not sample_failures:
        raise ValueError("no failed generations available for error-driven preference data")

    category_rows = defaultdict(list)
    for row in native_train:
        category_rows[row["category"]].append(row)
    base_sft: list[dict] = []
    base_selection: list[dict] = []
    for category in CATEGORIES:
        quota = args.target_per_category if category in TARGET_CATEGORIES else args.guard_per_category
        candidates = sorted(category_rows[category], key=lambda row: stable_key(args.seed, row["id"]))
        if len(candidates) < quota:
            raise ValueError(f"insufficient rows for {category}: need {quota}, got {len(candidates)}")
        for row in candidates[:quota]:
            base_sft.append({"conversations": conversation(row["prompt"], row["chosen"])})
            base_selection.append({
                "id": row["id"],
                "category": row["category"],
                "family": row.get("family"),
                "origin": "native_v2_balanced_chosen",
                "validator_pass": True,
            })

    by_prompt_failures: dict[str, list[dict]] = defaultdict(list)
    for failure in sample_failures:
        by_prompt_failures[failure["prompt_id"]].append(failure)
    hard_prompt_failures = []
    for prompt_id, rows in sorted(by_prompt_failures.items()):
        hard_prompt_failures.append(sorted(
            rows,
            key=lambda row: (
                -float(row.get("repetition_ratio_replayed", 0.0)),
                -int(row.get("generated_tokens", 0)),
                str(row.get("failure_reason", "")),
                str(row.get("sample_key", "")),
            ),
        )[0])

    error_sft: list[dict] = []
    error_selection: list[dict] = []
    for failure in hard_prompt_failures:
        target = by_id[failure["prompt_id"]]
        error_sft.append({"conversations": conversation(target["prompt"], target["chosen"])})
        error_selection.append({
            "id": target["id"],
            "category": target["category"],
            "family": target.get("family"),
            "origin": "error_driven_hard_prompt_repair",
            "validator_pass": True,
            "failure_reason": failure["failure_reason"],
            "source_sample_key": failure.get("sample_key"),
        })

    sft_rows = base_sft + error_sft
    sft_selection = base_selection + error_selection
    preference_rows: list[dict] = []
    for index, failure in enumerate(sorted(sample_failures, key=lambda row: str(row.get("sample_key", "")))):
        target = by_id[failure["prompt_id"]]
        rejected = str(failure.get("response", ""))
        if not rejected.strip() or rejected == target["chosen"]:
            raise ValueError(f"invalid rejected response for {failure.get('sample_key')}")
        preference_rows.append({
            "id": f"error_pair_{index:04d}_{failure.get('sample_key', 'unknown').replace(':', '_')}",
            "split": "train",
            "category": target["category"],
            "family": target.get("family"),
            "prompt": target["prompt"],
            "chosen": conversation(target["prompt"], target["chosen"]),
            "rejected": conversation(target["prompt"], rejected),
            "source": "native_v2_chosen_plus_corrected_grpo_failure",
            "source_sample_key": failure.get("sample_key"),
            "failure_reason": failure["failure_reason"],
            "rejected_repeat_3gram_ratio": failure["repetition_ratio_replayed"],
            "chosen_validator_pass": True,
            "rejected_validator_pass": False,
        })

    sft_path = output_dir / "error_driven_sft.jsonl"
    preference_path = output_dir / "preference_train.jsonl"
    failure_path_out = output_dir / "failure_pairs.jsonl"
    selection_path = output_dir / "selection_manifest.json"
    manifest_path = output_dir / "data_manifest.json"
    report_path = output_dir / "report.md"
    write_jsonl(sft_path, sft_rows)
    write_jsonl(preference_path, preference_rows)
    write_jsonl(failure_path_out, sample_failures)
    selection_path.write_text(json.dumps({
        "base_sft": base_selection,
        "error_sft": error_selection,
        "preference_count": len(preference_rows),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = {
        "schema_version": 1,
        "status": "ERROR_DRIVEN_DATA_READY",
        "seed": args.seed,
        "native_source": NATIVE_SOURCE,
        "source_hashes": {
            "train_manifest": sha256_file(train_path),
            "validation_manifest": sha256_file(validation_path),
            "release_manifest": sha256_file(release_path),
            "failure_samples": sha256_file(failure_path),
        },
        "native_train": {
            "rows": len(native_train),
            "categories": dict(Counter(row["category"] for row in native_train)),
            "families": len({row.get("family") for row in native_train}),
            "chosen_validator_pass": f"{len(native_train)}/{len(native_train)}",
        },
        "evaluation": {
            "validation_rows": len(validation),
            "release_rows": len(release),
            "leakage": leakage,
        },
        "failure_audit": {
            "sample_rows": len(samples),
            "failed_rows": len(sample_failures),
            "hard_prompt_count": len(hard_prompt_failures),
            "category_counts": dict(Counter(row["manifest_category"] for row in sample_failures)),
            "reason_counts": dict(Counter(row["failure_reason"] for row in sample_failures)),
            "linkage_errors": len(sample_linkage_errors),
            "family_missing_filled_from_manifest": sample_family_missing,
        },
        "sft": {
            "base_count": len(base_sft),
            "error_repair_count": len(error_sft),
            "total_count": len(sft_rows),
            "base_category_counts": dict(Counter(row["category"] for row in base_selection)),
            "total_category_counts": dict(Counter(row["category"] for row in sft_selection)),
            "selection_policy": "sha256(seed:id) stable order; target categories 96 each; guard categories 32 each; one hard-prompt chosen replay target appended per failing prompt",
            "validator_target_all_pass": True,
        },
        "preference": {
            "pair_count": len(preference_rows),
            "category_counts": dict(Counter(row["category"] for row in preference_rows)),
            "reason_counts": dict(Counter(row["failure_reason"] for row in preference_rows)),
            "chosen_all_pass": all(row["chosen_validator_pass"] for row in preference_rows),
            "rejected_all_fail": all(not row["rejected_validator_pass"] for row in preference_rows),
            "selection_policy": "all failed generated samples paired with the native v2 chosen response",
        },
        "artifacts": {
            "error_driven_sft": str(sft_path),
            "preference_train": str(preference_path),
            "failure_pairs": str(failure_path_out),
            "selection_manifest": str(selection_path),
        },
        "diagnostic_only": True,
        "default_model_changed": False,
        "gpu_wall_seconds": 0,
    }
    if not finite_tree(result):
        raise ValueError("non-finite result")
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_path.write_text(
        "# Error-driven SFT and preference data\n\n"
        f"- Status: {result['status']}\n"
        f"- Native chosen replay: {result['native_train']['chosen_validator_pass']}\n"
        f"- Failed generation rows used as rejected pairs: {len(sample_failures)}\n"
        f"- SFT rows: {len(sft_rows)} ({len(base_sft)} balanced chosen + {len(error_sft)} hard-prompt repairs)\n"
        f"- Preference pairs: {len(preference_rows)}\n"
        f"- Train/evaluation leakage: {leakage}\n"
        "- No reward, validator, default weight or existing experiment directory was modified.\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--release-manifest", required=True)
    parser.add_argument("--failure-samples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-native-train-count", type=int, default=1016)
    parser.add_argument("--expected-validation-count", type=int, default=160)
    parser.add_argument("--expected-release-count", type=int, default=32)
    parser.add_argument("--target-per-category", type=int, default=96)
    parser.add_argument("--guard-per-category", type=int, default=32)
    args = parser.parse_args()
    result = prepare(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
