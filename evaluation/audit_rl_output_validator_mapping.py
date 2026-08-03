"""Audit generated-output to validator/reward mapping without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from align.rl_rules import rule_reward
from dataset.alignment_v2.validators import repeat_3gram_ratio, validate_record


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
COMPONENTS = (
    "validator_reward",
    "parse_reward",
    "field_reward",
    "item_count_reward",
    "arithmetic_reward",
    "format_reward",
    "termination_reward",
    "repetition_penalty",
)
VALIDATOR_FUNCTIONS = {category: f"validate_{category}" for category in CATEGORIES}
REQUIRED_METADATA_KEYS = {
    "conciseness": {"max_chars", "required_terms"},
    "format": {"format_type", "expected"},
    "instruction": {"separator", "allowed_words", "count"},
    "reasoning": {"answer"},
    "repetition": {"count"},
    "safety": {"required_markers"},
    "termination": {"max_chars"},
    "uncertainty": {"required_markers"},
}
TASK_COMPONENTS = {
    "format": {"parse_reward", "field_reward", "item_count_reward", "format_reward"},
    "instruction": {"item_count_reward", "format_reward"},
    "reasoning": {"arithmetic_reward", "format_reward"},
    "repetition": {"item_count_reward"},
}
STRUCTURAL_FAILURES = {
    "extra_prefix_suffix_or_code_block",
    "parse_error",
    "newline_whitespace_or_numbering",
    "not_single_line_expression",
    "count_or_duplicate_mismatch",
    "count_or_repetition",
    "termination_constraint",
}


def finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ensure_empty_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty audit directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> tuple[list[dict], list[dict[str, object]]]:
    rows: list[dict] = []
    errors: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append({"path": path.as_posix(), "line": line_number, "error": str(exc)})
            continue
        if not isinstance(value, dict) or not finite_tree(value):
            errors.append({"path": path.as_posix(), "line": line_number, "error": "non_object_or_nonfinite"})
            continue
        rows.append(value)
    return rows, errors


def number(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(default if value is None else value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def close(left: object, right: object, tolerance: float = 1e-9) -> bool:
    return abs(number(left) - number(right)) <= tolerance


def classify_failure(reason: str) -> str:
    if reason in STRUCTURAL_FAILURES:
        return "structural"
    if reason:
        return "semantic_value"
    return "none"


def metadata_errors(row: dict) -> list[str]:
    category = str(row.get("category", ""))
    metadata = row.get("metadata")
    errors: list[str] = []
    if not isinstance(metadata, dict) or not metadata:
        return ["metadata_missing_or_empty"]
    for key in sorted(REQUIRED_METADATA_KEYS.get(category, set())):
        if key not in metadata:
            errors.append(f"missing:{key}")
    return errors


def component_contract_mismatches(category: str, validator_pass: bool, components: dict) -> list[str]:
    expected_value = 1.0 if validator_pass else 0.0
    active = TASK_COMPONENTS.get(category, set())
    routed_components = {"parse_reward", "field_reward", "item_count_reward", "arithmetic_reward", "format_reward"}
    return [
        component
        for component in sorted(routed_components)
        if not close(components.get(component), expected_value if component in active else 0.0)
    ]


def replay(manifest: dict, response: str) -> dict[str, object]:
    category = str(manifest.get("category", ""))
    metadata = manifest.get("metadata") or {}
    record = {
        "conversations": [
            {"role": "user", "content": str(manifest.get("prompt", ""))},
            {"role": "assistant", "content": response},
        ]
    }
    try:
        passed, reason = validate_record(record, {"category": category, "metadata": metadata})
        reward, components = rule_reward(category, str(manifest.get("prompt", "")), response, metadata)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return {
            "validator_pass": False,
            "validator_reason": f"exception:{type(exc).__name__}",
            "failure_class": "mapping_error",
            "reward": None,
            "components": {},
            "error": str(exc),
        }
    normalized_components = {key: number(components.get(key)) for key in COMPONENTS}
    return {
        "validator_pass": bool(passed),
        "validator_reason": str(reason),
        "failure_class": classify_failure(str(reason)) if not passed else "none",
        "reward": float(reward),
        "components": normalized_components,
        "error": None,
    }


def response_features(sample: dict, response: str) -> dict[str, object]:
    return {
        "char_count": len(response),
        "utf8_bytes": len(response.encode("utf-8")),
        "line_count": len(response.splitlines()) if response else 0,
        "has_newline": "\n" in response,
        "has_carriage_return": "\r" in response,
        "leading_or_trailing_whitespace": response != response.strip(),
        "empty": not response.strip(),
        "repeat_3gram_ratio": repeat_3gram_ratio(response),
        "generated_tokens": sample.get("generated_tokens"),
        "termination_reason": sample.get("termination_reason"),
        "eos_seen": sample.get("eos_seen"),
        "finished_naturally": sample.get("finished_naturally"),
        "max_length_hit": sample.get("max_length_hit"),
    }


def audit(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.experiment_root)
    run_dir = root / args.run_name
    output_dir = Path(args.output_dir)
    ensure_empty_output_dir(output_dir)
    manifest_path = Path(args.manifest) if args.manifest else run_dir / "resolved_train_manifest.jsonl"
    sample_path = run_dir / "samples.jsonl"
    validator_path = ROOT / "dataset/alignment_v2/validators.py"
    rules_path = ROOT / "align/rl_rules.py"
    required = [manifest_path, sample_path, validator_path, rules_path]
    missing = [path.as_posix() for path in required if not path.exists()]
    if missing:
        result = {"status": "TELEMETRY_INCOMPLETE", "missing": missing, "diagnostic_only": True}
        (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    manifest_rows, manifest_errors = read_jsonl(manifest_path)
    samples, sample_errors = read_jsonl(sample_path)
    manifest_by_id = {str(row.get("id")): row for row in manifest_rows if row.get("id") is not None}
    manifest_duplicate_ids = len(manifest_rows) - len(manifest_by_id)
    manifest_contract_errors: list[dict[str, object]] = []
    chosen_replays: list[dict[str, object]] = []
    for row in manifest_rows:
        category = str(row.get("category", ""))
        expected_validator = VALIDATOR_FUNCTIONS.get(category)
        errors = metadata_errors(row)
        if row.get("validator") != expected_validator:
            errors.append(f"validator_field:{row.get('validator')}!=:{expected_validator}")
        if row.get("source") != "alignment_v2_programmatic_v1":
            errors.append("source_mismatch")
        if errors:
            manifest_contract_errors.append({"id": row.get("id"), "category": category, "errors": errors})
        chosen_replays.append(replay(row, str(row.get("chosen", ""))))

    chosen_validator_pass = sum(bool(row["validator_pass"]) for row in chosen_replays)
    chosen_replay_errors = [row.get("error") for row in chosen_replays if row.get("error")]
    joined: list[dict[str, object]] = []
    missing_prompt_ids: list[str] = []
    category_mismatches: list[str] = []
    family_mismatches: list[str] = []
    sample_keys: set[str] = set()
    duplicate_sample_keys: list[str] = []
    replay_reward_mismatches: list[str] = []
    persisted_reward_mismatches: list[str] = []
    persisted_component_mismatches: list[str] = []
    persisted_validator_mismatches: list[str] = []
    component_routing_mismatches: list[dict[str, object]] = []
    mapping_errors: list[dict[str, object]] = []
    for index, sample in enumerate(samples):
        sample_key = str(sample.get("sample_key", ""))
        if not sample_key or sample_key in sample_keys:
            duplicate_sample_keys.append(sample_key or f"<missing:{index}>")
        sample_keys.add(sample_key)
        prompt_id = str(sample.get("prompt_id", ""))
        manifest = manifest_by_id.get(prompt_id)
        if manifest is None:
            missing_prompt_ids.append(prompt_id)
            continue
        category = str(manifest.get("category", ""))
        if str(sample.get("category")) != category:
            category_mismatches.append(sample_key)
        if sample.get("family") is not None and str(sample.get("family")) != str(manifest.get("family")):
            family_mismatches.append(sample_key)
        response = str(sample.get("response", ""))
        replay_result = replay(manifest, response)
        joined_row = {"sample": sample, "manifest": manifest, "replay": replay_result}
        joined.append(joined_row)
        if replay_result.get("error"):
            mapping_errors.append({"sample_key": sample_key, "error": replay_result.get("error")})
        if not close(sample.get("reward"), replay_result.get("reward")):
            replay_reward_mismatches.append(sample_key)
        if not close(sample.get("rule_reward"), replay_result.get("reward")):
            persisted_reward_mismatches.append(sample_key)
        persisted_components = sample.get("components") or {}
        if any(not close(persisted_components.get(component), replay_result["components"].get(component)) for component in COMPONENTS):
            persisted_component_mismatches.append(sample_key)
        if not close(persisted_components.get("validator_reward"), 1.0 if replay_result["validator_pass"] else 0.0):
            persisted_validator_mismatches.append(sample_key)
        routed = component_contract_mismatches(category, bool(replay_result["validator_pass"]), replay_result["components"])
        if routed:
            component_routing_mismatches.append({"sample_key": sample_key, "components": routed})

    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    failure_cases: list[dict[str, object]] = []
    failure_reasons: Counter[str] = Counter()
    failure_classes: Counter[str] = Counter()
    for row in joined:
        category = str(row["manifest"].get("category", ""))
        by_category[category].append(row)
        replay_result = row["replay"]
        if not replay_result["validator_pass"]:
            sample = row["sample"]
            manifest = row["manifest"]
            reason = str(replay_result["validator_reason"])
            failure_reasons[f"{category}:{reason}"] += 1
            failure_classes[str(replay_result["failure_class"])] += 1
            failure_cases.append({
                "sample_key": sample.get("sample_key"),
                "step": sample.get("step"),
                "micro_index": sample.get("micro_index"),
                "prompt_id": sample.get("prompt_id"),
                "category": category,
                "family": manifest.get("family"),
                "validator_branch": VALIDATOR_FUNCTIONS.get(category),
                "failure_reason": reason,
                "failure_class": replay_result["failure_class"],
                "response_sha256": sha256_text(str(sample.get("response", ""))),
                "response_features": response_features(sample, str(sample.get("response", ""))),
                "metadata_keys": sorted((manifest.get("metadata") or {}).keys()),
                "persisted_validator_reward": (sample.get("components") or {}).get("validator_reward"),
                "replayed_validator_reward": replay_result["components"].get("validator_reward"),
                "persisted_rule_reward": sample.get("rule_reward"),
                "replayed_rule_reward": replay_result.get("reward"),
            })

    category_summary: dict[str, dict[str, object]] = {}
    for category in CATEGORIES:
        rows = by_category.get(category, [])
        category_summary[category] = {
            "sample_count": len(rows),
            "prompt_count": len({str(row["sample"].get("prompt_id")) for row in rows}),
            "family_count": len({str(row["manifest"].get("family")) for row in rows}),
            "validator_pass": sum(bool(row["replay"]["validator_pass"]) for row in rows),
            "failure_reasons": dict(sorted(Counter(str(row["replay"]["validator_reason"]) for row in rows if not row["replay"]["validator_pass"]).items())),
            "failure_classes": dict(sorted(Counter(str(row["replay"]["failure_class"]) for row in rows if not row["replay"]["validator_pass"]).items())),
        }

    observed_prompt_ids = {str(row["sample"].get("prompt_id")) for row in joined}
    observed_families = {str(row["manifest"].get("family")) for row in joined}
    manifest_families = {str(row.get("family")) for row in manifest_rows}
    termination_zero_natural = sum(
        row["sample"].get("finished_naturally") is True
        and number(row["replay"]["components"].get("termination_reward")) == 0.0
        for row in joined
    )
    termination_reasons = Counter(str(row["sample"].get("termination_reason", "missing")) for row in joined)
    warnings: list[str] = []
    if len(observed_prompt_ids) < len(manifest_by_id):
        warnings.append("OUTPUT_COVERAGE_LIMITED_PROMPT_SCOPE")
    if len(observed_families) < len(manifest_families):
        warnings.append("OUTPUT_COVERAGE_LIMITED_FAMILY_SCOPE")
    if termination_zero_natural:
        warnings.append("TERMINATION_REWARD_IS_SINGLE_LINE_NOT_EOS")
    if failure_cases:
        warnings.append("FAILURES_ARE_MODEL_OUTPUT_CONTRACT_ERRORS_UNLESS_MAPPING_MISMATCH_FOUND")
    warnings.append("SINGLE_SEED_DIAGNOSTIC_NOT_PROMOTION_EVIDENCE")

    mapping_integrity_ok = not (
        manifest_errors
        or sample_errors
        or manifest_contract_errors
        or manifest_duplicate_ids
        or chosen_replay_errors
        or chosen_validator_pass != len(manifest_rows)
        or missing_prompt_ids
        or category_mismatches
        or family_mismatches
        or duplicate_sample_keys
        or mapping_errors
        or replay_reward_mismatches
        or persisted_reward_mismatches
        or persisted_component_mismatches
        or persisted_validator_mismatches
        or component_routing_mismatches
    )
    coverage_limited = len(observed_prompt_ids) < len(manifest_by_id) or len(observed_families) < len(manifest_families)
    status = (
        "TELEMETRY_INCOMPLETE"
        if manifest_errors or sample_errors
        else "OUTPUT_VALIDATOR_MAPPING_MISMATCH_BLOCKED"
        if not mapping_integrity_ok
        else "OUTPUT_VALIDATOR_MAPPING_CONSISTENT_LIMITED_DIAGNOSTIC"
        if coverage_limited
        else "OUTPUT_VALIDATOR_MAPPING_CONSISTENT_DIAGNOSTIC"
    )

    input_manifest = []
    for path in (manifest_path, sample_path, validator_path, rules_path):
        input_manifest.append({"path": path.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    result = {
        "status": status,
        "diagnostic_only": True,
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        "default_model_changed": False,
        "experiment_root": str(root),
        "run_name": args.run_name,
        "input_manifest": input_manifest,
        "manifest_contract": {
            "row_count": len(manifest_rows),
            "category_counts": dict(sorted(Counter(str(row.get("category")) for row in manifest_rows).items())),
            "family_count": len(manifest_families),
            "contract_error_count": len(manifest_contract_errors),
            "chosen_validator": f"{chosen_validator_pass}/{len(manifest_rows)}",
            "chosen_replay_error_count": len(chosen_replay_errors),
        },
        "sample_contract": {
            "row_count": len(samples),
            "unique_sample_key_count": len(sample_keys),
            "duplicate_sample_key_count": len(duplicate_sample_keys),
            "missing_prompt_id_count": len(missing_prompt_ids),
            "category_mismatch_count": len(category_mismatches),
            "family_mismatch_count": len(family_mismatches),
            "prompt_coverage": f"{len(observed_prompt_ids)}/{len(manifest_by_id)}",
            "family_coverage": f"{len(observed_families)}/{len(manifest_families)}",
            "category_counts": {category: len(by_category.get(category, [])) for category in CATEGORIES},
        },
        "replay_contract": {
            "mapping_error_count": len(mapping_errors),
            "reward_mismatch_count": len(replay_reward_mismatches),
            "persisted_rule_reward_mismatch_count": len(persisted_reward_mismatches),
            "persisted_component_mismatch_count": len(persisted_component_mismatches),
            "persisted_validator_mismatch_count": len(persisted_validator_mismatches),
            "component_routing_mismatch_count": len(component_routing_mismatches),
        },
        "failure_summary": {
            "failed_sample_count": len(failure_cases),
            "failure_reasons": dict(sorted(failure_reasons.items())),
            "failure_classes": dict(sorted(failure_classes.items())),
            "by_category": category_summary,
        },
        "termination_semantics": {
            "termination_reason_counts": dict(sorted(termination_reasons.items())),
            "natural_end_with_zero_termination_reward": termination_zero_natural,
            "interpretation": "termination_reward is a non-empty single-line reward, not an EOS/natural-end reward.",
        },
        "warnings": warnings,
        "mapping_integrity_ok": mapping_integrity_ok,
        "coverage_limited": coverage_limited,
        "execution": {"gpu_started": False, "gpu_wall_seconds": 0, "server_status": "RUNNING"},
        "next_decision": "If mapping integrity is consistent, the remaining failures are output-contract evidence rather than a validator/reward wiring defect. Keep reward and model unchanged; do not start formal RL from this single-seed, partial-prompt audit.",
    }
    (output_dir / "input_manifest.json").write_text(json.dumps(input_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output_dir / "failure_cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in failure_cases:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        "\n".join([
            "# Output to validator mapping audit",
            "",
            f"- status: `{status}`",
            f"- chosen validator replay: {chosen_validator_pass}/{len(manifest_rows)}",
            f"- generated samples: {len(samples)}; prompt coverage: {len(observed_prompt_ids)}/{len(manifest_by_id)}; family coverage: {len(observed_families)}/{len(manifest_families)}",
            f"- replay mismatches: reward={len(replay_reward_mismatches)}, components={len(persisted_component_mismatches)}, routing={len(component_routing_mismatches)}",
            f"- failed samples: {len(failure_cases)}; failure classes: {dict(sorted(failure_classes.items()))}",
            f"- natural-end samples with zero termination reward: {termination_zero_natural}",
            "",
            "The audit is artifact-only. It does not rewrite responses or authorize reward, optimizer, model or formal-RL changes.",
            "",
        ]),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest")
    result = audit(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] in {"TELEMETRY_INCOMPLETE", "OUTPUT_VALIDATOR_MAPPING_MISMATCH_BLOCKED"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
