"""Audit current prompt/output coverage and natural reward signal variability offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from align.rl_rules import rule_reward
from dataset.alignment_v2.validators import validate_record
from evaluation.audit_corrected_kl_gate import ensure_empty_output_dir


COMPONENT_KEYS = (
    "validator_reward",
    "parse_reward",
    "field_reward",
    "item_count_reward",
    "arithmetic_reward",
    "format_reward",
    "termination_reward",
    "repetition_penalty",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return True


def _float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _path(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def _read_jsonl(path: Path) -> tuple[list[dict], list[dict[str, object]]]:
    rows: list[dict] = []
    errors: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append({"path": path.as_posix(), "line": line_number, "error": str(exc)})
            continue
        if not isinstance(row, dict):
            errors.append({"path": path.as_posix(), "line": line_number, "error": "row_not_object"})
            continue
        rows.append(row)
    return rows, errors


def collect_sample_files(sample_roots: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for root in sample_roots:
        if root.is_file() and root.name == "samples.jsonl":
            files.add(root)
        elif root.is_dir():
            files.update(root.rglob("samples.jsonl"))
    return sorted(files)


def _manifest_record(row: dict) -> dict:
    return {
        "conversations": [
            {"role": "user", "content": str(row.get("prompt", ""))},
            {"role": "assistant", "content": str(row.get("chosen", ""))},
        ]
    }


def _validate(row: dict, response: str) -> tuple[bool, str]:
    manifest = {"category": row.get("category"), "metadata": row.get("metadata") or {}}
    record = {
        "conversations": [
            {"role": "user", "content": str(row.get("prompt", ""))},
            {"role": "assistant", "content": response},
        ]
    }
    try:
        return validate_record(record, manifest)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return False, f"validator_exception:{type(exc).__name__}"


def _score(row: dict, response: str) -> tuple[float, dict[str, float], bool, str]:
    passed, reason = _validate(row, response)
    try:
        reward, components = rule_reward(
            str(row.get("category", "")),
            str(row.get("prompt", "")),
            response,
            row.get("metadata") or {},
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return float("nan"), {}, passed, f"reward_exception:{type(exc).__name__}"
    return float(reward), {key: float(components.get(key, 0.0)) for key in COMPONENT_KEYS}, passed, reason


def _component_summary(rows: list[dict], component: str) -> dict[str, object]:
    values: list[float] = []
    missing = 0
    for row in rows:
        value = _float((row.get("components") or {}).get(component))
        if value is None:
            missing += 1
        else:
            values.append(value)
    nonzero = sum(abs(value) > 1e-12 for value in values)
    return {
        "observed_count": len(values),
        "missing_count": missing,
        "coverage_rate": len(values) / len(rows) if rows else 0.0,
        "nonzero_count": nonzero,
        "nonzero_rate": nonzero / len(values) if values else 0.0,
        "unique_count": len(set(values)),
        "mean": statistics.fmean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _probe_summary(manifest_rows: list[dict]) -> dict[str, object]:
    variants = ("chosen", "empty", "newline_suffix")
    outcomes: dict[str, list[dict[str, object]]] = {variant: [] for variant in variants}
    for row in manifest_rows:
        chosen = str(row.get("chosen", ""))
        texts = {
            "chosen": chosen,
            "empty": "",
            "newline_suffix": chosen + "\n",
        }
        for variant, text in texts.items():
            reward, components, passed, reason = _score(row, text)
            outcomes[variant].append({
                "reward": reward,
                "components": components,
                "validator_pass": passed,
                "validator_reason": reason,
            })

    variant_summary: dict[str, object] = {}
    for variant, rows in outcomes.items():
        rewards = [float(row["reward"]) for row in rows if math.isfinite(float(row["reward"]))]
        variant_summary[variant] = {
            "count": len(rows),
            "reward_unique_count": len(set(rewards)),
            "reward_mean": statistics.fmean(rewards) if rewards else None,
            "validator_pass_count": sum(bool(row["validator_pass"]) for row in rows),
            "validator_pass_rate": sum(bool(row["validator_pass"]) for row in rows) / len(rows) if rows else 0.0,
            "termination_reward_unique_count": len({
                float(row["components"].get("termination_reward", 0.0)) for row in rows
            }),
            "termination_reward_values": sorted({
                float(row["components"].get("termination_reward", 0.0)) for row in rows
            }),
            "failure_reasons": dict(Counter(
                str(row["validator_reason"]) for row in rows if not row["validator_pass"]
            )),
        }
    probe_reward_values = {
        float(row["reward"])
        for rows in outcomes.values()
        for row in rows
        if math.isfinite(float(row["reward"]))
    }
    probe_validator_values = {
        bool(row["validator_pass"])
        for rows in outcomes.values()
        for row in rows
    }
    probe_termination_values = {
        float(row["components"].get("termination_reward", 0.0))
        for rows in outcomes.values()
        for row in rows
    }
    return {
        "variant_summaries": variant_summary,
        "probe_reward_unique_count": len(probe_reward_values),
        "probe_validator_outcome_count": len(probe_validator_values),
        "probe_termination_reward_unique_count": len(probe_termination_values),
        "probe_signal_variation_present": (
            len(probe_reward_values) > 1
            and len(probe_validator_values) > 1
            and len(probe_termination_values) > 1
        ),
        "limitation": "Probes call the validator and rule_reward functions only; they are not model generations or quality evidence.",
    }


def _source_chosen_summary(manifest_rows: list[dict]) -> dict[str, object]:
    categories = Counter(str(row.get("category", "<missing>")) for row in manifest_rows)
    families = Counter(str(row.get("family", "<missing>")) for row in manifest_rows)
    difficulties = Counter(str(row.get("difficulty", "<missing>")) for row in manifest_rows)
    validator_reasons: Counter[str] = Counter()
    rewards: list[float] = []
    component_rows: list[dict] = []
    for row in manifest_rows:
        reward, components, passed, reason = _score(row, str(row.get("chosen", "")))
        validator_reasons[reason] += 1
        if math.isfinite(reward):
            rewards.append(reward)
        component_rows.append({"components": components})
    return {
        "sample_count": len(manifest_rows),
        "category_counts": dict(sorted(categories.items())),
        "family_count": len(families),
        "difficulty_counts": dict(sorted(difficulties.items())),
        "validator_pass_count": sum(
            _validate(row, str(row.get("chosen", "")))[0] for row in manifest_rows
        ),
        "validator_failure_reasons": dict(validator_reasons),
        "reward_unique_count": len(set(rewards)),
        "reward_mean": statistics.fmean(rewards) if rewards else None,
        "component_nonzero_keys": sorted({
            component
            for row in component_rows
            for component in COMPONENT_KEYS
            if abs(float(row["components"].get(component, 0.0))) > 1e-12
        }),
        "component_summary": {
            component: _component_summary(component_rows, component) for component in COMPONENT_KEYS
        },
    }


def _sample_summary(
    sample_files: list[Path],
    manifest_by_id: dict[str, dict],
    manifest_categories: set[str],
    manifest_families: set[str],
    max_gen_len: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict] = []
    parse_errors: list[dict[str, object]] = []
    input_manifest: list[dict[str, object]] = []
    for path in sample_files:
        input_manifest.append({"path": path.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
        parsed, errors = _read_jsonl(path)
        rows.extend(parsed)
        parse_errors.extend(errors)

    categories = Counter()
    families = Counter()
    prompt_ids: set[str] = set()
    missing_manifest: list[str] = []
    validator_pass = 0
    validator_outcomes: list[bool] = []
    validator_reasons: Counter[str] = Counter()
    persisted_reward_mismatches = 0
    persisted_component_mismatches = 0
    invalid_rows = 0
    termination_reasons = Counter()
    rewards: list[float] = []
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        if not finite_tree(row):
            invalid_rows += 1
        prompt_id = str(row.get("prompt_id", "<missing>"))
        prompt_ids.add(prompt_id)
        manifest = manifest_by_id.get(prompt_id)
        if manifest is None:
            missing_manifest.append(prompt_id)
            continue
        category = str(manifest.get("category", row.get("category", "<missing>")))
        categories[category] += 1
        families[str(manifest.get("family", "<missing>"))] += 1
        response = str(row.get("response", ""))
        reward, components, passed, reason = _score(manifest, response)
        if passed:
            validator_pass += 1
        validator_outcomes.append(bool(passed))
        validator_reasons[reason] += 1
        persisted_reward = _float(row.get("rule_reward", row.get("reward")))
        if persisted_reward is None or not math.isfinite(reward) or abs(persisted_reward - reward) > 1e-9:
            persisted_reward_mismatches += 1
        persisted_components = row.get("components") or {}
        for component in COMPONENT_KEYS:
            persisted = _float(persisted_components.get(component))
            if persisted is None or abs(persisted - components[component]) > 1e-9:
                persisted_component_mismatches += 1
                break
        if math.isfinite(reward):
            rewards.append(reward)
            groups[(row.get("step"), prompt_id, row.get("micro_index", "legacy"))].append(reward)
        termination_reasons[str(row.get("termination_reason", "missing"))] += 1

    eligible_groups = [values for values in groups.values() if len(values) >= 2]
    spread_groups = [values for values in eligible_groups if max(values) - min(values) > 1e-12]
    current_categories = set(categories)
    current_families = set(families)
    component_rows = [
        {"components": row.get("components") or {}}
        for path in sample_files
        for row in _read_jsonl(path)[0]
        if str(row.get("prompt_id", "<missing>")) in manifest_by_id
    ]
    generated_tokens = [
        int(row["generated_tokens"])
        for row in rows
        if isinstance(row.get("generated_tokens"), int) and not isinstance(row.get("generated_tokens"), bool)
    ]
    termination_values = [
        _float((row.get("components") or {}).get("termination_reward"))
        for row in rows
        if _float((row.get("components") or {}).get("termination_reward")) is not None
    ]
    summary = {
        "sample_file_count": len(sample_files),
        "sample_count": len(rows),
        "category_counts": dict(sorted(categories.items())),
        "category_coverage_count": len(current_categories),
        "category_coverage_total": len(manifest_categories),
        "category_coverage_rate": len(current_categories & manifest_categories) / len(manifest_categories) if manifest_categories else 0.0,
        "unobserved_categories": sorted(manifest_categories - current_categories),
        "family_coverage_count": len(current_families),
        "family_coverage_total": len(manifest_families),
        "family_coverage_rate": len(current_families & manifest_families) / len(manifest_families) if manifest_families else 0.0,
        "prompt_coverage_count": len(prompt_ids & set(manifest_by_id)),
        "prompt_coverage_total": len(manifest_by_id),
        "prompt_coverage_rate": len(prompt_ids & set(manifest_by_id)) / len(manifest_by_id) if manifest_by_id else 0.0,
        "missing_manifest_count": len(missing_manifest),
        "missing_manifest_prompt_ids": sorted(set(missing_manifest)),
        "validator_pass_count": validator_pass,
        "validator_pass_rate": validator_pass / len(rows) if rows else 0.0,
        "validator_outcome_count": len(set(validator_outcomes)),
        "validator_failure_reasons": dict(sorted(validator_reasons.items())),
        "persisted_reward_mismatch_count": persisted_reward_mismatches,
        "persisted_component_mismatch_count": persisted_component_mismatches,
        "reward_unique_count": len(set(rewards)),
        "reward_mean": statistics.fmean(rewards) if rewards else None,
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "component_summary": {
            component: _component_summary(component_rows, component) for component in COMPONENT_KEYS
        },
        "component_nonzero_keys": sorted({
            component
            for component in COMPONENT_KEYS
            if any(abs(_float((row.get("components") or {}).get(component)) or 0.0) > 1e-12 for row in rows)
        }),
        "termination_reward_unique_count": len(set(termination_values)),
        "termination_reward_values": sorted(set(termination_values)),
        "termination_reason_counts": dict(sorted(termination_reasons.items())),
        "termination_reason_count": len(termination_reasons),
        "natural_end_rate": sum(bool(row.get("finished_naturally")) for row in rows) / len(rows) if rows else 0.0,
        "max_length_hit_rate": sum(bool(row.get("max_length_hit")) for row in rows) / len(rows) if rows else 0.0,
        "empty_response_rate": sum(bool(row.get("empty_response")) for row in rows) / len(rows) if rows else 0.0,
        "generated_tokens_min": min(generated_tokens) if generated_tokens else None,
        "generated_tokens_max": max(generated_tokens) if generated_tokens else None,
        "generated_tokens_over_limit_count": sum(value > max_gen_len for value in generated_tokens),
        "configured_max_gen_len": max_gen_len,
        "group_count": len(groups),
        "eligible_group_count": len(eligible_groups),
        "collapsed_group_count": sum(max(values) - min(values) <= 1e-12 for values in eligible_groups),
        "collapsed_group_rate": (
            sum(max(values) - min(values) <= 1e-12 for values in eligible_groups) / len(eligible_groups)
            if eligible_groups else None
        ),
        "nonzero_reward_spread_group_count": len(spread_groups),
        "invalid_or_nonfinite_rows": invalid_rows,
    }
    return summary, parse_errors, input_manifest


def audit_paths(
    manifest_path: Path,
    sample_roots: list[Path],
    output_dir: Path,
    max_gen_len: int = 16,
) -> dict[str, object]:
    ensure_empty_output_dir(output_dir)
    manifest_path = _path(manifest_path)
    sample_roots = [_path(path) for path in sample_roots]
    manifest_rows, manifest_errors = _read_jsonl(manifest_path)
    manifest_by_id = {str(row.get("id")): row for row in manifest_rows if row.get("id") is not None}
    sample_files = collect_sample_files(sample_roots)
    manifest_categories = {str(row.get("category", "<missing>")) for row in manifest_rows}
    manifest_families = {str(row.get("family", "<missing>")) for row in manifest_rows}
    source_summary = _source_chosen_summary(manifest_rows)
    sample_summary, sample_errors, input_manifest = _sample_summary(
        sample_files, manifest_by_id, manifest_categories, manifest_families, max_gen_len
    )
    probe = _probe_summary(manifest_rows)
    parse_errors = manifest_errors + sample_errors
    warnings: list[str] = []
    if sample_summary["category_coverage_rate"] < 1.0:
        warnings.append("CURRENT_CATEGORY_COVERAGE_INCOMPLETE")
    if sample_summary["validator_pass_count"] == 0:
        warnings.append("CURRENT_VALIDATOR_PASS_NOT_OBSERVED")
    if sample_summary["termination_reward_unique_count"] <= 1:
        warnings.append("CURRENT_TERMINATION_REWARD_COLLAPSED")
    if sample_summary["termination_reason_count"] > 1 and sample_summary["termination_reward_unique_count"] == 1:
        warnings.append("TERMINATION_REWARD_NOT_SENSITIVE_TO_EOS_REASON")
    if probe["probe_signal_variation_present"]:
        warnings.append("RULE_REWARD_PROBE_VARIATION_PRESENT")
    if source_summary["validator_pass_count"] == source_summary["sample_count"]:
        warnings.append("SOURCE_CHOSEN_VALIDATOR_ALL_PASS")
    if sample_summary["nonzero_reward_spread_group_count"] == 0:
        warnings.append("NO_CURRENT_NONZERO_GROUP_REWARD_SPREAD")
    if parse_errors or sample_summary["missing_manifest_count"] or sample_summary["persisted_reward_mismatch_count"] or sample_summary["persisted_component_mismatch_count"]:
        status = "TELEMETRY_INCOMPLETE"
    elif sample_summary["category_coverage_rate"] < 1.0 or sample_summary["validator_pass_count"] == 0 or sample_summary["termination_reward_unique_count"] <= 1:
        status = "CURRENT_GENERATION_SIGNAL_COVERAGE_INSUFFICIENT_DIAGNOSTIC"
    else:
        status = "CURRENT_GENERATION_SIGNAL_VARIABILITY_OBSERVED_DIAGNOSTIC"
    result = {
        "status": status,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "sample_roots": [path.as_posix() for path in sample_roots],
        "max_gen_len": max_gen_len,
        "manifest_error_count": len(manifest_errors),
        "sample_file_count": len(sample_files),
        "input_manifest": [{"path": manifest_path.as_posix(), "sha256": sha256_file(manifest_path), "bytes": manifest_path.stat().st_size}] + input_manifest,
        "parse_errors": parse_errors,
        "source_chosen": source_summary,
        "current_generation": sample_summary,
        "rule_reward_probes": probe,
        "warnings": warnings,
        "diagnostic_only": True,
        "gpu_wall_seconds": 0,
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        "default_model_changed": False,
        "conclusion": "Current E027 generations do not cover the eight-category v2 input space or exercise both validator and termination reward outcomes. Deterministic probes show the rule functions are capable of variation, but probes are not model evidence. Do not infer an optimizer remedy or start formal RL from this audit.",
    }
    (output_dir / "input_manifest.json").write_text(
        json.dumps(result["input_manifest"], ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report = [
        "# Natural reward input/output coverage audit",
        "",
        f"- status: `{status}`",
        f"- manifest rows: {source_summary['sample_count']}; categories: {len(manifest_categories)}; families: {source_summary['family_count']}",
        f"- current samples: {sample_summary['sample_count']}; category coverage: {sample_summary['category_coverage_count']}/{sample_summary['category_coverage_total']}",
        f"- current validator pass: {sample_summary['validator_pass_count']}/{sample_summary['sample_count']}",
        f"- current termination reward values: {sample_summary['termination_reward_values']}",
        f"- termination reasons: {sample_summary['termination_reason_counts']}",
        f"- deterministic rule probes show signal variation: {probe['probe_signal_variation_present']}",
        "",
        "Warnings: " + (", ".join(warnings) if warnings else "none"),
        "",
        "Diagnostic only; probes do not represent model generations and no model or optimizer decision is made.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-root", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-gen-len", type=int, default=16)
    args = parser.parse_args()
    result = audit_paths(args.manifest, args.sample_root, args.output_dir, args.max_gen_len)
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] == "TELEMETRY_INCOMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
