"""Audit validator and output-quality coverage for an isolated balanced RL smoke."""

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

from dataset.alignment_v2.validators import repeat_3gram_ratio
from evaluation.audit_natural_reward_coverage import (
    COMPONENT_KEYS,
    _float,
    _path,
    _read_jsonl,
    _score,
    _source_chosen_summary,
    collect_sample_files,
    sha256_file,
)
from evaluation.audit_corrected_kl_gate import ensure_empty_output_dir


VALID_TERMINATION_REASONS = {"eos", "max_new_tokens", "no_eos_short_generation"}


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _numeric_stats(values: list[float]) -> dict[str, object]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _component_summary(rows: list[dict[str, object]], component: str) -> dict[str, object]:
    values = [
        float((row.get("components") or {}).get(component, 0.0))
        for row in rows
        if _float((row.get("components") or {}).get(component)) is not None
    ]
    return {
        "observed_count": len(values),
        "missing_count": len(rows) - len(values),
        "coverage_rate": _rate(len(values), len(rows)),
        "nonzero_count": sum(abs(value) > 1e-12 for value in values),
        "nonzero_rate": _rate(sum(abs(value) > 1e-12 for value in values), len(values)),
        "unique_count": len(set(values)),
        "mean": statistics.fmean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _telemetry_consistency(row: dict[str, object], response: str) -> tuple[bool, bool, str]:
    reason = str(row.get("termination_reason", "missing"))
    eos_seen = bool(row.get("eos_seen"))
    finished = bool(row.get("finished_naturally"))
    max_length = bool(row.get("max_length_hit"))
    if reason == "eos":
        termination_ok = eos_seen and finished and not max_length
    elif reason == "max_new_tokens":
        termination_ok = not eos_seen and not finished and max_length
    elif reason == "no_eos_short_generation":
        termination_ok = not eos_seen and not finished and not max_length
    else:
        termination_ok = False
    empty_ok = bool(row.get("empty_response")) == (not response.strip())
    return termination_ok, empty_ok, reason


def _sample_diagnostic(row: dict[str, object], manifest: dict[str, object]) -> dict[str, object]:
    response = str(row.get("response", ""))
    reward, components, passed, failure_reason = _score(manifest, response)
    persisted_reward = _float(row.get("rule_reward", row.get("reward")))
    persisted_components = row.get("components") or {}
    reward_match = persisted_reward is not None and abs(persisted_reward - reward) <= 1e-9
    component_match = all(
        _float(persisted_components.get(component)) is not None
        and abs(float(persisted_components[component]) - components[component]) <= 1e-9
        for component in COMPONENT_KEYS
    )
    termination_ok, empty_ok, termination_reason = _telemetry_consistency(row, response)
    generated_tokens = row.get("generated_tokens")
    generated_tokens_value = (
        int(generated_tokens)
        if isinstance(generated_tokens, int) and not isinstance(generated_tokens, bool)
        else None
    )
    return {
        "sample_key": row.get("sample_key"),
        "step": row.get("step"),
        "micro_index": row.get("micro_index"),
        "candidate_index": row.get("candidate_index"),
        "prompt_id": manifest.get("id"),
        "category": manifest.get("category"),
        "family": manifest.get("family"),
        "difficulty": manifest.get("difficulty"),
        "response": response,
        "response_chars": len(response),
        "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        "generated_tokens": generated_tokens_value,
        "termination_reason": termination_reason,
        "eos_seen": bool(row.get("eos_seen")),
        "finished_naturally": bool(row.get("finished_naturally")),
        "max_length_hit": bool(row.get("max_length_hit")),
        "empty_response": bool(row.get("empty_response")),
        "repeat_3gram_ratio": repeat_3gram_ratio(response),
        "rule_reward": reward,
        "validator_pass": bool(passed),
        "validator_failure_reason": "" if passed else failure_reason,
        "components": components,
        "persisted_reward_match": reward_match,
        "persisted_component_match": component_match,
        "termination_telemetry_consistent": termination_ok,
        "empty_telemetry_consistent": empty_ok,
    }


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    token_values = [int(row["generated_tokens"]) for row in rows if row.get("generated_tokens") is not None]
    char_values = [int(row["response_chars"]) for row in rows]
    repeat_values = [float(row["repeat_3gram_ratio"]) for row in rows]
    rewards = [float(row["rule_reward"]) for row in rows]
    failure_reasons = Counter(
        str(row["validator_failure_reason"]) for row in rows if not row["validator_pass"]
    )
    termination_reasons = Counter(str(row["termination_reason"]) for row in rows)
    return {
        "sample_count": len(rows),
        "validator_pass_count": sum(bool(row["validator_pass"]) for row in rows),
        "validator_pass_rate": _rate(sum(bool(row["validator_pass"]) for row in rows), len(rows)),
        "validator_failure_reasons": dict(sorted(failure_reasons.items())),
        "termination_reason_counts": dict(sorted(termination_reasons.items())),
        "unknown_termination_count": sum(
            str(row["termination_reason"]) not in VALID_TERMINATION_REASONS for row in rows
        ),
        "natural_end_count": sum(bool(row["finished_naturally"]) for row in rows),
        "natural_end_rate": _rate(sum(bool(row["finished_naturally"]) for row in rows), len(rows)),
        "max_length_hit_count": sum(bool(row["max_length_hit"]) for row in rows),
        "max_length_hit_rate": _rate(sum(bool(row["max_length_hit"]) for row in rows), len(rows)),
        "empty_response_count": sum(bool(row["empty_response"]) for row in rows),
        "empty_response_rate": _rate(sum(bool(row["empty_response"]) for row in rows), len(rows)),
        "generated_tokens": _numeric_stats([float(value) for value in token_values]),
        "response_chars": _numeric_stats([float(value) for value in char_values]),
        "repeat_3gram_ratio": _numeric_stats(repeat_values),
        "reward": {
            **_numeric_stats(rewards),
            "unique_count": len(set(rewards)),
        },
        "component_summary": {
            component: _component_summary(rows, component) for component in COMPONENT_KEYS
        },
    }


def _group_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, object], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("step"), row.get("prompt_id"))].append(row)
    result = []
    for (step, prompt_id), group in sorted(groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        rewards = [float(row["rule_reward"]) for row in group]
        result.append({
            "step": step,
            "prompt_id": prompt_id,
            "category": group[0].get("category"),
            "sample_count": len(group),
            "validator_pass_count": sum(bool(row["validator_pass"]) for row in group),
            "reward_values": sorted(set(rewards)),
            "reward_spread": max(rewards) - min(rewards) if rewards else None,
            "natural_end_rate": _rate(sum(bool(row["finished_naturally"]) for row in group), len(group)),
            "max_length_hit_rate": _rate(sum(bool(row["max_length_hit"]) for row in group), len(group)),
        })
    return result


def _tail(rows: list[dict[str, object]], field: str, limit: int = 5) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: float(row.get(field) or 0.0), reverse=True)
    return [
        {
            "sample_key": row.get("sample_key"),
            "step": row.get("step"),
            "prompt_id": row.get("prompt_id"),
            "category": row.get("category"),
            "value": row.get(field),
            "generated_tokens": row.get("generated_tokens"),
            "termination_reason": row.get("termination_reason"),
            "validator_pass": row.get("validator_pass"),
            "validator_failure_reason": row.get("validator_failure_reason"),
        }
        for row in ordered[:limit]
    ]


def audit_paths(
    manifest_path: Path,
    sample_root: Path,
    output_dir: Path,
    max_gen_len: int = 16,
) -> dict[str, object]:
    ensure_empty_output_dir(output_dir)
    manifest_path = _path(manifest_path)
    sample_root = _path(sample_root)
    manifest_rows, manifest_errors = _read_jsonl(manifest_path)
    manifest_by_id = {str(row.get("id")): row for row in manifest_rows if row.get("id") is not None}
    sample_files = collect_sample_files([sample_root])
    rows: list[dict[str, object]] = []
    parse_errors = list(manifest_errors)
    input_manifest = [{"path": manifest_path.as_posix(), "sha256": sha256_file(manifest_path), "bytes": manifest_path.stat().st_size}]
    for path in sample_files:
        input_manifest.append({"path": path.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
        parsed, errors = _read_jsonl(path)
        rows.extend(parsed)
        parse_errors.extend(errors)

    diagnostics: list[dict[str, object]] = []
    missing_manifest: list[str] = []
    for row in rows:
        prompt_id = str(row.get("prompt_id", "<missing>"))
        manifest = manifest_by_id.get(prompt_id)
        if manifest is None:
            missing_manifest.append(prompt_id)
            continue
        diagnostics.append(_sample_diagnostic(row, manifest))

    category_values = sorted({str(row.get("category")) for row in manifest_rows})
    family_values = sorted({str(row.get("family")) for row in manifest_rows})
    prompt_values = set(manifest_by_id)
    by_category = {
        category: _aggregate([row for row in diagnostics if row.get("category") == category])
        for category in category_values
    }
    by_step = {
        str(step): _aggregate([row for row in diagnostics if str(row.get("step")) == str(step)])
        for step in sorted({row.get("step") for row in diagnostics}, key=str)
    }
    by_prompt = {
        prompt_id: _aggregate([row for row in diagnostics if row.get("prompt_id") == prompt_id])
        for prompt_id in sorted({str(row.get("prompt_id")) for row in diagnostics})
    }
    replay_reward_mismatches = sum(not bool(row["persisted_reward_match"]) for row in diagnostics)
    replay_component_mismatches = sum(not bool(row["persisted_component_match"]) for row in diagnostics)
    termination_telemetry_mismatches = sum(not bool(row["termination_telemetry_consistent"]) for row in diagnostics)
    empty_telemetry_mismatches = sum(not bool(row["empty_telemetry_consistent"]) for row in diagnostics)
    validator_pass_count = sum(bool(row["validator_pass"]) for row in diagnostics)
    validator_rate = _rate(validator_pass_count, len(diagnostics))
    category_coverage = len({row.get("category") for row in diagnostics} & set(category_values))
    family_coverage = len({row.get("family") for row in diagnostics} & set(family_values))
    prompt_coverage = len({str(row.get("prompt_id")) for row in diagnostics} & prompt_values)
    unknown_termination_count = sum(
        str(row["termination_reason"]) not in VALID_TERMINATION_REASONS for row in diagnostics
    )
    failure_counts = Counter(
        str(row["validator_failure_reason"]) for row in diagnostics if not row["validator_pass"]
    )
    dominant_failure_rate = max(failure_counts.values(), default=0) / max(sum(failure_counts.values()), 1)
    warnings: list[str] = []
    if validator_rate < 0.10:
        warnings.append("VALIDATOR_SIGNAL_SPARSE")
    if category_coverage < len(category_values):
        warnings.append("CATEGORY_OUTPUT_COVERAGE_INCOMPLETE")
    if family_coverage < len(family_values) or prompt_coverage < len(prompt_values):
        warnings.append("FAMILY_OR_PROMPT_OUTPUT_COVERAGE_INCOMPLETE")
    if unknown_termination_count:
        warnings.append("UNKNOWN_TERMINATION_REASON")
    if termination_telemetry_mismatches:
        warnings.append("TERMINATION_TELEMETRY_INCONSISTENT")
    if empty_telemetry_mismatches:
        warnings.append("EMPTY_RESPONSE_TELEMETRY_INCONSISTENT")
    if replay_reward_mismatches or replay_component_mismatches:
        warnings.append("PERSISTED_REWARD_REPLAY_MISMATCH")
    if diagnostics and _rate(sum(bool(row["max_length_hit"]) for row in diagnostics), len(diagnostics)) >= 0.50:
        warnings.append("MAX_LENGTH_HIT_PREVALENT")
    if diagnostics and _rate(sum(bool(row["finished_naturally"]) for row in diagnostics), len(diagnostics)) < 0.50:
        warnings.append("NATURAL_END_RATE_LOW")
    if dominant_failure_rate >= 0.50:
        warnings.append("VALIDATOR_FAILURE_REASON_CONCENTRATED")
    elif failure_counts:
        warnings.append("VALIDATOR_FAILURE_REASON_DISTRIBUTED")
    if len({float(row["components"].get("termination_reward", 0.0)) for row in diagnostics}) > 1:
        warnings.append("TERMINATION_REWARD_VARIABILITY_OBSERVED")
    if _source_chosen_summary(manifest_rows)["validator_pass_count"] == len(manifest_rows):
        warnings.append("SOURCE_CHOSEN_VALIDATOR_ALL_PASS")

    if (
        parse_errors
        or missing_manifest
        or replay_reward_mismatches
        or replay_component_mismatches
        or termination_telemetry_mismatches
        or empty_telemetry_mismatches
        or unknown_termination_count
    ):
        status = "TELEMETRY_INCOMPLETE"
    elif category_coverage < len(category_values) or family_coverage < len(family_values) or prompt_coverage < len(prompt_values):
        status = "OUTPUT_QUALITY_COVERAGE_INCOMPLETE_DIAGNOSTIC"
    elif validator_rate < 0.10:
        status = "OUTPUT_QUALITY_SIGNAL_SPARSE_DIAGNOSTIC"
    elif dominant_failure_rate >= 0.50:
        status = "OUTPUT_QUALITY_FAILURE_CONCENTRATED_DIAGNOSTIC"
    else:
        status = "OUTPUT_QUALITY_VARIABILITY_OBSERVED_DIAGNOSTIC"

    source_summary = _source_chosen_summary(manifest_rows)
    result = {
        "status": status,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "sample_root": sample_root.as_posix(),
        "sample_file_count": len(sample_files),
        "max_gen_len": max_gen_len,
        "manifest_error_count": len(manifest_errors),
        "parse_errors": parse_errors,
        "input_manifest": input_manifest,
        "source_chosen": source_summary,
        "current_generation": {
            "sample_count": len(diagnostics),
            "category_coverage_count": category_coverage,
            "category_coverage_total": len(category_values),
            "family_coverage_count": family_coverage,
            "family_coverage_total": len(family_values),
            "prompt_coverage_count": prompt_coverage,
            "prompt_coverage_total": len(prompt_values),
            "missing_manifest_count": len(missing_manifest),
            "missing_manifest_prompt_ids": sorted(set(missing_manifest)),
            "validator_pass_count": validator_pass_count,
            "validator_pass_rate": validator_rate,
            "failure_reason_counts": dict(sorted(failure_counts.items())),
            "dominant_failure_reason_rate": dominant_failure_rate,
            "category_summary": by_category,
            "step_summary": by_step,
            "prompt_summary": by_prompt,
            "termination_reason_counts": dict(sorted(Counter(str(row["termination_reason"]) for row in diagnostics).items())),
            "reward_unique_count": len({float(row["rule_reward"]) for row in diagnostics}),
            "reward_values": sorted({float(row["rule_reward"]) for row in diagnostics}),
            "component_summary": {
                component: _component_summary(diagnostics, component) for component in COMPONENT_KEYS
            },
            "replay_reward_mismatch_count": replay_reward_mismatches,
            "replay_component_mismatch_count": replay_component_mismatches,
            "termination_telemetry_mismatch_count": termination_telemetry_mismatches,
            "empty_telemetry_mismatch_count": empty_telemetry_mismatches,
            "unknown_termination_count": unknown_termination_count,
            "group_summary": _group_summary(diagnostics),
            "quality_tails": {
                "repetition": _tail(diagnostics, "repeat_3gram_ratio"),
                "generated_tokens": _tail(diagnostics, "generated_tokens"),
                "response_chars": _tail(diagnostics, "response_chars"),
            },
        },
        "warnings": warnings,
        "diagnostic_only": True,
        "gpu_wall_seconds": 0,
        "cuda_disabled": True,
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        "default_model_changed": False,
        "conclusion": "E030 balanced outputs cover the input categories, but validator pass rate and natural-end/length behavior must be treated as output-quality diagnostics only. This audit does not justify reward, optimizer, checkpoint or default-model changes.",
    }
    (output_dir / "input_manifest.json").write_text(
        json.dumps(input_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "sample_diagnostics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in diagnostics),
        encoding="utf-8",
    )
    (output_dir / "category_summary.json").write_text(
        json.dumps(by_category, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report = [
        "# Balanced output-quality audit",
        "",
        f"- status: `{status}`",
        f"- samples: {len(diagnostics)}; category/family/prompt coverage: {category_coverage}/{len(category_values)}, {family_coverage}/{len(family_values)}, {prompt_coverage}/{len(prompt_values)}",
        f"- validator pass: {validator_pass_count}/{len(diagnostics)} ({validator_rate:.4f})",
        f"- failure reasons: {dict(sorted(failure_counts.items()))}",
        f"- termination reasons: {dict(sorted(Counter(str(row['termination_reason']) for row in diagnostics).items()))}",
        f"- replay mismatches: reward={replay_reward_mismatches}, components={replay_component_mismatches}",
        f"- quality warnings: {', '.join(warnings) if warnings else 'none'}",
        "",
        "Diagnostic only; no GPU task, reward change, optimizer change or model promotion was performed.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-gen-len", type=int, default=16)
    args = parser.parse_args()
    result = audit_paths(args.manifest, args.sample_root, args.output_dir, args.max_gen_len)
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] == "TELEMETRY_INCOMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
