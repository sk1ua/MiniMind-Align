"""Audit recurring RL prompt/category anomalies and reward components.

This audit is deliberately offline and diagnostic-only.  It consumes the
micro-batch and sample telemetry from completed runs, ranks the same prompt or
category when it recurs in the top-K KL/gradient/quality tails, and computes
descriptive reward-component associations.  It does not change checkpoint
selection, KL gates, model promotion, or any training artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.alignment_v2.validators import validate_record


DEFAULT_TOP_K = 10
REQUIRED_RUNS = (
    "grpo_control_seed42",
    "grpo_low_lr_seed42",
    "grpo_clip_half_seed42",
)
REWARD_COMPONENTS = (
    "validator_reward",
    "parse_reward",
    "field_reward",
    "item_count_reward",
    "arithmetic_reward",
    "format_reward",
    "termination_reward",
    "repetition_penalty",
)
TOP_METRICS = (
    "reference_kl_max",
    "micro_grad_norm_unscaled",
    "quality_anomaly_score",
)
CORRELATION_TARGETS = (
    "reference_kl_max",
    "micro_grad_norm_unscaled",
    "quality_anomaly_score",
    "max_length_hit_rate",
    "repetition_penalty_mean",
    "reward_mean",
)


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line, parse_constant=_reject_nonfinite)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _number(value: Any, default: float = 0.0) -> float:
    return float(value) if _finite(value) else default


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return mean(materialized) if materialized else 0.0


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _correlation(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    pairs = [
        (_number(row.get(left)), _number(row.get(right)))
        for row in rows
        if _finite(row.get(left)) and _finite(row.get(right))
    ]
    if len(pairs) < 2:
        return {"n": len(pairs), "pearson_r": None}
    left_values = [pair[0] for pair in pairs]
    right_values = [pair[1] for pair in pairs]
    left_mean = mean(left_values)
    right_mean = mean(right_values)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in pairs)
    left_var = sum((x - left_mean) ** 2 for x in left_values)
    right_var = sum((y - right_mean) ** 2 for y in right_values)
    denominator = math.sqrt(left_var * right_var)
    return {
        "n": len(pairs),
        "pearson_r": numerator / denominator if denominator else None,
    }


def _sample_key(row: dict[str, Any]) -> str | None:
    value = row.get("sample_key")
    if isinstance(value, str) and value:
        return value
    if all(_finite(row.get(name)) for name in ("step", "micro_index", "candidate_index")):
        return f"{int(row['step'])}:{int(row['micro_index'])}:{int(row['candidate_index'])}"
    return None


def _component_value(row: dict[str, Any], component: str) -> float:
    components = row.get("components")
    if not isinstance(components, dict):
        return 0.0
    return _number(components.get(component))


def _diagnostic_value(row: dict[str, Any], name: str, fallback: Any = 0.0) -> float:
    diagnostics = row.get("policy_diagnostics")
    if isinstance(diagnostics, dict) and name in diagnostics:
        return _number(diagnostics.get(name), _number(fallback))
    return _number(row.get(name), _number(fallback))


def _sample_rates(samples: list[dict[str, Any]]) -> dict[str, float]:
    count = len(samples)
    if not count:
        return {
            "validator_pass_rate": 0.0,
            "max_length_hit_rate": 0.0,
            "natural_end_rate": 0.0,
            "empty_response_rate": 0.0,
            "repetition_penalty_mean": 0.0,
            "average_tokens": 0.0,
        }
    return {
        "validator_pass_rate": sum(_component_value(row, "validator_reward") > 0 for row in samples) / count,
        "max_length_hit_rate": sum(bool(row.get("max_length_hit")) for row in samples) / count,
        "natural_end_rate": sum(bool(row.get("finished_naturally")) for row in samples) / count,
        "empty_response_rate": sum(bool(row.get("empty_response")) for row in samples) / count,
        "repetition_penalty_mean": _mean(_component_value(row, "repetition_penalty") for row in samples),
        "average_tokens": _mean(_number(row.get("generated_tokens")) for row in samples),
    }


def _sample_association(
    micro: dict[str, Any],
    sample_by_key: dict[str, dict[str, Any]],
    sample_key_owners: dict[str, tuple[int, int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    declared = micro.get("sample_keys")
    declared_keys = [item for item in declared if isinstance(item, str)] if isinstance(declared, list) else []
    duplicate_keys = sorted({key for key in declared_keys if declared_keys.count(key) > 1})
    linked = [sample_by_key[key] for key in declared_keys if key in sample_by_key]
    missing = sorted(set(declared_keys) - set(sample_by_key))
    mismatched: list[str] = []
    if _finite(micro.get("step")) and _finite(micro.get("micro_index")):
        expected_owner = (int(micro["step"]), int(micro["micro_index"]))
        for key in declared_keys:
            if key in sample_key_owners and sample_key_owners[key] != expected_owner:
                mismatched.append(key)
    association = {
        "declared_sample_count": len(declared_keys),
        "linked_sample_count": len(linked),
        "missing_sample_keys": missing,
        "duplicate_sample_keys": duplicate_keys,
        "mismatched_sample_keys": sorted(set(mismatched)),
        "sample_link_complete": bool(declared_keys)
        and not missing
        and not duplicate_keys
        and not mismatched
        and len(linked) == len(declared_keys),
    }
    return linked, association


def _normalize_micro(
    run_name: str,
    micro: dict[str, Any],
    sample_by_key: dict[str, dict[str, Any]],
    sample_key_owners: dict[str, tuple[int, int]],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not _finite(micro.get("step")) or not _finite(micro.get("micro_index")):
        errors.append("microbatch_missing_step_or_index")
        step = 0
        micro_index = 0
    else:
        step = int(micro["step"])
        micro_index = int(micro["micro_index"])
    linked, association = _sample_association(micro, sample_by_key, sample_key_owners)
    if not association["sample_link_complete"]:
        errors.append(f"sample_link_incomplete:{step}:{micro_index}")
    if not linked:
        errors.append(f"microbatch_has_no_samples:{step}:{micro_index}")
    rates = _sample_rates(linked)
    reward_components = {
        component: _mean(_component_value(sample, component) for sample in linked)
        for component in REWARD_COMPONENTS
    }
    quality_score = (
        rates["max_length_hit_rate"]
        + rates["repetition_penalty_mean"]
        + rates["empty_response_rate"]
        + (1.0 - rates["natural_end_rate"])
    )
    raw_sample_keys = micro.get("sample_keys")
    sample_keys = [key for key in raw_sample_keys if isinstance(key, str)] if isinstance(raw_sample_keys, list) else []
    row = {
        "run": run_name,
        "step": step,
        "micro_index": micro_index,
        "prompt_id": str(micro.get("prompt_id", "unknown")),
        "category": str(micro.get("category", "unknown")),
        "loss": _number(micro.get("loss")),
        "reward_mean": _number(micro.get("reward_mean"), _mean(_number(item.get("reward")) for item in linked)),
        "reward_components": reward_components,
        "reference_kl_mean": _diagnostic_value(micro, "reference_kl_mean", micro.get("kl")),
        "reference_kl_p95": _diagnostic_value(micro, "reference_kl_p95", micro.get("kl")),
        "reference_kl_max": _diagnostic_value(micro, "reference_kl_max", micro.get("kl")),
        "ratio_p95": _diagnostic_value(micro, "ratio_p95"),
        "ratio_max": _diagnostic_value(micro, "ratio_max"),
        "micro_grad_norm_scaled": _number(micro.get("micro_grad_norm_scaled")),
        "micro_grad_norm_unscaled": _number(micro.get("micro_grad_norm_unscaled")),
        "accumulated_grad_norm": _number(micro.get("accumulated_grad_norm")),
        "validator_pass_rate": rates["validator_pass_rate"],
        "max_length_hit_rate": rates["max_length_hit_rate"],
        "natural_end_rate": rates["natural_end_rate"],
        "empty_response_rate": rates["empty_response_rate"],
        "repetition_penalty_mean": rates["repetition_penalty_mean"],
        "average_tokens": rates["average_tokens"],
        "quality_anomaly_score": quality_score,
        "sample_keys": sample_keys,
        "sample_association": association,
        "top_k_metrics": [],
        "anomaly_metrics": [],
    }
    return row, errors


def _sort_top(rows: list[dict[str, Any]], metric: str, top_k: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -_number(row.get(metric)),
            int(row.get("step", 0)),
            int(row.get("micro_index", 0)),
            str(row.get("prompt_id", "")),
        ),
    )[:top_k]


def _public_top_row(row: dict[str, Any], metric: str) -> dict[str, Any]:
    return {
        "run": row["run"],
        "step": row["step"],
        "micro_index": row["micro_index"],
        "prompt_id": row["prompt_id"],
        "category": row["category"],
        "metric": metric,
        "metric_value": row[metric],
        "reference_kl_mean": row["reference_kl_mean"],
        "reference_kl_p95": row["reference_kl_p95"],
        "reference_kl_max": row["reference_kl_max"],
        "micro_grad_norm_unscaled": row["micro_grad_norm_unscaled"],
        "quality_anomaly_score": row["quality_anomaly_score"],
        "reward_mean": row["reward_mean"],
        "reward_components": row["reward_components"],
        "validator_pass_rate": row["validator_pass_rate"],
        "max_length_hit_rate": row["max_length_hit_rate"],
        "natural_end_rate": row["natural_end_rate"],
        "repetition_penalty_mean": row["repetition_penalty_mean"],
        "sample_keys": row["sample_keys"],
        "sample_association": row["sample_association"],
    }


def _group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    output: dict[str, Any] = {}
    numeric_fields = (
        "reference_kl_mean",
        "reference_kl_p95",
        "reference_kl_max",
        "micro_grad_norm_unscaled",
        "quality_anomaly_score",
        "reward_mean",
        "validator_pass_rate",
        "max_length_hit_rate",
        "natural_end_rate",
        "empty_response_rate",
        "repetition_penalty_mean",
        "average_tokens",
    )
    for group_key, group in sorted(grouped.items()):
        component_means = {
            component: _mean(row["reward_components"][component] for row in group)
            for component in REWARD_COMPONENTS
        }
        top_rows = [row for row in group if row["anomaly_metrics"]]
        top_conditions = sorted({row["run"] for row in top_rows})
        output[group_key] = {
            "key": group_key,
            "microbatch_count": len(group),
            "sample_count": sum(row["sample_association"]["linked_sample_count"] for row in group),
            "conditions_present": sorted({row["run"] for row in group}),
            "within_run_occurrence_count": {
                run: sum(row["run"] == run for row in group)
                for run in sorted({row["run"] for row in group})
            },
            "within_run_max_occurrence": max(
                (sum(row["run"] == run for row in group) for run in {row["run"] for row in group}),
                default=0,
            ),
            "cross_condition_condition_count": len({row["run"] for row in group}),
            "anomaly_condition_count": len(top_conditions),
            "anomaly_conditions": top_conditions,
            "top_k_hit_count": sum(bool(row["top_k_metrics"]) for row in group),
            "top_k_hit_conditions": sorted({row["run"] for row in group if row["top_k_metrics"]}),
            "anomaly_metric_counts": {
                metric: sum(metric in row["anomaly_metrics"] for row in group) for metric in TOP_METRICS
            },
            "sample_link_complete": all(row["sample_association"]["sample_link_complete"] for row in group),
            "metrics": {
                field: {
                    "mean": _mean(row[field] for row in group),
                    "p95": _percentile((row[field] for row in group), 0.95),
                    "max": max(row[field] for row in group),
                }
                for field in numeric_fields
            },
            "reward_components": component_means,
            "quality_telemetry_status_counts": {
                status: sum(row.get("quality_telemetry_status") == status for row in group)
                for status in sorted({row.get("quality_telemetry_status", "UNKNOWN") for row in group})
            },
        }
    return output


def _step_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["step"])].append(row)
    output: list[dict[str, Any]] = []
    for step, group in sorted(grouped.items()):
        output.append(
            {
                "step": step,
                "microbatch_count": len(group),
                "categories": sorted({row["category"] for row in group}),
                "prompt_ids": sorted({row["prompt_id"] for row in group}),
                "reward_mean": _mean(row["reward_mean"] for row in group),
                "reference_kl_mean": _mean(row["reference_kl_mean"] for row in group),
                "reference_kl_p95": _percentile((row["reference_kl_p95"] for row in group), 0.95),
                "reference_kl_max": max(row["reference_kl_max"] for row in group),
                "micro_grad_norm_unscaled_max": max(row["micro_grad_norm_unscaled"] for row in group),
                "validator_pass_rate": _mean(row["validator_pass_rate"] for row in group),
                "max_length_hit_rate": _mean(row["max_length_hit_rate"] for row in group),
                "natural_end_rate": _mean(row["natural_end_rate"] for row in group),
                "repetition_penalty_mean": _mean(row["repetition_penalty_mean"] for row in group),
                "quality_anomaly_score": _mean(row["quality_anomaly_score"] for row in group),
                "anomaly_microbatch_count": sum(bool(row["anomaly_metrics"]) for row in group),
            }
        )
    return output


def analyze_run_records(
    run_name: str,
    micro_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    sample_by_key: dict[str, dict[str, Any]] = {}
    sample_key_owners: dict[str, tuple[int, int]] = {}
    errors: list[str] = []
    for sample in sample_rows:
        key = _sample_key(sample)
        if key is None:
            errors.append("sample_missing_key")
            continue
        if key in sample_by_key:
            errors.append(f"duplicate_sample_key:{key}")
            continue
        sample_by_key[key] = sample
        if _finite(sample.get("step")) and _finite(sample.get("micro_index")):
            sample_key_owners[key] = (int(sample["step"]), int(sample["micro_index"]))
        else:
            errors.append(f"sample_missing_step_or_index:{key}")
    normalized: list[dict[str, Any]] = []
    for micro in micro_rows:
        row, row_errors = _normalize_micro(run_name, micro, sample_by_key, sample_key_owners)
        normalized.append(row)
        errors.extend(row_errors)
    if not normalized:
        errors.append("no_microbatch_rows")
    top_k_rows: dict[str, list[dict[str, Any]]] = {}
    for metric in TOP_METRICS:
        selected = _sort_top(normalized, metric, top_k)
        top_k_rows[metric] = selected
        for row in selected:
            row["top_k_metrics"].append(metric)
    for row in normalized:
        row["top_k_metrics"] = sorted(set(row["top_k_metrics"]))
        row["anomaly_metrics"] = list(row["top_k_metrics"])
    top_rows = {
        metric: [_public_top_row(row, metric) for row in rows]
        for metric, rows in top_k_rows.items()
    }
    return {
        "run": run_name,
        "status": "AUDIT_INCOMPLETE" if errors else "COMPLETE_DIAGNOSTIC",
        "microbatch_count": len(normalized),
        "sample_count": len(sample_rows),
        "errors": sorted(set(errors)),
        "linkage": {
            "sample_key_count": len(sample_by_key),
            "microbatch_with_complete_links": sum(
                row["sample_association"]["sample_link_complete"] for row in normalized
            ),
            "microbatch_count": len(normalized),
            "all_links_complete": bool(normalized)
            and all(row["sample_association"]["sample_link_complete"] for row in normalized),
        },
        "top_k": top_rows,
        "step_summary": _step_summary(normalized),
        "category_summary": _group_summary(normalized, "category"),
        "prompt_summary": _group_summary(normalized, "prompt_id"),
        "reward_component_correlations": _component_correlations(normalized),
        "_rows": normalized,
    }


def _component_correlations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for component in REWARD_COMPONENTS:
        component_rows = [
            {
                "component_value": row["reward_components"][component],
                **{target: row[target] for target in CORRELATION_TARGETS},
            }
            for row in rows
        ]
        output[component] = {
            "mean": _mean(row["component_value"] for row in component_rows),
            "nonzero_rate": _mean(row["component_value"] != 0 for row in component_rows),
            "correlations": {
                target: _correlation(component_rows, "component_value", target) for target in CORRELATION_TARGETS
            },
        }
    return output


def _merge_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for report in reports for row in report.get("_rows", [])]


def _selected_metrics(run_dir: Path) -> dict[str, Any]:
    selection = read_json(run_dir / "selection.json")
    selected_step = selection.get("selected_step")
    for checkpoint in selection.get("checkpoints", []):
        if checkpoint.get("step") == selected_step and isinstance(checkpoint.get("metrics"), dict):
            return dict(checkpoint["metrics"])
    return {}


def _validation_history(run_dir: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(run_dir / "validation_history.jsonl")
    return [
        {
            "step": row.get("step"),
            "metrics": row.get("metrics", {}),
        }
        for row in rows
        if isinstance(row.get("metrics"), dict)
    ]


def audit_run_directory(run_dir: Path, *, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    required = (
        "microbatch_summaries.jsonl",
        "samples.jsonl",
        "step_summaries.jsonl",
        "validation_history.jsonl",
        "selection.json",
        "baseline_validation.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        return {
            "run": run_dir.name,
            "status": "AUDIT_INCOMPLETE",
            "microbatch_count": 0,
            "sample_count": 0,
            "errors": [f"missing_file:{name}" for name in missing],
            "linkage": {"all_links_complete": False},
            "top_k": {},
            "step_summary": [],
            "category_summary": {},
            "prompt_summary": {},
            "reward_component_correlations": {},
            "selected_metrics": {},
            "baseline_metrics": {},
            "validation_history": [],
            "_rows": [],
        }
    try:
        # Parse the step file as part of completeness checking even though
        # micro-batch rows are the primary source for this audit.
        read_jsonl(run_dir / "step_summaries.jsonl")
        report = analyze_run_records(
            run_dir.name,
            read_jsonl(run_dir / "microbatch_summaries.jsonl"),
            read_jsonl(run_dir / "samples.jsonl"),
            top_k=top_k,
        )
        baseline = read_json(run_dir / "baseline_validation.json")
        report["selected_metrics"] = _selected_metrics(run_dir)
        report["baseline_metrics"] = dict(baseline.get("metrics", {}))
        report["validation_history"] = _validation_history(run_dir)
        return report
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "run": run_dir.name,
            "status": "AUDIT_INCOMPLETE",
            "microbatch_count": 0,
            "sample_count": 0,
            "errors": [f"read_or_parse_error:{error}"],
            "linkage": {"all_links_complete": False},
            "top_k": {},
            "step_summary": [],
            "category_summary": {},
            "prompt_summary": {},
            "reward_component_correlations": {},
            "selected_metrics": {},
            "baseline_metrics": {},
            "validation_history": [],
            "_rows": [],
        }


def discover_runs(experiment_root: Path, run_names: Iterable[str] = REQUIRED_RUNS) -> list[Path]:
    paths: list[Path] = []
    for run_name in run_names:
        path = experiment_root / run_name
        if not path.is_dir():
            raise ValueError(f"required run does not exist: {path}")
        paths.append(path)
    return paths


def _public_run_report(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "_rows"}


def _global_component_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        component: {
            "mean": _mean(row["reward_components"][component] for row in rows),
            "nonzero_rate": _mean(row["reward_components"][component] != 0 for row in rows),
            "correlations": {
                target: _correlation(
                    [
                        {
                            "component_value": row["reward_components"][component],
                            target: row[target],
                        }
                        for row in rows
                    ],
                    "component_value",
                    target,
                )
                for target in CORRELATION_TARGETS
            },
        }
        for component in REWARD_COMPONENTS
    }


def _recurring_groups(groups: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        group
        for group in groups.values()
        if group.get("anomaly_condition_count", 0) >= 2
        and group.get("top_k_hit_count", 0) > 0
    ]


def _markdown(summary: dict[str, Any], reports: list[dict[str, Any]]) -> str:
    lines = [
        "# RL Prompt and Reward-Component Audit",
        "",
        "本报告读取已完成的 update-scale telemetry，按 step、micro-batch、category 和 prompt 排名异常，并计算 reward component 的描述性相关性。该审计只提供诊断证据，不证明因果，不改变 checkpoint 选择、KL 门禁或默认模型。",
        "",
        f"- source experiment root: `{summary['source_experiment_root']}`",
        f"- audit output root: `{summary['experiment_root']}`",
        f"- status: `{summary['status']}`",
        f"- run count: {summary['run_count']}",
        f"- GPU wall time: {summary['gpu_wall_seconds']} seconds (offline audit)",
        "",
        "## Answer first",
        "",
        f"- recurring prompt count: {len(summary['recurring_prompts'])}",
        f"- recurring category count: {len(summary['recurring_categories'])}",
        f"- model gate: `{summary['model_change_gate']}`",
        "",
        "## Run coverage",
        "",
        "| run | micro-batches | samples | linkage | status |",
        "|---|---:|---:|---|---|",
    ]
    for report in reports:
        lines.append(
            f"| {report['run']} | {report['microbatch_count']} | {report['sample_count']} | "
            f"{report.get('linkage', {}).get('all_links_complete', False)} | {report['status']} |"
        )
    lines.extend(
        [
            "",
            "## Recurrence rule",
            "",
            "A prompt/category is called recurring only when it appears in the union of the top-K KL-max, unscaled-gradient, or quality-anomaly rows in at least two distinct conditions. This is a deterministic localization heuristic; it is not a causal test.",
            "",
            "## Recurring categories",
            "",
            "| category | conditions | top-K hits | anomaly metrics |",
            "|---|---:|---:|---|",
        ]
    )
    for item in summary["recurring_categories"]:
        lines.append(
            f"| {item['key']} | {item['anomaly_condition_count']} | {item['top_k_hit_count']} | "
            f"{item['anomaly_metric_counts']} |"
        )
    if not summary["recurring_categories"]:
        lines.append("| none | 0 | 0 | none |")
    lines.extend(
        [
            "",
            "## Recurring prompts",
            "",
            "| prompt | conditions | top-K hits | category |",
            "|---|---:|---:|---|",
        ]
    )
    for item in summary["recurring_prompts"][:20]:
        lines.append(
            f"| {item['key']} | {item['anomaly_condition_count']} | {item['top_k_hit_count']} | "
            f"{item.get('categories', ['unknown'])[0]} |"
        )
    if not summary["recurring_prompts"]:
        lines.append("| none | 0 | 0 | none |")
    lines.extend(
        [
            "",
            "## Reward-component interpretation",
            "",
            "Correlations below are descriptive Pearson coefficients over micro-batch aggregates. They are sensitive to the small, fixed run set and do not establish that a component caused a KL, gradient, truncation, or repetition spike.",
            "",
            "| component | mean | nonzero rate | r with KL max | r with grad | r with quality |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for component, item in summary["reward_component_summary"].items():
        correlations = item["correlations"]
        values = [correlations[target]["pearson_r"] for target in ("reference_kl_max", "micro_grad_norm_unscaled", "quality_anomaly_score")]
        formatted = ["null" if value is None else f"{value:.4f}" for value in values]
        lines.append(
            f"| {component} | {item['mean']:.6f} | {item['nonzero_rate']:.4f} | "
            f"{formatted[0]} | {formatted[1]} | {formatted[2]} |"
        )
    lines.extend(
        [
            "",
            "## Limits and next decision",
            "",
            "- Prompt/category recurrence means the same identifier is repeatedly present in an anomaly tail; it does not isolate token-level behavior or prove a data defect.",
            "- Reward components are measured on the generated samples and aggregated into micro-batches; the audit does not replay logits or gradients.",
            "- The current result remains diagnostic-only. Do not expand to CISPO or three-seed RL and do not change the default model until a separate controlled intervention and the existing promotion gate are satisfied.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(
    source_root: Path,
    output_dir: Path,
    *,
    run_names: Iterable[str] = REQUIRED_RUNS,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = [audit_run_directory(path, top_k=top_k) for path in discover_runs(source_root, run_names)]
    all_rows = _merge_rows(reports)
    category_groups = _group_summary(all_rows, "category") if all_rows else {}
    prompt_groups = _group_summary(all_rows, "prompt_id") if all_rows else {}
    for item in prompt_groups.values():
        matching = [row for row in all_rows if row["prompt_id"] == item["key"]]
        item["categories"] = sorted({row["category"] for row in matching})
    recurring_prompts = sorted(
        _recurring_groups(prompt_groups),
        key=lambda item: (-item["anomaly_condition_count"], -item["top_k_hit_count"], item["key"]),
    )
    recurring_categories = sorted(
        _recurring_groups(category_groups),
        key=lambda item: (-item["anomaly_condition_count"], -item["top_k_hit_count"], item["key"]),
    )
    incomplete = any(report["status"] == "AUDIT_INCOMPLETE" for report in reports)
    if incomplete:
        status = "AUDIT_INCOMPLETE"
    elif recurring_prompts:
        status = "RECURRING_PROMPT_DIAGNOSTIC"
    elif recurring_categories:
        status = "RECURRING_CATEGORY_DIAGNOSTIC"
    else:
        status = "NO_RECURRING_SOURCE_DIAGNOSTIC"
    summary = {
        "status": status,
        "task_ids": ["MM-E014", "MM-F022"],
        "experiment_root": str(output_dir),
        "source_experiment_root": str(source_root),
        "run_count": len(reports),
        "runs": [
            {
                "run": report["run"],
                "status": report["status"],
                "microbatch_count": report["microbatch_count"],
                "sample_count": report["sample_count"],
                "linkage_complete": report.get("linkage", {}).get("all_links_complete", False),
                "errors": report.get("errors", []),
            }
            for report in reports
        ],
        "top_k": top_k,
        "recurring_prompt_count": len(recurring_prompts),
        "recurring_category_count": len(recurring_categories),
        "recurring_prompts": recurring_prompts,
        "recurring_categories": recurring_categories,
        "all_json_finite": True,
        "diagnostic_only": True,
        "gpu_wall_seconds": 0,
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        "default_model_changed": False,
        "limits": [
            "Top-K recurrence is a deterministic localization heuristic, not causal evidence.",
            "Reward-component correlations are descriptive and are not a checkpoint-selection criterion.",
            "The audit uses persisted aggregate telemetry and sample metadata; raw logits, gradients, and responses are not copied into audit outputs.",
        ],
    }
    component_summary = _global_component_summary(all_rows)
    summary["reward_component_summary"] = component_summary
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "run_reports.jsonl").open("w", encoding="utf-8") as handle:
        for report in reports:
            handle.write(json.dumps(_public_run_report(report), ensure_ascii=False) + "\n")
    with (output_dir / "prompt_summary.jsonl").open("w", encoding="utf-8") as handle:
        for item in sorted(prompt_groups.values(), key=lambda value: value["key"]):
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (output_dir / "category_summary.json").write_text(
        json.dumps(category_groups, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "reward_component_summary.json").write_text(
        json.dumps(component_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_markdown(summary, reports), encoding="utf-8")
    return summary


V2_STRUCTURAL_METRICS = (
    "reference_kl_max",
    "micro_grad_norm_unscaled",
)


def _quality_telemetry_status(
    micro: dict[str, Any],
    linked_samples: list[dict[str, Any]],
) -> str:
    """Return the trust state of generation-end quality fields.

    The pre-E015 artifacts infer truncation from a padded batch width and do
    not contain an explicit termination reason.  They remain readable, but
    their quality-tail fields must not be used as corrected evidence.
    """
    if not isinstance(micro.get("termination_reason_counts"), dict):
        return "LEGACY_QUALITY_TELEMETRY_UNTRUSTED"
    if not linked_samples or any("termination_reason" not in sample for sample in linked_samples):
        return "LEGACY_QUALITY_TELEMETRY_UNTRUSTED"
    return "CORRECTED_QUALITY_TELEMETRY"


def _v2_mark_top_metrics(
    rows: list[dict[str, Any]],
    metrics: tuple[str, ...],
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    for row in rows:
        row["top_k_metrics"] = []
        row["anomaly_metrics"] = []
    selected_rows: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        selected = _sort_top(rows, metric, top_k)
        selected_rows[metric] = selected
        for row in selected:
            row["top_k_metrics"].append(metric)
    for row in rows:
        row["top_k_metrics"] = sorted(set(row["top_k_metrics"]))
        row["anomaly_metrics"] = list(row["top_k_metrics"])
    return selected_rows


def analyze_run_records_v2(
    run_name: str,
    micro_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Analyze a run while explicitly separating legacy quality telemetry."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    sample_by_key: dict[str, dict[str, Any]] = {}
    sample_key_owners: dict[str, tuple[int, int]] = {}
    errors: list[str] = []
    for sample in sample_rows:
        key = _sample_key(sample)
        if key is None:
            errors.append("sample_missing_key")
            continue
        if key in sample_by_key:
            errors.append(f"duplicate_sample_key:{key}")
            continue
        sample_by_key[key] = sample
        if _finite(sample.get("step")) and _finite(sample.get("micro_index")):
            sample_key_owners[key] = (int(sample["step"]), int(sample["micro_index"]))
        else:
            errors.append(f"sample_missing_step_or_index:{key}")

    normalized: list[dict[str, Any]] = []
    for micro in micro_rows:
        row, row_errors = _normalize_micro(run_name, micro, sample_by_key, sample_key_owners)
        linked, _ = _sample_association(micro, sample_by_key, sample_key_owners)
        row["quality_telemetry_status"] = _quality_telemetry_status(micro, linked)
        normalized.append(row)
        errors.extend(row_errors)
    if not normalized:
        errors.append("no_microbatch_rows")

    quality_statuses = {row.get("quality_telemetry_status") for row in normalized}
    quality_trusted = bool(normalized) and quality_statuses == {"CORRECTED_QUALITY_TELEMETRY"}
    metrics = V2_STRUCTURAL_METRICS + (("quality_anomaly_score",) if quality_trusted else ())
    top_k_rows = _v2_mark_top_metrics(normalized, metrics, top_k)
    top_rows = {
        metric: [_public_top_row(row, metric) for row in rows]
        for metric, rows in top_k_rows.items()
    }
    return {
        "run": run_name,
        "status": "AUDIT_INCOMPLETE" if errors else "COMPLETE_DIAGNOSTIC",
        "microbatch_count": len(normalized),
        "sample_count": len(sample_rows),
        "errors": sorted(set(errors)),
        "quality_telemetry_status": (
            "CORRECTED_QUALITY_TELEMETRY" if quality_trusted
            else "LEGACY_QUALITY_TELEMETRY_UNTRUSTED"
        ),
        "top_metrics_used": list(metrics),
        "linkage": {
            "sample_key_count": len(sample_by_key),
            "microbatch_with_complete_links": sum(
                row["sample_association"]["sample_link_complete"] for row in normalized
            ),
            "microbatch_count": len(normalized),
            "all_links_complete": bool(normalized)
            and all(row["sample_association"]["sample_link_complete"] for row in normalized),
        },
        "top_k": top_rows,
        "step_summary": _step_summary(normalized),
        "category_summary": _group_summary(normalized, "category"),
        "prompt_summary": _group_summary(normalized, "prompt_id"),
        "reward_component_correlations": _component_correlations(normalized),
        "_rows": normalized,
    }


def _audit_run_directory_v2(run_dir: Path, *, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    required = (
        "microbatch_summaries.jsonl",
        "samples.jsonl",
        "step_summaries.jsonl",
        "validation_history.jsonl",
        "selection.json",
        "baseline_validation.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        return {
            "run": run_dir.name,
            "status": "AUDIT_INCOMPLETE",
            "microbatch_count": 0,
            "sample_count": 0,
            "errors": [f"missing_file:{name}" for name in missing],
            "linkage": {"all_links_complete": False},
            "top_k": {},
            "step_summary": [],
            "category_summary": {},
            "prompt_summary": {},
            "reward_component_correlations": {},
            "selected_metrics": {},
            "baseline_metrics": {},
            "validation_history": [],
            "quality_telemetry_status": "LEGACY_QUALITY_TELEMETRY_UNTRUSTED",
            "_rows": [],
            "_samples": [],
        }
    try:
        read_jsonl(run_dir / "step_summaries.jsonl")
        samples = read_jsonl(run_dir / "samples.jsonl")
        report = analyze_run_records_v2(
            run_dir.name,
            read_jsonl(run_dir / "microbatch_summaries.jsonl"),
            samples,
            top_k=top_k,
        )
        baseline = read_json(run_dir / "baseline_validation.json")
        report["selected_metrics"] = _selected_metrics(run_dir)
        report["baseline_metrics"] = dict(baseline.get("metrics", {}))
        report["validation_history"] = _validation_history(run_dir)
        report["_samples"] = samples
        return report
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "run": run_dir.name,
            "status": "AUDIT_INCOMPLETE",
            "microbatch_count": 0,
            "sample_count": 0,
            "errors": [f"read_or_parse_error:{error}"],
            "linkage": {"all_links_complete": False},
            "top_k": {},
            "step_summary": [],
            "category_summary": {},
            "prompt_summary": {},
            "reward_component_correlations": {},
            "selected_metrics": {},
            "baseline_metrics": {},
            "validation_history": [],
            "quality_telemetry_status": "LEGACY_QUALITY_TELEMETRY_UNTRUSTED",
            "_rows": [],
            "_samples": [],
        }


def _source_manifest_info(source_manifest: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(source_manifest)
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    missing_metadata: list[str] = []
    chosen_failures: list[dict[str, str]] = []
    for row in rows:
        row_id = str(row.get("id", ""))
        if not row_id or row_id in by_id:
            duplicate_ids.append(row_id or "<missing>")
            continue
        by_id[row_id] = row
        if not isinstance(row.get("metadata"), dict) or not row["metadata"]:
            missing_metadata.append(row_id)
        record = {
            "conversations": [
                {"role": "user", "content": str(row.get("prompt", ""))},
                {"role": "assistant", "content": str(row.get("chosen", ""))},
            ]
        }
        try:
            passed, reason = validate_record(record, row)
        except (KeyError, TypeError, ValueError) as error:
            passed, reason = False, f"validator_error:{error}"
        if not passed:
            chosen_failures.append({"id": row_id, "reason": reason})
    digest = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    info = {
        "path": str(source_manifest),
        "sha256": digest,
        "row_count": len(rows),
        "unique_id_count": len(by_id),
        "duplicate_ids": sorted(set(duplicate_ids)),
        "metadata_missing_count": len(missing_metadata),
        "metadata_missing_ids": sorted(missing_metadata),
        "source_chosen_validator_pass_count": len(rows) - len(chosen_failures),
        "source_chosen_validator_total": len(rows),
        "source_chosen_validator_failures": chosen_failures,
        "by_id": by_id,
    }
    return rows, info


def _validator_replay(
    reports: list[dict[str, Any]],
    source_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    total = 0
    matched = 0
    mismatches: list[dict[str, Any]] = []
    missing_source: list[str] = []
    failure_reasons: dict[str, int] = defaultdict(int)
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0, "mismatches": 0})
    for report in reports:
        for sample in report.get("_samples", []):
            total += 1
            prompt_id = str(sample.get("prompt_id", ""))
            manifest = source_by_id.get(prompt_id)
            if manifest is None:
                missing_source.append(prompt_id)
                continue
            category = str(manifest.get("category", sample.get("category", "unknown")))
            by_category[category]["total"] += 1
            response = str(sample.get("response", ""))
            record = {
                "conversations": [
                    {"role": "user", "content": str(manifest.get("prompt", ""))},
                    {"role": "assistant", "content": response},
                ]
            }
            try:
                replay_passed, reason = validate_record(record, manifest)
            except (KeyError, TypeError, ValueError) as error:
                replay_passed, reason = False, f"validator_error:{error}"
            if replay_passed:
                by_category[category]["passed"] += 1
            else:
                failure_reasons[reason or "unknown"] += 1
            persisted = _component_value(sample, "validator_reward") > 0
            if persisted == replay_passed:
                matched += 1
            else:
                by_category[category]["mismatches"] += 1
                mismatches.append({
                    "run": report.get("run"),
                    "sample_key": _sample_key(sample),
                    "prompt_id": prompt_id,
                    "persisted_validator_pass": persisted,
                    "replay_validator_pass": replay_passed,
                    "replay_failure_reason": reason,
                })
    return {
        "sample_total": total,
        "replay_evaluated": total - len(missing_source),
        "persisted_replay_match_count": matched,
        "persisted_replay_mismatch_count": len(mismatches),
        "persisted_replay_exact": not missing_source and not mismatches,
        "missing_source_count": len(missing_source),
        "missing_source_prompt_ids": sorted(set(missing_source)),
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
        "by_category": dict(sorted(by_category.items())),
        "mismatches": mismatches,
    }


def _reward_coverage(
    reports: list[dict[str, Any]],
    source_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    all_samples = [sample for report in reports for sample in report.get("_samples", [])]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in all_samples:
        manifest = source_by_id.get(str(sample.get("prompt_id", "")), {})
        by_category[str(manifest.get("category", sample.get("category", "unknown")))].append(sample)
        by_family[str(manifest.get("family", "unknown"))].append(sample)

    def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(samples)
        return {
            "sample_count": count,
            "components": {
                component: {
                    "nonzero_count": sum(_component_value(sample, component) != 0 for sample in samples),
                    "nonzero_rate": (
                        sum(_component_value(sample, component) != 0 for sample in samples) / count
                        if count else 0.0
                    ),
                }
                for component in REWARD_COMPONENTS
            },
        }

    return {
        "global": summarize(all_samples),
        "by_category": {key: summarize(value) for key, value in sorted(by_category.items())},
        "by_family": {key: summarize(value) for key, value in sorted(by_family.items())},
    }


def _exposure_denominators(
    source_rows: list[dict[str, Any]],
    prompt_groups: dict[str, dict[str, Any]],
    category_groups: dict[str, dict[str, Any]],
    all_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    category_counts: dict[str, int] = defaultdict(int)
    family_counts: dict[str, int] = defaultdict(int)
    prompt_to_family: dict[str, str] = {}
    prompt_to_category: dict[str, str] = {}
    for row in source_rows:
        category = str(row.get("category", "unknown"))
        family = str(row.get("family", "unknown"))
        prompt_id = str(row.get("id", ""))
        category_counts[category] += 1
        family_counts[family] += 1
        prompt_to_family[prompt_id] = family
        prompt_to_category[prompt_id] = category

    def top_hits(groups: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
        return {
            group: {
                "source_prompt_count": count,
                "top_k_prompt_count": sum(
                    bool(prompt_groups.get(prompt, {}).get("top_k_hit_count"))
                    for prompt in prompt_to_category
                    if (prompt_to_category.get(prompt) if key == "category" else prompt_to_family.get(prompt)) == group
                ),
                "top_k_microbatch_hit_count": sum(
                    row.get("top_k_metrics") != []
                    and (row.get("category") if key == "category" else prompt_to_family.get(row.get("prompt_id", ""))) == group
                    for row in all_rows
                ),
                "cross_condition_group_count": sum(
                    item.get("cross_condition_condition_count", 0) >= 2
                    for item in groups.values()
                    if (item.get("key") if key == "category" else prompt_to_family.get(item.get("key", ""))) == group
                ),
            }
            for group, count in sorted((category_counts if key == "category" else family_counts).items())
        }

    return {
        "category": top_hits(category_groups, "category"),
        "family": top_hits(prompt_groups, "family"),
    }


def _cross_condition_groups(groups: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (
            group for group in groups.values()
            if group.get("cross_condition_condition_count", 0) >= 2
            and group.get("top_k_hit_count", 0) > 0
        ),
        key=lambda item: (
            -item.get("cross_condition_condition_count", 0),
            -item.get("top_k_hit_count", 0),
            item.get("key", ""),
        ),
    )


def _markdown_v2(summary: dict[str, Any], reports: list[dict[str, Any]]) -> str:
    source = summary["source_manifest"]
    replay = summary["validator_replay"]
    lines = [
        "# RL Input and Reward Coverage Audit v2",
        "",
        "## Technical summary",
        "",
        f"本次离线审计覆盖 {summary['sample_count']} 个生成样本；source chosen validator 为 {source['source_chosen_validator_pass_count']}/{source['source_chosen_validator_total']}，持久化 validator_reward 与 replay 一致为 {replay['persisted_replay_match_count']}/{replay['sample_total']}。旧 run 没有可回放的结束原因字段，因此旧质量尾部证据统一标记为 `{summary['quality_telemetry_status']}`，不参与新的 top-K 质量结论。",
        "",
        f"当前诊断状态为 `{summary['status']}`，模型门禁保持 `{summary['model_change_gate']}`；本轮 GPU wall time 为 0 秒，未改变 reward、KL 门禁、checkpoint selection 或默认模型。",
        "",
        "## Key findings",
        "",
        f"- cross-condition prompt groups: {summary['cross_condition_prompt_count']}；cross-condition category groups: {summary['cross_condition_category_count']}。这里的 cross-condition 表示同一标识出现在多个实验条件的 top-K 中，不表示同一 run 内重复采样。",
        f"- metadata missing: {source['metadata_missing_count']}；replay mismatch: {replay['persisted_replay_mismatch_count']}；sample-source missing: {replay['missing_source_count']}。",
        f"- quality interpretation: `{summary['quality_telemetry_status']}`；旧 `max_length_hit` 不再用于推导截断或 quality-tail 结论。",
        "",
        "## Scope and definitions",
        "",
        f"- source manifest: `{source['path']}`; SHA-256 `{source['sha256']}`",
        f"- source rows: {source['row_count']}; generated samples: {summary['sample_count']}; conditions: {summary['run_count']}",
        "- validator replay uses the same category validator and source metadata; persisted pass means `components.validator_reward > 0`.",
        "- within-run occurrence count is reported separately from cross-condition condition count.",
        "",
        "## Run coverage",
        "",
        "| run | micro-batches | samples | linkage | quality telemetry |",
        "|---|---:|---:|---|---|",
    ]
    for report in reports:
        lines.append(
            f"| {report['run']} | {report['microbatch_count']} | {report['sample_count']} | "
            f"{report.get('linkage', {}).get('all_links_complete', False)} | {report.get('quality_telemetry_status')} |"
        )
    lines.extend([
        "",
        "## Validator replay and reward coverage",
        "",
        f"Replay failure reasons: `{json.dumps(replay['failure_reason_counts'], ensure_ascii=False, sort_keys=True)}`。Reward component nonzero coverage is written to `reward_component_summary.json`; it is descriptive and does not change checkpoint selection.",
        "",
        "## Exposure and localization",
        "",
        f"Category/family denominators are written to `exposure_denominators.json`. The top-K localization uses only `{', '.join(summary['top_metrics_used'])}` because the input quality telemetry is legacy.",
        "",
        "## Limitations and next decision",
        "",
        "- The replay establishes validator-input consistency, not causal attribution for KL or gradient spikes.",
        "- Cross-condition concentration is a deterministic diagnostic cut; it is not evidence that a prompt or category caused the update anomaly.",
        "- The next permitted step is a corrected GRPO control smoke only after this offline audit passes; do not expand CISPO, three seeds, C-Eval, or change the default model in this phase.",
        "",
    ])
    return "\n".join(lines)


def run_audit_v2(
    source_root: Path,
    source_manifest: Path,
    output_dir: Path,
    *,
    run_names: Iterable[str] = REQUIRED_RUNS,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Run the non-mutating MM-E015/MM-F023 audit in a new output root."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows, source_info = _source_manifest_info(source_manifest)
    run_paths = discover_runs(source_root, run_names)
    reports = [_audit_run_directory_v2(path, top_k=top_k) for path in run_paths]
    all_rows = _merge_rows(reports)
    all_samples = [sample for report in reports for sample in report.get("_samples", [])]
    category_groups = _group_summary(all_rows, "category") if all_rows else {}
    prompt_groups = _group_summary(all_rows, "prompt_id") if all_rows else {}
    for item in prompt_groups.values():
        prompt_id = item["key"]
        matching = [row for row in all_rows if row["prompt_id"] == prompt_id]
        manifest = source_info["by_id"].get(prompt_id, {})
        item["categories"] = sorted({row["category"] for row in matching})
        item["family"] = manifest.get("family")
        item["difficulty"] = manifest.get("difficulty")
        item["metadata_present"] = bool(manifest.get("metadata"))
    cross_prompts = _cross_condition_groups(prompt_groups)
    cross_categories = _cross_condition_groups(category_groups)
    validator_replay = _validator_replay(reports, source_info["by_id"])
    reward_coverage = _reward_coverage(reports, source_info["by_id"])
    exposures = _exposure_denominators(source_rows, prompt_groups, category_groups, all_rows)
    incomplete = any(report["status"] == "AUDIT_INCOMPLETE" for report in reports)
    if incomplete:
        status = "AUDIT_INCOMPLETE"
    elif source_info["source_chosen_validator_pass_count"] != source_info["source_chosen_validator_total"]:
        status = "INPUT_VALIDATOR_INCONSISTENT"
    elif not validator_replay["persisted_replay_exact"]:
        status = "REWARD_VALIDATOR_REPLAY_MISMATCH"
    elif cross_prompts:
        status = "CROSS_CONDITION_PROMPT_DIAGNOSTIC"
    elif cross_categories:
        status = "CROSS_CONDITION_CATEGORY_DIAGNOSTIC"
    else:
        status = "NO_CROSS_CONDITION_SOURCE_DIAGNOSTIC"
    quality_statuses = {report.get("quality_telemetry_status") for report in reports}
    quality_status = (
        "CORRECTED_QUALITY_TELEMETRY"
        if quality_statuses == {"CORRECTED_QUALITY_TELEMETRY"}
        else "LEGACY_QUALITY_TELEMETRY_UNTRUSTED"
    )
    summary = {
        "status": status,
        "task_ids": ["MM-E015", "MM-F023"],
        "experiment_root": str(output_dir),
        "source_experiment_root": str(source_root),
        "source_manifest": source_info,
        "run_count": len(reports),
        "sample_count": len(all_samples),
        "runs": [
            {
                "run": report["run"],
                "status": report["status"],
                "microbatch_count": report["microbatch_count"],
                "sample_count": report["sample_count"],
                "linkage_complete": report.get("linkage", {}).get("all_links_complete", False),
                "quality_telemetry_status": report.get("quality_telemetry_status"),
                "errors": report.get("errors", []),
            }
            for report in reports
        ],
        "top_k": top_k,
        "top_metrics_used": list(
            sorted({metric for report in reports for metric in report.get("top_metrics_used", [])})
        ),
        "quality_telemetry_status": quality_status,
        "cross_condition_prompt_count": len(cross_prompts),
        "cross_condition_category_count": len(cross_categories),
        "cross_condition_prompts": cross_prompts,
        "cross_condition_categories": cross_categories,
        "validator_replay": validator_replay,
        "reward_component_nonzero_coverage": reward_coverage,
        "exposure_denominators": exposures,
        "all_json_finite": True,
        "diagnostic_only": True,
        "gpu_wall_seconds": 0,
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        "default_model_changed": False,
        "old_artifacts_untouched": True,
        "limits": [
            "Legacy max_length_hit and finished_naturally fields are readable but not trusted for corrected quality conclusions.",
            "Cross-condition grouping is a deterministic descriptive diagnostic, not causal evidence.",
            "Validator replay and reward coverage do not change checkpoint selection or the default model.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "run_reports.jsonl").open("w", encoding="utf-8") as handle:
        for report in reports:
            handle.write(json.dumps(_public_run_report(report), ensure_ascii=False) + "\n")
    with (output_dir / "cross_condition_prompt_summary.jsonl").open("w", encoding="utf-8") as handle:
        for item in cross_prompts:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (output_dir / "cross_condition_category_summary.json").write_text(
        json.dumps(cross_categories, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "exposure_denominators.json").write_text(
        json.dumps(exposures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "validator_replay_summary.json").write_text(
        json.dumps(validator_replay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "reward_component_summary.json").write_text(
        json.dumps(reward_coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_markdown_v2(summary, reports), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--audit-version", choices=("legacy", "v2"), default="legacy")
    parser.add_argument("--source-manifest", type=Path)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("top-k must be positive")
    if args.audit_version == "v2":
        if args.source_manifest is None:
            parser.error("--source-manifest is required for --audit-version v2")
        summary = run_audit_v2(
            args.source_root,
            args.source_manifest,
            args.output_dir,
            top_k=args.top_k,
        )
    else:
        summary = run_audit(args.source_root, args.output_dir, top_k=args.top_k)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
