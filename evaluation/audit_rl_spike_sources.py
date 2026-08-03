"""Forensic audit of RL KL spikes, reward transfer, and micro-batch sources.

This is diagnostic only. It does not alter checkpoint selection, KL gates, or
model promotion. Micro-batch telemetry, when available, is summarized without
persisting raw logits, log-prob tensors, gradients, or response text in the
audit outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


RUN_PATTERN = re.compile(r"^(grpo|cispo)_(control|low_lr|accum16)_seed42$")
MICRO_RUN_PATTERN = re.compile(r"^grpo_control_seed42(?:_diagnostic|_diagnostic_balanced)?$")
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_KL_THRESHOLD = 0.005
MICROBATCH_SCHEMA_VERSION = 2


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


def read_jsonl_with_completion(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Read JSONL and distinguish a truncated final line from valid rows."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, parse_constant=_reject_nonfinite)
        except json.JSONDecodeError:
            if line_number == len(lines) and not text.endswith("\n"):
                return rows, False
            raise
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows, True


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _number(value: Any, default: float = 0.0) -> float:
    return float(value) if _finite(value) else default


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if not _finite(numerator) or not _finite(denominator) or float(denominator) <= 0:
        return None
    return float(numerator) / float(denominator)


def _selected_metrics(selection: dict[str, Any]) -> dict[str, Any]:
    selected_step = selection.get("selected_step")
    for checkpoint in selection.get("checkpoints", []):
        if checkpoint.get("step") == selected_step and isinstance(checkpoint.get("metrics"), dict):
            return dict(checkpoint["metrics"])
    return {}


def _sample_stats(sample_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        if not _finite(row.get("step")):
            raise ValueError("sample row has no finite step")
        grouped[int(row["step"])].append(row)

    output: dict[int, dict[str, Any]] = {}
    for step, rows in sorted(grouped.items()):
        rewards = [_number(row.get("reward")) for row in rows]
        generated_tokens = [_number(row.get("generated_tokens")) for row in rows]
        repetition = [_number(dict(row.get("components", {})).get("repetition_penalty")) for row in rows]
        validator = [_number(dict(row.get("components", {})).get("validator_reward")) > 0 for row in rows]
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_category[str(row.get("category", "unknown"))].append(row)
        category_stats: dict[str, dict[str, Any]] = {}
        for category, category_rows in sorted(by_category.items()):
            category_validator = [
                _number(dict(item.get("components", {})).get("validator_reward")) > 0
                for item in category_rows
            ]
            category_stats[category] = {
                "count": len(category_rows),
                "validator_pass_rate": _rate(category_validator),
                "max_length_hit_rate": _rate([bool(item.get("max_length_hit")) for item in category_rows]),
                "natural_end_rate": _rate([bool(item.get("finished_naturally")) for item in category_rows]),
                "average_tokens": mean(_number(item.get("generated_tokens")) for item in category_rows),
                "repetition_penalty_mean": mean(
                    _number(dict(item.get("components", {})).get("repetition_penalty"))
                    for item in category_rows
                ),
            }
        output[step] = {
            "step": step,
            "sample_count": len(rows),
            "category_count": len(by_category),
            "validator_pass_rate": _rate(validator),
            "empty_response_rate": _rate([bool(row.get("empty_response")) for row in rows]),
            "max_length_hit_rate": _rate([bool(row.get("max_length_hit")) for row in rows]),
            "natural_end_rate": _rate([bool(row.get("finished_naturally")) for row in rows]),
            "average_tokens": mean(generated_tokens) if generated_tokens else 0.0,
            "reward_mean": mean(rewards) if rewards else 0.0,
            "reward_std": pstdev(rewards) if len(rewards) > 1 else 0.0,
            "repetition_penalty_mean": mean(repetition) if repetition else 0.0,
            "category_stats": category_stats,
        }
    return output


def _sample_key(row: dict[str, Any]) -> str | None:
    value = row.get("sample_key")
    if isinstance(value, str) and value:
        return value
    if all(_finite(row.get(name)) for name in ("step", "micro_index", "candidate_index")):
        return f"{int(row['step'])}:{int(row['micro_index'])}:{int(row['candidate_index'])}"
    return None


def _micro_diag(row: dict[str, Any], name: str, fallback: Any = 0.0) -> float:
    diagnostics = row.get("policy_diagnostics")
    if isinstance(diagnostics, dict) and name in diagnostics:
        return _number(diagnostics.get(name), _number(fallback))
    return _number(row.get(name), _number(fallback))


def _top_concentration(rows: list[dict[str, Any]], metric: str, top_k: int = 10) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda item: _number(item.get(metric)), reverse=True)[:top_k]
    category_counts: dict[str, int] = defaultdict(int)
    prompt_counts: dict[str, int] = defaultdict(int)
    for row in ranked:
        category_counts[str(row.get("category", "unknown"))] += 1
        prompt_counts[str(row.get("prompt_id", "unknown"))] += 1
    count = len(ranked)
    return {
        "metric": metric,
        "top_k": count,
        "category_counts": dict(sorted(category_counts.items())),
        "prompt_counts": dict(sorted(prompt_counts.items())),
        "max_category_share": max(category_counts.values(), default=0) / count if count else 0.0,
        "max_prompt_share": max(prompt_counts.values(), default=0) / count if count else 0.0,
        "top_rows": [
            {
                "step": row["step"],
                "micro_index": row["micro_index"],
                "prompt_id": row["prompt_id"],
                "category": row["category"],
                metric: row.get(metric),
                "reference_kl_max": row.get("reference_kl_max"),
                "reference_kl_p95": row.get("reference_kl_p95"),
                "micro_grad_norm_unscaled": row.get("micro_grad_norm_unscaled"),
                "max_length_hit_rate": row.get("max_length_hit_rate"),
                "repetition_penalty_mean": row.get("repetition_penalty_mean"),
                "sample_link_count": row.get("sample_link_count"),
            }
            for row in ranked
        ],
    }


def _microbatch_attribution(
    micro_rows: list[dict[str, Any]] | None,
    sample_rows: list[dict[str, Any]],
    *,
    available: bool,
    complete: bool,
    required: bool,
    kl_threshold: float,
) -> dict[str, Any]:
    if not available:
        return {
            "status": "TELEMETRY_INCOMPLETE" if required else "NOT_AVAILABLE_LEGACY",
            "reason": "microbatch_summaries.jsonl is missing",
            "microbatch_count": 0,
            "category_summary": {},
            "prompt_summary": {},
            "top_kl": {},
            "top_gradient": {},
            "top_quality": {},
            "overlap": {},
        }
    if not micro_rows:
        return {
            "status": "NO_SEPARATION_UNRESOLVED" if complete else "TELEMETRY_INCOMPLETE",
            "reason": "no complete micro-batch rows were available",
            "microbatch_count": 0,
            "category_summary": {},
            "prompt_summary": {},
            "top_kl": {},
            "top_gradient": {},
            "top_quality": {},
            "overlap": {},
        }

    samples_by_key = {key: row for row in sample_rows if (key := _sample_key(row)) is not None}
    normalized: list[dict[str, Any]] = []
    for row in micro_rows:
        if not _finite(row.get("step")) or not _finite(row.get("micro_index")):
            raise ValueError("micro-batch row has no finite step or micro_index")
        diagnostics = row.get("policy_diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        declared_keys = row.get("sample_keys")
        if not isinstance(declared_keys, list):
            declared_keys = []
        linked_samples = [samples_by_key[key] for key in declared_keys if key in samples_by_key]
        if not linked_samples:
            linked_samples = [
                sample
                for sample in sample_rows
                if sample.get("step") == row.get("step")
                and sample.get("micro_index") == row.get("micro_index")
                and sample.get("prompt_id") == row.get("prompt_id")
            ]
        repetition = [
            _number(dict(sample.get("components", {})).get("repetition_penalty"))
            for sample in linked_samples
        ]
        normalized.append(
            {
                "step": int(row["step"]),
                "micro_index": int(row["micro_index"]),
                "prompt_id": str(row.get("prompt_id", "unknown")),
                "category": str(row.get("category", "unknown")),
                "reference_kl_mean": _micro_diag(row, "reference_kl_mean", row.get("kl")),
                "reference_kl_p95": _micro_diag(row, "reference_kl_p95", row.get("kl")),
                "reference_kl_max": _micro_diag(row, "reference_kl_max", row.get("kl")),
                "ratio_p95": _micro_diag(row, "ratio_p95"),
                "ratio_max": _micro_diag(row, "ratio_max"),
                "micro_grad_norm_scaled": _number(row.get("micro_grad_norm_scaled")),
                "micro_grad_norm_unscaled": _number(row.get("micro_grad_norm_unscaled")),
                "accumulated_grad_norm": _number(row.get("accumulated_grad_norm")),
                "reward_mean": _number(row.get("reward_mean")),
                "train_validator_pass_rate": _number(row.get("train_validator_pass_rate")),
                "max_length_hit_rate": _number(
                    row.get("train_max_length_hit_rate"),
                ),
                "natural_end_rate": _number(row.get("train_natural_end_rate")),
                "repetition_penalty_mean": _number(
                    row.get("train_repetition_penalty_mean"),
                    mean(repetition) if repetition else 0.0,
                ),
                "empty_response_rate": _number(row.get("train_empty_response_rate")),
                "sample_link_count": len(linked_samples),
                "declared_sample_count": len(declared_keys),
                "sample_link_complete": len(linked_samples) == len(declared_keys) if declared_keys else False,
            }
        )

    gradient_values = [row["micro_grad_norm_unscaled"] for row in normalized if row["micro_grad_norm_unscaled"] > 0]
    gradient_median = sorted(gradient_values)[len(gradient_values) // 2] if gradient_values else 0.0
    for row in normalized:
        row["gradient_tail"] = bool(gradient_median and row["micro_grad_norm_unscaled"] >= 2.0 * gradient_median)
        row["kl_tail"] = bool(
            row["reference_kl_max"] >= 0.1
            and (_ratio(row["reference_kl_max"], row["reference_kl_mean"]) or 0.0) >= 10.0
        ) or bool(
            row["reference_kl_p95"] >= kl_threshold
            and (_ratio(row["reference_kl_p95"], row["reference_kl_mean"]) or 0.0) >= 3.0
        )
        row["quality_tail"] = bool(
            row["max_length_hit_rate"] >= 0.5
            or row["repetition_penalty_mean"] >= 0.1
            or row["empty_response_rate"] > 0.05
        )

    category_summary: dict[str, dict[str, Any]] = {}
    prompt_summary: dict[str, dict[str, Any]] = {}
    for dimension, output in (("category", category_summary), ("prompt_id", prompt_summary)):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in normalized:
            grouped[row[dimension]].append(row)
        for key, group in sorted(grouped.items()):
            output[key] = {
                "microbatch_count": len(group),
                "kl_tail_count": sum(bool(row["kl_tail"]) for row in group),
                "gradient_tail_count": sum(bool(row["gradient_tail"]) for row in group),
                "quality_tail_count": sum(bool(row["quality_tail"]) for row in group),
                "reference_kl_max": max(row["reference_kl_max"] for row in group),
                "reference_kl_p95": max(row["reference_kl_p95"] for row in group),
                "micro_grad_norm_unscaled_max": max(row["micro_grad_norm_unscaled"] for row in group),
                "reward_mean": mean(row["reward_mean"] for row in group),
                "validator_pass_rate": mean(row["train_validator_pass_rate"] for row in group),
                "max_length_hit_rate": mean(row["max_length_hit_rate"] for row in group),
                "natural_end_rate": mean(row["natural_end_rate"] for row in group),
                "repetition_penalty_mean": mean(row["repetition_penalty_mean"] for row in group),
                "sample_link_complete": all(bool(row["sample_link_complete"]) for row in group),
            }

    top_kl = _top_concentration(normalized, "reference_kl_max")
    top_gradient = _top_concentration(normalized, "micro_grad_norm_unscaled")
    for row in normalized:
        row["quality_score"] = (
            row["max_length_hit_rate"]
            + row["repetition_penalty_mean"]
            + row["empty_response_rate"]
        )
    top_quality = _top_concentration(normalized, "quality_score")
    kl_keys = {(row["step"], row["micro_index"]) for row in top_kl["top_rows"][:3]}
    gradient_keys = {(row["step"], row["micro_index"]) for row in top_gradient["top_rows"][:3]}
    quality_keys = {(row["step"], row["micro_index"]) for row in top_quality["top_rows"][:3]}
    overlap = {
        "top3_kl_gradient_overlap": sorted(kl_keys & gradient_keys),
        "top3_kl_quality_overlap": sorted(kl_keys & quality_keys),
        "top3_kl_gradient_category_overlap": sorted(
            set(top_kl.get("category_counts", {})) & set(top_gradient.get("category_counts", {}))
        ),
        "top3_kl_quality_category_overlap": sorted(
            set(top_kl.get("category_counts", {})) & set(top_quality.get("category_counts", {}))
        ),
        "linked_sample_count": sum(row["sample_link_count"] for row in normalized),
        "declared_sample_count": sum(row["declared_sample_count"] for row in normalized),
    }
    localized = max(
        top_kl.get("max_category_share", 0.0),
        top_kl.get("max_prompt_share", 0.0),
        top_gradient.get("max_category_share", 0.0),
        top_gradient.get("max_prompt_share", 0.0),
    ) >= 0.5
    status = "SOURCE_LOCALIZED_DIAGNOSTIC" if localized else "BROAD_SPIKE_DIAGNOSTIC"
    if not complete or not all(bool(row["sample_link_complete"]) for row in normalized):
        status = "TELEMETRY_INCOMPLETE"
    return {
        "status": status,
        "reason": "top-K micro-batch concentration and sample linkage were evaluated",
        "microbatch_count": len(normalized),
        "gradient_median_unscaled": gradient_median,
        "category_summary": category_summary,
        "prompt_summary": prompt_summary,
        "top_kl": top_kl,
        "top_gradient": top_gradient,
        "top_quality": top_quality,
        "overlap": overlap,
        "heuristics": {
            "top_k": 10,
            "localized_share_threshold": 0.5,
            "kl_absolute_min": 0.1,
            "kl_max_to_mean_min": 10.0,
            "kl_p95_to_mean_min": 3.0,
            "quality_max_length_min": 0.5,
            "quality_repetition_min": 0.1,
        },
    }


def _validation_by_step(validation_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for row in validation_rows:
        if _finite(row.get("step")) and isinstance(row.get("metrics"), dict):
            output[int(row["step"])] = dict(row["metrics"])
    return output


def _classify_step(
    row: dict[str, Any],
    sample: dict[str, Any] | None,
    validation: dict[str, Any] | None,
    *,
    max_grad_norm: float,
    kl_threshold: float,
) -> dict[str, Any]:
    kl_mean = _number(row.get("kl_mean"))
    kl_p95 = _number(row.get("kl_p95"))
    kl_max = _number(row.get("kl_max"))
    grad_pre = _number(row.get("grad_norm_pre_clip"))
    ratio_max = _number(row.get("ratio_max"))
    flags: list[str] = []
    if kl_mean > kl_threshold:
        flags.append("kl_mean_above_threshold")
    if kl_max >= 0.1 and (_ratio(kl_max, kl_mean) or 0.0) >= 10.0:
        flags.append("kl_max_tail_concentration")
    if kl_p95 >= kl_threshold and (_ratio(kl_p95, kl_mean) or 0.0) >= 3.0:
        flags.append("kl_p95_tail_concentration")
    if grad_pre > max_grad_norm:
        flags.append("gradient_clipping_active")
    if ratio_max >= 2.0:
        flags.append("ratio_tail")
    if _number(row.get("train_max_length_hit_rate")) >= 0.5:
        flags.append("truncation_dominant")
    if _number(row.get("train_repetition_penalty_mean")) >= 0.1:
        flags.append("repetition_penalty_elevated")
    if _number(row.get("train_empty_response_rate")) > 0.05:
        flags.append("empty_response_elevated")
    return {
        "step": int(row.get("step", 0)),
        "kl_mean": row.get("kl_mean"),
        "kl_p95": row.get("kl_p95"),
        "kl_max": row.get("kl_max"),
        "kl_max_to_mean": _ratio(row.get("kl_max"), row.get("kl_mean")),
        "kl_p95_to_mean": _ratio(row.get("kl_p95"), row.get("kl_mean")),
        "grad_norm_pre_clip": row.get("grad_norm_pre_clip"),
        "ratio_max": row.get("ratio_max"),
        "train_reward_mean": row.get("reward_mean"),
        "train_validator_pass_rate": row.get("train_validator_pass_rate"),
        "train_max_length_hit_rate": row.get("train_max_length_hit_rate"),
        "train_repetition_penalty_mean": row.get("train_repetition_penalty_mean"),
        "sample_stats": sample,
        "validation_metrics": validation,
        "flags": flags,
    }


def analyze_run_data(
    run_name: str,
    step_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    baseline_metrics: dict[str, Any],
    validation_rows: list[dict[str, Any]],
    selected_metrics: dict[str, Any],
    *,
    micro_rows: list[dict[str, Any]] | None = None,
    microbatch_available: bool = False,
    microbatch_complete: bool = True,
    microbatch_required: bool = False,
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
    kl_threshold: float = DEFAULT_KL_THRESHOLD,
) -> dict[str, Any]:
    if not step_rows:
        raise ValueError(f"no step rows for {run_name}")
    sample_by_step = _sample_stats(sample_rows)
    validation_by_step = _validation_by_step(validation_rows)
    steps = [
        _classify_step(
            row,
            sample_by_step.get(int(row.get("step", 0))),
            validation_by_step.get(int(row.get("step", 0))),
            max_grad_norm=max_grad_norm,
            kl_threshold=kl_threshold,
        )
        for row in step_rows
    ]
    baseline_pass = _number(baseline_metrics.get("validator_pass_rate"))
    train_validator_values = [_number(row.get("train_validator_pass_rate")) for row in step_rows]
    reward_values = [_number(row.get("reward_mean")) for row in step_rows]
    first_train_validator = train_validator_values[0]
    first_reward = reward_values[0]
    max_train_validator = max(train_validator_values)
    max_reward = max(reward_values)
    signal_counts: dict[str, int] = defaultdict(int)
    for step in steps:
        for flag in step["flags"]:
            signal_counts[flag] += 1
    max_length_values = [_number(row.get("train_max_length_hit_rate")) for row in step_rows]
    repeat_values = [_number(row.get("train_repetition_penalty_mean")) for row in step_rows]
    selected_pass = _number(selected_metrics.get("validator_pass"))
    selected_safety = _number(selected_metrics.get("safety_pass"))
    selected_termination = _number(selected_metrics.get("termination_pass"))
    reward_validation_gap = max_train_validator > baseline_pass + 0.05 and selected_pass <= _number(
        baseline_metrics.get("validator_pass")
    )
    validator_gain = max_train_validator - first_train_validator
    reward_gain = max_reward - first_reward
    warnings: list[str] = []
    if reward_validation_gap:
        warnings.append("train_validator_gain_without_validation_gain")
    if reward_gain >= 0.1 and selected_pass <= _number(baseline_metrics.get("validator_pass")):
        warnings.append("reward_gain_without_validation_gain")
    if max(max_length_values, default=0.0) - max_length_values[0] >= 0.2:
        warnings.append("max_length_hit_increase")
    if max(repeat_values, default=0.0) - repeat_values[0] >= 0.05:
        warnings.append("repetition_penalty_increase")
    if signal_counts.get("empty_response_elevated", 0):
        warnings.append("empty_response_elevated")

    microbatch_attribution = _microbatch_attribution(
        micro_rows,
        sample_rows,
        available=microbatch_available,
        complete=microbatch_complete,
        required=microbatch_required,
        kl_threshold=kl_threshold,
    )

    top_spikes = sorted(
        steps,
        key=lambda item: (_number(item.get("kl_max")), _number(item.get("kl_p95"))),
        reverse=True,
    )[:5]
    if signal_counts.get("kl_max_tail_concentration") and signal_counts.get("gradient_clipping_active"):
        interpretation = (
            "A rare high-KL tail co-occurs with clipped aggregate gradients; this is a plausible "
            "interaction, not a causal finding."
        )
    elif signal_counts.get("kl_mean_above_threshold"):
        interpretation = "The KL mean crosses the gate in the observed tail; the available data do not isolate the source."
    else:
        interpretation = "No step-level KL mean breach was observed in the available rows."
    return {
        "run": run_name,
        "status": "DIAGNOSTIC_ONLY_WITH_SIGNALS" if warnings or signal_counts else "DIAGNOSTIC_ONLY_NO_SIGNAL",
        "steps_completed": len(step_rows),
        "sample_rows": len(sample_rows),
        "selected_metrics": {
            "validator_pass": selected_metrics.get("validator_pass"),
            "safety_pass": selected_metrics.get("safety_pass"),
            "termination_pass": selected_metrics.get("termination_pass"),
            "natural_end": selected_metrics.get("natural_end"),
        },
        "baseline_metrics": {
            "validator_pass": baseline_metrics.get("validator_pass"),
            "validator_pass_rate": baseline_metrics.get("validator_pass_rate"),
            "safety_pass": baseline_metrics.get("safety_pass"),
            "termination_pass": baseline_metrics.get("termination_pass"),
        },
        "train_signal_extrema": {
            "first_validator_pass_rate": first_train_validator,
            "max_validator_pass_rate": max_train_validator,
            "validator_gain_within_train": validator_gain,
            "first_reward_mean": first_reward,
            "max_reward_mean": max_reward,
            "reward_gain_within_train": reward_gain,
            "max_length_hit_rate": max(max_length_values, default=0.0),
            "max_repetition_penalty_mean": max(repeat_values, default=0.0),
        },
        "validation_transfer": {
            "selected_validator_pass": selected_pass,
            "selected_safety_pass": selected_safety,
            "selected_termination_pass": selected_termination,
            "train_validator_gain_without_validation_gain": reward_validation_gap,
            "reward_gain_without_validation_gain": "reward_gain_without_validation_gain" in warnings,
        },
        "signal_counts": dict(sorted(signal_counts.items())),
        "warnings": warnings,
        "top_spike_steps": top_spikes,
        "step_analysis": steps,
        "microbatch_attribution": microbatch_attribution,
        "interpretation": interpretation,
        "limits": [
            "The persisted artifacts are step-level summaries plus per-sample outputs.",
            "Micro-batch attribution uses aggregate KL/ratio statistics and gradient norms, not raw token log-prob tensors or gradient vectors.",
            "The interpretation is diagnostic inference and does not establish causality or alter model gates.",
        ],
    }


def audit_run(
    run_dir: Path,
    *,
    max_grad_norm: float,
    kl_threshold: float,
    require_microbatch: bool = False,
) -> dict[str, Any]:
    selection = read_json(run_dir / "selection.json")
    baseline = read_json(run_dir / "baseline_validation.json")
    step_rows = read_jsonl(run_dir / "step_summaries.jsonl")
    sample_rows = read_jsonl(run_dir / "samples.jsonl")
    validation_rows = read_jsonl(run_dir / "validation_history.jsonl")
    microbatch_path = run_dir / "microbatch_summaries.jsonl"
    micro_rows = None
    microbatch_available = microbatch_path.exists()
    microbatch_complete = True
    if microbatch_available:
        micro_rows, microbatch_complete = read_jsonl_with_completion(microbatch_path)
    return analyze_run_data(
        run_dir.name,
        step_rows,
        sample_rows,
        dict(baseline.get("metrics", {})),
        validation_rows,
        _selected_metrics(selection),
        micro_rows=micro_rows,
        microbatch_available=microbatch_available,
        microbatch_complete=microbatch_complete,
        microbatch_required=require_microbatch,
        max_grad_norm=max_grad_norm,
        kl_threshold=kl_threshold,
    )


def discover_runs(
    root: Path,
    include_smoke: bool = False,
    run_name: str | None = None,
) -> list[Path]:
    if run_name is not None:
        path = root / run_name
        if not path.is_dir():
            raise ValueError(f"requested run does not exist: {path}")
        return [path]
    runs = [
        path
        for path in root.iterdir()
        if path.is_dir() and (RUN_PATTERN.fullmatch(path.name) or MICRO_RUN_PATTERN.fullmatch(path.name))
    ]
    if include_smoke:
        smoke = root / "grpo_smoke_control_seed42"
        if smoke.is_dir():
            runs.append(smoke)
    return sorted(runs)


def _compact(report: dict[str, Any]) -> dict[str, Any]:
    attribution = report["microbatch_attribution"]
    return {
        "run": report["run"],
        "status": report["status"],
        "steps_completed": report["steps_completed"],
        "selected_metrics": report["selected_metrics"],
        "train_signal_extrema": report["train_signal_extrema"],
        "validation_transfer": report["validation_transfer"],
        "signal_counts": report["signal_counts"],
        "warnings": report["warnings"],
        "attribution_status": attribution["status"],
        "microbatch_count": attribution["microbatch_count"],
        "top3_kl_gradient_overlap": attribution.get("overlap", {}).get("top3_kl_gradient_overlap", []),
        "top3_kl_gradient_category_overlap": attribution.get("overlap", {}).get(
            "top3_kl_gradient_category_overlap", []
        ),
        "top_spike_steps": [
            {
                "step": item["step"],
                "kl_mean": item["kl_mean"],
                "kl_p95": item["kl_p95"],
                "kl_max": item["kl_max"],
                "flags": item["flags"],
            }
            for item in report["top_spike_steps"]
        ],
    }


def _markdown(summary: dict[str, Any], reports: list[dict[str, Any]]) -> str:
    lines = [
        "# RL KL Spike-Source Forensics",
        "",
        "本报告只做 step/sample-level 诊断，不改变 KL、quality、checkpoint 或模型晋级门禁。",
        "",
        f"- experiment root: `{summary['experiment_root']}`",
        f"- run count: {summary['run_count']}",
        f"- status: `{summary['status']}`",
        "",
        "## Run summary",
        "",
        "| run | steps | selected validation | max train validator | attribution | KL tail signals | warnings |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for report in reports:
        selected = report["selected_metrics"].get("validator_pass")
        extrema = report["train_signal_extrema"]
        tail = report["signal_counts"].get("kl_max_tail_concentration", 0) + report["signal_counts"].get(
            "kl_p95_tail_concentration", 0
        )
        attribution_status = report["microbatch_attribution"]["status"]
        lines.append(
            f"| {report['run']} | {report['steps_completed']} | {selected} | "
            f"{extrema['max_validator_pass_rate']:.4f} | {attribution_status} | {tail} | "
            f"{', '.join(report['warnings']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- KL tail concentration means the persisted aggregate max/P95 is much larger than the same-step mean; it identifies concentration, not the exact token or micro-batch.",
            "- A train validator/reward gain with unchanged selected validation is a transfer gap, not evidence of model improvement.",
            "- Micro-batch attribution ranks aggregate KL, gradient-norm, and quality tails by category and prompt; it is diagnostic localization rather than causal proof.",
            "",
            "## Decision",
            "",
            "继续暂停正式三 seed RL 扩展；本审计不改变 checkpoint 或模型晋级门禁。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_audit(
    experiment_root: Path,
    output_dir: Path,
    *,
    include_smoke: bool = False,
    run_name: str | None = None,
    require_microbatch: bool = False,
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
    kl_threshold: float = DEFAULT_KL_THRESHOLD,
) -> dict[str, Any]:
    run_dirs = discover_runs(experiment_root, include_smoke=include_smoke, run_name=run_name)
    if not run_dirs:
        raise ValueError(f"no matching stability runs under {experiment_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = [
        audit_run(
            path,
            max_grad_norm=max_grad_norm,
            kl_threshold=kl_threshold,
            require_microbatch=require_microbatch,
        )
        for path in run_dirs
    ]
    signal_union = sorted({signal for report in reports for signal in report["signal_counts"]})
    attribution_statuses = [report["microbatch_attribution"]["status"] for report in reports]
    if "TELEMETRY_INCOMPLETE" in attribution_statuses:
        overall_attribution_status = "TELEMETRY_INCOMPLETE"
    elif "SOURCE_LOCALIZED_DIAGNOSTIC" in attribution_statuses:
        overall_attribution_status = "SOURCE_LOCALIZED_DIAGNOSTIC"
    elif "BROAD_SPIKE_DIAGNOSTIC" in attribution_statuses:
        overall_attribution_status = "BROAD_SPIKE_DIAGNOSTIC"
    else:
        overall_attribution_status = "NO_SEPARATION_UNRESOLVED"
    summary = {
        "status": overall_attribution_status if any(
            report["microbatch_attribution"]["microbatch_count"] for report in reports
        ) or any(status == "TELEMETRY_INCOMPLETE" for status in attribution_statuses)
        else ("DIAGNOSTIC_ONLY_WITH_SIGNALS" if signal_union else "DIAGNOSTIC_ONLY_NO_SIGNAL"),
        "experiment_root": str(experiment_root),
        "output_dir": str(output_dir),
        "run_count": len(reports),
        "runs": [_compact(report) for report in reports],
        "attribution_statuses": attribution_statuses,
        "observed_signal_codes": signal_union,
        "all_json_finite": True,
        "diagnostic_only": True,
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        "thresholds": {
            "max_grad_norm": max_grad_norm,
            "kl_threshold": kl_threshold,
            "kl_tail_absolute_min": 0.1,
            "kl_max_to_mean_min": 10.0,
            "kl_p95_to_mean_min": 3.0,
            "truncation_dominant_rate": 0.5,
        },
        "limits": reports[0]["limits"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "run_reports.jsonl").open("w", encoding="utf-8") as handle:
        for report in reports:
            handle.write(json.dumps(report, ensure_ascii=False) + "\n")
    (output_dir / "microbatch_attribution.json").write_text(
        json.dumps(
            {
                "status": overall_attribution_status,
                "runs": {
                    report["run"]: report["microbatch_attribution"]
                    for report in reports
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "category_summary.json").write_text(
        json.dumps(
            {
                report["run"]: report["microbatch_attribution"].get("category_summary", {})
                for report in reports
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "prompt_summary.json").write_text(
        json.dumps(
            {
                report["run"]: report["microbatch_attribution"].get("prompt_summary", {})
                for report in reports
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_markdown(summary, reports), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--require-microbatch", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--kl-threshold", type=float, default=DEFAULT_KL_THRESHOLD)
    args = parser.parse_args()
    if args.max_grad_norm <= 0 or args.kl_threshold <= 0:
        parser.error("max-grad-norm and kl-threshold must be positive")
    summary = run_audit(
        args.experiment_root,
        args.output_dir,
        include_smoke=args.include_smoke,
        run_name=args.run_name,
        require_microbatch=args.require_microbatch,
        max_grad_norm=args.max_grad_norm,
        kl_threshold=args.kl_threshold,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
