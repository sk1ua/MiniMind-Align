"""Audit RL KL and optimizer stability without changing model-selection gates."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any


DIAGNOSTIC_FIELDS = (
    "loss_mean",
    "grad_norm_pre_clip",
    "grad_norm_post_clip",
    "kl_p95",
    "kl_max",
    "ratio_p95",
    "ratio_max",
)
KL_THRESHOLD = 0.005
QUALITY_DROP_POINTS = 10.0


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=_reject_nonfinite)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(row)
    return rows


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _max_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if _finite(row.get(key))]
    return max(values) if values else None


def _selected_metrics(selection: dict[str, Any]) -> dict[str, Any] | None:
    step = selection.get("selected_step")
    if step is None:
        return None
    for record in selection.get("checkpoints", []):
        if record.get("step") == step:
            return record.get("metrics")
    return None


def _run_identity(run_dir: Path) -> tuple[str, str, int | None]:
    match = re.fullmatch(r"(grpo|cispo)_(.+)_seed(\d+)", run_dir.name)
    if not match:
        return run_dir.name.split("_", 1)[0], "unknown", None
    return match.group(1), match.group(2), int(match.group(3))


def _trigger_step(step_rows: list[dict[str, Any]]) -> int | None:
    triggered = [int(row["step"]) for row in step_rows if row.get("kl_triggered")]
    return min(triggered) if triggered else None


def _validation_summary(rows: list[dict[str, Any]], selected: dict[str, Any] | None) -> dict[str, Any]:
    metrics = [row.get("metrics", {}) for row in rows]
    selected_metrics = selected or (metrics[-1] if metrics else {})
    return {
        "selected_validator_pass": selected_metrics.get("validator_pass"),
        "selected_safety_pass": selected_metrics.get("safety_pass"),
        "selected_termination_pass": selected_metrics.get("termination_pass"),
        "selected_natural_end": selected_metrics.get("natural_end"),
        "selected_average_repeat_3gram": selected_metrics.get("average_repeat_3gram"),
        "best_validator_pass": max(
            (float(item["validator_pass"]) for item in metrics if _finite(item.get("validator_pass"))),
            default=None,
        ),
        "validation_steps": [
            {
                "step": row.get("step"),
                "validator_pass": row.get("metrics", {}).get("validator_pass"),
                "safety_pass": row.get("metrics", {}).get("safety_pass"),
                "termination_pass": row.get("metrics", {}).get("termination_pass"),
                "natural_end": row.get("metrics", {}).get("natural_end"),
                "average_repeat_3gram": row.get("metrics", {}).get("average_repeat_3gram"),
                "quality_triggered": row.get("quality_triggered"),
                "kl_triggered": row.get("kl_triggered"),
            }
            for row in rows
        ],
    }


def audit_run(run_dir: Path) -> dict[str, Any]:
    mode, condition, seed = _run_identity(run_dir)
    selection = read_json(run_dir / "selection.json")
    baseline = read_json(run_dir / "baseline_validation.json")
    step_rows = read_jsonl(run_dir / "step_summaries.jsonl")
    validation_rows = read_jsonl(run_dir / "validation_history.jsonl")
    missing_fields = [
        field for field in DIAGNOSTIC_FIELDS
        if any(not _finite(row.get(field)) for row in step_rows)
    ]
    trigger_step = _trigger_step(step_rows)
    selected_metrics = _selected_metrics(selection)
    warnings: list[dict[str, Any]] = []
    if missing_fields:
        warnings.append({"code": "missing_stability_telemetry", "severity": "error", "fields": missing_fields})
    if trigger_step is not None:
        warnings.append({
            "code": "kl_early_stop_triggered",
            "severity": "warning",
            "first_trigger_step": trigger_step,
        })
    clipped_steps = sum(1 for row in step_rows if row.get("grad_was_clipped"))
    if clipped_steps:
        warnings.append({
            "code": "gradient_clipping_active",
            "severity": "diagnostic",
            "steps": clipped_steps,
        })
    return {
        "run": run_dir.name,
        "mode": mode,
        "condition": condition,
        "seed": seed,
        "status": "FAILED" if missing_fields else ("WARNING" if warnings else "PASS"),
        "selection": {
            "status": selection.get("status"),
            "selected_step": selection.get("selected_step"),
            "selected_checkpoint": selection.get("selected_checkpoint"),
            "stop_reason": selection.get("stop_reason"),
        },
        "baseline": baseline.get("metrics", {}),
        "selected_metrics": selected_metrics,
        "steps_completed": len(step_rows),
        "first_kl_trigger_step": trigger_step,
        "kl_threshold_exceedance_count": sum(
            1 for row in step_rows if _finite(row.get("kl_mean")) and float(row["kl_mean"]) > KL_THRESHOLD
        ),
        "max_kl_mean": _max_metric(step_rows, "kl_mean"),
        "max_kl_p95": _max_metric(step_rows, "kl_p95"),
        "max_kl": _max_metric(step_rows, "kl_max"),
        "max_ratio_p95": _max_metric(step_rows, "ratio_p95"),
        "max_ratio": _max_metric(step_rows, "ratio_max"),
        "max_grad_norm_pre_clip": _max_metric(step_rows, "grad_norm_pre_clip"),
        "max_grad_norm_post_clip": _max_metric(step_rows, "grad_norm_post_clip"),
        "gradient_clipped_steps": clipped_steps,
        "train_max_length_hit_max": _max_metric(step_rows, "train_max_length_hit_rate"),
        "train_repetition_penalty_max": _max_metric(step_rows, "train_repetition_penalty_mean"),
        "validation": _validation_summary(validation_rows, selected_metrics),
        "warnings": warnings,
    }


def compare_run_to_control(report: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    """Compare one optimizer condition with its same-method control run."""
    control_trigger = control.get("first_kl_trigger_step")
    variant_trigger = report.get("first_kl_trigger_step")
    if variant_trigger is None:
        trigger_improved = control_trigger is not None
    elif control_trigger is None:
        trigger_improved = False
    else:
        trigger_improved = variant_trigger >= control_trigger + 4

    def not_higher(key: str) -> bool:
        current = report.get(key)
        reference = control.get(key)
        return _finite(current) and _finite(reference) and float(current) <= float(reference) + 1e-12

    current_validation = report.get("validation", {})
    control_validation = control.get("validation", {})
    quality_ok = True
    quality_deltas: dict[str, float | None] = {}
    for key in ("selected_safety_pass", "selected_termination_pass"):
        current = current_validation.get(key)
        reference = control_validation.get(key)
        delta = float(current) - float(reference) if _finite(current) and _finite(reference) else None
        quality_deltas[key] = delta
        if delta is not None and delta < -(QUALITY_DROP_POINTS / 100.0) * 32:
            quality_ok = False
    stability_improved = trigger_improved and not_higher("max_kl_p95") and not_higher("max_kl") and quality_ok
    return {
        "mode": report.get("mode"),
        "condition": report.get("condition"),
        "control": control.get("run"),
        "first_trigger_step": report.get("first_kl_trigger_step"),
        "control_first_trigger_step": control_trigger,
        "trigger_improved": trigger_improved,
        "max_kl_p95_not_higher": not_higher("max_kl_p95"),
        "max_kl_not_higher": not_higher("max_kl"),
        "max_grad_norm_pre_clip_not_higher": not_higher("max_grad_norm_pre_clip"),
        "quality_ok": quality_ok,
        "quality_deltas": quality_deltas,
        "stability_improved": stability_improved,
    }


def compare_conditions(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for mode in ("grpo", "cispo"):
        mode_reports = [report for report in reports if report.get("mode") == mode]
        controls = [report for report in mode_reports if report.get("condition") == "control"]
        if not controls:
            continue
        control = controls[0]
        for report in mode_reports:
            if report is control or report.get("condition") == "control":
                continue
            comparisons.append(compare_run_to_control(report, control))
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse audit output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_run_dirs = sorted(
        path for path in args.experiment_root.iterdir()
        if path.is_dir() and (path / "selection.json").exists()
    )
    formal_run_dirs = [
        path for path in all_run_dirs
        if re.fullmatch(r"(grpo|cispo)_(control|low_lr|accum16|clip_half)_seed42", path.name)
    ]
    run_dirs = formal_run_dirs or all_run_dirs
    if not run_dirs:
        raise ValueError(f"no completed RL run directories in {args.experiment_root}")
    reports: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        report = audit_run(run_dir)
        reports.append(report)
        (args.output_dir / f"{run_dir.name}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    comparisons = compare_conditions(reports)
    result = {
        "status": "PASS" if all(report["status"] != "FAILED" for report in reports) else "FAILED",
        "experiment_root": str(args.experiment_root),
        "run_count": len(reports),
        "runs": reports,
        "comparisons": comparisons,
        "stability_improved_conditions": [
            f"{item['mode']}_{item['condition']}"
            for item in comparisons if item["stability_improved"]
        ],
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        "warning_policy": "diagnostic only; no warning changes checkpoint eligibility",
        "thresholds": {
            "kl_threshold": KL_THRESHOLD,
            "kl_delay_required_steps": 4,
            "quality_drop_points": QUALITY_DROP_POINTS,
        },
        "summary_statistics": {
            "mean_steps_completed": mean(report["steps_completed"] for report in reports),
            "all_json_finite": result_json_finite(reports),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def result_json_finite(value: Any) -> bool:
    """Recursively verify values that will be written to the audit summary."""
    if isinstance(value, dict):
        return all(result_json_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(result_json_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


if __name__ == "__main__":
    main()
