"""Audit RL training/validation divergence without changing checkpoint selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line, parse_constant=_reject_nonfinite))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _max_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if _finite(row.get(key))]
    return max(values) if values else None


def _step_metrics(step_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "step",
        "reward_mean",
        "train_validator_pass_rate",
        "train_empty_response_rate",
        "train_max_length_hit_rate",
        "train_natural_end_rate",
        "train_repetition_penalty_mean",
        "kl_mean",
    )
    return [{key: row.get(key) for key in fields} for row in step_rows]


def detect_warnings(
    step_rows: list[dict[str, Any]],
    baseline_metrics: dict[str, Any],
    selected_metrics: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return diagnostic flags; these flags never alter checkpoint eligibility."""
    if not step_rows:
        return [{"code": "missing_training_steps", "severity": "error"}]
    first = step_rows[0]
    warnings: list[dict[str, Any]] = []
    best_validation_pass = (
        float(selected_metrics["validator_pass"])
        if selected_metrics and _finite(selected_metrics.get("validator_pass"))
        else None
    )
    baseline_pass = float(baseline_metrics.get("validator_pass", 0))
    train_validator_gain = _max_metric(step_rows, "train_validator_pass_rate")
    if (
        train_validator_gain is not None
        and _finite(first.get("train_validator_pass_rate"))
        and train_validator_gain - float(first["train_validator_pass_rate"]) >= 0.10
        and best_validation_pass is not None
        and best_validation_pass <= baseline_pass
    ):
        warnings.append(
            {
                "code": "train_validator_gain_without_validation_gain",
                "severity": "warning",
                "train_gain": train_validator_gain - float(first["train_validator_pass_rate"]),
                "validation_best": best_validation_pass,
                "validation_baseline": baseline_pass,
            }
        )
    train_reward_gain = _max_metric(step_rows, "reward_mean")
    if (
        train_reward_gain is not None
        and _finite(first.get("reward_mean"))
        and train_reward_gain - float(first["reward_mean"]) >= 0.10
        and best_validation_pass is not None
        and best_validation_pass <= baseline_pass
    ):
        warnings.append(
            {
                "code": "reward_gain_without_validation_gain",
                "severity": "warning",
                "train_gain": train_reward_gain - float(first["reward_mean"]),
                "validation_best": best_validation_pass,
                "validation_baseline": baseline_pass,
            }
        )
    for key, code in (
        ("train_empty_response_rate", "empty_response_increase"),
        ("train_max_length_hit_rate", "max_length_hit_increase"),
    ):
        maximum = _max_metric(step_rows, key)
        if maximum is not None and _finite(first.get(key)) and maximum - float(first[key]) >= 0.10:
            warnings.append(
                {
                    "code": code,
                    "severity": "warning",
                    "increase": maximum - float(first[key]),
                }
            )
    maximum_repeat = _max_metric(step_rows, "train_repetition_penalty_mean")
    if maximum_repeat is not None and _finite(first.get("train_repetition_penalty_mean")):
        increase = maximum_repeat - float(first["train_repetition_penalty_mean"])
        if increase >= 0.02:
            warnings.append(
                {
                    "code": "repetition_penalty_increase",
                    "severity": "warning",
                    "increase": increase,
                }
            )
    if selected_metrics:
        for key, code in (
            ("natural_end_rate", "validation_natural_end_drop"),
            ("average_repeat_3gram", "validation_repetition_increase"),
        ):
            baseline_key = key
            current = selected_metrics.get(key)
            reference = baseline_metrics.get(baseline_key)
            if _finite(current) and _finite(reference):
                delta = float(current) - float(reference)
                threshold = -0.10 if key == "natural_end_rate" else 0.02
                if (key == "natural_end_rate" and delta <= threshold) or (
                    key == "average_repeat_3gram" and delta >= threshold
                ):
                    warnings.append(
                        {
                            "code": code,
                            "severity": "warning",
                            "delta": delta,
                        }
                    )
    return warnings


def selected_metrics_from_selection(selection: dict[str, Any]) -> dict[str, Any] | None:
    selected_step = selection.get("selected_step")
    if selected_step is None:
        return None
    for record in selection.get("checkpoints", []):
        if record.get("step") == selected_step:
            return record.get("metrics")
    return None


def audit_run(run_dir: Path) -> dict[str, Any]:
    selection = read_json(run_dir / "selection.json")
    baseline = read_json(run_dir / "baseline_validation.json")
    step_rows = read_jsonl(run_dir / "step_summaries.jsonl")
    validation_rows = read_jsonl(run_dir / "validation_history.jsonl")
    baseline_metrics = baseline["metrics"]
    selected_metrics = selected_metrics_from_selection(selection)
    warnings = detect_warnings(step_rows, baseline_metrics, selected_metrics)
    return {
        "run": run_dir.name,
        "mode": run_dir.name.split("_", 1)[0],
        "status": "WARNING" if any(item["severity"] == "warning" for item in warnings) else "PASS",
        "selection": {
            "status": selection.get("status"),
            "selected_step": selection.get("selected_step"),
            "selected_checkpoint": selection.get("selected_checkpoint"),
            "stop_reason": selection.get("stop_reason"),
        },
        "baseline": baseline_metrics,
        "selected_metrics": selected_metrics,
        "steps": _step_metrics(step_rows),
        "validation_steps": [
            {
                "step": row.get("step"),
                "validator_pass": row.get("metrics", {}).get("validator_pass"),
                "safety_pass": row.get("metrics", {}).get("safety_pass"),
                "termination_pass": row.get("metrics", {}).get("termination_pass"),
                "natural_end": row.get("metrics", {}).get("natural_end"),
                "average_tokens": row.get("metrics", {}).get("average_tokens"),
                "average_repeat_3gram": row.get("metrics", {}).get("average_repeat_3gram"),
                "quality_triggered": row.get("quality_triggered"),
                "kl_triggered": row.get("kl_triggered"),
            }
            for row in validation_rows
        ],
        "diagnostic_thresholds": {
            "train_gain_without_validation_gain_points": 10,
            "empty_or_max_length_increase_points": 10,
            "repetition_penalty_increase": 0.02,
            "validation_natural_end_drop_points": 10,
            "validation_repeat_increase": 0.02,
        },
        "warnings": warnings,
    }


def aggregate_gate(run_reports: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for report in run_reports:
        groups.setdefault(report["mode"], []).append(report)
    methods: dict[str, Any] = {}
    for mode, reports in sorted(groups.items()):
        values = [
            float((report["selected_metrics"] or report["baseline"])["validator_pass"])
            for report in reports
        ]
        baselines = [float(report["baseline"]["validator_pass"]) for report in reports]
        safety_values = [
            float((report["selected_metrics"] or report["baseline"])["safety_pass"])
            for report in reports
        ]
        termination_values = [
            float((report["selected_metrics"] or report["baseline"])["termination_pass"])
            for report in reports
        ]
        baseline_safety = [float(report["baseline"]["safety_pass"]) for report in reports]
        baseline_termination = [float(report["baseline"]["termination_pass"]) for report in reports]
        mean_pass = mean(values)
        baseline_mean = mean(baselines)
        qualified = (
            len(reports) == 3
            and mean_pass - baseline_mean >= 3.0
            and mean(safety_values) >= mean(baseline_safety)
            and mean(termination_values) >= mean(baseline_termination)
        )
        methods[mode] = {
            "seed_count": len(reports),
            "validator_pass_by_seed": values,
            "validator_mean": mean_pass,
            "validator_std_population": pstdev(values) if len(values) > 1 else 0.0,
            "baseline_validator_mean": baseline_mean,
            "delta_vs_baseline": mean_pass - baseline_mean,
            "safety_mean": mean(safety_values),
            "baseline_safety_mean": mean(baseline_safety),
            "termination_mean": mean(termination_values),
            "baseline_termination_mean": mean(baseline_termination),
            "qualified_for_model_change": qualified,
            "status": "IMPROVEMENT_GATE_MET" if qualified else "NOT_MET_NO_MODEL_CHANGE",
        }
    qualified_methods = [mode for mode, data in methods.items() if data["qualified_for_model_change"]]
    return {
        "methods": methods,
        "qualified_methods": qualified_methods,
        "status": "IMPROVEMENT_GATE_MET" if qualified_methods else "NOT_MET_NO_MODEL_CHANGE",
    }


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
        if path.name.startswith(("grpo_seed", "cispo_seed"))
    ]
    run_dirs = formal_run_dirs or all_run_dirs
    if not run_dirs:
        raise ValueError(f"no completed RL run directories in {args.experiment_root}")
    reports = []
    for run_dir in run_dirs:
        report = audit_run(run_dir)
        reports.append(report)
        (args.output_dir / f"{run_dir.name}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    result = {
        "status": "PASS" if all(report["status"] in {"PASS", "WARNING"} for report in reports) else "FAILED",
        "experiment_root": str(args.experiment_root),
        "run_count": len(reports),
        "runs": reports,
        "gate": aggregate_gate(reports),
        "warning_policy": "diagnostic only; no warning changes checkpoint eligibility",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
