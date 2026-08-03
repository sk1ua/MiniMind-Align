"""Audit a full 32-row corrected-GRPO quality-evidence diagnostic offline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_CATEGORIES = (
    "conciseness",
    "format",
    "instruction",
    "reasoning",
    "repetition",
    "safety",
    "termination",
    "uncertainty",
)


def finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return True


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not finite_tree(value):
        raise ValueError(f"invalid or non-finite JSON: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not finite_tree(value):
            raise ValueError(f"invalid or non-finite JSONL row {path}:{line_number}")
        rows.append(value)
    return rows


def ensure_empty_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty audit directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def category_snapshot(metrics: dict, details: list[dict], expected_count: int) -> dict[str, object]:
    categories = metrics.get("categories", {})
    detail_counts: dict[str, int] = {}
    detail_ids: list[str] = []
    for row in details:
        category = row.get("category")
        if category:
            detail_counts[str(category)] = detail_counts.get(str(category), 0) + 1
        if row.get("id"):
            detail_ids.append(str(row["id"]))
    metric_counts = {
        category: (categories.get(category, {}) if isinstance(categories, dict) else {}).get("count")
        for category in EXPECTED_CATEGORIES
    }
    return {
        "metric_category_counts": metric_counts,
        "detail_category_counts": detail_counts,
        "expected_category_counts": {category: expected_count for category in EXPECTED_CATEGORIES},
        "balanced_metrics": all(metric_counts.get(category) == expected_count for category in EXPECTED_CATEGORIES),
        "balanced_details": all(detail_counts.get(category, 0) == expected_count for category in EXPECTED_CATEGORIES),
        "detail_count": len(details),
        "unique_detail_ids": len(detail_ids) == len(set(detail_ids)),
    }


def metric_rates(metrics: dict, details: list[dict], max_new_tokens: int) -> dict[str, object]:
    if details:
        natural_end = sum(bool(row.get("finished_naturally")) for row in details)
        max_length_hit = sum(
            int(row.get("generated_tokens", -1)) >= max_new_tokens and not bool(row.get("finished_naturally"))
            for row in details
        )
    else:
        natural_end = int(metrics.get("natural_end", 0) or 0)
        max_length_hit = int(metrics.get("max_length_hit", 0) or 0)
    repeat_values = [float(row.get("repeat_3gram_ratio", 0.0)) for row in details]
    count = len(details) or int(metrics.get("count", 0) or 0)
    return {
        "count": count,
        "validator_pass": int(metrics.get("validator_pass", 0) or 0),
        "validator_pass_rate": float(metrics.get("validator_pass_rate", 0.0) or 0.0),
        "safety_count": int(metrics.get("safety_count", 0) or 0),
        "safety_pass": int(metrics.get("safety_pass", 0) or 0),
        "termination_count": int(metrics.get("termination_count", 0) or 0),
        "termination_pass": int(metrics.get("termination_pass", 0) or 0),
        "natural_end": natural_end,
        "natural_end_rate": (natural_end / count if count else None),
        "max_length_hit": max_length_hit,
        "max_length_rate": (max_length_hit / count if count else None),
        "average_tokens": float(metrics.get("average_tokens", 0.0) or 0.0),
        "average_repeat_3gram": (
            sum(repeat_values) / len(repeat_values)
            if repeat_values
            else float(metrics.get("average_repeat_3gram", 0.0) or 0.0)
        ),
    }


def record_quality(
    record: dict,
    expected_total_count: int,
    expected_category_count: int,
    max_new_tokens: int,
) -> dict[str, object]:
    metrics = record.get("metrics", {})
    details = record.get("details", [])
    if not isinstance(metrics, dict) or not isinstance(details, list):
        return {"valid": False, "reason": "metrics_or_details_missing"}
    balance = category_snapshot(metrics, details, expected_category_count)
    rates = metric_rates(metrics, details, max_new_tokens)
    valid = (
        rates["count"] == expected_total_count
        and balance["balanced_metrics"] is True
        and balance["balanced_details"] is True
        and balance["unique_detail_ids"] is True
    )
    return {"valid": valid, "balance": balance, "rates": rates}


def audit(args: argparse.Namespace) -> dict[str, object]:
    experiment_root = Path(args.experiment_root)
    run_dir = experiment_root / args.run_name
    output_dir = Path(args.output_dir)
    ensure_empty_output_dir(output_dir)
    precision_path = Path(args.precision_summary) if args.precision_summary else None
    required = [
        experiment_root / "matrix.json",
        run_dir / "run_metadata.json",
        run_dir / "step_summaries.jsonl",
        run_dir / "validation_history.jsonl",
        run_dir / "selection.json",
        run_dir / "samples.jsonl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if precision_path is not None and not precision_path.exists():
        missing.append(str(precision_path))
    if missing:
        result = {
            "status": "QUALITY_EVIDENCE_TELEMETRY_INCOMPLETE",
            "missing": missing,
            "diagnostic_only": True,
            "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        }
        (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    try:
        matrix = load_json(experiment_root / "matrix.json")
        metadata = load_json(run_dir / "run_metadata.json")
        steps = load_jsonl(run_dir / "step_summaries.jsonl")
        validation = load_jsonl(run_dir / "validation_history.jsonl")
        selection = load_json(run_dir / "selection.json")
        samples = load_jsonl(run_dir / "samples.jsonl")
        precision = load_json(precision_path) if precision_path is not None else None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "status": "QUALITY_EVIDENCE_TELEMETRY_INCOMPLETE",
            "error": str(exc),
            "diagnostic_only": True,
            "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        }
        (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    expected_steps = int(matrix.get("max_steps", 4))
    expected_count = int(matrix.get("validation_max_prompts", 32))
    max_new_tokens = int(matrix.get("validation_max_new_tokens", matrix.get("max_gen_len", 128)))
    expected_category_count = expected_count // len(EXPECTED_CATEGORIES)
    step_numbers = sorted(int(row.get("step", -1)) for row in steps)
    accepted_steps = [
        int(row["step"])
        for row in steps
        if row.get("optimizer_step_applied") is True and row.get("post_step_kl_gate_passed") is True
    ]
    active_gate_ok = all(
        row.get("post_step_kl_gate_passed") is True
        and float(row.get("post_step_kl_gate_mean", math.inf)) <= float(matrix.get("post_step_kl_target", 0.005))
        for row in steps
    )
    precision_contract_ok = all(
        row.get("precision_contract_valid") is True
        and row.get("precision_contract_mode") == "no_autocast_v1"
        and row.get("active_loss_variant") == row.get("active_gate_variant")
        for row in [metadata, *steps]
    )
    checkpoint_records = [row for row in validation if row.get("checkpoint")]
    baseline_raw = selection.get("baseline", {})
    baseline_record = {"metrics": baseline_raw, "details": []}
    baseline_quality = record_quality(baseline_record, expected_count, expected_category_count, max_new_tokens)
    # The trainer stores baseline category metrics but not baseline detail rows in selection.json.
    # Validation history provides detail rows for reloaded checkpoints, so balance is checked there.
    baseline_quality["valid"] = (
        int(baseline_raw.get("count", 0) or 0) == expected_count
        and isinstance(baseline_raw.get("categories"), dict)
        and all(
            baseline_raw.get("categories", {}).get(category, {}).get("count") == expected_category_count
            for category in EXPECTED_CATEGORIES
        )
    )
    checkpoint_quality = []
    for row in checkpoint_records:
        quality = record_quality(row, expected_count, expected_category_count, max_new_tokens)
        quality.update({"step": row.get("step"), "checkpoint": row.get("checkpoint"), "evaluation_source": row.get("evaluation_source")})
        checkpoint_quality.append(quality)
    selected_checkpoint = selection.get("selected_checkpoint")
    selected = next((row for row in checkpoint_quality if row.get("checkpoint") == selected_checkpoint), None)
    quality_scope_complete = (
        baseline_quality["valid"]
        and len(checkpoint_quality) >= 1
        and all(row.get("valid") is True for row in checkpoint_quality)
        and selected is not None
    )
    baseline_rates = baseline_quality.get("rates", {})
    selected_rates = selected.get("rates", {}) if selected else {}
    def safe_rate(rates: dict, key: str) -> float | None:
        value = rates.get(key)
        return float(value) if value is not None else None

    baseline_safety_rate = (
        float(baseline_raw["safety_pass"]) / float(baseline_raw["safety_count"])
        if baseline_raw.get("safety_count")
        else None
    )
    selected_safety_rate = (
        float(selected_rates["safety_pass"]) / float(selected_rates["safety_count"])
        if selected and selected_rates.get("safety_count")
        else None
    )
    baseline_termination_rate = (
        float(baseline_raw["termination_pass"]) / float(baseline_raw["termination_count"])
        if baseline_raw.get("termination_count")
        else None
    )
    selected_termination_rate = (
        float(selected_rates["termination_pass"]) / float(selected_rates["termination_count"])
        if selected and selected_rates.get("termination_count")
        else None
    )
    delta = {
        "validator_pass": (selected_rates.get("validator_pass", 0) - baseline_raw.get("validator_pass", 0)) if selected else None,
        "safety_drop_points": (baseline_safety_rate - selected_safety_rate) * 100 if baseline_safety_rate is not None and selected_safety_rate is not None else None,
        "termination_drop_points": (baseline_termination_rate - selected_termination_rate) * 100 if baseline_termination_rate is not None and selected_termination_rate is not None else None,
        "natural_end_delta_points": (safe_rate(selected_rates, "natural_end_rate") - safe_rate(baseline_rates, "natural_end_rate")) * 100 if selected and safe_rate(selected_rates, "natural_end_rate") is not None and safe_rate(baseline_rates, "natural_end_rate") is not None else None,
        "max_length_delta_points": (safe_rate(selected_rates, "max_length_rate") - safe_rate(baseline_rates, "max_length_rate")) * 100 if selected and safe_rate(selected_rates, "max_length_rate") is not None and safe_rate(baseline_rates, "max_length_rate") is not None else None,
        "repeat_3gram_delta": (selected_rates.get("average_repeat_3gram", 0.0) - baseline_rates.get("average_repeat_3gram", 0.0)) if selected else None,
    }
    safety_guard_ok = delta["safety_drop_points"] is None or delta["safety_drop_points"] <= 10.0
    termination_guard_ok = delta["termination_drop_points"] is None or delta["termination_drop_points"] <= 10.0
    steps_complete = step_numbers == list(range(1, expected_steps + 1)) and accepted_steps == step_numbers
    sample_linkage_ok = bool(samples) and all(row.get("sample_key") or row.get("id") for row in samples)
    status = (
        "QUALITY_EVIDENCE_DIAGNOSTIC_COMPLETE"
        if steps_complete and active_gate_ok and precision_contract_ok and quality_scope_complete and safety_guard_ok and termination_guard_ok and sample_linkage_ok
        else "QUALITY_EVIDENCE_TELEMETRY_INCOMPLETE"
    )
    result = {
        "status": status,
        "experiment_root": str(experiment_root),
        "run_name": args.run_name,
        "diagnostic_only": True,
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        "default_model_changed": False,
        "expected_steps": expected_steps,
        "steps_completed": len(steps),
        "accepted_steps": accepted_steps,
        "active_gate_ok": active_gate_ok,
        "precision_contract_ok": precision_contract_ok,
        "quality_scope": {
            "validation_count": expected_count,
            "expected_category_count": expected_category_count,
            "baseline": baseline_quality,
            "checkpoints": checkpoint_quality,
            "selected_checkpoint": selected_checkpoint,
            "selected": selected,
            "complete": quality_scope_complete,
        },
        "directional_delta_vs_source_sft": delta,
        "guards": {
            "safety_ok": safety_guard_ok,
            "termination_ok": termination_guard_ok,
            "max_length_rate": selected_rates.get("max_length_rate") if selected else None,
            "natural_end_rate": selected_rates.get("natural_end_rate") if selected else None,
            "repeat_3gram": selected_rates.get("average_repeat_3gram") if selected else None,
        },
        "sample_count": len(samples),
        "sample_linkage_ok": sample_linkage_ok,
        "precision_audit_status": precision.get("status") if precision else None,
        "all_json_finite": True,
        "formal_rl_ready": False,
        "automatic_gpu_start": False,
        "next_decision": "This is directional single-seed quality evidence only; it does not authorize three-seed promotion or default-model replacement.",
    }
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        "\n".join([
            "# Corrected-GRPO quality evidence audit",
            "",
            f"- status: {status}",
            f"- accepted steps: {accepted_steps}",
            f"- validation scope: {expected_count} rows; {expected_category_count} per category",
            f"- selected checkpoint: {selected_checkpoint}",
            f"- directional validator delta vs source SFT: {delta['validator_pass']}",
            "",
            "Diagnostic only. The result is not an RL promotion or default-model decision.",
            "",
        ]),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--precision-summary")
    print(json.dumps(audit(parser.parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
