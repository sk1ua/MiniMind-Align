"""Offline audit for KL guard attempt, dtype, update, and rollback telemetry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from align.rl_rules import dtype_gap_threshold
def load_json(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    json.dumps(value, allow_nan=False)
    return value


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            json.dumps(value, allow_nan=False)
            rows.append(value)
    return rows


def classify_attempt(attempt: dict, target: float) -> dict:
    bf16 = attempt["post_step_kl_bfloat16"]
    fp32 = attempt["post_step_kl_float32"]
    gap = abs(float(bf16["reference_kl_mean"]) - float(fp32["reference_kl_mean"]))
    threshold = dtype_gap_threshold(float(fp32["reference_kl_mean"]), target)
    enriched = dict(attempt)
    enriched.update({
        "bfloat16_gate_passed": float(bf16["reference_kl_mean"]) <= target,
        "float32_gate_passed": float(fp32["reference_kl_mean"]) <= target,
        "dtype_gap_mean_recomputed": gap,
        "dtype_gap_threshold_recomputed": threshold,
        "dtype_sensitive_recomputed": gap > threshold,
    })
    return enriched


def audit(args: argparse.Namespace) -> dict:
    root = Path(args.experiment_root)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty audit directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = root / args.run_name
    required = {
        "matrix": root / "matrix.json",
        "baseline": run_dir / "baseline_validation.json",
        "selection": run_dir / "selection.json",
        "steps": run_dir / "step_summaries.jsonl",
        "attempts": run_dir / "kl_guard_attempts.jsonl",
        "microbatches": run_dir / "microbatch_summaries.jsonl",
        "samples": run_dir / "samples.jsonl",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        summary = {
            "status": "TELEMETRY_INCOMPLETE",
            "experiment_root": str(root),
            "output_dir": str(output_dir),
            "run": {"name": args.run_name, "steps": 0, "attempts": 0},
            "missing": missing,
            "diagnostic_only": True,
            "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    matrix = load_json(required["matrix"])
    baseline = load_json(required["baseline"])
    selection = load_json(required["selection"])
    steps = load_jsonl(required["steps"])
    attempts = load_jsonl(required["attempts"])
    microbatches = load_jsonl(required["microbatches"])
    samples = load_jsonl(required["samples"])
    warnings: list[str] = []
    target = float(matrix["guard"]["post_step_kl_target"])
    attempt_rows = [row for row in attempts if row.get("record_type") == "kl_guard_attempt"]
    required_attempt_fields = {
        "step",
        "attempt_index",
        "lr_multiplier",
        "post_step_kl_bfloat16",
        "post_step_kl_float32",
        "parameter_delta",
        "pre_attempt_policy_state_digest",
        "pre_attempt_optimizer_state_digest",
        "post_attempt_policy_state_digest",
        "post_attempt_optimizer_state_digest",
        "rollback_after_attempt",
    }
    schema_complete = all(required_attempt_fields.issubset(row) for row in attempt_rows)
    if not schema_complete:
        warnings.append("one or more attempt records lack required telemetry fields")
    if not attempt_rows:
        warnings.append("no KL guard attempts were recorded")

    enriched_attempts = []
    if schema_complete:
        for attempt in attempt_rows:
            enriched_attempts.append(classify_attempt(attempt, target))
    dtype_sensitive_count = sum(bool(row.get("dtype_sensitive_recomputed")) for row in enriched_attempts)
    gate_disagreement = any(
        row.get("bfloat16_gate_passed") != row.get("float32_gate_passed")
        for row in enriched_attempts
    )
    both_over_target_count = sum(
        not row.get("bfloat16_gate_passed", True) and not row.get("float32_gate_passed", True)
        for row in enriched_attempts
    )

    all_sample_keys = {row.get("sample_key") for row in samples}
    expected_sample_keys = {key for row in microbatches for key in row.get("sample_keys", [])}
    linkage_ok = all_sample_keys == expected_sample_keys and None not in all_sample_keys
    if not linkage_ok:
        warnings.append("micro-batch and sample key sets do not match")

    rejected_steps = [row for row in steps if row.get("optimizer_step_rejected") is True]
    rollback_ok = all(
        row.get("rollback_exact_match") is True
        and row.get("rollback_verification_failed") is False
        and (row.get("rollback_parameter_delta") or {}).get("parameter_delta_max_abs", 0.0) == 0.0
        for row in rejected_steps
    )
    if rejected_steps and not rollback_ok:
        warnings.append("one or more rejected steps lack exact rollback evidence")
    if any(row.get("optimizer_step_rejected") is True for row in steps):
        if selection.get("status") != "baseline_retained" or selection.get("checkpoints"):
            warnings.append("rejected run did not retain baseline without checkpoints")

    if not schema_complete or not attempt_rows or not linkage_ok or (rejected_steps and not rollback_ok):
        status = "TELEMETRY_INCOMPLETE"
    elif dtype_sensitive_count >= 2 and gate_disagreement:
        status = "BF16_MEASUREMENT_SENSITIVE"
    elif both_over_target_count >= 2 and dtype_sensitive_count < 2:
        status = "REAL_UPDATE_SENSITIVITY"
    else:
        status = "MIXED_UNRESOLVED"

    baseline_metrics = baseline.get("metrics", {})
    selected_metrics = baseline_metrics
    baseline_validator = int(baseline_metrics.get("validator_pass", 0))
    selected_validator = int(selected_metrics.get("validator_pass", baseline_validator))
    summary = {
        "status": status,
        "experiment_root": str(root),
        "output_dir": str(output_dir),
        "run": {
            "name": args.run_name,
            "steps": len(steps),
            "attempts": len(enriched_attempts),
            "target": target,
            "dtype_sensitive_attempts": dtype_sensitive_count,
            "gate_disagreement": gate_disagreement,
            "both_dtype_over_target_attempts": both_over_target_count,
            "rejected_steps": len(rejected_steps),
            "accepted_steps": sum(row.get("optimizer_step_applied") is True for row in steps),
            "baseline_validator_pass": baseline_validator,
            "selected_validator_pass": selected_validator,
        },
        "attempts": enriched_attempts,
        "checkpoint_selection": selection,
        "rollback": {
            "verified": rollback_ok,
            "rejected_steps": len(rejected_steps),
        },
        "sample_linkage_complete": linkage_ok,
        "all_json_finite": True,
        "warnings": warnings,
        "diagnostic_only": True,
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "attempt_analysis.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in enriched_attempts),
        encoding="utf-8",
    )
    lines = [
        "# KL guard telemetry audit",
        "",
        f"- status: `{status}`",
        f"- run: `{args.run_name}`",
        f"- attempts: `{len(enriched_attempts)}`",
        f"- dtype-sensitive attempts: `{dtype_sensitive_count}`",
        f"- gate disagreement: `{gate_disagreement}`",
        f"- rollback verified: `{rollback_ok}`",
        "",
        "This audit is diagnostic-only and cannot promote a checkpoint or change the default model.",
    ]
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
