"""Offline audit for post-update KL trust-region guard diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    json.dumps(value, allow_nan=False)
    return value


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                json.dumps(row, allow_nan=False)
                rows.append(row)
    return rows


def numeric(value, default: float = 0.0) -> float:
    return default if value is None else float(value)


def write_report(output_dir: Path, summary: dict) -> None:
    run = summary["run"]
    lines = [
        "# RL KL guard audit",
        "",
        f"- status: `{summary['status']}`",
        f"- run: `{run['name']}`",
        f"- steps: `{run['steps']}`; accepted optimizer steps: `{run['accepted_optimizer_steps']}`",
        f"- guard target: `{run['target']}`",
        f"- guard attempts/backoffs: `{run['guard_attempts']}` / `{run['backoff_count']}`",
        f"- baseline/selected validation: `{run['baseline_validator_pass']}` / `{run['selected_validator_pass']}`",
        "",
        "This audit is diagnostic-only. It does not promote a checkpoint or change the default model.",
        "",
    ]
    if summary["warnings"]:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(args: argparse.Namespace) -> dict:
    root = Path(args.experiment_root)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty audit directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = root / args.run_name
    required = [
        root / "matrix.json",
        run_dir / "baseline_validation.json",
        run_dir / "selection.json",
        run_dir / "step_summaries.jsonl",
        run_dir / "microbatch_summaries.jsonl",
        run_dir / "samples.jsonl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        summary = {
            "status": "TELEMETRY_INCOMPLETE",
            "experiment_root": str(root),
            "output_dir": str(output_dir),
            "run": {
                "name": args.run_name,
                "steps": 0,
                "accepted_optimizer_steps": 0,
                "target": None,
                "guard_attempts": 0,
                "backoff_count": 0,
                "baseline_validator_pass": None,
                "selected_validator_pass": None,
            },
            "missing": missing,
            "diagnostic_only": True,
            "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        write_report(output_dir, {**summary, "warnings": ["required artifact missing"]})
        return summary

    warnings: list[str] = []
    matrix = load_json(required[0])
    baseline = load_json(required[1])
    selection = load_json(required[2])
    steps = load_jsonl(required[3])
    validation_path = run_dir / "validation_history.jsonl"
    validation = load_jsonl(validation_path) if validation_path.exists() else []
    if not validation and selection.get("status") != "baseline_retained":
        warnings.append("validation history is absent for a run without baseline-retained selection")
    microbatches = load_jsonl(required[4])
    samples = load_jsonl(required[5])
    target = float(matrix["guard"]["post_step_kl_target"])
    guard_enabled = all(bool(step.get("kl_guard_enabled")) for step in steps)
    required_step_fields = {
        "post_step_kl_target",
        "post_step_kl_mean",
        "post_step_kl_p95",
        "post_step_kl_max",
        "kl_guard_enabled",
        "kl_guard_attempts",
        "kl_guard_backoff_count",
        "optimizer_step_applied",
        "optimizer_step_rejected",
    }
    schema_complete = all(required_step_fields.issubset(step) for step in steps)
    if not schema_complete:
        warnings.append("one or more step summaries lack KL guard fields")
    accepted = [step for step in steps if step.get("optimizer_step_applied") is True]
    rejected = [step for step in steps if step.get("optimizer_step_rejected") is True]
    accepted_over_target = [
        step for step in accepted
        if step.get("post_step_kl_mean") is None or float(step["post_step_kl_mean"]) > target
    ]
    guard_attempts = sum(int(step.get("kl_guard_attempts", 0)) for step in steps)
    backoff_count = sum(int(step.get("kl_guard_backoff_count", 0)) for step in steps)
    all_sample_keys = {row.get("sample_key") for row in samples}
    expected_sample_keys = {
        key
        for micro in microbatches
        for key in micro.get("sample_keys", [])
    }
    linkage_ok = all_sample_keys == expected_sample_keys and None not in all_sample_keys
    if not linkage_ok:
        warnings.append("micro-batch and sample key sets do not match")
    selected_step = selection.get("selected_step")
    selected_record = next((record for record in validation if record.get("step") == selected_step), None)
    baseline_metrics = baseline.get("metrics", {})
    selected_metrics = selected_record.get("metrics", {}) if selected_record else baseline_metrics
    baseline_validator = int(baseline_metrics.get("validator_pass", 0))
    selected_validator = int(selected_metrics.get("validator_pass", baseline_validator))
    safety_drop = numeric(baseline_metrics.get("safety_pass_rate")) - numeric(selected_metrics.get("safety_pass_rate"))
    termination_drop = numeric(baseline_metrics.get("termination_pass_rate")) - numeric(selected_metrics.get("termination_pass_rate"))
    if safety_drop > 0.10 or termination_drop > 0.10:
        warnings.append("selected validation safety/termination quality drop exceeded 10 percentage points")

    if not steps or not schema_complete or not linkage_ok:
        status = "TELEMETRY_INCOMPLETE"
    elif rejected:
        status = "GUARD_UNRESOLVED_BASELINE_RETAINED" if selection.get("status") == "baseline_retained" else "TELEMETRY_INCOMPLETE"
    elif accepted_over_target:
        status = "TELEMETRY_INCOMPLETE"
        warnings.append("an accepted optimizer step exceeded the post-update KL target")
    elif backoff_count > 0:
        status = "GUARD_EFFECTIVE_NO_MODEL_CHANGE"
    else:
        status = "GUARD_NOT_TRIGGERED_DIAGNOSTIC"

    reference = None
    if args.reference_root:
        reference_root = Path(args.reference_root)
        reference_steps = sorted(reference_root.rglob("step_summaries.jsonl"))
        if reference_steps:
            reference_rows = load_jsonl(reference_steps[-1])
            reference = {
                "path": str(reference_steps[-1]),
                "max_kl_mean": max((float(row.get("kl_mean", 0.0)) for row in reference_rows), default=0.0),
                "max_kl_p95": max((float(row.get("kl_p95", 0.0)) for row in reference_rows), default=0.0),
                "max_kl": max((float(row.get("kl_max", 0.0)) for row in reference_rows), default=0.0),
            }

    summary = {
        "status": status,
        "experiment_root": str(root),
        "output_dir": str(output_dir),
        "run": {
            "name": args.run_name,
            "steps": len(steps),
            "accepted_optimizer_steps": len(accepted),
            "rejected_optimizer_steps": len(rejected),
            "target": target,
            "guard_enabled": guard_enabled,
            "guard_attempts": guard_attempts,
            "backoff_count": backoff_count,
            "accepted_post_step_kl_mean_max": max((float(step["post_step_kl_mean"]) for step in accepted), default=0.0),
            "accepted_post_step_kl_p95_max": max((float(step["post_step_kl_p95"]) for step in accepted), default=0.0),
            "accepted_post_step_kl_max_max": max((float(step["post_step_kl_max"]) for step in accepted), default=0.0),
            "baseline_validator_pass": baseline_validator,
            "selected_validator_pass": selected_validator,
            "safety_drop": safety_drop,
            "termination_drop": termination_drop,
        },
        "checkpoint_selection": selection,
        "validation_records": len(validation),
        "microbatch_count": len(microbatches),
        "sample_count": len(samples),
        "sample_linkage_complete": linkage_ok,
        "reference_control": reference,
        "warnings": warnings,
        "all_json_finite": True,
        "diagnostic_only": True,
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "run_reports.jsonl").write_text(json.dumps(summary, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(output_dir, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--reference-root")
    args = parser.parse_args()
    summary = audit(args)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
