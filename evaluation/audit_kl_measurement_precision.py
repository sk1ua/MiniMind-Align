"""Audit bfloat16, no-autocast, and true-float32 KL measurements."""

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
    no_autocast = attempt["post_step_kl_float32"]
    full_fp32 = attempt["post_step_kl_full_float32"]
    bf16_mean = float(bf16["reference_kl_mean"])
    no_autocast_mean = float(no_autocast["reference_kl_mean"])
    full_fp32_mean = float(full_fp32["reference_kl_mean"])
    threshold = dtype_gap_threshold(full_fp32_mean, target)
    return {
        "step": attempt.get("step"),
        "attempt_index": attempt.get("attempt_index"),
        "lr_multiplier": attempt.get("lr_multiplier"),
        "bfloat16_mean": bf16_mean,
        "no_autocast_bfloat16_weights_mean": no_autocast_mean,
        "full_float32_copy_mean": full_fp32_mean,
        "bfloat16_gate_passed": bf16_mean <= target,
        "no_autocast_gate_passed": no_autocast_mean <= target,
        "full_float32_gate_passed": full_fp32_mean <= target,
        "bfloat16_full_fp32_gap": abs(bf16_mean - full_fp32_mean),
        "no_autocast_full_fp32_gap": abs(no_autocast_mean - full_fp32_mean),
        "gap_threshold": threshold,
        "bfloat16_sensitive": abs(bf16_mean - full_fp32_mean) > threshold,
        "no_autocast_matches_full_fp32": abs(no_autocast_mean - full_fp32_mean) <= threshold,
        "parameter_delta": attempt.get("parameter_delta"),
        "rollback_exact_match": (attempt.get("rollback_after_attempt") or {}).get("exact_match"),
    }


def audit(args: argparse.Namespace) -> dict:
    root = Path(args.experiment_root)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty audit directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = root / args.run_name
    matrix_path = root / "matrix.json"
    attempts_path = run_dir / "kl_guard_attempts.jsonl"
    required = [matrix_path, attempts_path, run_dir / "selection.json", run_dir / "samples.jsonl"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        summary = {"status": "TELEMETRY_INCOMPLETE", "missing": missing, "diagnostic_only": True}
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    matrix = load_json(matrix_path)
    attempts = [row for row in load_jsonl(attempts_path) if row.get("record_type") == "kl_guard_attempt"]
    target = float(matrix["guard"]["post_step_kl_target"])
    required_fields = {"post_step_kl_bfloat16", "post_step_kl_float32", "post_step_kl_full_float32", "parameter_delta", "rollback_after_attempt"}
    schema_complete = bool(attempts) and all(required_fields.issubset(row) for row in attempts)
    rows = [classify_attempt(row, target) for row in attempts] if schema_complete else []
    sensitive_count = sum(row["bfloat16_sensitive"] for row in rows)
    no_autocast_matches = sum(row["no_autocast_matches_full_fp32"] for row in rows)
    full_passes = sum(row["full_float32_gate_passed"] for row in rows)
    bf16_rejects = sum(not row["bfloat16_gate_passed"] for row in rows)
    gate_disagreement = sum(row["bfloat16_gate_passed"] != row["full_float32_gate_passed"] for row in rows)
    rollback_ok = all(row["rollback_exact_match"] is True for row in rows)

    if not schema_complete or not rollback_ok:
        status = "TELEMETRY_INCOMPLETE"
    elif sensitive_count >= 2 and full_passes >= 2 and no_autocast_matches >= 2 and bf16_rejects >= 2:
        status = "BF16_AUTOCAST_SENSITIVE"
    elif full_passes >= 2 and no_autocast_matches < 2:
        status = "BF16_WEIGHT_SENSITIVE"
    elif len(rows) >= 2 and full_passes < 2:
        status = "FULL_FP32_KL_OVER_TARGET"
    else:
        status = "MEASUREMENT_VARIANT_UNRESOLVED"

    summary = {
        "status": status,
        "experiment_root": str(root),
        "output_dir": str(output_dir),
        "target": target,
        "measurement_variants": {
            "bfloat16": "policy bfloat16 weights + bfloat16 autocast",
            "float32": "policy bfloat16 weights + autocast disabled",
            "full_float32": "detached policy copy converted to float32 + autocast disabled",
        },
        "attempts": len(rows),
        "bfloat16_sensitive_attempts": sensitive_count,
        "no_autocast_matches_full_fp32_attempts": no_autocast_matches,
        "full_float32_gate_pass_attempts": full_passes,
        "bfloat16_gate_reject_attempts": bf16_rejects,
        "full_fp32_gate_disagreement_attempts": gate_disagreement,
        "rollback_verified": rollback_ok,
        "attempt_analysis": rows,
        "diagnostic_only": True,
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        "next_decision": "validate_or_correct_KL_measurement_before_optimizer_update_scale_change",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "attempt_analysis.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        "\n".join([
            "# KL measurement precision audit",
            "",
            f"- status: `{status}`",
            f"- attempts: `{len(rows)}`",
            f"- bfloat16-sensitive attempts: `{sensitive_count}`",
            f"- full-float32 gate disagreements: `{gate_disagreement}`",
            f"- rollback verified: `{rollback_ok}`",
            "",
            "The audit is diagnostic-only. It does not change the production bfloat16 gate, reward, checkpoint selection, optimizer, or default model.",
            "",
        ]),
        encoding="utf-8",
    )
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
