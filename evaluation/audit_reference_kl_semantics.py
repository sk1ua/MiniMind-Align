"""Offline contract audit for the reference-KL measurement path.

This audit deliberately does not load a model or execute CUDA. It validates
the persisted guard telemetry, the mathematical KL contract, and the source
path that produces the three E019 measurement variants. Because historical
telemetry stores aggregate statistics rather than token-level log-probability
vectors, the report distinguishes semantic validation from true token replay.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(key) and finite_tree(item) for key, item in value.items())
    return True


def load_json(path: Path) -> object:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not finite_tree(value):
        raise ValueError(f"non-finite JSON value: {path}")
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


def independent_reference_kl_values(
    ref_log_probs: list[float],
    new_log_probs: list[float],
    completion_mask: list[int | bool],
) -> list[float]:
    """Compute the scalar reference-KL surrogate without importing torch."""
    if not (len(ref_log_probs) == len(new_log_probs) == len(completion_mask)):
        raise ValueError("reference, policy, and completion-mask lengths must match")
    values = []
    for ref, new, mask in zip(ref_log_probs, new_log_probs, completion_mask):
        if not bool(mask):
            continue
        delta = float(ref) - float(new)
        values.append(math.exp(delta) - delta - 1.0)
    return values


def linear_quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def independent_aggregate(values: list[list[float]]) -> dict[str, float]:
    merged = [float(item) for row in values for item in row]
    if not merged:
        return {
            "reference_kl_mean": 0.0,
            "reference_kl_p95": 0.0,
            "reference_kl_max": 0.0,
            "token_count": 0.0,
        }
    return {
        "reference_kl_mean": sum(merged) / len(merged),
        "reference_kl_p95": linear_quantile(merged, 0.95),
        "reference_kl_max": max(merged),
        "token_count": float(len(merged)),
    }


def analytic_fixtures() -> dict[str, bool]:
    log_two = math.log(2.0)
    masked = independent_reference_kl_values(
        [log_two, 100.0, 0.0],
        [0.0, 0.0, 0.0],
        [1, 0, 1],
    )
    expected = 2.0 - log_two - 1.0
    aggregate = independent_aggregate([[0.0, 2.0], [4.0]])
    return {
        "zero_delta_is_zero": independent_reference_kl_values([0.0], [0.0], [1]) == [0.0],
        "completion_mask_excludes_padding": len(masked) == 2 and math.isclose(masked[0], expected, abs_tol=1e-12),
        "aggregate_is_token_weighted": math.isclose(aggregate["reference_kl_mean"], 2.0, abs_tol=1e-12),
        "aggregate_p95_is_linear": math.isclose(aggregate["reference_kl_p95"], 3.8, abs_tol=1e-12),
    }


def source_contract_checks(trainer_path: Path, rules_path: Path) -> dict[str, bool]:
    trainer = trainer_path.read_text(encoding="utf-8")
    rules = rules_path.read_text(encoding="utf-8")
    checks = {
        "formula_reference_minus_new": "ref_log_probs - new_log_probs" in rules,
        "formula_exp_minus_delta_minus_one": "torch.exp((ref_log_probs - new_log_probs).float())" in rules,
        "completion_mask_applied_before_aggregation": "values[completion_mask.bool()]" in rules,
        "all_microbatch_tokens_aggregated": "torch.cat([value.float().reshape(-1) for value in values])" in rules,
        "reference_log_probs_computed_from_reference_model": "ref_log_probs = completion_log_probs(reference" in trainer,
        "reference_log_probs_detached_into_guard_rollouts": "ref_log_probs.detach().cpu()" in trainer,
        "production_variant_uses_guard_rollouts": "post_update_kl_stats(" in trainer and "guard_rollouts" in trainer,
        "full_fp32_variant_is_optional": "--post-step-kl-diagnostic-full-fp32" in trainer,
        "production_gate_reads_production_variant": 'post_step_kl["reference_kl_mean"] <= args.post_step_kl_target' in trainer,
    }
    checks["all_source_checks_pass"] = all(checks.values())
    return checks


def _close(left: float, right: float, *, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol)


def check_attempt(attempt: dict, target: float) -> dict[str, object]:
    bf16 = attempt["post_step_kl_bfloat16"]
    no_autocast = attempt["post_step_kl_float32"]
    full_fp32 = attempt["post_step_kl_full_float32"]
    bf16_mean = float(bf16["reference_kl_mean"])
    no_autocast_mean = float(no_autocast["reference_kl_mean"])
    full_fp32_mean = float(full_fp32["reference_kl_mean"])
    production_gate_expected = bf16_mean <= target
    threshold = max(1e-4, 0.1 * max(full_fp32_mean, target))
    return {
        "step": attempt.get("step"),
        "attempt_index": attempt.get("attempt_index"),
        "lr_multiplier": attempt.get("lr_multiplier"),
        "bfloat16_mean": bf16_mean,
        "no_autocast_bfloat16_weights_mean": no_autocast_mean,
        "full_float32_copy_mean": full_fp32_mean,
        "production_gate_expected": production_gate_expected,
        "production_gate_recorded": attempt.get("production_gate_passed"),
        "production_gate_matches": attempt.get("production_gate_passed") == production_gate_expected,
        "no_autocast_matches_full_fp32": _close(no_autocast_mean, full_fp32_mean),
        "full_fp32_gate_passed": full_fp32_mean <= target,
        "bfloat16_gate_passed": production_gate_expected,
        "gap_to_full_fp32": abs(bf16_mean - full_fp32_mean),
        "gap_threshold": threshold,
        "bfloat16_autocast_sensitive": abs(bf16_mean - full_fp32_mean) > threshold,
        "rollback_exact_match": (attempt.get("rollback_after_attempt") or {}).get("exact_match") is True,
        "parameter_delta_l2": (attempt.get("parameter_delta") or {}).get("parameter_delta_l2"),
    }


def ensure_empty_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty audit directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def audit(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.experiment_root)
    output_dir = Path(args.output_dir)
    ensure_empty_output_dir(output_dir)
    run_dir = root / args.run_name
    required = [
        root / "matrix.json",
        run_dir / "kl_guard_attempts.jsonl",
        run_dir / "step_summaries.jsonl",
        run_dir / "selection.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        summary = {
            "status": "TELEMETRY_INCOMPLETE",
            "missing": missing,
            "diagnostic_only": True,
            "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
            "gpu_wall_seconds": 0,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    matrix = load_json(root / "matrix.json")
    attempts = load_jsonl(run_dir / "kl_guard_attempts.jsonl")
    summaries = load_jsonl(run_dir / "step_summaries.jsonl")
    target = float(matrix.get("guard", {}).get("post_step_kl_target", 0.005))
    guard = matrix.get("guard", {})
    factor = float(guard.get("backoff_factor", guard.get("kl_backoff_factor", 0.5)))
    max_backoffs = int(guard.get("max_backoffs", 3))
    expected_multipliers = [factor**index for index in range(max_backoffs + 1)]
    required_attempt_fields = {
        "post_step_kl_bfloat16",
        "post_step_kl_float32",
        "post_step_kl_full_float32",
        "rollback_after_attempt",
    }
    schema_complete = bool(attempts) and all(required_attempt_fields.issubset(row) for row in attempts)
    rows = [check_attempt(row, target) for row in attempts] if schema_complete else []
    multipliers_match = (
        len(rows) == len(expected_multipliers)
        and all(_close(float(row["lr_multiplier"]), expected) for row, expected in zip(rows, expected_multipliers))
    )
    indices_match = [row.get("attempt_index") for row in attempts] == list(range(len(attempts)))
    gate_matches = all(bool(row["production_gate_matches"]) for row in rows)
    rollback_verified = bool(rows) and all(bool(row["rollback_exact_match"]) for row in rows)
    no_autocast_matches = sum(bool(row["no_autocast_matches_full_fp32"]) for row in rows)
    full_fp32_passes = sum(bool(row["full_fp32_gate_passed"]) for row in rows)
    bf16_rejects = sum(not bool(row["bfloat16_gate_passed"]) for row in rows)
    raw_token_level_replay_available = any(
        any(key in attempt for key in ("token_values", "rollout_log_probs", "completion_mask_values"))
        for attempt in attempts
    )
    summary_alignment = True
    by_step: dict[int, list[dict]] = {}
    for attempt in attempts:
        by_step.setdefault(int(attempt["step"]), []).append(attempt)
    for step_summary in summaries:
        step = int(step_summary["step"])
        step_attempts = by_step.get(step, [])
        if not step_attempts:
            summary_alignment = False
            continue
        last = step_attempts[-1]
        summary_alignment &= _close(
            float(step_summary["post_step_kl_mean"]),
            float(last["post_step_kl_bfloat16"]["reference_kl_mean"]),
        )
        summary_alignment &= _close(
            float(step_summary["post_step_kl_full_float32_mean"]),
            float(last["post_step_kl_full_float32"]["reference_kl_mean"]),
        )
    fixtures = analytic_fixtures()
    source_checks = source_contract_checks(Path(args.trainer_path), Path(args.rules_path))
    telemetry_checks = {
        "schema_complete": schema_complete,
        "backoff_multipliers_match": multipliers_match,
        "attempt_indices_match": indices_match,
        "production_gate_matches_formula": gate_matches,
        "rollback_verified": rollback_verified,
        "step_summary_alignment": summary_alignment,
        "all_json_finite": True,
        "raw_token_level_replay_available": raw_token_level_replay_available,
    }
    if not schema_complete or not source_checks["all_source_checks_pass"] or not gate_matches or not rollback_verified:
        status = "REFERENCE_KL_SEMANTICS_INCONSISTENT"
    elif not multipliers_match or not indices_match or not summary_alignment:
        status = "TELEMETRY_INCOMPLETE"
    elif not raw_token_level_replay_available:
        status = "REFERENCE_KL_SEMANTICS_CONSISTENT_LIMITED"
    else:
        status = "REFERENCE_KL_SEMANTICS_REPLAYED"
    result = {
        "status": status,
        "experiment_root": str(root),
        "run_name": args.run_name,
        "target": target,
        "formula": "exp(ref_log_prob - new_log_prob) - (ref_log_prob - new_log_prob) - 1",
        "mask_scope": "completion_mask true tokens only",
        "aggregation_scope": "all valid tokens concatenated across all rollouts and micro-batches",
        "source_checks": source_checks,
        "analytic_fixtures": fixtures,
        "telemetry_checks": telemetry_checks,
        "attempts": len(rows),
        "expected_backoff_multipliers": expected_multipliers,
        "bfloat16_reject_attempts": bf16_rejects,
        "full_fp32_pass_attempts": full_fp32_passes,
        "no_autocast_matches_full_fp32_attempts": no_autocast_matches,
        "rollback_verified": rollback_verified,
        "diagnostic_only": True,
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        "gpu_wall_seconds": 0,
        "limitation": (
            "Historical E019 telemetry stores aggregate KL statistics, not token-level "
            "log-probabilities and masks; this validates the contract and persisted "
            "variant alignment, but is not token-level replay."
        ),
        "next_decision": (
            "If a corrected smoke is authorized, measure guard KL with autocast disabled "
            "and retain token-level replay fields; do not change optimizer/update-scale yet."
        ),
        "attempt_analysis": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "attempt_analysis.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_dir / "contract.json").write_text(
        json.dumps(
            {
                "formula": result["formula"],
                "mask_scope": result["mask_scope"],
                "aggregation_scope": result["aggregation_scope"],
                "analytic_fixtures": fixtures,
                "source_checks": source_checks,
                "token_level_replay": raw_token_level_replay_available,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# Reference-KL semantics audit",
                "",
                f"- status: {status}",
                f"- attempts: {len(rows)}",
                f"- bfloat16 gate rejects: {bf16_rejects}",
                f"- full-float32 gate passes: {full_fp32_passes}",
                f"- no-autocast/full-float32 matches: {no_autocast_matches}",
                f"- rollback verified: {rollback_verified}",
                f"- token-level replay available: {raw_token_level_replay_available}",
                "",
                "The audit is offline and diagnostic-only. It does not change the production gate, reward, optimizer, checkpoint selection, or default model.",
                "",
                "The persisted artifact does not contain token-level log-probabilities and masks, so the result validates the formula, mask/aggregation contract, source-path alignment, and summary consistency; it does not claim token-level replay.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--trainer-path", default="trainer/train_grpo_lite.py")
    parser.add_argument("--rules-path", default="align/rl_rules.py")
    args = parser.parse_args()
    print(json.dumps(audit(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
