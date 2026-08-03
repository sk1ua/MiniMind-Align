"""Audit an opt-in fp32/no-autocast KL-gate smoke without changing model gates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.audit_kl_token_replay import replay_row_check, replay_variant_groups


EXPECTED_VARIANTS = {
    "bfloat16_autocast",
    "bfloat16_no_autocast",
    "full_float32_no_autocast",
}


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
    rows = []
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


def close(left: object, right: object, tolerance: float = 5e-7) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def validate_attempts(attempts: list[dict], target: float) -> dict[str, object]:
    checks = []
    accepted_by_step: dict[int, dict] = {}
    rejected_rollback_ok = True
    gate_consistent = True
    accepted_state_ok = True
    for attempt in attempts:
        mode = attempt.get("post_step_kl_gate_mode")
        gate = attempt.get("post_step_kl_gate") or {}
        selected = (
            attempt.get("post_step_kl_float32")
            if mode == "fp32_no_autocast"
            else attempt.get("post_step_kl_bfloat16")
        ) or {}
        gate_matches_selected = all(
            close(gate.get(key), selected.get(key))
            for key in ("reference_kl_mean", "reference_kl_p95", "reference_kl_max", "token_count")
        )
        expected_pass = bool(
            gate_matches_selected
            and float(gate.get("reference_kl_mean", float("inf"))) <= target
        )
        pass_fields_match = (
            attempt.get("post_step_kl_gate_passed") is expected_pass
            and attempt.get("production_gate_passed") is expected_pass
            and attempt.get("accepted") is expected_pass
        )
        delta = attempt.get("parameter_delta") or {}
        accepted = attempt.get("accepted") is True
        rollback = attempt.get("rollback_after_attempt")
        if accepted:
            accepted_by_step[int(attempt["step"])] = attempt
            accepted_ok = (
                float(delta.get("parameter_delta_l2", 0.0)) > 0.0
                and rollback is None
                and attempt.get("post_attempt_policy_state_digest")
                != attempt.get("pre_attempt_policy_state_digest")
                and attempt.get("post_attempt_optimizer_state_digest")
                != attempt.get("pre_attempt_optimizer_state_digest")
            )
            accepted_state_ok &= bool(accepted_ok)
        else:
            rollback_delta = (rollback or {}).get("parameter_delta") or {}
            rollback_ok = bool(
                rollback
                and rollback.get("exact_match") is True
                and close(rollback_delta.get("parameter_delta_l2"), 0.0)
                and close(rollback_delta.get("parameter_delta_max_abs"), 0.0)
            )
            rejected_rollback_ok &= rollback_ok
        gate_consistent &= gate_matches_selected and pass_fields_match
        checks.append({
            "step": attempt.get("step"),
            "attempt_index": attempt.get("attempt_index"),
            "mode": mode,
            "accepted": accepted,
            "gate_matches_selected": gate_matches_selected,
            "pass_fields_match": pass_fields_match,
            "parameter_delta_l2": delta.get("parameter_delta_l2"),
            "rollback_exact": (rollback or {}).get("exact_match"),
        })

    continuity_ok = True
    accepted_steps = sorted(accepted_by_step)
    for previous_step, current_step in zip(accepted_steps, accepted_steps[1:]):
        previous = accepted_by_step[previous_step]
        current_attempts = [row for row in attempts if int(row["step"]) == current_step]
        first_current = min(current_attempts, key=lambda row: int(row["attempt_index"]))
        continuity_ok &= (
            previous.get("post_attempt_policy_state_digest")
            == first_current.get("pre_attempt_policy_state_digest")
            and previous.get("post_attempt_optimizer_state_digest")
            == first_current.get("pre_attempt_optimizer_state_digest")
        )
    return {
        "checks": checks,
        "accepted_by_step": accepted_by_step,
        "accepted_steps": accepted_steps,
        "gate_consistent": bool(gate_consistent and attempts),
        "rejected_rollback_ok": rejected_rollback_ok,
        "accepted_state_ok": accepted_state_ok,
        "state_continuity_ok": bool(continuity_ok and accepted_steps),
    }


def determine_status(
    *,
    integrity_ok: bool,
    accepted_steps: list[int],
    accepted_attempts: list[dict],
    optimizer_rejected: bool,
) -> str:
    if not integrity_ok:
        return "TELEMETRY_INCOMPLETE"
    if accepted_steps == [1, 2]:
        if any(int(row.get("attempt_index", 0)) > 0 for row in accepted_attempts):
            return "CORRECTED_GATE_BACKOFF_REQUIRED_DIAGNOSTIC"
        return "CORRECTED_GATE_ACCEPTED_2_STEPS_DIAGNOSTIC"
    if not accepted_steps and optimizer_rejected:
        return "CORRECTED_GATE_UNRESOLVED_BASELINE_RETAINED"
    return "CORRECTED_GATE_PARTIAL_DIAGNOSTIC"


def audit(args: argparse.Namespace) -> dict[str, object]:
    experiment_root = Path(args.experiment_root)
    run_dir = experiment_root / args.run_name
    output_dir = Path(args.output_dir)
    ensure_empty_output_dir(output_dir)
    required = [
        experiment_root / "matrix.json",
        run_dir / "kl_guard_attempts.jsonl",
        run_dir / "kl_guard_token_replay.jsonl",
        run_dir / "step_summaries.jsonl",
        run_dir / "selection.json",
        run_dir / "validation_history.jsonl",
        run_dir / "run.log",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        result = {
            "status": "TELEMETRY_INCOMPLETE",
            "missing": missing,
            "warnings": [],
            "diagnostic_only": True,
            "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        }
        (output_dir / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result

    matrix = load_json(experiment_root / "matrix.json")
    attempts = load_jsonl(run_dir / "kl_guard_attempts.jsonl")
    replay_rows = load_jsonl(run_dir / "kl_guard_token_replay.jsonl")
    steps = load_jsonl(run_dir / "step_summaries.jsonl")
    validation = load_jsonl(run_dir / "validation_history.jsonl")
    selection = load_json(run_dir / "selection.json")
    target = float(matrix["guard"]["post_step_kl_target"])

    attempt_result = validate_attempts(attempts, target)
    accepted_attempts = [row for row in attempts if row.get("accepted") is True]
    replay_checks = [replay_row_check(row) for row in replay_rows]
    replay_rows_valid = bool(replay_checks) and all(row.get("valid") for row in replay_checks)
    groups = replay_variant_groups(replay_rows) if replay_rows_valid else {}
    replay_groups_complete = bool(groups) and all(set(group) == EXPECTED_VARIANTS for group in groups.values())
    replay_sources_match = True
    for group in groups.values():
        for field in ("generated_sha256", "mask_sha256", "reference_log_probs_sha256"):
            replay_sources_match &= len({row[field] for row in group.values()}) == 1
        replay_sources_match &= len({json.dumps(row["sample_keys"], sort_keys=True) for row in group.values()}) == 1
    attempt_keys = {(int(row["step"]), int(row["attempt_index"])) for row in attempts}
    replay_attempt_keys = {(int(row["step"]), int(row["attempt_index"])) for row in replay_rows}
    replay_attempt_alignment = attempt_keys == replay_attempt_keys

    checkpoint_records = [row for row in validation if row.get("checkpoint")]
    checkpoint_reload_ok = bool(checkpoint_records) and all(
        row.get("evaluation_source") == "reloaded_checkpoint"
        and Path(row["checkpoint"]).exists()
        for row in checkpoint_records
    )
    step_gate_ok = bool(steps) and all(
        row.get("post_step_kl_gate_mode") == "fp32_no_autocast"
        and row.get("post_step_kl_gate_passed") is True
        and float(row.get("post_step_kl_gate_mean", float("inf"))) <= target
        for row in steps
        if row.get("optimizer_step_applied") is True
    )
    pre_step_complete = bool(steps) and all(
        row.get("pre_step_kl_bfloat16_shadow_mean") is not None
        and row.get("pre_step_kl_fp32_no_autocast_mean") is not None
        for row in steps
    )

    warnings = ["CHECKPOINT_SELECTION_SMOKE_ONLY"]
    if any(row.get("shadow_gate_disagreement") is True for row in attempts):
        warnings.append("LEGACY_BF16_SHADOW_DISAGREEMENT")
    if any(row.get("pre_step_shadow_disagreement") is True for row in steps):
        warnings.append("PRESTEP_PRECISION_DIVERGENCE")

    continuity_requirement_ok = (
        attempt_result["state_continuity_ok"]
        if len(attempt_result["accepted_steps"]) >= 2
        else True
    )
    checkpoint_requirement_ok = (
        checkpoint_reload_ok
        if 2 in attempt_result["accepted_steps"]
        else True
    )
    integrity_ok = all([
        attempt_result["gate_consistent"],
        attempt_result["rejected_rollback_ok"],
        attempt_result["accepted_state_ok"],
        continuity_requirement_ok,
        replay_rows_valid,
        replay_groups_complete,
        replay_sources_match,
        replay_attempt_alignment,
        checkpoint_requirement_ok,
        step_gate_ok,
        pre_step_complete,
    ])
    status = determine_status(
        integrity_ok=integrity_ok,
        accepted_steps=attempt_result["accepted_steps"],
        accepted_attempts=accepted_attempts,
        optimizer_rejected=any(row.get("optimizer_step_rejected") is True for row in steps),
    )
    result = {
        "status": status,
        "experiment_root": str(experiment_root),
        "run_name": args.run_name,
        "target": target,
        "steps_completed": len(steps),
        "accepted_steps": attempt_result["accepted_steps"],
        "attempts": len(attempts),
        "accepted_attempts": len(accepted_attempts),
        "replay_rows": len(replay_rows),
        "gate_consistent": attempt_result["gate_consistent"],
        "accepted_state_ok": attempt_result["accepted_state_ok"],
        "rejected_rollback_ok": attempt_result["rejected_rollback_ok"],
        "state_continuity_ok": attempt_result["state_continuity_ok"],
        "replay_rows_valid": replay_rows_valid,
        "replay_groups_complete": replay_groups_complete,
        "replay_sources_match": replay_sources_match,
        "replay_attempt_alignment": replay_attempt_alignment,
        "checkpoint_reload_ok": checkpoint_reload_ok,
        "checkpoint_count": len(checkpoint_records),
        "step_gate_ok": step_gate_ok,
        "pre_step_complete": pre_step_complete,
        "selection_status": selection.get("status"),
        "warnings": warnings,
        "all_json_finite": True,
        "diagnostic_only": True,
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        "default_model_changed": False,
        "attempt_analysis": attempt_result["checks"],
        "limitation": "Two validation prompts verify plumbing only and cannot establish safety, generalization, or model improvement.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "attempt_analysis.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in attempt_result["checks"]),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join([
            "# Corrected KL gate smoke audit",
            "",
            f"- status: {status}",
            f"- accepted steps: {attempt_result['accepted_steps']}",
            f"- attempts: {len(attempts)}",
            f"- replay rows: {len(replay_rows)}",
            f"- checkpoint reload: {checkpoint_reload_ok}",
            f"- state continuity: {attempt_result['state_continuity_ok']}",
            f"- warnings: {', '.join(warnings)}",
            "",
            "Diagnostic only; no default-model or promotion decision is made.",
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
    print(json.dumps(audit(parser.parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
