"""Audit the isolated natural rule-reward corrected active-path smoke."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.audit_corrected_kl_gate import ensure_empty_output_dir


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return True


def close(left: object, right: object, tolerance: float = 1e-6) -> bool:
    if left is None or right is None:
        return False
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def _accepted_attempts(step_rows: list[dict]) -> list[dict]:
    return [
        attempt
        for step in step_rows
        for attempt in step.get("kl_guard_attempt_history", [])
        if attempt.get("accepted") is True
    ]


def _state_continuity(step_rows: list[dict]) -> bool:
    if not step_rows:
        return False
    for index, step in enumerate(step_rows):
        accepted = [
            attempt
            for attempt in step.get("kl_guard_attempt_history", [])
            if attempt.get("accepted") is True
        ]
        if len(accepted) != 1:
            return False
        attempt = accepted[0]
        if not attempt.get("post_attempt_policy_state_digest") or not attempt.get("post_attempt_optimizer_state_digest"):
            return False
        if index + 1 < len(step_rows):
            next_step = step_rows[index + 1]
            if next_step.get("pre_step_policy_state_digest") != attempt.get("post_attempt_policy_state_digest"):
                return False
            if next_step.get("pre_step_optimizer_state_digest") != attempt.get("post_attempt_optimizer_state_digest"):
                return False
    return True


def _rule_reward_source_checks(
    micro_rows: list[dict],
    sample_rows: list[dict],
    pre_step_rows: list[dict],
    step_rows: list[dict],
) -> dict[str, bool]:
    micro_ok = bool(micro_rows) and all(
        row.get("reward_source") == "rule_reward"
        and row.get("controlled_reward_pattern") is None
        and row.get("controlled_reward_values") is None
        and isinstance(row.get("rule_reward_values"), list)
        for row in micro_rows
    )
    sample_ok = bool(sample_rows) and all(
        row.get("reward_source") == "rule_reward"
        and row.get("controlled_reward") is None
        and row.get("controlled_reward_delta") == 0.0
        and isinstance(row.get("rule_reward"), (int, float))
        for row in sample_rows
    )
    pre_step_ok = bool(pre_step_rows) and all(
        row.get("reward_source") == "rule_reward"
        and row.get("controlled_reward_pattern") is None
        and row.get("controlled_reward_values") is None
        and isinstance(row.get("rule_reward_values"), list)
        for row in pre_step_rows
    )
    step_ok = bool(step_rows) and all(
        row.get("reward_source") == "rule_reward"
        and row.get("controlled_reward_override_enabled") is False
        and row.get("controlled_reward_pattern") is None
        for row in step_rows
    )
    return {
        "micro_rule_reward_ok": micro_ok,
        "sample_rule_reward_ok": sample_ok,
        "pre_step_rule_reward_ok": pre_step_ok,
        "step_rule_reward_ok": step_ok,
    }


def _step_check(row: dict, expected_mode: str) -> dict[str, object]:
    delta = row.get("final_parameter_delta") or {}
    precision = row.get("pre_step_loss_precision") or {}
    gate_mean = row.get("post_step_kl_gate_mean")
    target = row.get("post_step_kl_target")
    active_precision_ok = (
        precision.get("active_variant") == expected_mode
        and close(precision.get("active_loss_mean"), precision.get(f"{expected_mode}_loss_mean"))
        and close(precision.get("active_kl_mean"), precision.get(f"{expected_mode}_kl_mean"))
        and precision.get("active_gradient_matches_selected_variant") is True
    )
    return {
        "step": row.get("step"),
        "active_mode_ok": row.get("training_forward_mode") == expected_mode
        and row.get("active_loss_precision") == expected_mode,
        "active_gate_ok": row.get("post_step_kl_gate_mode") == expected_mode
        and row.get("post_step_kl_gate_passed") is True
        and gate_mean is not None
        and target is not None
        and float(gate_mean) <= float(target) + 1e-9,
        "optimizer_step_ok": row.get("optimizer_step_applied") is True
        and row.get("optimizer_step_rejected") is False,
        "active_precision_ok": active_precision_ok,
        "parameter_update_nonzero": float(delta.get("parameter_delta_l2", 0.0)) > 0.0
        and float(delta.get("parameter_delta_max_abs", 0.0)) > 0.0,
        "advantage_nonzero": float(row.get("advantage_nonzero_rate", 0.0)) > 0.0,
    }


def audit_run(
    experiment_root: Path,
    run_name: str,
    output_dir: Path,
    expected_mode: str = "fp32_no_autocast",
    expected_steps: int = 2,
) -> dict[str, object]:
    ensure_empty_output_dir(output_dir)
    run_dir = experiment_root / run_name
    paths = {
        "micro": run_dir / "microbatch_summaries.jsonl",
        "steps": run_dir / "step_summaries.jsonl",
        "samples": run_dir / "samples.jsonl",
        "pre_step": run_dir / "pre_step_loss_replay.jsonl",
        "attempts": run_dir / "kl_guard_attempts.jsonl",
        "token_replay": run_dir / "kl_guard_token_replay.jsonl",
        "validation": run_dir / "validation_history.jsonl",
        "selection": run_dir / "selection.json",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        result = {
            "status": "TELEMETRY_INCOMPLETE",
            "missing_artifacts": missing,
            "expected_training_forward_mode": expected_mode,
            "reward_source": "rule_reward",
            "diagnostic_only": True,
            "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
            "default_model_changed": False,
        }
    else:
        micro_rows = load_jsonl(paths["micro"])
        step_rows = load_jsonl(paths["steps"])
        sample_rows = load_jsonl(paths["samples"])
        pre_step_rows = load_jsonl(paths["pre_step"])
        attempts = load_jsonl(paths["attempts"])
        token_replay_rows = load_jsonl(paths["token_replay"])
        validation_rows = load_jsonl(paths["validation"])
        selection = json.loads(paths["selection"].read_text(encoding="utf-8"))

        source_checks = _rule_reward_source_checks(
            micro_rows, sample_rows, pre_step_rows, step_rows
        )
        expected_step_numbers = list(range(1, expected_steps + 1))
        step_numbers = [row.get("step") for row in step_rows]
        step_checks = [_step_check(row, expected_mode) for row in step_rows]
        accepted_attempts = _accepted_attempts(step_rows)
        rejected_attempts = [attempt for attempt in attempts if attempt.get("accepted") is False]
        rejected_rollback_ok = all(
            (attempt.get("rollback_after_attempt") or {}).get("exact_match") is True
            for attempt in rejected_attempts
        )
        checkpoint_records = [
            row for row in validation_rows
            if row.get("evaluation_source") == "reloaded_checkpoint"
            and row.get("checkpoint")
        ]
        checkpoint_reload_path_ok = all(
            (ROOT / row["checkpoint"]).exists()
            if not Path(row["checkpoint"]).is_absolute()
            else Path(row["checkpoint"]).exists()
            for row in checkpoint_records
        ) and bool(checkpoint_records)
        all_finite = all(
            finite_tree(row)
            for rows in (
                micro_rows,
                step_rows,
                sample_rows,
                pre_step_rows,
                attempts,
                token_replay_rows,
                validation_rows,
            )
            for row in rows
        ) and finite_tree(selection)
        active_path_ok = bool(step_checks) and all(
            all(value for key, value in item.items() if key not in {"step", "parameter_update_nonzero", "advantage_nonzero"})
            for item in step_checks
        )
        checks = {
            "steps_complete": step_numbers == expected_step_numbers,
            **source_checks,
            "active_path_ok": active_path_ok,
            "step_checks": step_checks,
            "accepted_attempt_count": len(accepted_attempts),
            "rejected_attempt_count": len(rejected_attempts),
            "rejected_rollback_ok": rejected_rollback_ok,
            "state_continuity_ok": _state_continuity(step_rows),
            "checkpoint_reload_path_ok": checkpoint_reload_path_ok,
            "all_finite": all_finite,
            "selection": selection,
        }
        base_pass = all([
            checks["steps_complete"],
            checks["micro_rule_reward_ok"],
            checks["sample_rule_reward_ok"],
            checks["pre_step_rule_reward_ok"],
            checks["step_rule_reward_ok"],
            checks["active_path_ok"],
            checks["accepted_attempt_count"] == expected_steps,
            checks["rejected_rollback_ok"],
            checks["state_continuity_ok"],
            checks["checkpoint_reload_path_ok"],
            checks["all_finite"],
        ])
        nonzero_microbatches = sum(
            float(row.get("advantage_nonzero_rate", 0.0)) > 0.0 for row in micro_rows
        )
        nonzero_updates = sum(
            bool(item["parameter_update_nonzero"]) for item in step_checks
        )
        if base_pass and nonzero_microbatches and nonzero_updates:
            status = "NATURAL_RULE_REWARD_ACTIVE_PATH_NONZERO_UPDATE_PASS_2_STEPS"
        elif base_pass and not nonzero_microbatches:
            status = "NATURAL_RULE_REWARD_ZERO_ADVANTAGE_DIAGNOSTIC"
        elif base_pass:
            status = "NATURAL_RULE_REWARD_NONZERO_ADVANTAGE_NO_UPDATE_DIAGNOSTIC"
        else:
            status = "NATURAL_RULE_REWARD_PARTIAL_DIAGNOSTIC"
        result = {
            "status": status,
            "expected_training_forward_mode": expected_mode,
            "reward_source": "rule_reward",
            "microbatch_count": len(micro_rows),
            "sample_count": len(sample_rows),
            "pre_step_replay_rows": len(pre_step_rows),
            "token_replay_rows": len(token_replay_rows),
            "step_count": len(step_rows),
            "nonzero_advantage_microbatches": nonzero_microbatches,
            "nonzero_parameter_updates": nonzero_updates,
            "checks": checks,
            "diagnostic_only": True,
            "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
            "default_model_changed": False,
            "limitation": "Natural rule rewards are evaluated only for implementation and telemetry linkage; the two validation prompts are plumbing evidence, not a quality claim.",
        }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join([
            "# Natural rule-reward corrected active-path audit",
            "",
            f"- status: {result['status']}",
            f"- expected mode: {expected_mode}",
            "- reward source: rule_reward",
            "",
            "Diagnostic only; validation is plumbing evidence and does not select a model.",
            "",
        ]),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-training-forward-mode", default="fp32_no_autocast")
    parser.add_argument("--expected-steps", type=int, default=2)
    args = parser.parse_args()
    result = audit_run(
        args.experiment_root,
        args.run_name,
        args.output_dir,
        expected_mode=args.expected_training_forward_mode,
        expected_steps=args.expected_steps,
    )
    print(json.dumps(result, ensure_ascii=False))
    accepted_statuses = {
        "NATURAL_RULE_REWARD_ACTIVE_PATH_NONZERO_UPDATE_PASS_2_STEPS",
        "NATURAL_RULE_REWARD_ZERO_ADVANTAGE_DIAGNOSTIC",
        "NATURAL_RULE_REWARD_NONZERO_ADVANTAGE_NO_UPDATE_DIAGNOSTIC",
    }
    if result["status"] == "TELEMETRY_INCOMPLETE":
        raise SystemExit(2)
    if result["status"] not in accepted_statuses:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
