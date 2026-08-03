"""Audit the isolated controlled nonzero-advantage GRPO active-path smoke."""

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


def parse_pattern(pattern: str) -> list[float]:
    values = [float(piece.strip()) for piece in pattern.split(",")]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("expected a finite comma-separated reward pattern")
    return values


def _list_close(left: object, right: list[float]) -> bool:
    return isinstance(left, list) and len(left) == len(right) and all(
        close(value, expected) for value, expected in zip(left, right)
    )


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


def audit_run(
    experiment_root: Path,
    run_name: str,
    output_dir: Path,
    expected_pattern: list[float],
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
            "expected_controlled_reward_pattern": expected_pattern,
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

        expected_step_numbers = list(range(1, expected_steps + 1))
        step_numbers = [row.get("step") for row in step_rows]
        micro_control_checks = [
            row.get("reward_source") == "controlled_reward_pattern"
            and row.get("training_forward_mode") == expected_mode
            and _list_close(row.get("controlled_reward_pattern"), expected_pattern)
            and _list_close(row.get("controlled_reward_values"), expected_pattern)
            and float(row.get("advantage_nonzero_rate", 0.0)) > 0.0
            for row in micro_rows
        ]
        sample_control_checks = [
            row.get("reward_source") == "controlled_reward_pattern"
            and row.get("controlled_reward") is not None
            and close(row.get("controlled_reward"), expected_pattern[int(row["candidate_index"]) % len(expected_pattern)])
            for row in sample_rows
        ]
        pre_step_control_checks = [
            row.get("reward_source") == "controlled_reward_pattern"
            and max(abs(float(value)) for value in row.get("advantages", [0.0])) > 0.0
            for row in pre_step_rows
        ]
        step_control_checks = []
        for row in step_rows:
            delta = row.get("final_parameter_delta") or {}
            precision = row.get("pre_step_loss_precision") or {}
            step_control_checks.append({
                "step": row.get("step"),
                "active_mode_ok": row.get("training_forward_mode") == expected_mode
                and row.get("active_loss_precision") == expected_mode,
                "controlled_reward_ok": row.get("reward_source") == "controlled_reward_pattern"
                and _list_close(row.get("controlled_reward_pattern"), expected_pattern),
                "nonzero_advantage_ok": float(row.get("advantage_nonzero_rate", 0.0)) > 0.0,
                "active_gate_ok": row.get("post_step_kl_gate_mode") == expected_mode
                and row.get("post_step_kl_gate_passed") is True
                and float(row.get("post_step_kl_gate_mean")) <= float(row.get("post_step_kl_target")) + 1e-9,
                "optimizer_step_ok": row.get("optimizer_step_applied") is True
                and row.get("optimizer_step_rejected") is False,
                "parameter_update_nonzero": float(delta.get("parameter_delta_l2", 0.0)) > 0.0
                and float(delta.get("parameter_delta_max_abs", 0.0)) > 0.0,
                "active_precision_ok": precision.get("active_variant") == expected_mode
                and close(
                    precision.get("active_loss_mean"),
                    precision.get(f"{expected_mode}_loss_mean"),
                )
                and close(
                    precision.get("active_kl_mean"),
                    precision.get(f"{expected_mode}_kl_mean"),
                )
                and precision.get("active_gradient_matches_selected_variant") is True,
            })
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
            (ROOT / row["checkpoint"]).exists() if not Path(row["checkpoint"]).is_absolute()
            else Path(row["checkpoint"]).exists()
            for row in checkpoint_records
        ) and bool(checkpoint_records)
        all_finite = all(
            finite_tree(row)
            for rows in (micro_rows, step_rows, sample_rows, pre_step_rows, attempts, token_replay_rows, validation_rows)
            for row in rows
        ) and finite_tree(selection)
        checks = {
            "steps_complete": step_numbers == expected_step_numbers,
            "micro_control_ok": bool(micro_rows) and all(micro_control_checks),
            "sample_control_ok": bool(sample_rows) and all(sample_control_checks),
            "pre_step_control_ok": bool(pre_step_rows) and all(pre_step_control_checks),
            "step_control_checks": step_control_checks,
            "active_path_ok": bool(step_control_checks) and all(
                all(value for key, value in item.items() if key != "step") for item in step_control_checks
            ),
            "accepted_attempt_count": len(accepted_attempts),
            "rejected_attempt_count": len(rejected_attempts),
            "rejected_rollback_ok": rejected_rollback_ok,
            "state_continuity_ok": _state_continuity(step_rows),
            "checkpoint_reload_path_ok": checkpoint_reload_path_ok,
            "all_finite": all_finite,
            "selection": selection,
        }
        passed = all([
            checks["steps_complete"],
            checks["micro_control_ok"],
            checks["sample_control_ok"],
            checks["pre_step_control_ok"],
            checks["active_path_ok"],
            checks["accepted_attempt_count"] == expected_steps,
            checks["rejected_rollback_ok"],
            checks["state_continuity_ok"],
            checks["checkpoint_reload_path_ok"],
            checks["all_finite"],
        ])
        result = {
            "status": "CONTROLLED_NONZERO_ADVANTAGE_ACTIVE_PATH_PASS_2_STEPS" if passed else "CONTROLLED_NONZERO_ADVANTAGE_PARTIAL_DIAGNOSTIC",
            "expected_training_forward_mode": expected_mode,
            "expected_controlled_reward_pattern": expected_pattern,
            "microbatch_count": len(micro_rows),
            "sample_count": len(sample_rows),
            "pre_step_replay_rows": len(pre_step_rows),
            "token_replay_rows": len(token_replay_rows),
            "step_count": len(step_rows),
            "checks": checks,
            "diagnostic_only": True,
            "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
            "default_model_changed": False,
            "limitation": "Controlled reward injection validates the active update path only; it is not a quality or promotion result.",
        }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join([
            "# Controlled nonzero-advantage active-path audit",
            "",
            f"- status: {result['status']}",
            f"- expected mode: {expected_mode}",
            f"- controlled reward pattern: {expected_pattern}",
            "",
            "Diagnostic only; controlled rewards are not a model-quality or promotion signal.",
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
    parser.add_argument("--expected-pattern", default="1.0,0.0")
    parser.add_argument("--expected-training-forward-mode", default="fp32_no_autocast")
    parser.add_argument("--expected-steps", type=int, default=2)
    args = parser.parse_args()
    result = audit_run(
        args.experiment_root,
        args.run_name,
        args.output_dir,
        parse_pattern(args.expected_pattern),
        expected_mode=args.expected_training_forward_mode,
        expected_steps=args.expected_steps,
    )
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] == "TELEMETRY_INCOMPLETE":
        raise SystemExit(2)
    if result["status"] != "CONTROLLED_NONZERO_ADVANTAGE_ACTIVE_PATH_PASS_2_STEPS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
