"""Audit an opt-in FP32/no-autocast training-forward smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.audit_corrected_kl_gate import close, load_jsonl
from evaluation.audit_prestep_precision import audit as audit_prestep


def _active_precision_checks(
    micro_rows: list[dict],
    step_rows: list[dict],
    expected_mode: str,
) -> dict[str, object]:
    micro_checks = []
    for row in micro_rows:
        precision = row.get("pre_step_loss_precision") or {}
        active_mode_ok = (
            row.get("training_forward_mode") == expected_mode
            and precision.get("training_forward_mode") == expected_mode
            and precision.get("active_variant") == expected_mode
        )
        selected_loss = precision.get(f"{expected_mode}_loss")
        selected_kl = precision.get(f"{expected_mode}_kl")
        active_loss_ok = close(row.get("loss"), selected_loss, tolerance=1e-6)
        active_kl_ok = close(row.get("kl"), selected_kl, tolerance=1e-6)
        micro_checks.append({
            "step": row.get("step"),
            "micro_index": row.get("micro_index"),
            "active_mode_ok": active_mode_ok,
            "active_loss_ok": active_loss_ok,
            "active_kl_ok": active_kl_ok,
            "active_gradient_ok": precision.get("active_gradient_matches_selected_variant") is True,
        })
    step_checks = []
    for row in step_rows:
        precision = row.get("pre_step_loss_precision") or {}
        step_checks.append({
            "step": row.get("step"),
            "active_mode_ok": (
                row.get("training_forward_mode") == expected_mode
                and precision.get("training_forward_mode") == expected_mode
                and precision.get("active_variant") == expected_mode
            ),
            "active_loss_ok": close(
                precision.get("active_loss_mean"),
                precision.get(f"{expected_mode}_loss_mean"),
                tolerance=1e-6,
            ),
            "active_kl_ok": close(
                precision.get("active_kl_mean"),
                precision.get(f"{expected_mode}_kl_mean"),
                tolerance=1e-6,
            ),
            "active_gradient_ok": precision.get("active_gradient_matches_selected_variant") is True,
        })
    all_checks = micro_checks + step_checks
    return {
        "micro_checks": micro_checks,
        "step_checks": step_checks,
        "mode_ok": bool(all_checks) and all(item["active_mode_ok"] for item in all_checks),
        "active_loss_ok": bool(all_checks) and all(item["active_loss_ok"] for item in all_checks),
        "active_kl_ok": bool(all_checks) and all(item["active_kl_ok"] for item in all_checks),
        "active_gradient_ok": bool(all_checks) and all(item["active_gradient_ok"] for item in all_checks),
    }


def audit(args: argparse.Namespace) -> dict[str, object]:
    base = audit_prestep(args)
    experiment_root = Path(args.experiment_root)
    run_dir = experiment_root / args.run_name
    output_dir = Path(args.output_dir)
    expected_mode = args.expected_training_forward_mode
    required_artifacts_ok = all([
        (run_dir / "microbatch_summaries.jsonl").exists(),
        (run_dir / "step_summaries.jsonl").exists(),
        (run_dir / "kl_guard_attempts.jsonl").exists(),
        (run_dir / "kl_guard_token_replay.jsonl").exists(),
        (run_dir / "selection.json").exists(),
    ])
    if not required_artifacts_ok:
        result = {
            **base,
            "status": "FP32_TRAINING_FORWARD_TELEMETRY_INCOMPLETE",
            "expected_training_forward_mode": expected_mode,
            "diagnostic_only": True,
            "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        }
    else:
        micro_rows = load_jsonl(run_dir / "microbatch_summaries.jsonl")
        step_rows = load_jsonl(run_dir / "step_summaries.jsonl")
        attempts = load_jsonl(run_dir / "kl_guard_attempts.jsonl")
        checks = _active_precision_checks(micro_rows, step_rows, expected_mode)
        accepted_steps = base.get("accepted_steps", [])
        complete_two_steps = accepted_steps == [1, 2]
        base_integrity = all([
            base.get("pre_step_replay_valid") is True,
            base.get("sample_linkage_ok") is True,
            base.get("shadow_grad_isolation_ok") is True,
            base.get("post_gate_ok") is True,
            base.get("post_replay_valid") is True,
            base.get("state_continuity_ok") is True,
            base.get("checkpoint_reload_ok") is True,
            base.get("run_exit_ok") is True,
        ])
        accepted_attempts = [row for row in attempts if row.get("accepted") is True]
        no_op_due_to_zero_advantage = (
            base.get("nonzero_advantage_microbatches") == 0
            and bool(accepted_attempts)
            and all(
                close((row.get("parameter_delta") or {}).get("parameter_delta_l2"), 0.0)
                and close((row.get("parameter_delta") or {}).get("parameter_delta_max_abs"), 0.0)
                and row.get("rollback_after_attempt") is None
                for row in accepted_attempts
            )
        )
        accepted_step_flags_ok = bool(step_rows) and all(
            row.get("optimizer_step_applied") is True
            and row.get("optimizer_step_rejected") is False
            for row in step_rows
        )
        accepted_state_evidence = base.get("state_continuity_ok") is True and (
            base.get("accepted_steps") == [1, 2]
            and (base.get("checkpoint_reload_ok") is True)
            and (
                no_op_due_to_zero_advantage
                or all(
                    float((row.get("parameter_delta") or {}).get("parameter_delta_l2", 0.0)) > 0.0
                    and row.get("post_attempt_policy_state_digest")
                    != row.get("pre_attempt_policy_state_digest")
                    for row in accepted_attempts
                )
            )
        )
        active_ok = all([
            checks["mode_ok"],
            checks["active_loss_ok"],
            checks["active_kl_ok"],
            checks["active_gradient_ok"],
        ])
        if base_integrity and accepted_step_flags_ok and accepted_state_evidence and active_ok and complete_two_steps:
            status = "FP32_TRAINING_FORWARD_ACCEPTED_2_STEPS_DIAGNOSTIC"
        elif base_integrity and accepted_steps:
            status = "FP32_TRAINING_FORWARD_BACKOFF_OR_PARTIAL_DIAGNOSTIC"
        else:
            status = "FP32_TRAINING_FORWARD_TELEMETRY_INCOMPLETE"
        warnings = list(base.get("warnings", []))
        if any(row.get("loss_disagreement") is True or row.get("kl_disagreement") is True for row in base.get("step_precision", [])):
            if "LEGACY_BF16_SHADOW_DISAGREEMENT" not in warnings:
                warnings.append("LEGACY_BF16_SHADOW_DISAGREEMENT")
        if no_op_due_to_zero_advantage:
            warnings.append("ZERO_ADVANTAGE_NO_PARAMETER_UPDATE")
        result = {
            **base,
            "status": status,
            "expected_training_forward_mode": expected_mode,
            "active_precision_checks": checks,
            "active_precision_ok": active_ok,
            "base_integrity_ok": base_integrity,
            "accepted_step_flags_ok": accepted_step_flags_ok,
            "accepted_state_evidence": accepted_state_evidence,
            "no_op_due_to_zero_advantage": no_op_due_to_zero_advantage,
            "two_steps_accepted": complete_two_steps,
            "warnings": warnings,
            "diagnostic_only": True,
            "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
            "default_model_changed": False,
            "limitation": "This 2-step smoke validates the opt-in forward/loss precision path; zero-advantage groups still do not validate ratio clipping or model quality.",
        }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join([
            "# FP32 training-forward smoke audit",
            "",
            f"- status: {result['status']}",
            f"- expected training forward: {expected_mode}",
            f"- accepted steps: {result.get('accepted_steps', [])}",
            f"- active precision ok: {result.get('active_precision_ok', False)}",
            f"- warnings: {', '.join(result.get('warnings', []))}",
            "",
            "Diagnostic only; no reward, checkpoint-selection, or default-model change.",
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
    parser.add_argument(
        "--expected-training-forward-mode",
        choices=("legacy_bfloat16_autocast", "fp32_no_autocast"),
        default="fp32_no_autocast",
    )
    print(json.dumps(audit(parser.parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
