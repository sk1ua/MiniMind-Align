"""Audit the explicit no-autocast precision-contract smoke."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CONTRACT_FIELDS = (
    "precision_contract_version",
    "policy_parameter_dtype",
    "reference_parameter_dtype",
    "active_loss_variant",
    "active_gate_variant",
    "active_gate_source",
    "active_loss_autocast_enabled",
    "active_gate_autocast_enabled",
    "legacy_bfloat16_shadow_variant",
    "full_float32_shadow_variant",
    "full_float32_shadow_only",
    "precision_contract_valid",
)
EXPECTED_REPLAY_VARIANTS = {
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


def contract_of(row: dict) -> dict:
    nested = row.get("precision_contract")
    if isinstance(nested, dict):
        return nested
    return {key: row.get(key) for key in CONTRACT_FIELDS if key in row}


def precision_contract_check(row: dict, *, require_opt_in: bool = True) -> tuple[bool, list[str]]:
    contract = contract_of(row)
    errors: list[str] = []
    missing = [key for key in CONTRACT_FIELDS if key not in contract]
    if missing:
        errors.append("missing:" + ",".join(missing))
        return False, errors
    if require_opt_in and row.get("precision_contract_mode", contract.get("precision_contract_mode")) != "no_autocast_v1":
        errors.append("wrong_mode")
    if contract.get("precision_contract_valid") is not True:
        errors.append("contract_not_valid")
    if contract.get("active_loss_variant") != contract.get("active_gate_variant"):
        errors.append("active_variant_mismatch")
    if not str(contract.get("active_loss_variant", "")).endswith("_no_autocast"):
        errors.append("active_loss_not_no_autocast")
    if not str(contract.get("active_gate_variant", "")).endswith("_no_autocast"):
        errors.append("active_gate_not_no_autocast")
    if contract.get("active_gate_source") != "post_step_kl_float32":
        errors.append("wrong_active_gate_source")
    if contract.get("active_loss_autocast_enabled") is not False:
        errors.append("active_loss_autocast_enabled")
    if contract.get("active_gate_autocast_enabled") is not False:
        errors.append("active_gate_autocast_enabled")
    if contract.get("full_float32_shadow_only") is not True:
        errors.append("full_float32_shadow_not_shadow_only")
    if contract.get("active_gate_variant") == contract.get("legacy_bfloat16_shadow_variant"):
        errors.append("legacy_shadow_is_active")
    return not errors, errors


def dtype_gap(left: float, right: float, target: float = 0.005) -> tuple[float, float, bool]:
    gap = abs(float(left) - float(right))
    threshold = max(1e-4, 0.1 * max(float(right), float(target)))
    return gap, threshold, gap <= threshold


def _stats(row: dict, key: str) -> dict:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def replay_integrity(rows: list[dict]) -> tuple[bool, dict[str, object]]:
    groups: dict[tuple[int, int, int], dict[str, dict]] = defaultdict(dict)
    errors: list[str] = []
    for row in rows:
        try:
            key = (int(row["step"]), int(row["attempt_index"]), int(row["rollout_index"]))
        except (KeyError, TypeError, ValueError):
            errors.append("replay_key_missing")
            continue
        variant = row.get("variant")
        if variant not in EXPECTED_REPLAY_VARIANTS:
            errors.append(f"unexpected_variant:{variant}")
            continue
        groups[key][variant] = row
        if not all(row.get(field) for field in ("generated_sha256", "mask_sha256", "reference_log_probs_sha256")):
            errors.append("replay_source_hash_missing")
        ok, contract_errors = precision_contract_check(row)
        if not ok:
            errors.extend("contract:" + item for item in contract_errors)
    for key, variants in groups.items():
        if set(variants) != EXPECTED_REPLAY_VARIANTS:
            errors.append(f"replay_variant_group_incomplete:{key}")
            continue
        source_fields = ("generated_sha256", "mask_sha256", "reference_log_probs_sha256", "sample_keys")
        for field in source_fields:
            values = {
                json.dumps(row.get(field), sort_keys=True)
                for row in variants.values()
            }
            if len(values) != 1:
                errors.append(f"replay_source_mismatch:{key}:{field}")
    return not errors and bool(groups), {
        "groups": len(groups),
        "rows": len(rows),
        "errors": errors,
        "complete": not errors and bool(groups),
    }


def accepted_state_check(attempts: list[dict], steps: list[dict]) -> dict[str, object]:
    accepted = [row for row in attempts if row.get("accepted") is True]
    accepted_by_step = {int(row["step"]): row for row in accepted if "step" in row}
    rejected_ok = True
    accepted_nonzero = True
    for row in attempts:
        if row.get("accepted") is True:
            delta = _stats(row, "parameter_delta")
            accepted_nonzero &= float(delta.get("parameter_delta_l2", 0.0)) > 0.0
            rejected_ok &= row.get("rollback_after_attempt") is None
        else:
            rollback = row.get("rollback_after_attempt") or {}
            delta = _stats(rollback, "parameter_delta")
            rejected_ok &= rollback.get("exact_match") is True
            rejected_ok &= abs(float(delta.get("parameter_delta_l2", 1.0))) <= 1e-7
            rejected_ok &= abs(float(delta.get("parameter_delta_max_abs", 1.0))) <= 1e-7
    continuity = True
    for previous_step, current_step in zip(sorted(accepted_by_step), sorted(accepted_by_step)[1:]):
        previous = accepted_by_step[previous_step]
        current_attempts = [row for row in attempts if int(row.get("step", -1)) == current_step]
        if not current_attempts:
            continuity = False
            continue
        first = min(current_attempts, key=lambda row: int(row.get("attempt_index", 0)))
        continuity &= previous.get("post_attempt_policy_state_digest") == first.get("pre_attempt_policy_state_digest")
        continuity &= previous.get("post_attempt_optimizer_state_digest") == first.get("pre_attempt_optimizer_state_digest")
    step_gate_ok = True
    for row in steps:
        if row.get("optimizer_step_applied") is True:
            step_gate_ok &= row.get("post_step_kl_gate_passed") is True
            step_gate_ok &= float(row.get("post_step_kl_gate_mean", float("inf"))) <= 0.005
            step_gate_ok &= precision_contract_check(row)[0]
    return {
        "accepted_steps": sorted(accepted_by_step),
        "accepted_count": len(accepted),
        "accepted_nonzero": accepted_nonzero,
        "rejected_rollback_ok": rejected_ok,
        "state_continuity_ok": continuity,
        "step_gate_ok": step_gate_ok,
    }


def audit(args: argparse.Namespace) -> dict[str, object]:
    experiment_root = Path(args.experiment_root)
    run_dir = experiment_root / args.run_name
    output_dir = Path(args.output_dir)
    expected_steps = int(getattr(args, "expected_steps", 2))
    if expected_steps < 1:
        raise ValueError("expected_steps must be positive")
    ensure_empty_output_dir(output_dir)
    required = [
        experiment_root / "matrix.json",
        run_dir / "run_metadata.json",
        run_dir / "step_summaries.jsonl",
        run_dir / "microbatch_summaries.jsonl",
        run_dir / "pre_step_loss_replay.jsonl",
        run_dir / "kl_guard_attempts.jsonl",
        run_dir / "kl_guard_token_replay.jsonl",
        run_dir / "validation_history.jsonl",
        run_dir / "selection.json",
        run_dir / "run.log",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        result = {
            "status": "TELEMETRY_INCOMPLETE",
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
        micros = load_jsonl(run_dir / "microbatch_summaries.jsonl")
        pre_step = load_jsonl(run_dir / "pre_step_loss_replay.jsonl")
        attempts = load_jsonl(run_dir / "kl_guard_attempts.jsonl")
        replay = load_jsonl(run_dir / "kl_guard_token_replay.jsonl")
        validation = load_jsonl(run_dir / "validation_history.jsonl")
        selection = load_json(run_dir / "selection.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "status": "TELEMETRY_INCOMPLETE",
            "error": str(exc),
            "diagnostic_only": True,
            "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        }
        (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    contract_rows = [metadata, *steps, *micros, *pre_step, *attempts, *replay]
    contract_checks = [precision_contract_check(row) for row in contract_rows]
    contract_ok = bool(contract_checks) and all(ok for ok, _ in contract_checks)
    contract_errors = [error for _, errors in contract_checks for error in errors]

    target = float(matrix.get("post_step_kl_target", matrix.get("guard", {}).get("post_step_kl_target", 0.005)))
    attempt_gap_checks: list[dict[str, object]] = []
    shadow_warning = False
    attempt_precision_ok = True
    for row in attempts:
        active = _stats(row, "post_step_kl_float32")
        full = _stats(row, "post_step_kl_full_float32")
        legacy = _stats(row, "post_step_kl_bfloat16")
        if not active or not full or not legacy:
            attempt_precision_ok = False
            continue
        gap, threshold, close = dtype_gap(active.get("reference_kl_mean"), full.get("reference_kl_mean"), target)
        shadow_gap, shadow_threshold, shadow_close = dtype_gap(legacy.get("reference_kl_mean"), active.get("reference_kl_mean"), target)
        shadow_warning |= not shadow_close
        attempt_precision_ok &= close
        attempt_gap_checks.append({
            "step": row.get("step"),
            "attempt_index": row.get("attempt_index"),
            "active_mean": active.get("reference_kl_mean"),
            "full_float32_mean": full.get("reference_kl_mean"),
            "active_full_gap": gap,
            "active_full_threshold": threshold,
            "active_full_within_threshold": close,
            "legacy_shadow_mean": legacy.get("reference_kl_mean"),
            "legacy_active_gap": shadow_gap,
            "legacy_active_threshold": shadow_threshold,
            "legacy_shadow_within_threshold": shadow_close,
        })

    state = accepted_state_check(attempts, steps)
    replay_ok, replay_info = replay_integrity(replay)
    pre_step_contract_ok = bool(pre_step) and all(precision_contract_check(row)[0] for row in pre_step)
    checkpoint_records = [row for row in validation if row.get("checkpoint")]
    checkpoint_reload_ok = bool(checkpoint_records) and all(
        row.get("evaluation_source") == "reloaded_checkpoint"
        and Path(row["checkpoint"]).exists()
        for row in checkpoint_records
    )
    contract_acceptance = (
        state["accepted_steps"] == list(range(1, expected_steps + 1))
        and state["accepted_nonzero"]
        and state["rejected_rollback_ok"]
        and state["state_continuity_ok"]
        and state["step_gate_ok"]
        and attempt_precision_ok
        and pre_step_contract_ok
        and replay_ok
        and checkpoint_reload_ok
        and len(checkpoint_records) >= 1
    )
    if not contract_ok:
        status = "PRECISION_CONTRACT_MISMATCH_STOPPED"
    elif not contract_acceptance:
        status = "TELEMETRY_INCOMPLETE"
    elif shadow_warning:
        status = (
            "PRECISION_CONTRACT_PASS_WITH_BF16_SHADOW_WARNING"
            if expected_steps == 2
            else f"PRECISION_CONTRACT_PASS_WITH_BF16_SHADOW_WARNING_{expected_steps}_STEPS"
        )
    else:
        status = (
            "PRECISION_CONTRACT_PASS_2_STEPS_DIAGNOSTIC"
            if expected_steps == 2
            else f"PRECISION_CONTRACT_PASS_{expected_steps}_STEPS_DIAGNOSTIC"
        )

    result = {
        "status": status,
        "experiment_root": str(experiment_root),
        "run_name": args.run_name,
        "expected_steps": expected_steps,
        "target": target,
        "steps_completed": len(steps),
        "accepted_steps": state["accepted_steps"],
        "accepted_count": state["accepted_count"],
        "contract_rows": len(contract_rows),
        "contract_ok": contract_ok,
        "contract_errors": sorted(set(contract_errors)),
        "attempts": len(attempts),
        "attempt_precision_ok": attempt_precision_ok,
        "attempt_gap_checks": attempt_gap_checks,
        "state": state,
        "replay": replay_info,
        "pre_step_contract_ok": pre_step_contract_ok,
        "checkpoint_reload_ok": checkpoint_reload_ok,
        "checkpoint_count": len(checkpoint_records),
        "selection_status": selection.get("status"),
        "warnings": (["BF16_SHADOW_MEASUREMENT_WARNING"] if shadow_warning else []),
        "all_json_finite": True,
        "diagnostic_only": True,
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        "default_model_changed": False,
        "matrix": matrix,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "attempt_gap_checks.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in attempt_gap_checks),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join([
            "# Precision contract smoke audit",
            "",
            f"- status: {status}",
            f"- accepted steps: {state['accepted_steps']}",
            f"- active/full-FP32 attempt precision: {attempt_precision_ok}",
            f"- replay complete: {replay_ok}",
            f"- checkpoint reload: {checkpoint_reload_ok}",
            f"- warnings: {', '.join(result['warnings']) or 'none'}",
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
    parser.add_argument("--expected-steps", type=int, default=2)
    print(json.dumps(audit(parser.parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
