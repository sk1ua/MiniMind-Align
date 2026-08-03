"""Offline precision and quality-scope audit for the four-step contract run."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


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


def close(left: object, right: object, *, atol: float = 1e-6) -> bool:
    try:
        return abs(float(left) - float(right)) <= atol
    except (TypeError, ValueError):
        return False


def nested_sample_keys(rows: list[dict]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        values = row.get("sample_keys")
        if isinstance(values, list):
            keys.update(str(value) for value in values)
        elif row.get("sample_key") is not None:
            keys.add(str(row["sample_key"]))
    return keys


def audit(args: argparse.Namespace) -> dict[str, object]:
    experiment_root = Path(args.experiment_root)
    run_dir = experiment_root / args.run_name
    output_dir = Path(args.output_dir)
    ensure_empty_output_dir(output_dir)
    expected_steps = list(range(1, int(args.expected_steps) + 1))
    required = [
        experiment_root / "matrix.json",
        experiment_root / "precision_contract_audit_final" / "summary.json",
        run_dir / "run_metadata.json",
        run_dir / "step_summaries.jsonl",
        run_dir / "pre_step_loss_replay.jsonl",
        run_dir / "microbatch_summaries.jsonl",
        run_dir / "kl_guard_attempts.jsonl",
        run_dir / "kl_guard_token_replay.jsonl",
        run_dir / "samples.jsonl",
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
        contract_summary = load_json(experiment_root / "precision_contract_audit_final" / "summary.json")
        metadata = load_json(run_dir / "run_metadata.json")
        steps = load_jsonl(run_dir / "step_summaries.jsonl")
        pre_step = load_jsonl(run_dir / "pre_step_loss_replay.jsonl")
        micros = load_jsonl(run_dir / "microbatch_summaries.jsonl")
        attempts = load_jsonl(run_dir / "kl_guard_attempts.jsonl")
        token_replay = load_jsonl(run_dir / "kl_guard_token_replay.jsonl")
        samples = load_jsonl(run_dir / "samples.jsonl")
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

    step_rows = sorted(steps, key=lambda row: int(row.get("step", -1)))
    attempt_rows = sorted(
        attempts,
        key=lambda row: (int(row.get("step", -1)), int(row.get("attempt_index", -1))),
    )
    step_ids = [int(row.get("step", -1)) for row in step_rows]
    accepted_steps = [int(row["step"]) for row in attempt_rows if row.get("accepted") is True]
    target = float(matrix.get("post_step_kl_target", 0.005))

    step_precision: list[dict[str, object]] = []
    pre_step_disagreement = Counter()
    active_gate_ok = True
    contract_ok = True
    shadow_warning = False
    for row in step_rows:
        contract = row.get("precision_contract")
        contract_valid = isinstance(contract, dict) and contract.get("precision_contract_valid") is True
        contract_valid &= row.get("precision_contract_mode") == "no_autocast_v1"
        contract_valid &= contract.get("active_loss_variant") == contract.get("active_gate_variant") if isinstance(contract, dict) else False
        gate_mean = row.get("post_step_kl_gate_mean")
        full_mean = row.get("post_step_kl_full_float32_mean")
        legacy_mean = row.get("post_step_kl_mean")
        active_full_gap = math.inf
        active_full_ok = False
        if gate_mean is not None and full_mean is not None:
            active_full_gap = abs(float(gate_mean) - float(full_mean))
            active_full_ok = active_full_gap <= max(1e-4, 0.1 * max(float(full_mean), target))
        gate_ok = row.get("post_step_kl_gate_passed") is True and float(gate_mean) <= target
        active_gate_ok &= gate_ok and active_full_ok
        shadow_gap = math.inf
        if legacy_mean is not None and gate_mean is not None:
            shadow_gap = abs(float(legacy_mean) - float(gate_mean))
            shadow_warning |= shadow_gap > max(1e-4, 0.1 * max(float(gate_mean), target))
        pre = row.get("pre_step_loss_precision")
        if isinstance(pre, dict):
            for name in ("loss_disagreement", "kl_disagreement", "gradient_disagreement"):
                if pre.get(name) is True:
                    pre_step_disagreement[name] += 1
        else:
            contract_ok = False
        contract_ok &= contract_valid
        step_precision.append({
            "step": row.get("step"),
            "contract_valid": contract_valid,
            "active_gate_mean": gate_mean,
            "active_full_float32_mean": full_mean,
            "legacy_bfloat16_mean": legacy_mean,
            "active_full_gap": active_full_gap,
            "active_full_within_threshold": active_full_ok,
            "active_gate_passed": gate_ok,
            "legacy_shadow_gap": shadow_gap,
            "pre_step_loss_precision": pre,
        })

    accepted_attempts = [row for row in attempt_rows if row.get("accepted") is True]
    rejected_attempts = [row for row in attempt_rows if row.get("accepted") is not True]
    continuity_ok = True
    for previous, current in zip(accepted_attempts, accepted_attempts[1:]):
        if int(current.get("step", -1)) != int(previous.get("step", -1)) + 1:
            continuity_ok = False
        continuity_ok &= previous.get("post_attempt_policy_state_digest") == current.get("pre_attempt_policy_state_digest")
        continuity_ok &= previous.get("post_attempt_optimizer_state_digest") == current.get("pre_attempt_optimizer_state_digest")
    rollback_ok = all(
        (row.get("rollback_after_attempt") or {}).get("exact_match") is True
        for row in rejected_attempts
    )

    sample_keys = {str(row.get("sample_key")) for row in samples if row.get("sample_key") is not None}
    sample_linkage = {
        "sample_count": len(samples),
        "unique_sample_keys": len(sample_keys),
        "sample_keys_present": len(samples) == len(sample_keys) and bool(sample_keys),
        "pre_step_match": nested_sample_keys(pre_step) == sample_keys,
        "microbatch_match": nested_sample_keys(micros) == sample_keys,
        "token_replay_match": nested_sample_keys(token_replay) == sample_keys,
    }
    sample_linkage["complete"] = all(sample_linkage.values())

    validator_pass = 0
    max_length = 0
    natural_end = 0
    empty = 0
    repetition_values: list[float] = []
    categories = Counter()
    termination_reasons = Counter()
    reward_replay_ok = True
    for row in samples:
        categories[str(row.get("category"))] += 1
        components = row.get("components") if isinstance(row.get("components"), dict) else {}
        validator_pass += float(components.get("validator_reward", row.get("validator_reward", 0.0))) > 0.0
        max_length += bool(row.get("max_length_hit"))
        natural_end += bool(row.get("finished_naturally"))
        empty += bool(row.get("empty_response"))
        repetition_values.append(float(components.get("repetition_penalty", row.get("repetition_penalty", 0.0))))
        termination_reasons[str(row.get("termination_reason"))] += 1
        reward_replay_ok &= close(row.get("reward"), row.get("rule_reward"), atol=1e-8)
        reward_replay_ok &= row.get("reward_source") == "rule_reward"

    checkpoint_rows = [row for row in validation if row.get("checkpoint")]
    checkpoint_reload_ok = bool(checkpoint_rows) and all(
        row.get("evaluation_source") == "reloaded_checkpoint" and Path(row["checkpoint"]).exists()
        for row in checkpoint_rows
    )
    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    run_exit_ok = "EXIT_CODE=0" in run_log
    final_contract_ok = (
        contract_summary.get("status") == "PRECISION_CONTRACT_PASS_WITH_BF16_SHADOW_WARNING_4_STEPS"
        and contract_summary.get("accepted_steps") == expected_steps
    )
    integrity_ok = all([
        step_ids == expected_steps,
        accepted_steps == expected_steps,
        len(accepted_attempts) == len(expected_steps),
        len(rejected_attempts) == 0,
        contract_ok,
        active_gate_ok,
        continuity_ok,
        rollback_ok,
        sample_linkage["complete"],
        reward_replay_ok,
        checkpoint_reload_ok,
        run_exit_ok,
        final_contract_ok,
    ])
    quality_scope_limited = len(validation) <= 2
    if not integrity_ok:
        status = "TELEMETRY_INCOMPLETE"
    elif shadow_warning or any(pre_step_disagreement.values()) or quality_scope_limited:
        status = "PRECISION_DIVERGENCE_PERSISTS_QUALITY_SCOPE_LIMITED_DIAGNOSTIC"
    else:
        status = "PRECISION_QUALITY_AUDIT_COMPLETE_DIAGNOSTIC"

    result = {
        "status": status,
        "experiment_root": str(experiment_root),
        "run_name": args.run_name,
        "expected_steps": expected_steps,
        "steps": step_precision,
        "accepted_steps": accepted_steps,
        "attempt_count": len(attempt_rows),
        "backoff_count": sum(int(row.get("attempt_index", 0)) for row in attempt_rows),
        "rejected_attempts": len(rejected_attempts),
        "active_gate_ok": active_gate_ok,
        "contract_ok": contract_ok,
        "active_full_fp32_agreement": all(row["active_full_within_threshold"] for row in step_precision),
        "bf16_shadow_warning": shadow_warning,
        "pre_step_disagreement_counts": dict(pre_step_disagreement),
        "state_continuity_ok": continuity_ok,
        "rollback_ok": rollback_ok,
        "sample_linkage": sample_linkage,
        "reward_replay_ok": reward_replay_ok,
        "quality_scope": {
            "sample_count": len(samples),
            "validation_record_count": len(validation),
            "validation_scope_limited": quality_scope_limited,
            "validator_pass": validator_pass,
            "validator_pass_rate": validator_pass / len(samples) if samples else 0.0,
            "max_length_hit": max_length,
            "natural_end": natural_end,
            "empty_response": empty,
            "mean_repetition_penalty": sum(repetition_values) / len(repetition_values) if repetition_values else 0.0,
            "categories": dict(categories),
            "termination_reasons": dict(termination_reasons),
            "quality_claim_allowed": False,
        },
        "checkpoint_reload_ok": checkpoint_reload_ok,
        "checkpoint_count": len(checkpoint_rows),
        "selection_status": selection.get("status"),
        "run_exit_ok": run_exit_ok,
        "final_contract_audit_ok": final_contract_ok,
        "all_json_finite": True,
        "warnings": [
            *(["BF16_SHADOW_DIVERGENCE"] if shadow_warning else []),
            *(["PRESTEP_PRECISION_DIVERGENCE"] if any(pre_step_disagreement.values()) else []),
            *(["VALIDATION_SCOPE_TWO_PROMPTS"] if quality_scope_limited else []),
            "NO_RL_QUALITY_CLAIM",
        ],
        "diagnostic_only": True,
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        "default_model_changed": False,
        "matrix": matrix,
        "metadata_run_id": metadata.get("run_id"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "step_precision.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in step_precision),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join([
            "# Four-step precision and quality-scope audit",
            "",
            f"- status: {status}",
            f"- accepted steps: {accepted_steps}",
            f"- active/full-FP32 agreement: {result['active_full_fp32_agreement']}",
            f"- BF16 shadow warning: {shadow_warning}",
            f"- pre-step disagreement counts: {dict(pre_step_disagreement)}",
            f"- sample linkage: {sample_linkage['complete']}",
            f"- checkpoint reload: {checkpoint_reload_ok}",
            f"- quality scope: {len(validation)} validation records; no quality claim permitted",
            "",
            "Offline diagnostic only; no model, reward, gate or default weight was changed.",
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
    parser.add_argument("--expected-steps", type=int, default=4)
    result = audit(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
