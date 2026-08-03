"""Audit same-token training-loss and pre-step KL precision telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from align.rl_rules import clipped_policy_diagnostic_terms
from evaluation.audit_corrected_kl_gate import (
    close,
    ensure_empty_output_dir,
    load_json,
    load_jsonl,
    validate_attempts,
)
from evaluation.audit_kl_token_replay import replay_row_check, replay_variant_groups


EXPECTED_POST_VARIANTS = {
    "bfloat16_autocast",
    "bfloat16_no_autocast",
    "full_float32_no_autocast",
}
TERM_NAMES = {"ratio", "reference_kl", "policy_objective", "kl_penalty", "token_loss"}


def json_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode("utf-8")).hexdigest()


def tensors_close(left: object, right: torch.Tensor, tolerance: float = 5e-7) -> bool:
    try:
        observed = torch.tensor(left, dtype=torch.float32)
    except (TypeError, ValueError):
        return False
    return observed.shape == right.shape and bool(
        torch.allclose(observed, right.detach().float().cpu(), rtol=1e-6, atol=tolerance)
    )


def masked_means(terms: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, float]:
    selected = mask.bool()
    return {
        name: float(value[selected].mean()) if bool(selected.any()) else 0.0
        for name, value in terms.items()
    }


def validate_prestep_replay_row(row: dict) -> dict[str, object]:
    required = {
        "generated_ids",
        "completion_mask",
        "old_log_probs",
        "ref_log_probs",
        "advantages",
        "bfloat16_autocast_new_log_probs",
        "fp32_no_autocast_new_log_probs",
        "bfloat16_autocast_terms",
        "fp32_no_autocast_terms",
        "source_sha256",
    }
    missing = sorted(required - set(row))
    if missing:
        return {"valid": False, "step": row.get("step"), "micro_index": row.get("micro_index"), "missing": missing}
    try:
        mask = torch.tensor(row["completion_mask"], dtype=torch.float32)
        old = torch.tensor(row["old_log_probs"], dtype=torch.float32)
        reference = torch.tensor(row["ref_log_probs"], dtype=torch.float32)
        advantages = torch.tensor(row["advantages"], dtype=torch.float32)
        variants = {
            "bfloat16_autocast": torch.tensor(row["bfloat16_autocast_new_log_probs"], dtype=torch.float32),
            "fp32_no_autocast": torch.tensor(row["fp32_no_autocast_new_log_probs"], dtype=torch.float32),
        }
        shape_ok = (
            mask.shape == old.shape == reference.shape
            and all(value.shape == mask.shape for value in variants.values())
            and advantages.ndim == 1
            and advantages.shape[0] == mask.shape[0]
        )
        term_checks = {}
        mean_checks = {}
        loss_checks = {}
        kl_checks = {}
        for variant, new_log_probs in variants.items():
            terms = clipped_policy_diagnostic_terms(
                new_log_probs,
                old,
                reference,
                advantages,
                mode=str(row["mode"]),
                beta=float(row["beta"]),
                epsilon=float(row["epsilon"]),
                epsilon_high=float(row["epsilon_high"]),
            )
            persisted_terms = row[f"{variant}_terms"]
            term_checks[variant] = set(persisted_terms) == TERM_NAMES and all(
                tensors_close(persisted_terms[name], terms[name]) for name in TERM_NAMES
            )
            means = masked_means(terms, mask)
            persisted_means = row[f"{variant}_means"]
            mean_checks[variant] = all(close(persisted_means.get(name), value) for name, value in means.items())
            loss_checks[variant] = close(row[f"{variant}_loss"], means["token_loss"])
            kl_checks[variant] = close(row[f"{variant}_kl"], means["reference_kl"])
        source = row["source_sha256"]
        hash_checks = {
            "generated": source.get("generated") == json_sha256(row["generated_ids"]),
            "completion_mask": source.get("completion_mask") == json_sha256(row["completion_mask"]),
            "old_log_probs": source.get("old_log_probs") == json_sha256(row["old_log_probs"]),
            "ref_log_probs": source.get("ref_log_probs") == json_sha256(row["ref_log_probs"]),
            "advantages": source.get("advantages") == json_sha256(row["advantages"]),
        }
        finite = all(
            math.isfinite(float(value))
            for value in (
                row["bfloat16_autocast_loss"],
                row["fp32_no_autocast_loss"],
                row["bfloat16_autocast_kl"],
                row["fp32_no_autocast_kl"],
                row["fp32_shadow_gradient_norm_scaled"],
                row["fp32_shadow_gradient_norm_unscaled"],
            )
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        return {
            "valid": False,
            "step": row.get("step"),
            "micro_index": row.get("micro_index"),
            "error": str(error),
        }
    valid = all([
        shape_ok,
        all(term_checks.values()),
        all(mean_checks.values()),
        all(loss_checks.values()),
        all(kl_checks.values()),
        all(hash_checks.values()),
        finite,
        row.get("fp32_shadow_grad_isolation_ok") is True,
    ])
    return {
        "valid": valid,
        "step": int(row["step"]),
        "micro_index": int(row["micro_index"]),
        "shape_ok": shape_ok,
        "term_checks": term_checks,
        "mean_checks": mean_checks,
        "loss_checks": loss_checks,
        "kl_checks": kl_checks,
        "hash_checks": hash_checks,
        "finite": finite,
        "shadow_grad_isolation_ok": row.get("fp32_shadow_grad_isolation_ok") is True,
    }


def determine_status(
    *,
    integrity_ok: bool,
    step_precision: list[dict],
    micro_precision: list[dict],
) -> str:
    if not integrity_ok:
        return "TELEMETRY_INCOMPLETE"
    step_loss_or_kl = any(
        row.get("loss_disagreement") is True or row.get("kl_disagreement") is True
        for row in step_precision
    )
    gradient_rows = sum(row.get("gradient_disagreement") is True for row in micro_precision)
    if step_loss_or_kl and gradient_rows >= 2:
        return "TRAINING_AUTOCAST_PRECISION_SENSITIVE"
    if step_loss_or_kl:
        return "PRESTEP_KL_ONLY_PRECISION_SENSITIVE"
    if not any(row.get("gradient_disagreement") is True for row in micro_precision):
        return "PRESTEP_PRECISION_CONSISTENT"
    return "MIXED_PRESTEP_PRECISION_UNRESOLVED"


def audit(args: argparse.Namespace) -> dict[str, object]:
    experiment_root = Path(args.experiment_root)
    run_dir = experiment_root / args.run_name
    output_dir = Path(args.output_dir)
    ensure_empty_output_dir(output_dir)
    required = [
        experiment_root / "matrix.json",
        run_dir / "pre_step_loss_replay.jsonl",
        run_dir / "microbatch_summaries.jsonl",
        run_dir / "step_summaries.jsonl",
        run_dir / "kl_guard_attempts.jsonl",
        run_dir / "kl_guard_token_replay.jsonl",
        run_dir / "selection.json",
        run_dir / "validation_history.jsonl",
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

    matrix = load_json(experiment_root / "matrix.json")
    replay_rows = load_jsonl(run_dir / "pre_step_loss_replay.jsonl")
    micro_rows = load_jsonl(run_dir / "microbatch_summaries.jsonl")
    step_rows = load_jsonl(run_dir / "step_summaries.jsonl")
    attempts = load_jsonl(run_dir / "kl_guard_attempts.jsonl")
    post_replay_rows = load_jsonl(run_dir / "kl_guard_token_replay.jsonl")
    validation_rows = load_jsonl(run_dir / "validation_history.jsonl")
    selection = load_json(run_dir / "selection.json")
    target = float(matrix["guard"]["post_step_kl_target"])

    replay_checks = [validate_prestep_replay_row(row) for row in replay_rows]
    replay_valid = bool(replay_checks) and all(row["valid"] for row in replay_checks)
    replay_keys = {(int(row["step"]), int(row["micro_index"])) for row in replay_rows}
    micro_keys = {(int(row["step"]), int(row["micro_index"])) for row in micro_rows}
    linkage_ok = replay_keys == micro_keys and all(
        next(item for item in micro_rows if (int(item["step"]), int(item["micro_index"])) == key).get("sample_keys")
        == next(item for item in replay_rows if (int(item["step"]), int(item["micro_index"])) == key).get("sample_keys")
        for key in replay_keys
    )
    micro_precision = [row.get("pre_step_loss_precision") for row in micro_rows]
    step_precision = [row.get("pre_step_loss_precision") for row in step_rows]
    precision_complete = all(isinstance(row, dict) for row in micro_precision + step_precision)
    shadow_isolation_ok = precision_complete and all(row.get("shadow_grad_isolation_ok") is True for row in micro_precision)

    attempt_result = validate_attempts(attempts, target)
    accepted_steps = attempt_result["accepted_steps"]
    continuity_ok = attempt_result["state_continuity_ok"] if len(accepted_steps) >= 2 else True
    post_replay_checks = [replay_row_check(row) for row in post_replay_rows]
    post_replay_valid = bool(post_replay_checks) and all(row.get("valid") for row in post_replay_checks)
    post_groups = replay_variant_groups(post_replay_rows) if post_replay_valid else {}
    post_groups_complete = bool(post_groups) and all(set(group) == EXPECTED_POST_VARIANTS for group in post_groups.values())
    checkpoint_rows = [row for row in validation_rows if row.get("checkpoint")]
    checkpoint_reload_ok = bool(checkpoint_rows) and all(
        row.get("evaluation_source") == "reloaded_checkpoint" and Path(row["checkpoint"]).exists()
        for row in checkpoint_rows
    )
    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    run_exit_ok = "EXIT_CODE=0" in run_log
    post_gate_ok = accepted_steps == [1, 2] and all(
        row.get("post_step_kl_gate_mode") == "fp32_no_autocast"
        and row.get("post_step_kl_gate_passed") is True
        for row in step_rows
    )
    integrity_ok = all([
        replay_valid,
        linkage_ok,
        precision_complete,
        shadow_isolation_ok,
        attempt_result["gate_consistent"],
        attempt_result["accepted_state_ok"],
        continuity_ok,
        post_replay_valid,
        post_groups_complete,
        checkpoint_reload_ok,
        run_exit_ok,
        post_gate_ok,
    ])
    status = determine_status(
        integrity_ok=integrity_ok,
        step_precision=step_precision if precision_complete else [],
        micro_precision=micro_precision if precision_complete else [],
    )
    nonzero_advantage_microbatches = (
        sum(int(row.get("nonzero_advantage_microbatches", 0)) for row in step_precision)
        if precision_complete else 0
    )
    warnings = ["CHECKPOINT_SELECTION_SMOKE_ONLY"]
    if nonzero_advantage_microbatches == 0:
        warnings.append("ZERO_ADVANTAGE_KL_ONLY_SMOKE")
    if any(row.get("shadow_gate_disagreement") is True for row in attempts):
        warnings.append("LEGACY_BF16_POSTSTEP_SHADOW_DISAGREEMENT")

    result = {
        "status": status,
        "experiment_root": str(experiment_root),
        "run_name": args.run_name,
        "steps_completed": len(step_rows),
        "accepted_steps": accepted_steps,
        "pre_step_replay_rows": len(replay_rows),
        "pre_step_replay_valid": replay_valid,
        "sample_linkage_ok": linkage_ok,
        "shadow_grad_isolation_ok": shadow_isolation_ok,
        "step_precision": step_precision,
        "microbatch_loss_disagreement_count": sum(row.get("loss_disagreement") is True for row in micro_precision) if precision_complete else 0,
        "microbatch_kl_disagreement_count": sum(row.get("kl_disagreement") is True for row in micro_precision) if precision_complete else 0,
        "microbatch_gradient_disagreement_count": sum(row.get("gradient_disagreement") is True for row in micro_precision) if precision_complete else 0,
        "nonzero_advantage_microbatches": nonzero_advantage_microbatches,
        "post_gate_ok": post_gate_ok,
        "post_replay_rows": len(post_replay_rows),
        "post_replay_valid": post_replay_valid,
        "state_continuity_ok": attempt_result["state_continuity_ok"],
        "checkpoint_reload_ok": checkpoint_reload_ok,
        "checkpoint_count": len(checkpoint_rows),
        "selection_status": selection.get("status"),
        "run_exit_ok": run_exit_ok,
        "warnings": warnings,
        "all_json_finite": True,
        "diagnostic_only": True,
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        "default_model_changed": False,
        "limitation": "Zero-advantage smoke isolates the KL-loss path and does not validate nonzero-advantage ratio clipping or model quality.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "replay_checks.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in replay_checks),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join([
            "# Pre-step precision audit",
            "",
            f"- status: {status}",
            f"- accepted steps: {accepted_steps}",
            f"- replay rows: {len(replay_rows)}",
            f"- loss disagreements: {result['microbatch_loss_disagreement_count']}",
            f"- KL disagreements: {result['microbatch_kl_disagreement_count']}",
            f"- gradient disagreements: {result['microbatch_gradient_disagreement_count']}",
            f"- warnings: {', '.join(warnings)}",
            "",
            "Diagnostic only; production loss and the default model are unchanged.",
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
