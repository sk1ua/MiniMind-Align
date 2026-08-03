"""Offline validator for token-level reference-KL replay telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

# Allow this file to be invoked directly from the repository root while still
# importing the sibling evaluation module as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.audit_reference_kl_semantics import independent_aggregate


VARIANTS = {
    "bfloat16_autocast": ("post_step_kl_bfloat16",),
    "bfloat16_no_autocast": ("post_step_kl_float32",),
    "full_float32_no_autocast": ("post_step_kl_full_float32",),
}

# The trainer evaluates exp(delta) - delta - 1 in torch.float32 and then
# serializes the result. Recomputing the expression with Python float64 can
# differ by a few 1e-7 because of cancellation near delta == 0. Keep the
# replay check strict enough to catch real mismatches while allowing that
# documented float32 serialization round-off.
TOKEN_KL_ABS_TOL = 5e-7


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
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not finite_tree(value):
            raise ValueError(f"invalid or non-finite JSONL row {path}:{line_number}")
        rows.append(value)
    return rows


def payload_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def close(left: float, right: float, *, rel_tol: float = 1e-6, abs_tol: float = TOKEN_KL_ABS_TOL) -> bool:
    return math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol)


def replay_row_check(row: dict) -> dict[str, object]:
    required = {
        "step",
        "attempt_index",
        "variant",
        "rollout_index",
        "generated_ids",
        "completion_mask",
        "ref_log_probs",
        "new_log_probs",
        "token_kl_values",
        "valid_token_kl_values",
        "sample_keys",
        "generated_sha256",
        "mask_sha256",
        "reference_log_probs_sha256",
        "new_log_probs_sha256",
    }
    if not required.issubset(row):
        return {"valid": False, "reason": "missing_replay_fields"}
    generated = row["generated_ids"]
    mask = row["completion_mask"]
    refs = row["ref_log_probs"]
    news = row["new_log_probs"]
    token_values = row["token_kl_values"]
    valid_values = row["valid_token_kl_values"]
    if not (isinstance(mask, list) and isinstance(refs, list) and isinstance(news, list)):
        return {"valid": False, "reason": "replay_arrays_not_lists"}
    if not (len(mask) == len(refs) == len(news) == len(token_values)):
        return {"valid": False, "reason": "batch_dimension_mismatch"}
    expected_valid = []
    for batch_mask, batch_ref, batch_new, batch_token in zip(mask, refs, news, token_values):
        if not (len(batch_mask) == len(batch_ref) == len(batch_new) == len(batch_token)):
            return {"valid": False, "reason": "token_dimension_mismatch"}
        for mask_value, ref, new, actual in zip(batch_mask, batch_ref, batch_new, batch_token):
            delta = float(ref) - float(new)
            expected = math.exp(delta) - delta - 1.0
            if not close(actual, expected):
                return {"valid": False, "reason": "token_kl_formula_mismatch"}
            if bool(mask_value):
                expected_valid.append(expected)
    if len(valid_values) != len(expected_valid) or not all(
        close(actual, expected) for actual, expected in zip(valid_values, expected_valid)
    ):
        return {"valid": False, "reason": "masked_token_kl_mismatch"}
    digest_checks = {
        "generated_sha256": payload_sha(generated) == row["generated_sha256"],
        "mask_sha256": payload_sha(mask) == row["mask_sha256"],
        "reference_log_probs_sha256": payload_sha(refs) == row["reference_log_probs_sha256"],
        "new_log_probs_sha256": payload_sha(news) == row["new_log_probs_sha256"],
    }
    if not all(digest_checks.values()):
        return {"valid": False, "reason": "replay_digest_mismatch", **digest_checks}
    if not row["sample_keys"]:
        return {"valid": False, "reason": "sample_linkage_missing"}
    return {
        "valid": True,
        "key": [row["step"], row["attempt_index"], row["rollout_index"]],
        "variant": row["variant"],
        "step": row["step"],
        "attempt_index": row["attempt_index"],
        "rollout_index": row["rollout_index"],
        "micro_index": row.get("micro_index"),
        "prompt_id": row.get("prompt_id"),
        "category": row.get("category"),
        "sample_keys": row["sample_keys"],
        "valid_token_count": len(expected_valid),
        "valid_token_kl_mean": sum(expected_valid) / max(1, len(expected_valid)),
    }


def replay_variant_groups(rows: list[dict]) -> dict[tuple[int, int, int], dict[str, dict]]:
    groups: dict[tuple[int, int, int], dict[str, dict]] = {}
    for row in rows:
        key = (int(row["step"]), int(row["attempt_index"]), int(row["rollout_index"]))
        groups.setdefault(key, {})[str(row["variant"])] = row
    return groups


def aggregate_variant(rows: list[dict], variant: str) -> dict[str, float]:
    values = []
    for row in rows:
        if row.get("variant") == variant:
            values.append([float(value) for value in row["valid_token_kl_values"]])
    return independent_aggregate(values)


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
        run_dir / "kl_guard_token_replay.jsonl",
        run_dir / "selection.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        result = {
            "status": "TELEMETRY_INCOMPLETE",
            "missing": missing,
            "diagnostic_only": True,
            "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
            "gpu_wall_seconds": 0,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result

    matrix = load_json(root / "matrix.json")
    attempts = load_jsonl(run_dir / "kl_guard_attempts.jsonl")
    replay_rows = load_jsonl(run_dir / "kl_guard_token_replay.jsonl")
    target = float(matrix.get("guard", {}).get("post_step_kl_target", 0.005))
    row_checks = [replay_row_check(row) for row in replay_rows]
    rows_valid = bool(row_checks) and all(check["valid"] for check in row_checks)
    groups = replay_variant_groups(replay_rows) if rows_valid else {}
    expected_variants = set(VARIANTS)
    complete_groups = bool(groups) and all(set(group) == expected_variants for group in groups.values())
    same_rollout = True
    for group in groups.values():
        generated_hashes = {row["generated_sha256"] for row in group.values()}
        mask_hashes = {row["mask_sha256"] for row in group.values()}
        reference_hashes = {row["reference_log_probs_sha256"] for row in group.values()}
        sample_key_sets = {json.dumps(row.get("sample_keys", []), sort_keys=True) for row in group.values()}
        same_rollout &= len(generated_hashes) == 1 and len(mask_hashes) == 1
        same_rollout &= len(reference_hashes) == 1 and len(sample_key_sets) == 1

    attempt_by_key = {(int(row["step"]), int(row["attempt_index"])): row for row in attempts}
    replay_attempt_keys = {(int(row["step"]), int(row["attempt_index"])) for row in replay_rows}
    attempt_alignment = replay_attempt_keys == set(attempt_by_key)
    aggregate_checks = []
    for key, attempt in attempt_by_key.items():
        step, attempt_index = key
        selected_rows = [
            row for row in replay_rows
            if int(row["step"]) == step and int(row["attempt_index"]) == attempt_index
        ]
        for variant, (attempt_field,) in VARIANTS.items():
            aggregate = aggregate_variant(selected_rows, variant)
            expected = attempt[attempt_field]
            aggregate_checks.append({
                "step": step,
                "attempt_index": attempt_index,
                "variant": variant,
                "token_count_matches": float(aggregate["token_count"]) > 0,
                "mean_matches": close(aggregate["reference_kl_mean"], expected["reference_kl_mean"]),
                "p95_matches": close(aggregate["reference_kl_p95"], expected["reference_kl_p95"], rel_tol=2e-5, abs_tol=2e-7),
                "max_matches": close(aggregate["reference_kl_max"], expected["reference_kl_max"], rel_tol=2e-5, abs_tol=2e-7),
                "replay": aggregate,
                "attempt": expected,
            })
    aggregate_consistent = bool(aggregate_checks) and all(
        check["token_count_matches"]
        and check["mean_matches"]
        and check["p95_matches"]
        and check["max_matches"]
        for check in aggregate_checks
    )
    rollback_verified = bool(attempts) and all(
        (row.get("rollback_after_attempt") or {}).get("exact_match") is True
        for row in attempts
    )
    source_and_schema = rows_valid and complete_groups and same_rollout and attempt_alignment and aggregate_consistent
    if not source_and_schema or not rollback_verified:
        status = "TOKEN_REPLAY_INCONSISTENT"
    else:
        status = "TOKEN_REPLAY_VALIDATED"
    result = {
        "status": status,
        "experiment_root": str(root),
        "run_name": args.run_name,
        "target": target,
        "attempts": len(attempts),
        "replay_rows": len(replay_rows),
        "variants": sorted(expected_variants),
        "row_checks_pass": rows_valid,
        "complete_variant_groups": complete_groups,
        "same_rollout_mask_reference": same_rollout,
        "attempt_alignment": attempt_alignment,
        "aggregate_consistent": aggregate_consistent,
        "rollback_verified": rollback_verified,
        "all_json_finite": True,
        "diagnostic_only": True,
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        "gpu_wall_seconds": 0,
        "limitation": "Replay validates persisted token-level arithmetic and variant alignment; it does not prove training causality.",
        "next_decision": "Only after replay validation may a corrected no-autocast KL gate smoke be considered; optimizer/update-scale remains unchanged.",
        "row_analysis": row_checks,
        "aggregate_checks": aggregate_checks,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "row_analysis.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in row_checks),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join([
            "# KL token replay audit",
            "",
            f"- status: {status}",
            f"- attempts: {len(attempts)}",
            f"- replay rows: {len(replay_rows)}",
            f"- complete variant groups: {complete_groups}",
            f"- same rollout/mask/reference: {same_rollout}",
            f"- aggregate consistency: {aggregate_consistent}",
            f"- rollback verified: {rollback_verified}",
            "",
            "The audit is offline and diagnostic-only. It does not change the production gate, reward, optimizer/update-scale, checkpoint selection, or default model.",
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
    args = parser.parse_args()
    print(json.dumps(audit(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
