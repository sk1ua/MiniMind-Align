"""Replay a deterministic nonzero-advantage GRPO/CISPO contract fixture."""

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

from align.rl_rules import (
    clipped_policy_diagnostic_terms,
    clipped_policy_loss,
    group_advantages,
    training_forward_uses_autocast,
)
from evaluation.audit_corrected_kl_gate import ensure_empty_output_dir


def json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return True


def _mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask.bool()]
    return float(selected.mean().detach().cpu()) if selected.numel() else 0.0


def replay_variant(
    fixture: dict,
    mode: str,
    variant: dict,
) -> dict[str, object]:
    mask = torch.tensor(fixture["completion_mask"], dtype=torch.float32)
    old = torch.tensor(fixture["old_log_probs"], dtype=torch.float32)
    reference = torch.tensor(fixture["ref_log_probs"], dtype=torch.float32)
    advantages = torch.tensor(fixture["advantages"], dtype=torch.float32)
    new = torch.tensor(variant["new_log_probs"], dtype=torch.float32, requires_grad=True)
    loss, kl = clipped_policy_loss(
        new,
        old,
        reference,
        advantages,
        mask,
        mode=mode,
        beta=float(variant["beta"]),
        epsilon=float(variant["epsilon"]),
        epsilon_high=float(variant["epsilon_high"]),
    )
    terms = clipped_policy_diagnostic_terms(
        new,
        old,
        reference,
        advantages,
        mode=mode,
        beta=float(variant["beta"]),
        epsilon=float(variant["epsilon"]),
        epsilon_high=float(variant["epsilon_high"]),
    )
    ratio = torch.exp(new.detach() - old)
    if mode == "grpo":
        clipped_ratio = torch.clamp(ratio, 1.0 - float(variant["epsilon"]), 1.0 + float(variant["epsilon"]))
        raw_objective = ratio * advantages.unsqueeze(1)
        clipped_objective = clipped_ratio * advantages.unsqueeze(1)
        clipped_token_count = int(((raw_objective - clipped_objective).abs() > 1e-6).sum().item())
        expected_objective = torch.minimum(raw_objective, clipped_objective)
    else:
        clipped_ratio = torch.clamp(ratio, max=float(variant["epsilon_high"]))
        clipped_token_count = int((ratio > float(variant["epsilon_high"])).sum().item())
        expected_objective = clipped_ratio.detach() * advantages.unsqueeze(1) * new.detach()
    diagnostic_loss = _mean(terms["token_loss"], mask)
    diagnostic_kl = _mean(terms["reference_kl"], mask)
    objective_matches = bool(torch.allclose(
        terms["policy_objective"], expected_objective.float(), rtol=1e-6, atol=1e-6
    ))
    loss.backward()
    fp32_gradient_norm = float(new.grad.detach().float().norm().cpu())

    quantized = torch.tensor(variant["new_log_probs"], dtype=torch.bfloat16).float().detach().requires_grad_()
    quantized_loss, quantized_kl = clipped_policy_loss(
        quantized,
        old,
        reference,
        advantages,
        mask,
        mode=mode,
        beta=float(variant["beta"]),
        epsilon=float(variant["epsilon"]),
        epsilon_high=float(variant["epsilon_high"]),
    )
    quantized_loss.backward()
    bfloat16_gradient_norm = float(quantized.grad.detach().float().norm().cpu())
    loss_match = math.isclose(float(loss.detach()), diagnostic_loss, rel_tol=1e-6, abs_tol=1e-6)
    kl_match = math.isclose(float(kl.detach()), diagnostic_kl, rel_tol=1e-6, abs_tol=1e-6)
    finite = all(math.isfinite(value) for value in (
        float(loss.detach()), float(kl.detach()), diagnostic_loss, diagnostic_kl,
        fp32_gradient_norm, bfloat16_gradient_norm,
        float(quantized_loss.detach()), float(quantized_kl.detach()),
    ))
    return {
        "mode": mode,
        "advantage_nonzero": bool(float(advantages.abs().max()) > 0.0),
        "token_count": int(mask.sum().item()),
        "ratio_min": float(ratio.min()),
        "ratio_max": float(ratio.max()),
        "clipped_token_count": clipped_token_count,
        "expected_clipped_token_count": int(variant["expected_clipped_token_count"]),
        "clipping_observed": clipped_token_count == int(variant["expected_clipped_token_count"]),
        "objective_matches_clipped_formula": objective_matches,
        "production_loss": float(loss.detach()),
        "diagnostic_loss": diagnostic_loss,
        "loss_match": loss_match,
        "production_kl": float(kl.detach()),
        "diagnostic_kl": diagnostic_kl,
        "kl_match": kl_match,
        "fp32_gradient_norm": fp32_gradient_norm,
        "bfloat16_quantized_gradient_norm": bfloat16_gradient_norm,
        "fp32_gradient_nonzero": fp32_gradient_norm > 0.0,
        "bfloat16_quantized_gradient_nonzero": bfloat16_gradient_norm > 0.0,
        "finite": finite,
    }


def evaluate_fixture(fixture: dict) -> dict[str, object]:
    rewards = torch.tensor(fixture["rewards"], dtype=torch.float32)
    expected_advantages = torch.tensor(fixture["advantages"], dtype=torch.float32)
    computed_advantages = group_advantages(rewards, int(fixture["num_generations"]))
    advantage_replay_ok = bool(torch.allclose(computed_advantages, expected_advantages, rtol=1e-6, atol=1e-6))
    records = [
        replay_variant(fixture, mode, fixture["variants"][mode])
        for mode in ("grpo", "cispo")
    ]
    active_mode_contract = {
        "legacy_bfloat16_autocast": training_forward_uses_autocast("legacy_bfloat16_autocast", True),
        "fp32_no_autocast": training_forward_uses_autocast("fp32_no_autocast", True),
    }
    passed = advantage_replay_ok and all(
        record["advantage_nonzero"]
        and record["clipping_observed"]
        and record["objective_matches_clipped_formula"]
        and record["loss_match"]
        and record["kl_match"]
        and record["fp32_gradient_nonzero"]
        and record["bfloat16_quantized_gradient_nonzero"]
        and record["finite"]
        for record in records
    )
    return {
        "status": "NONZERO_ADVANTAGE_CONTRACT_PASS" if passed else "NONZERO_ADVANTAGE_CONTRACT_FAIL",
        "fixture_id": fixture["fixture_id"],
        "seed": fixture["seed"],
        "advantage_replay_ok": advantage_replay_ok,
        "computed_advantages": computed_advantages.tolist(),
        "active_mode_contract": active_mode_contract,
        "records": records,
        "all_json_finite": finite_tree(records),
        "diagnostic_only": True,
        "model_change_gate": "NOT_MET_NO_MODEL_CHANGE",
        "default_model_changed": False,
    }


def audit(args: argparse.Namespace) -> dict[str, object]:
    fixture_path = Path(args.fixture)
    output_dir = Path(args.output_dir)
    ensure_empty_output_dir(output_dir)
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not finite_tree(fixture):
        raise ValueError("fixture contains non-finite values")
    result = evaluate_fixture(fixture)
    result.update({
        "fixture": str(fixture_path),
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
    })
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join([
            "# Nonzero advantage contract audit",
            "",
            f"- status: {result['status']}",
            f"- advantage replay: {result['advantage_replay_ok']}",
            f"- GRPO clipped tokens: {result['records'][0]['clipped_token_count']}",
            f"- CISPO clipped tokens: {result['records'][1]['clipped_token_count']}",
            "",
            "Offline contract only; no model weights or default model changed.",
            "",
        ]),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output-dir", required=True)
    print(json.dumps(audit(parser.parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
