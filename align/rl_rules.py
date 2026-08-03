"""Rule rewards and clipped policy losses for the controlled RL pilot."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from dataset.alignment_v2.validators import repeat_3gram_ratio, validate_record


POST_STEP_KL_GATE_MODES = {
    "legacy_bfloat16_autocast",
    "fp32_no_autocast",
}

TRAINING_FORWARD_MODES = {
    "legacy_bfloat16_autocast",
    "fp32_no_autocast",
}


def parse_controlled_reward_pattern(pattern: str | None, num_generations: int) -> list[float] | None:
    """Parse an explicit diagnostic reward pattern for one prompt group.

    The default ``None`` path preserves rule rewards exactly.  A controlled
    pattern is deliberately strict: it must have one finite value per sampled
    candidate and must differ within the group so group-relative advantages
    cannot silently collapse to zero.
    """
    if pattern is None:
        return None
    if num_generations < 2:
        raise ValueError("controlled reward pattern requires at least two generations")
    pieces = [piece.strip() for piece in pattern.split(",")]
    if len(pieces) != num_generations or any(piece == "" for piece in pieces):
        raise ValueError(
            f"controlled reward pattern must contain exactly {num_generations} comma-separated values"
        )
    try:
        values = [float(piece) for piece in pieces]
    except ValueError as exc:
        raise ValueError("controlled reward pattern values must be numeric") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("controlled reward pattern values must be finite")
    if max(values) == min(values):
        raise ValueError("controlled reward pattern must produce nonzero group-relative advantages")
    return values


def training_forward_uses_autocast(mode: str, use_autocast: bool) -> bool:
    """Return whether the selected training forward uses bfloat16 autocast.

    The helper is deliberately pure so the default legacy path can be tested
    without constructing a model.  ``use_autocast`` captures device/dtype
    capability; the explicit mode is the only new opt-in behavior.
    """
    if mode not in TRAINING_FORWARD_MODES:
        raise ValueError(f"unsupported training forward mode: {mode}")
    return bool(use_autocast and mode == "legacy_bfloat16_autocast")


def rule_reward(category: str, prompt: str, response: str, metadata: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Return a bounded rule reward and auditable components.

    The validator is the main task signal.  Length and repetition are small
    guardrails, so the first RL pilot cannot optimize an open-ended prose
    judge or an external reward model by accident.
    """
    record = {"conversations": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]}
    passed, _ = validate_record(record, {"category": category, "metadata": metadata})
    validator_reward = 1.0 if passed else 0.0
    repetition_penalty = min(0.5, repeat_3gram_ratio(response))
    termination_reward = 0.1 if response.strip() and "\n" not in response else 0.0
    components = {
        "validator_reward": validator_reward,
        "parse_reward": validator_reward if category == "format" else 0.0,
        "field_reward": validator_reward if category == "format" else 0.0,
        "item_count_reward": validator_reward if category in {"format", "instruction", "repetition"} else 0.0,
        "arithmetic_reward": validator_reward if category == "reasoning" else 0.0,
        "format_reward": validator_reward if category in {"format", "instruction", "reasoning"} else 0.0,
        "termination_reward": termination_reward,
        "repetition_penalty": repetition_penalty,
    }
    return validator_reward + termination_reward - repetition_penalty, components


def group_advantages(rewards: Tensor, num_generations: int) -> Tensor:
    """Normalize rewards within each prompt group."""
    if rewards.numel() % num_generations:
        raise ValueError("reward count must be divisible by num_generations")
    groups = rewards.view(-1, num_generations)
    mean = groups.mean(dim=1, keepdim=True)
    std = groups.std(dim=1, unbiased=False, keepdim=True)
    return ((groups - mean) / (std + 1e-4)).reshape(-1)


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Average values over valid completion tokens."""
    return (values * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def reference_kl_values(
    new_log_probs: Tensor,
    ref_log_probs: Tensor,
    completion_mask: Tensor,
) -> Tensor:
    """Return token-level reference KL values under a completion mask."""
    values = torch.exp((ref_log_probs - new_log_probs).float()) - (ref_log_probs - new_log_probs).float() - 1.0
    masked = values[completion_mask.bool()]
    if masked.numel() == 0:
        return torch.zeros(1, dtype=torch.float32, device=new_log_probs.device)
    return masked


def reference_kl_stats(
    new_log_probs: Tensor,
    ref_log_probs: Tensor,
    completion_mask: Tensor,
) -> dict[str, float]:
    """Aggregate reference KL statistics for a post-update guard."""
    return aggregate_reference_kl_values([
        reference_kl_values(new_log_probs, ref_log_probs, completion_mask),
    ])


def aggregate_reference_kl_values(values: list[Tensor]) -> dict[str, float]:
    """Aggregate KL values from every micro-batch in one optimizer step."""
    if not values:
        return {
            "reference_kl_mean": 0.0,
            "reference_kl_p95": 0.0,
            "reference_kl_max": 0.0,
            "token_count": 0.0,
        }
    merged = torch.cat([value.float().reshape(-1) for value in values])
    return {
        "reference_kl_mean": float(merged.mean().detach().cpu()),
        "reference_kl_p95": float(torch.quantile(merged, 0.95).detach().cpu()),
        "reference_kl_max": float(merged.max().detach().cpu()),
        "token_count": float(merged.numel()),
    }


def kl_backoff_multipliers(backoff_factor: float, max_backoffs: int) -> list[float]:
    """Return the initial learning-rate multiplier and bounded backoffs."""
    if not 0.0 < backoff_factor < 1.0:
        raise ValueError("backoff_factor must be between 0 and 1")
    if max_backoffs < 0:
        raise ValueError("max_backoffs must be non-negative")
    return [backoff_factor**index for index in range(max_backoffs + 1)]


def finite_diagnostics(values: dict[str, float]) -> bool:
    """Return whether all scalar diagnostic values are finite."""
    return bool(torch.isfinite(torch.tensor(list(values.values()), dtype=torch.float32)).all())


def select_reference_kl_gate_stats(
    mode: str,
    legacy_bfloat16_stats: dict[str, float],
    fp32_no_autocast_stats: dict[str, float] | None,
) -> dict[str, float]:
    """Select the authoritative post-step KL stats without mutating either input."""
    if mode not in POST_STEP_KL_GATE_MODES:
        raise ValueError(f"unsupported post-step KL gate mode: {mode}")
    if mode == "legacy_bfloat16_autocast":
        return dict(legacy_bfloat16_stats)
    if fp32_no_autocast_stats is None:
        raise ValueError("fp32_no_autocast gate requires fp32 KL statistics")
    return dict(fp32_no_autocast_stats)


def dtype_gap_threshold(float32_mean: float, target: float) -> float:
    """Return the fixed bfloat16/float32 diagnostic gap threshold."""
    return max(1e-4, 0.1 * max(float32_mean, target))


def policy_diagnostics(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    ref_log_probs: Tensor,
    completion_mask: Tensor,
) -> dict[str, float]:
    """Return detached token-level diagnostics for a policy update.

    The values use the same completion mask as :func:`clipped_policy_loss`.
    This helper is intentionally detached and does not participate in the
    objective, so adding telemetry cannot change GRPO/CISPO behavior.
    """
    mask = completion_mask.bool()
    log_ratio = (new_log_probs - old_log_probs).detach().float()[mask]
    ref_delta = (ref_log_probs - new_log_probs).detach().float()[mask]
    if log_ratio.numel() == 0:
        log_ratio = torch.zeros(1, dtype=torch.float32, device=new_log_probs.device)
        ref_delta = torch.zeros(1, dtype=torch.float32, device=new_log_probs.device)
    ratio = torch.exp(log_ratio)
    reference_kl = torch.exp(ref_delta) - ref_delta - 1.0

    def value(tensor: Tensor, quantile: float | None = None) -> float:
        result = tensor if quantile is None else torch.quantile(tensor, quantile)
        return float(result.detach().cpu())

    return {
        "token_count": float(mask.sum().detach().cpu()),
        "ratio_mean": value(ratio.mean()),
        "ratio_p50": value(ratio, 0.50),
        "ratio_p95": value(ratio, 0.95),
        "ratio_max": value(ratio.max()),
        "log_ratio_mean": value(log_ratio.mean()),
        "log_ratio_abs_p95": value(log_ratio.abs(), 0.95),
        "reference_kl_mean": value(reference_kl.mean()),
        "reference_kl_p50": value(reference_kl, 0.50),
        "reference_kl_p95": value(reference_kl, 0.95),
        "reference_kl_max": value(reference_kl.max()),
    }


def clipped_policy_diagnostic_terms(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    ref_log_probs: Tensor,
    advantages: Tensor,
    *,
    mode: str,
    beta: float = 0.02,
    epsilon: float = 0.2,
    epsilon_high: float = 5.0,
) -> dict[str, Tensor]:
    """Return detached token terms matching :func:`clipped_policy_loss`.

    This helper is telemetry-only.  Keeping the production loss implementation
    below unchanged makes it possible to compare autocast and no-autocast
    forwards without changing the gradient-bearing objective.
    """
    if mode not in {"grpo", "cispo"}:
        raise ValueError(f"unknown mode: {mode}")
    ratio = torch.exp(new_log_probs - old_log_probs)
    reference_kl = torch.exp(ref_log_probs - new_log_probs) - (ref_log_probs - new_log_probs) - 1.0
    if mode == "cispo":
        weight = torch.clamp(ratio, max=epsilon_high).detach()
        policy_objective = weight * advantages.unsqueeze(1) * new_log_probs
    else:
        clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
        unclipped = ratio * advantages.unsqueeze(1)
        policy_objective = torch.minimum(unclipped, clipped * advantages.unsqueeze(1))
    kl_penalty = beta * reference_kl
    token_loss = -policy_objective + kl_penalty
    return {
        "ratio": ratio.detach().float(),
        "reference_kl": reference_kl.detach().float(),
        "policy_objective": policy_objective.detach().float(),
        "kl_penalty": kl_penalty.detach().float(),
        "token_loss": token_loss.detach().float(),
    }


def clipped_policy_loss(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    ref_log_probs: Tensor,
    advantages: Tensor,
    completion_mask: Tensor,
    *,
    mode: str,
    beta: float = 0.02,
    epsilon: float = 0.2,
    epsilon_high: float = 5.0,
) -> tuple[Tensor, Tensor]:
    """Return GRPO or CISPO loss and mean reference KL estimate.

    CISPO follows the repository's existing convention: the importance ratio
    is upper-clipped and detached, while the policy log-prob remains the
    gradient-bearing term.  GRPO uses the symmetric PPO-style ratio clip.
    """
    if mode not in {"grpo", "cispo"}:
        raise ValueError(f"unknown mode: {mode}")
    ratio = torch.exp(new_log_probs - old_log_probs)
    kl = torch.exp(ref_log_probs - new_log_probs) - (ref_log_probs - new_log_probs) - 1.0
    if mode == "cispo":
        weight = torch.clamp(ratio, max=epsilon_high).detach()
        token_loss = -(weight * advantages.unsqueeze(1) * new_log_probs - beta * kl)
    else:
        clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
        unclipped = ratio * advantages.unsqueeze(1)
        token_loss = -(torch.minimum(unclipped, clipped * advantages.unsqueeze(1)) - beta * kl)
    return masked_mean(token_loss, completion_mask), masked_mean(kl, completion_mask)
