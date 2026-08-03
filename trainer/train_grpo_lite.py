"""Controlled rule-reward GRPO/CISPO pilot for MiniMind-Align."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import random
import re
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from align.rl_rules import (
    POST_STEP_KL_GATE_MODES,
    TRAINING_FORWARD_MODES,
    clipped_policy_diagnostic_terms,
    clipped_policy_loss,
    aggregate_reference_kl_values,
    finite_diagnostics,
    group_advantages,
    kl_backoff_multipliers,
    policy_diagnostics,
    parse_controlled_reward_pattern,
    reference_kl_values,
    rule_reward,
    select_reference_kl_gate_stats,
    training_forward_uses_autocast,
)
from evaluation.rl_selection import quality_triggered, select_best_checkpoint
from evaluation.rl_validation import evaluate_policy
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def read_manifest(path: Path, categories: set[str], max_prompts: int) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["category"] in categories:
                if not row.get("metadata"):
                    row["metadata"] = legacy_metadata(row)
                    row["metadata_source"] = "derived_from_legacy_chosen"
                rows.append(row)
            if len(rows) >= max_prompts:
                break
    if not rows:
        raise ValueError("manifest filter produced no prompts")
    return rows


def interleave_rows_by_category(
    rows: list[dict[str, object]],
    category_order: list[str],
) -> list[dict[str, object]]:
    """Round-robin rows by category without changing row contents or seed order."""
    grouped: dict[str, list[dict[str, object]]] = {category: [] for category in category_order}
    for row in rows:
        category = str(row["category"])
        grouped.setdefault(category, []).append(row)
    ordered_categories = [category for category in category_order if grouped.get(category)]
    ordered_categories.extend(
        category for category in grouped if category not in ordered_categories and grouped[category]
    )
    output: list[dict[str, object]] = []
    index = 0
    while True:
        added = False
        for category in ordered_categories:
            bucket = grouped[category]
            if index < len(bucket):
                output.append(bucket[index])
                added = True
        if not added:
            return output
        index += 1


def legacy_metadata(row: dict[str, object]) -> dict[str, object]:
    """Derive conservative rule metadata for the legacy v1 train manifest.

    The registered 128-row train manifest predates Alignment v2 and carries an
    empty metadata object.  The original file is never modified.  This adapter
    only makes the existing prompt/answer constraints explicit in the resolved
    in-memory copy; validation and checkpoint selection still use the v2
    validation manifest with native metadata.
    """
    category = str(row["category"])
    prompt = str(row.get("prompt", ""))
    chosen = str(row.get("chosen", ""))
    if category == "conciseness":
        match = re.search(r"(?:不超过|以内)(\d+)字", prompt)
        return {"max_chars": int(match.group(1)) if match else 160, "required_terms": []}
    if category == "format":
        if chosen.lstrip().startswith("|"):
            expected = [[cell.strip() for cell in line.strip("|").split("|")] for line in chosen.splitlines() if line.strip()]
            return {"format_type": "markdown", "expected": expected}
        try:
            return {"format_type": "json_object", "expected": json.loads(chosen)}
        except json.JSONDecodeError:
            return {"format_type": "csv", "expected": list(csv.reader(io.StringIO(chosen), strict=True))}
    if category == "instruction":
        parts = [part.strip() for part in chosen.splitlines() if part.strip()]
        return {"count": len(parts), "separator": "\n", "allowed_words": parts}
    if category == "reasoning":
        match = re.search(r"=\s*(-?\d+(?:\.\d+)?)", chosen)
        return {"answer": float(match.group(1)) if match else 0.0}
    if category == "safety":
        markers = [marker for marker in ("不安全", "风险", "请立即", "靠边停车") if marker in chosen]
        return {"required_markers": markers or ["不安全", "风险"]}
    if category == "repetition":
        lines = [line.strip() for line in chosen.splitlines() if line.strip()]
        return {"count": len(lines)}
    if category == "termination":
        match = re.search(r"不超过(\d+)字", prompt)
        return {"max_chars": int(match.group(1)) if match else 160}
    if category == "uncertainty":
        markers = [marker for marker in ("无法确定", "不能确定", "实时数据", "外部数据") if marker in chosen]
        return {"required_markers": markers or ["无法确定"]}
    raise ValueError(f"unsupported legacy category: {category}")


def prompt_tokens(tokenizer, prompt: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    return encoded.input_ids.to(device), encoded.attention_mask.to(device)


def completion_log_probs(
    model: MiniMindForCausalLM,
    full_ids: torch.Tensor,
    prompt_length: int,
    completion_length: int,
    device: torch.device,
    requires_grad: bool,
) -> torch.Tensor:
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        attention = torch.ones_like(full_ids, device=device)
        logits = model(full_ids, attention_mask=attention).logits
        token_log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = full_ids[:, 1:].unsqueeze(-1)
        selected = token_log_probs.gather(-1, targets).squeeze(-1)
        start = prompt_length - 1
        return selected[:, start : start + completion_length]


def gradient_norm(model: MiniMindForCausalLM) -> float:
    """Return the finite L2 norm of currently accumulated gradients."""
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().norm(2).item()
        squared += value * value
    return squared**0.5


def gradient_snapshot(model: MiniMindForCausalLM) -> dict[int, torch.Tensor]:
    """Copy accumulated gradients so a later backward pass can be differenced."""
    return {
        id(parameter): parameter.grad.detach().float().clone()
        for parameter in model.parameters()
        if parameter.grad is not None
    }


def gradient_delta_norm(
    model: MiniMindForCausalLM,
    before: dict[int, torch.Tensor],
) -> float:
    """Return the L2 norm of the gradient contribution from one backward pass."""
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        current = parameter.grad.detach().float()
        previous = before.get(id(parameter))
        delta = current if previous is None else current - previous
        value = delta.norm(2).item()
        squared += value * value
    return squared**0.5


def gradient_sequence_norm(gradients: tuple[torch.Tensor | None, ...]) -> float:
    """Return an L2 norm for gradients returned by ``torch.autograd.grad``."""
    squared = 0.0
    for gradient in gradients:
        if gradient is None:
            continue
        value = gradient.detach().float().norm(2).item()
        squared += value * value
    return squared**0.5


def masked_diagnostic_means(
    terms: dict[str, torch.Tensor],
    completion_mask: torch.Tensor,
) -> dict[str, float]:
    """Aggregate detached diagnostic terms over the completion mask."""
    mask = completion_mask.bool()
    return {
        name: float(values[mask].mean().detach().cpu()) if mask.any() else 0.0
        for name, values in terms.items()
    }


def json_tensor_sha256(value: torch.Tensor) -> str:
    """Hash the JSON representation used by same-token replay records."""
    payload = value.detach().cpu().tolist()
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def snapshot_training_state(
    model: MiniMindForCausalLM,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Clone policy and optimizer state before a guarded optimizer step."""
    model_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    return model_state, copy.deepcopy(optimizer.state_dict())


def restore_training_state(
    model: MiniMindForCausalLM,
    optimizer: torch.optim.Optimizer,
    snapshot: tuple[dict[str, torch.Tensor], dict],
) -> None:
    """Restore a policy and optimizer snapshot without changing gradients."""
    model_state, optimizer_state = snapshot
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))


def _digest_update_tensor(digest: "hashlib._Hash", name: str, tensor: torch.Tensor) -> None:
    """Add a deterministic tensor representation to a state digest."""
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())


def policy_state_digest(model: MiniMindForCausalLM) -> str:
    """Return a deterministic digest of policy parameters and buffers."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if torch.is_tensor(value):
            _digest_update_tensor(digest, name, value)
        else:
            digest.update(f"{name}:{value!r}".encode("utf-8"))
    return digest.hexdigest()


def optimizer_state_digest(optimizer: torch.optim.Optimizer) -> str:
    """Return a deterministic digest of AdamW groups and state tensors."""
    state = optimizer.state_dict()
    digest = hashlib.sha256()
    digest.update(json.dumps(state["param_groups"], sort_keys=True, default=str, separators=(",", ":")).encode("utf-8"))
    for parameter_id, values in sorted(state["state"].items(), key=lambda item: int(item[0])):
        digest.update(f"param:{parameter_id}".encode("ascii"))
        for name, value in sorted(values.items()):
            if torch.is_tensor(value):
                _digest_update_tensor(digest, f"{parameter_id}:{name}", value)
            else:
                digest.update(f"{parameter_id}:{name}:{value!r}".encode("utf-8"))
    return digest.hexdigest()


def parameter_delta_metrics(
    model: MiniMindForCausalLM,
    before_state: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Measure the policy update relative to a pre-step state snapshot."""
    squared_delta = 0.0
    squared_before = 0.0
    max_abs = 0.0
    tensor_count = 0
    for name, value in model.state_dict().items():
        if not torch.is_tensor(value):
            continue
        before = before_state[name].to(value.device)
        delta = value.detach().float() - before.detach().float()
        squared_delta += float(torch.sum(delta * delta).cpu())
        squared_before += float(torch.sum(before.detach().float() * before.detach().float()).cpu())
        max_abs = max(max_abs, float(delta.abs().max().cpu()))
        tensor_count += 1
    delta_l2 = squared_delta**0.5
    before_l2 = squared_before**0.5
    return {
        "parameter_delta_l2": delta_l2,
        "parameter_delta_max_abs": max_abs,
        "parameter_relative_l2": delta_l2 / max(before_l2, 1e-12),
        "parameter_norm_before": before_l2,
        "parameter_tensor_count": float(tensor_count),
    }


def post_update_kl_stats(
    model: MiniMindForCausalLM,
    rollouts: list[tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]],
    device: torch.device,
    autocast_dtype: torch.dtype,
    use_autocast: bool,
    *,
    replay_path: Path | None = None,
    replay_context: list[dict[str, object]] | None = None,
    replay_variant: str | None = None,
    step: int | None = None,
    attempt_index: int | None = None,
) -> dict[str, float]:
    """Measure reference KL on every rollout used by one optimizer step."""
    values = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for rollout_index, (generated_cpu, mask_cpu, prompt_length, ref_log_probs_cpu) in enumerate(rollouts):
                generated = generated_cpu.to(device)
                completion_mask = mask_cpu.to(device)
                ref_log_probs = ref_log_probs_cpu.to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=use_autocast,
                ):
                    new_log_probs = completion_log_probs(
                        model,
                        generated,
                        prompt_length,
                        completion_mask.shape[1],
                        device,
                        False,
                    )
                delta = (ref_log_probs - new_log_probs).float()
                token_kl = torch.exp(delta) - delta - 1.0
                masked_token_kl = token_kl[completion_mask.bool()].detach().float().cpu()
                values.append(masked_token_kl)
                if replay_path is not None:
                    context = replay_context[rollout_index] if replay_context is not None else {}
                    generated_values = generated_cpu.detach().cpu().tolist()
                    mask_values = mask_cpu.detach().cpu().tolist()
                    reference_values = ref_log_probs.detach().float().cpu().tolist()
                    new_values = new_log_probs.detach().float().cpu().tolist()
                    replay_row = {
                        "schema_version": 1,
                        "record_type": "kl_guard_token_replay",
                        "step": step,
                        "attempt_index": attempt_index,
                        "variant": replay_variant,
                        "rollout_index": rollout_index,
                        "micro_index": context.get("micro_index"),
                        "prompt_id": context.get("prompt_id"),
                        "category": context.get("category"),
                        "sample_keys": context.get("sample_keys", []),
                        "generated_ids": generated_values,
                        "completion_mask": mask_values,
                        "ref_log_probs": reference_values,
                        "new_log_probs": new_values,
                        "token_kl_values": token_kl.detach().float().cpu().tolist(),
                        "valid_token_kl_values": masked_token_kl.tolist(),
                        "generated_sha256": hashlib.sha256(
                            json.dumps(generated_values, separators=(",", ":")).encode("utf-8")
                        ).hexdigest(),
                        "mask_sha256": hashlib.sha256(
                            json.dumps(mask_values, separators=(",", ":")).encode("utf-8")
                        ).hexdigest(),
                        "reference_log_probs_sha256": hashlib.sha256(
                            json.dumps(reference_values, separators=(",", ":")).encode("utf-8")
                        ).hexdigest(),
                        "new_log_probs_sha256": hashlib.sha256(
                            json.dumps(new_values, separators=(",", ":")).encode("utf-8")
                        ).hexdigest(),
                    }
                    with replay_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(replay_row, ensure_ascii=False, allow_nan=False) + "\n")
                        handle.flush()
    finally:
        if was_training:
            model.train()
    if not values:
        return {
            "reference_kl_mean": 0.0,
            "reference_kl_p95": 0.0,
            "reference_kl_max": 0.0,
            "token_count": 0.0,
        }
    return aggregate_reference_kl_values(values)


def build_full_fp32_measurement_model(
    model: MiniMindForCausalLM,
    device: torch.device,
) -> MiniMindForCausalLM:
    """Build a detached float32 copy used only for KL measurement diagnostics."""
    measurement_model = copy.deepcopy(model).to(device=device, dtype=torch.float32)
    measurement_model.eval().requires_grad_(False)
    return measurement_model


def classify_generation_outcome(
    *,
    eos_position: int | None,
    completion_width: int,
    max_gen_len: int,
) -> dict[str, object]:
    """Classify one generated completion without using batch padding width.

    ``completion_width`` is the number of completion tokens returned for this
    generation before EOS masking.  A completion containing EOS is natural,
    including when EOS happens to be the last token in a padded batch.  A
    completion without EOS is a max-token termination only when it actually
    reaches the configured generation budget; shorter no-EOS generations are
    retained as an explicit unknown case.
    """
    if completion_width < 0:
        raise ValueError("completion_width must be non-negative")
    if max_gen_len <= 0:
        raise ValueError("max_gen_len must be positive")
    if eos_position is not None:
        if eos_position < 0 or eos_position >= completion_width:
            raise ValueError("eos_position must be within the completion")
        return {
            "termination_reason": "eos",
            "eos_seen": True,
            "finished_naturally": True,
            "max_length_hit": False,
        }
    if completion_width >= max_gen_len:
        return {
            "termination_reason": "max_new_tokens",
            "eos_seen": False,
            "finished_naturally": False,
            "max_length_hit": True,
        }
    return {
        "termination_reason": "no_eos_short_generation",
        "eos_seen": False,
        "finished_naturally": False,
        "max_length_hit": False,
    }


def generate_group(
    model: MiniMindForCausalLM,
    tokenizer,
    row: dict[str, object],
    device: torch.device,
    num_generations: int,
    max_gen_len: int,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[str],
    list[float],
    list[dict[str, float]],
    int,
    list[dict[str, object]],
]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    prompt_ids, prompt_attention = prompt_tokens(tokenizer, str(row["prompt"]), device)
    prompt_length = prompt_ids.shape[1]
    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_attention,
            num_return_sequences=num_generations,
            max_new_tokens=max_gen_len,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )
    # MiniMind.generate is decorated with inference_mode; clone before the
    # full-sequence policy forward so autograd can save the input tensor.
    generated = generated.detach().clone()
    completion_ids = generated[:, prompt_length:]
    completion_length = completion_ids.shape[1]
    mask = torch.ones_like(completion_ids, dtype=torch.float32)
    responses = []
    rewards = []
    components = []
    outcomes: list[dict[str, object]] = []
    eos_id = tokenizer.eos_token_id
    for index, ids in enumerate(completion_ids):
        ids_list = ids.tolist()
        if eos_id is not None and eos_id in ids_list:
            eos_position = ids_list.index(eos_id)
            mask[index, eos_position + 1 :] = 0
            decode_ids = ids_list[:eos_position]
        else:
            eos_position = None
            decode_ids = ids_list
        outcomes.append(
            classify_generation_outcome(
                eos_position=eos_position,
                completion_width=len(ids_list),
                max_gen_len=max_gen_len,
            )
        )
        response = tokenizer.decode(decode_ids, skip_special_tokens=True).strip()
        score, score_components = rule_reward(str(row["category"]), str(row["prompt"]), response, dict(row["metadata"]))
        responses.append(response)
        rewards.append(score)
        components.append(score_components)
    return generated, mask, responses, rewards, components, prompt_length, outcomes


def save_weight(model: MiniMindForCausalLM, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.detach().half().cpu() for key, value in model.state_dict().items()}, path)


def save_checkpoint(model: MiniMindForCausalLM, path: Path) -> None:
    """Save a non-overwriting, CPU half-precision checkpoint."""
    save_weight(model, path)


def load_checkpoint_for_evaluation(
    config: MiniMindConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> MiniMindForCausalLM:
    """Load the serialized checkpoint used for validation and selection."""
    evaluation_model = MiniMindForCausalLM(config)
    evaluation_model.load_state_dict(
        torch.load(checkpoint_path, map_location=device),
        strict=True,
    )
    evaluation_model.to(device).eval()
    return evaluation_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--categories", default="format,instruction,reasoning,termination")
    parser.add_argument("--max-prompts", type=int, default=16)
    parser.add_argument("--validation-max-prompts", type=int, default=32)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--save-weight", default="grpo_v1_lite")
    parser.add_argument("--from-weight", default="align_sft_v2_pilot")
    parser.add_argument("--model-dir", type=Path, default=Path("out"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--max-gen-len", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--kl-threshold", type=float, default=0.005)
    parser.add_argument("--kl-patience", type=int, default=2)
    parser.add_argument("--post-step-kl-target", type=float)
    parser.add_argument(
        "--post-step-kl-gate-mode",
        choices=sorted(POST_STEP_KL_GATE_MODES),
        default="legacy_bfloat16_autocast",
    )
    parser.add_argument("--kl-backoff-factor", type=float, default=0.5)
    parser.add_argument("--kl-max-backoffs", type=int, default=3)
    parser.add_argument("--kl-guard-attempt-log-path", type=Path)
    parser.add_argument("--kl-guard-token-replay-path", type=Path)
    parser.add_argument("--post-step-kl-diagnostic-fp32", action="store_true")
    parser.add_argument("--post-step-kl-diagnostic-full-fp32", action="store_true")
    parser.add_argument("--pre-step-kl-diagnostic-fp32", action="store_true")
    parser.add_argument(
        "--training-forward-mode",
        choices=sorted(TRAINING_FORWARD_MODES),
        default="legacy_bfloat16_autocast",
        help="precision used by the gradient-bearing policy forward; legacy is unchanged by default",
    )
    parser.add_argument("--pre-step-loss-diagnostic-fp32", action="store_true")
    parser.add_argument("--pre-step-loss-replay-path", type=Path)
    parser.add_argument("--quality-drop-points", type=float, default=10.0)
    parser.add_argument("--checkpoint-every", type=int, default=4)
    parser.add_argument("--validation-max-new-tokens", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-7)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--microbatch-log-path", type=Path)
    parser.add_argument("--microbatch-gradient-norm", action="store_true")
    parser.add_argument("--interleave-categories", action="store_true")
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon-high", type=float, default=5.0)
    parser.add_argument(
        "--controlled-reward-pattern",
        help="diagnostic-only comma-separated per-generation rewards; omitted keeps rule_reward unchanged",
    )
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--mode", choices=["grpo", "cispo"], default="grpo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size != 1:
        raise ValueError("controlled pilot requires batch_size=1")
    if args.eval_every < 1 or args.checkpoint_every < 1 or args.kl_patience < 1:
        raise ValueError("eval/checkpoint intervals and KL patience must be positive")
    if args.max_grad_norm <= 0:
        raise ValueError("max-grad-norm must be positive")
    if args.post_step_kl_target is not None and args.post_step_kl_target <= 0:
        raise ValueError("post-step-kl-target must be positive when enabled")
    if args.post_step_kl_gate_mode != "legacy_bfloat16_autocast" and args.post_step_kl_target is None:
        raise ValueError("non-legacy post-step-kl-gate-mode requires post-step-kl-target")
    if args.kl_guard_attempt_log_path is not None and args.post_step_kl_target is None:
        raise ValueError("kl-guard-attempt-log-path requires post-step-kl-target")
    if args.kl_guard_token_replay_path is not None and args.post_step_kl_target is None:
        raise ValueError("kl-guard-token-replay-path requires post-step-kl-target")
    if args.post_step_kl_diagnostic_fp32 and args.post_step_kl_target is None:
        raise ValueError("post-step-kl-diagnostic-fp32 requires post-step-kl-target")
    if args.post_step_kl_diagnostic_full_fp32 and args.post_step_kl_target is None:
        raise ValueError("post-step-kl-diagnostic-full-fp32 requires post-step-kl-target")
    if args.pre_step_kl_diagnostic_fp32 and args.post_step_kl_target is None:
        raise ValueError("pre-step-kl-diagnostic-fp32 requires post-step-kl-target")
    if args.pre_step_loss_diagnostic_fp32 and args.post_step_kl_target is None:
        raise ValueError("pre-step-loss-diagnostic-fp32 requires post-step-kl-target")
    if args.pre_step_loss_diagnostic_fp32 != (args.pre_step_loss_replay_path is not None):
        raise ValueError("pre-step loss diagnostic and replay path must be enabled together")
    if args.pre_step_loss_diagnostic_fp32 and not args.microbatch_gradient_norm:
        raise ValueError("pre-step loss diagnostic requires microbatch-gradient-norm")
    if not 0.0 < args.kl_backoff_factor < 1.0:
        raise ValueError("kl-backoff-factor must be between 0 and 1")
    if args.kl_max_backoffs < 0:
        raise ValueError("kl-max-backoffs must be non-negative")
    if args.microbatch_gradient_norm and args.microbatch_log_path is None:
        raise ValueError("microbatch-gradient-norm requires microbatch-log-path")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    category_order = [category.strip() for category in args.categories.split(",") if category.strip()]
    controlled_reward_pattern = parse_controlled_reward_pattern(
        args.controlled_reward_pattern,
        args.num_generations,
    )
    rows = read_manifest(args.manifest, set(category_order), args.max_prompts)
    if args.interleave_categories:
        rows = interleave_rows_by_category(rows, category_order)
    validation_rows = None
    if args.validation_manifest:
        validation_rows = read_manifest(args.validation_manifest, set(args.categories.split(",")), args.validation_max_prompts)
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    policy = MiniMindForCausalLM(config)
    policy.load_state_dict(torch.load(args.model_dir / f"{args.from_weight}_768.pth", map_location=device), strict=True)
    reference = MiniMindForCausalLM(config)
    reference.load_state_dict(torch.load(args.model_dir / f"{args.from_weight}_768.pth", map_location=device), strict=True)
    policy.to(device).train()
    reference.to(device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    autocast_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    use_autocast = device.type == "cuda" and args.dtype == "bfloat16"
    training_use_autocast = training_forward_uses_autocast(args.training_forward_mode, use_autocast)
    fp32_post_step_required = (
        args.post_step_kl_diagnostic_fp32
        or args.post_step_kl_gate_mode == "fp32_no_autocast"
    )
    output_dir = args.save_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    microbatch_log_path = args.microbatch_log_path
    if microbatch_log_path is not None:
        if microbatch_log_path.exists():
            raise FileExistsError(f"refusing to overwrite {microbatch_log_path}")
        microbatch_log_path.parent.mkdir(parents=True, exist_ok=True)
    kl_guard_attempt_log_path = args.kl_guard_attempt_log_path
    if kl_guard_attempt_log_path is not None:
        if kl_guard_attempt_log_path.exists():
            raise FileExistsError(f"refusing to overwrite {kl_guard_attempt_log_path}")
        kl_guard_attempt_log_path.parent.mkdir(parents=True, exist_ok=True)
    kl_guard_token_replay_path = args.kl_guard_token_replay_path
    if kl_guard_token_replay_path is not None:
        if kl_guard_token_replay_path.exists():
            raise FileExistsError(f"refusing to overwrite {kl_guard_token_replay_path}")
        kl_guard_token_replay_path.parent.mkdir(parents=True, exist_ok=True)
    pre_step_loss_replay_path = args.pre_step_loss_replay_path
    if pre_step_loss_replay_path is not None:
        if pre_step_loss_replay_path.exists():
            raise FileExistsError(f"refusing to overwrite {pre_step_loss_replay_path}")
        pre_step_loss_replay_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_train_manifest.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    samples_path = output_dir / "samples.jsonl"
    step_log_path = output_dir / "step_summaries.jsonl"
    validation_history_path = output_dir / "validation_history.jsonl"
    checkpoints_dir = output_dir / "checkpoints"
    step_summaries = []
    validation_records: list[dict] = []
    prompt_cursor = 0
    kl_consecutive = 0
    stop_reason = None

    baseline_validation = None
    if validation_rows is not None:
        baseline_metrics, baseline_details = evaluate_policy(
            policy, tokenizer, validation_rows, device, max_new_tokens=args.validation_max_new_tokens
        )
        baseline_validation = {"metrics": baseline_metrics, "details": baseline_details}
        (output_dir / "baseline_validation.json").write_text(
            json.dumps(baseline_validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"validation": "baseline", **baseline_metrics}, ensure_ascii=False))
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        micro_summaries = []
        guard_rollouts: list[tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]] = []
        guard_rollout_contexts: list[dict[str, object]] = []
        for micro in range(args.accumulation_steps):
            row = rows[prompt_cursor % len(rows)]
            prompt_cursor += 1
            generated, completion_mask, responses, reward_values, components, prompt_length, outcomes = generate_group(
                policy, tokenizer, row, device, args.num_generations, args.max_gen_len, args.seed + step * 100 + micro,
            )
            rule_reward_values = list(reward_values)
            controlled_reward_delta = [0.0 for _ in rule_reward_values]
            reward_source = "rule_reward"
            if controlled_reward_pattern is not None:
                reward_values = [
                    controlled_reward_pattern[index % args.num_generations]
                    for index in range(len(rule_reward_values))
                ]
                controlled_reward_delta = [
                    float(controlled - original)
                    for controlled, original in zip(reward_values, rule_reward_values)
                ]
                reward_source = "controlled_reward_pattern"
            full_attention = torch.ones_like(generated, device=device)
            with torch.no_grad():
                old_log_probs = completion_log_probs(policy, generated, prompt_length, completion_mask.shape[1], device, False)
                ref_log_probs = completion_log_probs(reference, generated, prompt_length, completion_mask.shape[1], device, False)
            sample_count = len(responses)
            sample_keys = [f"{step}:{micro}:{index}" for index in range(sample_count)]
            if args.post_step_kl_target is not None:
                guard_rollouts.append(
                    (
                        generated.detach().cpu(),
                        completion_mask.detach().cpu(),
                        prompt_length,
                        ref_log_probs.detach().cpu(),
                    )
                )
                guard_rollout_contexts.append({
                    "micro_index": micro,
                    "prompt_id": row["id"],
                    "category": row["category"],
                    "sample_keys": sample_keys,
                })
            reward_tensor = torch.tensor(reward_values, dtype=torch.float32, device=device)
            advantages = group_advantages(reward_tensor, args.num_generations)
            legacy_new_log_probs = None
            legacy_loss = None
            legacy_kl = None
            fp32_new_log_probs = None
            fp32_loss = None
            fp32_kl = None
            if (
                args.training_forward_mode == "legacy_bfloat16_autocast"
                or args.pre_step_loss_diagnostic_fp32
            ):
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=(
                        training_use_autocast
                        if args.training_forward_mode == "legacy_bfloat16_autocast"
                        else use_autocast
                    ),
                ):
                    legacy_new_log_probs = completion_log_probs(
                        policy, generated, prompt_length, completion_mask.shape[1], device, True
                    )
                    legacy_loss, legacy_kl = clipped_policy_loss(
                        legacy_new_log_probs, old_log_probs, ref_log_probs, advantages, completion_mask,
                        mode=args.mode, beta=args.beta, epsilon=args.epsilon, epsilon_high=args.epsilon_high,
                    )
            if (
                args.training_forward_mode == "fp32_no_autocast"
                or args.pre_step_loss_diagnostic_fp32
            ):
                with torch.autocast(device_type=device.type, dtype=torch.float32, enabled=False):
                    fp32_new_log_probs = completion_log_probs(
                        policy, generated, prompt_length, completion_mask.shape[1], device, True
                    )
                    fp32_loss, fp32_kl = clipped_policy_loss(
                        fp32_new_log_probs, old_log_probs, ref_log_probs, advantages, completion_mask,
                        mode=args.mode, beta=args.beta, epsilon=args.epsilon, epsilon_high=args.epsilon_high,
                    )
            if args.training_forward_mode == "fp32_no_autocast":
                new_log_probs, loss, kl = fp32_new_log_probs, fp32_loss, fp32_kl
                active_variant = "fp32_no_autocast"
            else:
                new_log_probs, loss, kl = legacy_new_log_probs, legacy_loss, legacy_kl
                active_variant = "bfloat16_autocast"
            legacy_terms = None
            legacy_term_means = None
            fp32_term_means = None
            fp32_shadow_gradient_norm_scaled = None
            fp32_shadow_gradient_norm_unscaled = None
            legacy_shadow_gradient_norm_scaled = None
            legacy_shadow_gradient_norm_unscaled = None
            fp32_shadow_grad_isolation_ok = None
            if args.pre_step_loss_diagnostic_fp32:
                legacy_terms = clipped_policy_diagnostic_terms(
                    legacy_new_log_probs,
                    old_log_probs,
                    ref_log_probs,
                    advantages,
                    mode=args.mode,
                    beta=args.beta,
                    epsilon=args.epsilon,
                    epsilon_high=args.epsilon_high,
                )
                legacy_term_means = masked_diagnostic_means(legacy_terms, completion_mask)
                fp32_terms = clipped_policy_diagnostic_terms(
                    fp32_new_log_probs,
                    old_log_probs,
                    ref_log_probs,
                    advantages,
                    mode=args.mode,
                    beta=args.beta,
                    epsilon=args.epsilon,
                    epsilon_high=args.epsilon_high,
                )
                fp32_term_means = masked_diagnostic_means(fp32_terms, completion_mask)
                accumulated_gradient_before_shadow = gradient_norm(policy)
                trainable_parameters = tuple(parameter for parameter in policy.parameters() if parameter.requires_grad)
                legacy_shadow_gradients = torch.autograd.grad(
                    legacy_loss / args.accumulation_steps,
                    trainable_parameters,
                    retain_graph=active_variant == "bfloat16_autocast",
                    allow_unused=True,
                )
                fp32_shadow_gradients = torch.autograd.grad(
                    fp32_loss / args.accumulation_steps,
                    trainable_parameters,
                    retain_graph=active_variant == "fp32_no_autocast",
                    allow_unused=True,
                )
                legacy_shadow_gradient_norm_scaled = gradient_sequence_norm(legacy_shadow_gradients)
                legacy_shadow_gradient_norm_unscaled = (
                    legacy_shadow_gradient_norm_scaled * args.accumulation_steps
                )
                fp32_shadow_gradient_norm_scaled = gradient_sequence_norm(fp32_shadow_gradients)
                fp32_shadow_gradient_norm_unscaled = (
                    fp32_shadow_gradient_norm_scaled * args.accumulation_steps
                )
                del legacy_shadow_gradients
                del fp32_shadow_gradients
                fp32_shadow_grad_isolation_ok = (
                    gradient_norm(policy) == accumulated_gradient_before_shadow
                )
                replay_row = {
                    "schema_version": 1,
                    "record_type": "pre_step_loss_replay",
                    "run_id": output_dir.name,
                    "step": step,
                    "micro_index": micro,
                    "prompt_id": row["id"],
                    "category": row["category"],
                    "sample_keys": sample_keys,
                    "mode": args.mode,
                    "training_forward_mode": args.training_forward_mode,
                    "active_variant": active_variant,
                    "beta": args.beta,
                    "epsilon": args.epsilon,
                    "epsilon_high": args.epsilon_high,
                    "reward_source": reward_source,
                    "rule_reward_values": rule_reward_values,
                    "controlled_reward_values": reward_values if controlled_reward_pattern is not None else None,
                    "controlled_reward_pattern": controlled_reward_pattern,
                    "controlled_reward_delta": controlled_reward_delta,
                    "generated_ids": generated.detach().cpu().tolist(),
                    "completion_mask": completion_mask.detach().float().cpu().tolist(),
                    "old_log_probs": old_log_probs.detach().float().cpu().tolist(),
                    "ref_log_probs": ref_log_probs.detach().float().cpu().tolist(),
                    "advantages": advantages.detach().float().cpu().tolist(),
                    "bfloat16_autocast_new_log_probs": legacy_new_log_probs.detach().float().cpu().tolist(),
                    "fp32_no_autocast_new_log_probs": fp32_new_log_probs.detach().float().cpu().tolist(),
                    "bfloat16_autocast_terms": {
                        name: value.detach().float().cpu().tolist()
                        for name, value in legacy_terms.items()
                    },
                    "fp32_no_autocast_terms": {
                        name: value.detach().float().cpu().tolist()
                        for name, value in fp32_terms.items()
                    },
                    "bfloat16_autocast_means": legacy_term_means,
                    "fp32_no_autocast_means": fp32_term_means,
                    "bfloat16_autocast_loss": float(legacy_loss.detach().cpu()),
                    "fp32_no_autocast_loss": float(fp32_loss.detach().cpu()),
                    "bfloat16_autocast_kl": float(legacy_kl.detach().cpu()),
                    "fp32_no_autocast_kl": float(fp32_kl.detach().cpu()),
                    "fp32_shadow_gradient_norm_scaled": fp32_shadow_gradient_norm_scaled,
                    "fp32_shadow_gradient_norm_unscaled": fp32_shadow_gradient_norm_unscaled,
                    "bfloat16_shadow_gradient_norm_scaled": legacy_shadow_gradient_norm_scaled,
                    "bfloat16_shadow_gradient_norm_unscaled": legacy_shadow_gradient_norm_unscaled,
                    "fp32_shadow_grad_isolation_ok": fp32_shadow_grad_isolation_ok,
                    "source_sha256": {
                        "generated": json_tensor_sha256(generated),
                        "completion_mask": json_tensor_sha256(completion_mask),
                        "old_log_probs": json_tensor_sha256(old_log_probs.detach().float()),
                        "ref_log_probs": json_tensor_sha256(ref_log_probs.detach().float()),
                        "advantages": json_tensor_sha256(advantages.detach().float()),
                    },
                }
                with pre_step_loss_replay_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(replay_row, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
            diagnostics = policy_diagnostics(new_log_probs, old_log_probs, ref_log_probs, completion_mask)
            gradients_before = gradient_snapshot(policy) if args.microbatch_gradient_norm else {}
            (loss / args.accumulation_steps).backward()
            micro_grad_norm_scaled = (
                gradient_delta_norm(policy, gradients_before)
                if args.microbatch_gradient_norm
                else None
            )
            micro_grad_norm_unscaled = (
                micro_grad_norm_scaled * args.accumulation_steps
                if micro_grad_norm_scaled is not None
                else None
            )
            accumulated_grad_norm = gradient_norm(policy) if microbatch_log_path is not None else None
            sample_count = len(responses)
            generated_tokens = [int(completion_mask[index].sum().item()) for index in range(sample_count)]
            finished_naturally = [bool(outcome["finished_naturally"]) for outcome in outcomes]
            max_length_hit = [bool(outcome["max_length_hit"]) for outcome in outcomes]
            termination_reason_counts = {
                reason: sum(outcome["termination_reason"] == reason for outcome in outcomes)
                for reason in ("eos", "max_new_tokens", "no_eos_short_generation")
            }
            reward_component_means = {
                name: sum(float(component.get(name, 0.0)) for component in components) / sample_count
                for name in ("validator_reward", "termination_reward", "repetition_penalty")
            }
            micro_summary = {
                "schema_version": 5 if args.pre_step_loss_diagnostic_fp32 else 3,
                "telemetry_schema_version": 5 if args.pre_step_loss_diagnostic_fp32 else 3,
                "run_id": output_dir.name,
                "step": step,
                "micro_index": micro,
                "prompt_id": row["id"],
                "category": row["category"],
                "training_forward_mode": args.training_forward_mode,
                "active_variant": active_variant,
                "reward_source": reward_source,
                "rule_reward_values": rule_reward_values,
                "controlled_reward_values": reward_values if controlled_reward_pattern is not None else None,
                "controlled_reward_pattern": controlled_reward_pattern,
                "controlled_reward_delta": controlled_reward_delta,
                "loss": float(loss.detach().cpu()),
                "rewards": reward_values,
                "reward_mean": sum(reward_values) / len(reward_values),
                "reward_std": float(torch.tensor(reward_values).std(unbiased=False)),
                "rule_reward_mean": sum(rule_reward_values) / len(rule_reward_values),
                "controlled_reward_delta_mean": sum(controlled_reward_delta) / len(controlled_reward_delta),
                "advantage_abs_max": float(advantages.detach().abs().max().cpu()),
                "advantage_nonzero_rate": float((advantages.detach().abs() > 0).float().mean().cpu()),
                "kl": float(kl.detach().cpu()),
                "kl_measurement_phase": "pre_optimizer_step",
                "policy_diagnostics": diagnostics,
                "average_completion_tokens": float(completion_mask.sum(dim=1).float().mean().detach().cpu()),
                "sample_count": sample_count,
                "train_validator_pass_rate": reward_component_means["validator_reward"],
                "train_empty_response_rate": sum(not response.strip() for response in responses) / sample_count,
                "train_max_length_hit_rate": sum(max_length_hit) / sample_count,
                "train_natural_end_rate": sum(finished_naturally) / sample_count,
                "termination_reason_counts": termination_reason_counts,
                "termination_unknown_rate": termination_reason_counts["no_eos_short_generation"] / sample_count,
                "train_repetition_penalty_mean": reward_component_means["repetition_penalty"],
                "train_reward_components": reward_component_means,
                "micro_grad_norm_scaled": micro_grad_norm_scaled,
                "micro_grad_norm_unscaled": micro_grad_norm_unscaled,
                "accumulated_grad_norm": accumulated_grad_norm,
            }
            sample_keys = [f"{step}:{micro}:{index}" for index in range(sample_count)]
            micro_summary["sample_keys"] = sample_keys
            if args.pre_step_loss_diagnostic_fp32:
                fp32_loss_value = float(fp32_loss.detach().cpu())
                fp32_kl_value = float(fp32_kl.detach().cpu())
                legacy_loss_value = float(legacy_loss.detach().cpu())
                legacy_kl_value = float(legacy_kl.detach().cpu())
                kl_gap_threshold = max(
                    1e-4,
                    0.1 * max(fp32_kl_value, float(args.post_step_kl_target)),
                )
                loss_gap_threshold = max(1e-7, args.beta * kl_gap_threshold)
                gradient_gap = abs(
                    float(legacy_shadow_gradient_norm_unscaled)
                    - float(fp32_shadow_gradient_norm_unscaled)
                )
                gradient_ratio = float(legacy_shadow_gradient_norm_unscaled) / max(
                    float(fp32_shadow_gradient_norm_unscaled),
                    1e-12,
                )
                selected_shadow_gradient = (
                    fp32_shadow_gradient_norm_unscaled
                    if active_variant == "fp32_no_autocast"
                    else legacy_shadow_gradient_norm_unscaled
                )
                micro_summary["pre_step_loss_precision"] = {
                    "training_forward_mode": args.training_forward_mode,
                    "active_variant": active_variant,
                    "bfloat16_autocast_loss": legacy_loss_value,
                    "fp32_no_autocast_loss": fp32_loss_value,
                    "loss_abs_gap": abs(legacy_loss_value - fp32_loss_value),
                    "loss_gap_threshold": loss_gap_threshold,
                    "loss_disagreement": abs(legacy_loss_value - fp32_loss_value) > loss_gap_threshold,
                    "bfloat16_autocast_kl": legacy_kl_value,
                    "fp32_no_autocast_kl": fp32_kl_value,
                    "kl_abs_gap": abs(legacy_kl_value - fp32_kl_value),
                    "kl_gap_threshold": kl_gap_threshold,
                    "kl_disagreement": abs(legacy_kl_value - fp32_kl_value) > kl_gap_threshold,
                    "bfloat16_autocast_term_means": legacy_term_means,
                    "fp32_no_autocast_term_means": fp32_term_means,
                    "bfloat16_autocast_gradient_norm_scaled": legacy_shadow_gradient_norm_scaled,
                    "bfloat16_autocast_gradient_norm_unscaled": legacy_shadow_gradient_norm_unscaled,
                    "fp32_no_autocast_gradient_norm_scaled": fp32_shadow_gradient_norm_scaled,
                    "fp32_no_autocast_gradient_norm_unscaled": fp32_shadow_gradient_norm_unscaled,
                    "active_gradient_norm_scaled": micro_grad_norm_scaled,
                    "active_gradient_norm_unscaled": micro_grad_norm_unscaled,
                    "active_loss_matches_selected_variant": abs(
                        float(loss.detach().cpu()) - float(
                            fp32_loss.detach().cpu()
                            if active_variant == "fp32_no_autocast"
                            else legacy_loss.detach().cpu()
                        )
                    ) <= max(1e-7, 1e-5 * max(abs(float(loss.detach().cpu())), 1e-12)),
                    "active_kl_matches_selected_variant": abs(
                        float(kl.detach().cpu()) - float(
                            fp32_kl.detach().cpu()
                            if active_variant == "fp32_no_autocast"
                            else legacy_kl.detach().cpu()
                        )
                    ) <= max(1e-7, 1e-5 * max(abs(float(kl.detach().cpu())), 1e-12)),
                    "active_gradient_matches_selected_variant": abs(
                        float(micro_grad_norm_unscaled) - float(selected_shadow_gradient)
                    ) <= max(1e-6, 1e-4 * max(float(selected_shadow_gradient), 1e-12)),
                    "gradient_norm_abs_gap": gradient_gap,
                    "gradient_norm_ratio": gradient_ratio,
                    "gradient_disagreement": gradient_gap > 1e-6 and gradient_ratio > 5.0,
                    "shadow_grad_isolation_ok": fp32_shadow_grad_isolation_ok,
                    "same_rollout_mask_old_ref_advantages": True,
                    "advantage_abs_max": float(advantages.detach().abs().max().cpu()),
                    "replay_path": str(pre_step_loss_replay_path),
                }
            if microbatch_log_path is not None:
                finite_values = [
                    micro_summary["loss"],
                    micro_summary["reward_mean"],
                    micro_summary["reward_std"],
                    micro_summary["kl"],
                    micro_summary["average_completion_tokens"],
                    micro_summary["train_validator_pass_rate"],
                    micro_summary["train_empty_response_rate"],
                    micro_summary["train_max_length_hit_rate"],
                    micro_summary["train_natural_end_rate"],
                    micro_summary["train_repetition_penalty_mean"],
                    micro_summary["accumulated_grad_norm"],
                ]
                if micro_grad_norm_scaled is not None:
                    finite_values.extend([micro_grad_norm_scaled, micro_grad_norm_unscaled])
                if args.pre_step_loss_diagnostic_fp32:
                    precision = micro_summary["pre_step_loss_precision"]
                    finite_values.extend(
                        float(value)
                        for key, value in precision.items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool)
                    )
                    for term_means in (
                        precision["bfloat16_autocast_term_means"],
                        precision["fp32_no_autocast_term_means"],
                    ):
                        finite_values.extend(term_means.values())
                finite_values.extend(diagnostics.values())
                if not all(torch.isfinite(torch.tensor(value)) for value in finite_values):
                    raise FloatingPointError(f"non-finite microbatch telemetry at step {step}, micro {micro}")
            micro_summaries.append(micro_summary)
            with samples_path.open("a", encoding="utf-8") as handle:
                for index, response in enumerate(responses):
                    handle.write(json.dumps({
                        "step": step,
                        "micro_index": micro,
                        "sample_key": sample_keys[index],
                        "prompt_id": row["id"],
                        "category": row["category"],
                        "candidate_index": index,
                        "reward": reward_values[index],
                        "rule_reward": rule_reward_values[index],
                        "reward_source": reward_source,
                        "controlled_reward": (
                            reward_values[index] if controlled_reward_pattern is not None else None
                        ),
                        "controlled_reward_delta": controlled_reward_delta[index],
                    "components": components[index],
                    "response": response,
                    "generated_tokens": generated_tokens[index],
                    "termination_reason": outcomes[index]["termination_reason"],
                    "eos_seen": outcomes[index]["eos_seen"],
                    "finished_naturally": finished_naturally[index],
                        "empty_response": not response.strip(),
                        "max_length_hit": max_length_hit[index],
                    }, ensure_ascii=False) + "\n")
            if microbatch_log_path is not None:
                with microbatch_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(micro_summary, ensure_ascii=False) + "\n")
                    handle.flush()
        pre_step_kl_bfloat16_shadow = None
        pre_step_kl_fp32_no_autocast = None
        pre_step_shadow_disagreement = None
        if args.pre_step_kl_diagnostic_fp32:
            pre_step_kl_bfloat16_shadow = post_update_kl_stats(
                policy,
                guard_rollouts,
                device,
                autocast_dtype,
                use_autocast,
            )
            pre_step_kl_fp32_no_autocast = post_update_kl_stats(
                policy,
                guard_rollouts,
                device,
                torch.float32,
                False,
            )
            pre_step_gap_threshold = max(
                1e-4,
                0.1 * max(
                    float(pre_step_kl_fp32_no_autocast["reference_kl_mean"]),
                    float(args.post_step_kl_target),
                ),
            )
            pre_step_shadow_disagreement = (
                abs(
                    float(pre_step_kl_bfloat16_shadow["reference_kl_mean"])
                    - float(pre_step_kl_fp32_no_autocast["reference_kl_mean"])
                )
                > pre_step_gap_threshold
            )
        grad_norm_pre_clip = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm).detach().cpu())
        grad_norm_post_clip = gradient_norm(policy)
        kl_guard_enabled = args.post_step_kl_target is not None
        kl_guard_attempts = 0
        kl_guard_backoff_count = 0
        kl_guard_accepted_lr_multiplier = None
        optimizer_step_applied = True
        optimizer_step_rejected = False
        kl_guard_attempt_history: list[dict[str, object]] = []
        pre_step_policy_state_digest = None
        pre_step_optimizer_state_digest = None
        rollback_policy_state_digest = None
        rollback_optimizer_state_digest = None
        rollback_parameter_delta = None
        rollback_exact_match = None
        rollback_verification_failed = False
        post_step_kl = {
            "reference_kl_mean": None,
            "reference_kl_p95": None,
            "reference_kl_max": None,
        }
        post_step_kl_float32 = None
        post_step_kl_full_float32 = None
        post_step_kl_gate = None
        post_step_kl_gate_passed = None
        legacy_bfloat16_gate_passed = None
        fp32_no_autocast_gate_passed = None
        shadow_gate_disagreement = None
        final_parameter_delta = None
        full_fp32_measurement_model = None
        if kl_guard_enabled:
            state_snapshot = snapshot_training_state(policy, optimizer)
            pre_step_policy_state_digest = policy_state_digest(policy)
            pre_step_optimizer_state_digest = optimizer_state_digest(optimizer)
            if args.post_step_kl_diagnostic_full_fp32:
                full_fp32_measurement_model = build_full_fp32_measurement_model(policy, device)
            base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
            accepted = False
            for attempt, multiplier in enumerate(
                kl_backoff_multipliers(args.kl_backoff_factor, args.kl_max_backoffs)
            ):
                pre_attempt_policy_state_digest = policy_state_digest(policy)
                pre_attempt_optimizer_state_digest = optimizer_state_digest(optimizer)
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base_lr * multiplier
                optimizer.step()
                kl_guard_attempts = attempt + 1
                post_step_kl = post_update_kl_stats(
                    policy,
                    guard_rollouts,
                    device,
                    autocast_dtype,
                    use_autocast,
                    replay_path=kl_guard_token_replay_path,
                    replay_context=guard_rollout_contexts,
                    replay_variant="bfloat16_autocast",
                    step=step,
                    attempt_index=attempt,
                )
                if fp32_post_step_required:
                    post_step_kl_float32 = post_update_kl_stats(
                        policy,
                        guard_rollouts,
                        device,
                        torch.float32,
                        False,
                        replay_path=kl_guard_token_replay_path,
                        replay_context=guard_rollout_contexts,
                        replay_variant="bfloat16_no_autocast",
                        step=step,
                        attempt_index=attempt,
                    )
                if full_fp32_measurement_model is not None:
                    full_fp32_measurement_model.load_state_dict(policy.state_dict(), strict=True)
                    post_step_kl_full_float32 = post_update_kl_stats(
                        full_fp32_measurement_model,
                        guard_rollouts,
                        device,
                        torch.float32,
                        False,
                        replay_path=kl_guard_token_replay_path,
                        replay_context=guard_rollout_contexts,
                        replay_variant="full_float32_no_autocast",
                        step=step,
                        attempt_index=attempt,
                    )
                legacy_bfloat16_finite = finite_diagnostics({
                    key: float(value)
                    for key, value in post_step_kl.items()
                })
                fp32_no_autocast_finite = (
                    post_step_kl_float32 is None
                    or finite_diagnostics({key: float(value) for key, value in post_step_kl_float32.items()})
                )
                full_float32_finite = (
                    post_step_kl_full_float32 is None
                    or finite_diagnostics({
                        key: float(value)
                        for key, value in post_step_kl_full_float32.items()
                    })
                )
                post_step_kl_gate = select_reference_kl_gate_stats(
                    args.post_step_kl_gate_mode,
                    post_step_kl,
                    post_step_kl_float32,
                )
                selected_gate_finite = finite_diagnostics({
                    key: float(value)
                    for key, value in post_step_kl_gate.items()
                })
                diagnostic_finite = (
                    legacy_bfloat16_finite
                    and fp32_no_autocast_finite
                    and full_float32_finite
                )
                final_parameter_delta = parameter_delta_metrics(policy, state_snapshot[0])
                post_attempt_policy_state_digest = policy_state_digest(policy)
                post_attempt_optimizer_state_digest = optimizer_state_digest(optimizer)
                dtype_gap = None
                dtype_gap_threshold = None
                dtype_sensitive = None
                if post_step_kl_float32 is not None and legacy_bfloat16_finite and fp32_no_autocast_finite:
                    dtype_gap = abs(
                        float(post_step_kl["reference_kl_mean"])
                        - float(post_step_kl_float32["reference_kl_mean"])
                    )
                    dtype_gap_threshold = max(
                        1e-4,
                        0.1 * max(float(post_step_kl_float32["reference_kl_mean"]), args.post_step_kl_target),
                    )
                    dtype_sensitive = dtype_gap > dtype_gap_threshold
                legacy_bfloat16_gate_passed = bool(
                    legacy_bfloat16_finite
                    and post_step_kl["reference_kl_mean"] <= args.post_step_kl_target
                )
                fp32_no_autocast_gate_passed = (
                    bool(
                        fp32_no_autocast_finite
                        and post_step_kl_float32["reference_kl_mean"] <= args.post_step_kl_target
                    )
                    if post_step_kl_float32 is not None
                    else None
                )
                post_step_kl_gate_passed = bool(
                    selected_gate_finite
                    and post_step_kl_gate["reference_kl_mean"] <= args.post_step_kl_target
                )
                shadow_gate_disagreement = (
                    legacy_bfloat16_gate_passed != fp32_no_autocast_gate_passed
                    if fp32_no_autocast_gate_passed is not None
                    else None
                )
                production_gate_passed = post_step_kl_gate_passed
                attempt_record: dict[str, object] = {
                    "schema_version": 2,
                    "record_type": "kl_guard_attempt",
                    "run_id": output_dir.name,
                    "step": step,
                    "attempt_index": attempt,
                    "lr_multiplier": multiplier,
                    "pre_attempt_policy_state_digest": pre_attempt_policy_state_digest,
                    "pre_attempt_optimizer_state_digest": pre_attempt_optimizer_state_digest,
                    "pre_attempt_matches_step_snapshot": (
                        pre_attempt_policy_state_digest == pre_step_policy_state_digest
                        and pre_attempt_optimizer_state_digest == pre_step_optimizer_state_digest
                    ),
                    "post_step_kl_bfloat16": post_step_kl,
                    "post_step_kl_float32": post_step_kl_float32,
                    "post_step_kl_full_float32": post_step_kl_full_float32,
                    "post_step_kl_gate_mode": args.post_step_kl_gate_mode,
                    "post_step_kl_gate": post_step_kl_gate,
                    "post_step_kl_gate_passed": post_step_kl_gate_passed,
                    "legacy_bfloat16_gate_passed": legacy_bfloat16_gate_passed,
                    "fp32_no_autocast_gate_passed": fp32_no_autocast_gate_passed,
                    "shadow_gate_disagreement": shadow_gate_disagreement,
                    "dtype_gap_mean": dtype_gap,
                    "dtype_gap_threshold": dtype_gap_threshold,
                    "dtype_sensitive": dtype_sensitive,
                    "parameter_delta": final_parameter_delta,
                    "post_attempt_policy_state_digest": post_attempt_policy_state_digest,
                    "post_attempt_optimizer_state_digest": post_attempt_optimizer_state_digest,
                    "production_gate_passed": production_gate_passed,
                    "diagnostic_finite": diagnostic_finite,
                    "accepted": False,
                    "rollback_after_attempt": None,
                }
                if not selected_gate_finite or not diagnostic_finite:
                    attempt_record["failure_reason"] = "non_finite_kl"
                    restore_training_state(policy, optimizer, state_snapshot)
                    rollback_policy_state_digest = policy_state_digest(policy)
                    rollback_optimizer_state_digest = optimizer_state_digest(optimizer)
                    rollback_parameter_delta = parameter_delta_metrics(policy, state_snapshot[0])
                    rollback_exact_match = (
                        rollback_policy_state_digest == pre_step_policy_state_digest
                        and rollback_optimizer_state_digest == pre_step_optimizer_state_digest
                    )
                    attempt_record["rollback_after_attempt"] = {
                        "policy_state_digest": rollback_policy_state_digest,
                        "optimizer_state_digest": rollback_optimizer_state_digest,
                        "parameter_delta": rollback_parameter_delta,
                        "exact_match": rollback_exact_match,
                    }
                    attempt_record["accepted"] = False
                    kl_guard_attempt_history.append(attempt_record)
                    if kl_guard_attempt_log_path is not None:
                        with kl_guard_attempt_log_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(attempt_record, ensure_ascii=False) + "\n")
                            handle.flush()
                    rollback_verification_failed = not rollback_exact_match
                    break
                if production_gate_passed:
                    accepted = True
                    kl_guard_backoff_count = attempt
                    kl_guard_accepted_lr_multiplier = multiplier
                    attempt_record["accepted"] = True
                    kl_guard_attempt_history.append(attempt_record)
                    if kl_guard_attempt_log_path is not None:
                        with kl_guard_attempt_log_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(attempt_record, ensure_ascii=False) + "\n")
                            handle.flush()
                    break
                restore_training_state(policy, optimizer, state_snapshot)
                rollback_policy_state_digest = policy_state_digest(policy)
                rollback_optimizer_state_digest = optimizer_state_digest(optimizer)
                rollback_parameter_delta = parameter_delta_metrics(policy, state_snapshot[0])
                rollback_exact_match = (
                    rollback_policy_state_digest == pre_step_policy_state_digest
                    and rollback_optimizer_state_digest == pre_step_optimizer_state_digest
                )
                attempt_record["rollback_after_attempt"] = {
                    "policy_state_digest": rollback_policy_state_digest,
                    "optimizer_state_digest": rollback_optimizer_state_digest,
                    "parameter_delta": rollback_parameter_delta,
                    "exact_match": rollback_exact_match,
                }
                kl_guard_attempt_history.append(attempt_record)
                if kl_guard_attempt_log_path is not None:
                    with kl_guard_attempt_log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(attempt_record, ensure_ascii=False) + "\n")
                        handle.flush()
                if not rollback_exact_match:
                    rollback_verification_failed = True
                    break
            if not accepted:
                restore_training_state(policy, optimizer, state_snapshot)
                rollback_policy_state_digest = policy_state_digest(policy)
                rollback_optimizer_state_digest = optimizer_state_digest(optimizer)
                rollback_parameter_delta = parameter_delta_metrics(policy, state_snapshot[0])
                rollback_exact_match = (
                    rollback_policy_state_digest == pre_step_policy_state_digest
                    and rollback_optimizer_state_digest == pre_step_optimizer_state_digest
                )
                optimizer.zero_grad(set_to_none=True)
                optimizer_step_applied = False
                optimizer_step_rejected = True
                kl_guard_backoff_count = max(0, kl_guard_attempts - 1)
            else:
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base_lr
        else:
            optimizer.step()
        if full_fp32_measurement_model is not None:
            del full_fp32_measurement_model
        total_samples = sum(item["sample_count"] for item in micro_summaries)

        def weighted_mean(key: str) -> float:
            return sum(item[key] * item["sample_count"] for item in micro_summaries) / total_samples

        termination_reason_counts = {
            reason: sum(item["termination_reason_counts"].get(reason, 0) for item in micro_summaries)
            for reason in ("eos", "max_new_tokens", "no_eos_short_generation")
        }

        summary = {
            "step": step,
            "mode": args.mode,
            "diagnostic_schema_version": 4 if args.pre_step_loss_diagnostic_fp32 else 2,
            "training_forward_mode": args.training_forward_mode,
            "active_loss_precision": args.training_forward_mode,
            "reward_source": "controlled_reward_pattern" if controlled_reward_pattern is not None else "rule_reward",
            "controlled_reward_override_enabled": controlled_reward_pattern is not None,
            "controlled_reward_pattern": controlled_reward_pattern,
            "reward_mean": sum(item["reward_mean"] for item in micro_summaries) / len(micro_summaries),
            "reward_std_mean": sum(item["reward_std"] for item in micro_summaries) / len(micro_summaries),
            "rule_reward_mean": weighted_mean("rule_reward_mean"),
            "controlled_reward_delta_mean": weighted_mean("controlled_reward_delta_mean"),
            "advantage_abs_max": max(item["advantage_abs_max"] for item in micro_summaries),
            "advantage_nonzero_rate": weighted_mean("advantage_nonzero_rate"),
            "loss_mean": sum(item["loss"] for item in micro_summaries) / len(micro_summaries),
            "kl_mean": sum(item["kl"] for item in micro_summaries) / len(micro_summaries),
            "kl_measurement_phase": "pre_optimizer_step",
            "pre_step_kl_bfloat16_shadow_mean": (
                pre_step_kl_bfloat16_shadow["reference_kl_mean"]
                if pre_step_kl_bfloat16_shadow is not None else None
            ),
            "pre_step_kl_bfloat16_shadow_p95": (
                pre_step_kl_bfloat16_shadow["reference_kl_p95"]
                if pre_step_kl_bfloat16_shadow is not None else None
            ),
            "pre_step_kl_bfloat16_shadow_max": (
                pre_step_kl_bfloat16_shadow["reference_kl_max"]
                if pre_step_kl_bfloat16_shadow is not None else None
            ),
            "pre_step_kl_fp32_no_autocast_mean": (
                pre_step_kl_fp32_no_autocast["reference_kl_mean"]
                if pre_step_kl_fp32_no_autocast is not None else None
            ),
            "pre_step_kl_fp32_no_autocast_p95": (
                pre_step_kl_fp32_no_autocast["reference_kl_p95"]
                if pre_step_kl_fp32_no_autocast is not None else None
            ),
            "pre_step_kl_fp32_no_autocast_max": (
                pre_step_kl_fp32_no_autocast["reference_kl_max"]
                if pre_step_kl_fp32_no_autocast is not None else None
            ),
            "pre_step_shadow_disagreement": pre_step_shadow_disagreement,
            "completion_tokens_mean": sum(item["average_completion_tokens"] for item in micro_summaries) / len(micro_summaries),
            "unique_prompt_count": len({item["prompt_id"] for item in micro_summaries}),
            "train_sample_count": total_samples,
            "train_validator_pass_rate": weighted_mean("train_validator_pass_rate"),
            "train_empty_response_rate": weighted_mean("train_empty_response_rate"),
            "train_max_length_hit_rate": weighted_mean("train_max_length_hit_rate"),
            "train_natural_end_rate": weighted_mean("train_natural_end_rate"),
            "termination_reason_counts": termination_reason_counts,
            "termination_unknown_rate": termination_reason_counts["no_eos_short_generation"] / total_samples,
            "train_repetition_penalty_mean": weighted_mean("train_repetition_penalty_mean"),
            "train_reward_components": {
                name: sum(
                    item["train_reward_components"][name] * item["sample_count"]
                    for item in micro_summaries
                ) / total_samples
                for name in ("validator_reward", "termination_reward", "repetition_penalty")
            },
            "grad_norm_pre_clip": grad_norm_pre_clip,
            "grad_norm_post_clip": grad_norm_post_clip,
            "grad_was_clipped": grad_norm_pre_clip > args.max_grad_norm,
            "post_step_kl_target": args.post_step_kl_target,
            "post_step_kl_mean": post_step_kl["reference_kl_mean"],
            "post_step_kl_p95": post_step_kl["reference_kl_p95"],
            "post_step_kl_max": post_step_kl["reference_kl_max"],
            "post_step_kl_float32_mean": (
                post_step_kl_float32["reference_kl_mean"] if post_step_kl_float32 is not None else None
            ),
            "post_step_kl_float32_p95": (
                post_step_kl_float32["reference_kl_p95"] if post_step_kl_float32 is not None else None
            ),
            "post_step_kl_float32_max": (
                post_step_kl_float32["reference_kl_max"] if post_step_kl_float32 is not None else None
            ),
            "post_step_kl_full_float32_mean": (
                post_step_kl_full_float32["reference_kl_mean"]
                if post_step_kl_full_float32 is not None else None
            ),
            "post_step_kl_full_float32_p95": (
                post_step_kl_full_float32["reference_kl_p95"]
                if post_step_kl_full_float32 is not None else None
            ),
            "post_step_kl_full_float32_max": (
                post_step_kl_full_float32["reference_kl_max"]
                if post_step_kl_full_float32 is not None else None
            ),
            "post_step_kl_gate_mode": args.post_step_kl_gate_mode,
            "post_step_kl_gate_mean": (
                post_step_kl_gate["reference_kl_mean"] if post_step_kl_gate is not None else None
            ),
            "post_step_kl_gate_p95": (
                post_step_kl_gate["reference_kl_p95"] if post_step_kl_gate is not None else None
            ),
            "post_step_kl_gate_max": (
                post_step_kl_gate["reference_kl_max"] if post_step_kl_gate is not None else None
            ),
            "post_step_kl_gate_passed": post_step_kl_gate_passed,
            "legacy_bfloat16_gate_passed": legacy_bfloat16_gate_passed,
            "fp32_no_autocast_gate_passed": fp32_no_autocast_gate_passed,
            "shadow_gate_disagreement": shadow_gate_disagreement,
            "kl_guard_enabled": kl_guard_enabled,
            "kl_guard_attempts": kl_guard_attempts,
            "kl_guard_backoff_count": kl_guard_backoff_count,
            "kl_guard_accepted_lr_multiplier": kl_guard_accepted_lr_multiplier,
            "optimizer_step_applied": optimizer_step_applied,
            "optimizer_step_rejected": optimizer_step_rejected,
            "kl_guard_attempt_history": kl_guard_attempt_history,
            "kl_guard_attempt_log_path": str(kl_guard_attempt_log_path) if kl_guard_attempt_log_path else None,
            "kl_guard_token_replay_path": str(kl_guard_token_replay_path) if kl_guard_token_replay_path else None,
            "pre_step_policy_state_digest": pre_step_policy_state_digest,
            "pre_step_optimizer_state_digest": pre_step_optimizer_state_digest,
            "final_parameter_delta": final_parameter_delta,
            "rollback_policy_state_digest": rollback_policy_state_digest,
            "rollback_optimizer_state_digest": rollback_optimizer_state_digest,
            "rollback_parameter_delta": rollback_parameter_delta,
            "rollback_exact_match": rollback_exact_match,
            "rollback_verification_failed": rollback_verification_failed,
        }
        if args.pre_step_loss_diagnostic_fp32:
            precision_rows = [item["pre_step_loss_precision"] for item in micro_summaries]

            def precision_mean(key: str) -> float:
                return sum(float(item[key]) for item in precision_rows) / len(precision_rows)

            legacy_loss_mean = precision_mean("bfloat16_autocast_loss")
            fp32_loss_mean = precision_mean("fp32_no_autocast_loss")
            legacy_kl_mean = precision_mean("bfloat16_autocast_kl")
            fp32_kl_mean = precision_mean("fp32_no_autocast_kl")
            legacy_grad_mean = precision_mean("bfloat16_autocast_gradient_norm_unscaled")
            fp32_grad_mean = precision_mean("fp32_no_autocast_gradient_norm_unscaled")
            active_loss_key = f"{args.training_forward_mode}_loss"
            active_kl_key = f"{args.training_forward_mode}_kl"
            active_grad_key = f"{args.training_forward_mode}_gradient_norm_unscaled"
            active_loss_mean = precision_mean(active_loss_key)
            active_kl_mean = precision_mean(active_kl_key)
            active_grad_mean = precision_mean(active_grad_key)
            step_kl_gap_threshold = max(
                1e-4,
                0.1 * max(fp32_kl_mean, float(args.post_step_kl_target)),
            )
            step_loss_gap_threshold = max(1e-7, args.beta * step_kl_gap_threshold)
            step_gradient_gap = abs(legacy_grad_mean - fp32_grad_mean)
            step_gradient_ratio = legacy_grad_mean / max(fp32_grad_mean, 1e-12)
            summary["pre_step_loss_precision"] = {
                "training_forward_mode": args.training_forward_mode,
                "active_variant": args.training_forward_mode,
                "active_loss_mean": active_loss_mean,
                "active_kl_mean": active_kl_mean,
                "active_gradient_norm_unscaled_mean": active_grad_mean,
                "active_gradient_matches_selected_variant": all(
                    row["active_gradient_matches_selected_variant"] for row in precision_rows
                ),
                "bfloat16_autocast_loss_mean": legacy_loss_mean,
                "fp32_no_autocast_loss_mean": fp32_loss_mean,
                "loss_abs_gap": abs(legacy_loss_mean - fp32_loss_mean),
                "loss_gap_threshold": step_loss_gap_threshold,
                "loss_disagreement": abs(legacy_loss_mean - fp32_loss_mean) > step_loss_gap_threshold,
                "bfloat16_autocast_kl_mean": legacy_kl_mean,
                "fp32_no_autocast_kl_mean": fp32_kl_mean,
                "kl_abs_gap": abs(legacy_kl_mean - fp32_kl_mean),
                "kl_gap_threshold": step_kl_gap_threshold,
                "kl_disagreement": abs(legacy_kl_mean - fp32_kl_mean) > step_kl_gap_threshold,
                "bfloat16_autocast_gradient_norm_unscaled_mean": legacy_grad_mean,
                "fp32_no_autocast_gradient_norm_unscaled_mean": fp32_grad_mean,
                "gradient_norm_abs_gap": step_gradient_gap,
                "gradient_norm_ratio": step_gradient_ratio,
                "gradient_disagreement": step_gradient_gap > 1e-6 and step_gradient_ratio > 5.0,
                "microbatch_count": len(precision_rows),
                "microbatch_loss_disagreement_count": sum(item["loss_disagreement"] for item in precision_rows),
                "microbatch_kl_disagreement_count": sum(item["kl_disagreement"] for item in precision_rows),
                "microbatch_gradient_disagreement_count": sum(item["gradient_disagreement"] for item in precision_rows),
                "shadow_grad_isolation_ok": all(item["shadow_grad_isolation_ok"] for item in precision_rows),
                "same_token_replay_rows": len(precision_rows),
                "nonzero_advantage_microbatches": sum(float(item["advantage_abs_max"]) > 0.0 for item in precision_rows),
                "replay_path": str(pre_step_loss_replay_path),
            }
        diagnostic_mean_keys = (
            "ratio_mean",
            "ratio_p50",
            "ratio_p95",
            "log_ratio_mean",
            "log_ratio_abs_p95",
            "reference_kl_mean",
            "reference_kl_p50",
            "reference_kl_p95",
        )
        diagnostic_max_keys = ("ratio_max", "reference_kl_max")
        summary["policy_diagnostics"] = {
            "token_count": sum(item["policy_diagnostics"]["token_count"] for item in micro_summaries),
            **{
                key: sum(item["policy_diagnostics"][key] for item in micro_summaries) / len(micro_summaries)
                for key in diagnostic_mean_keys
            },
            **{
                key: max(item["policy_diagnostics"][key] for item in micro_summaries)
                for key in diagnostic_max_keys
            },
        }
        summary["kl_p95"] = summary["policy_diagnostics"]["reference_kl_p95"]
        summary["kl_max"] = summary["policy_diagnostics"]["reference_kl_max"]
        summary["ratio_p95"] = summary["policy_diagnostics"]["ratio_p95"]
        summary["ratio_max"] = summary["policy_diagnostics"]["ratio_max"]
        finite_keys = (
            "reward_mean",
            "reward_std_mean",
            "rule_reward_mean",
            "controlled_reward_delta_mean",
            "advantage_abs_max",
            "advantage_nonzero_rate",
            "loss_mean",
            "kl_mean",
            "kl_p95",
            "kl_max",
            "ratio_p95",
            "ratio_max",
            "completion_tokens_mean",
            "train_validator_pass_rate",
            "train_empty_response_rate",
            "train_max_length_hit_rate",
            "train_natural_end_rate",
            "train_repetition_penalty_mean",
            "grad_norm_pre_clip",
            "grad_norm_post_clip",
        )
        diagnostic_values = list(summary["policy_diagnostics"].values())
        if not all(torch.isfinite(torch.tensor(summary[key])) for key in finite_keys + ("grad_norm_pre_clip", "grad_norm_post_clip")) or not all(
            torch.isfinite(torch.tensor(value)) for value in diagnostic_values
        ):
            raise FloatingPointError(f"non-finite training summary at step {step}: {summary}")
        if args.pre_step_loss_diagnostic_fp32:
            precision_values = [
                float(value)
                for value in summary["pre_step_loss_precision"].values()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            if not finite_diagnostics({str(index): value for index, value in enumerate(precision_values)}):
                raise FloatingPointError(f"non-finite pre-step loss precision summary at step {step}: {summary}")
        if kl_guard_enabled and not finite_diagnostics({
            key: float(value)
            for key, value in post_step_kl.items()
        }):
            raise FloatingPointError(f"non-finite post-update summary at step {step}: {summary}")
        if kl_guard_enabled and post_step_kl_float32 is not None and not finite_diagnostics({
            key: float(value)
            for key, value in post_step_kl_float32.items()
        }):
            raise FloatingPointError(f"non-finite float32 post-update summary at step {step}: {summary}")
        if kl_guard_enabled and post_step_kl_full_float32 is not None and not finite_diagnostics({
            key: float(value)
            for key, value in post_step_kl_full_float32.items()
        }):
            raise FloatingPointError(f"non-finite full-float32 post-update summary at step {step}: {summary}")
        if kl_guard_enabled and post_step_kl_gate is not None and not finite_diagnostics({
            key: float(value)
            for key, value in post_step_kl_gate.items()
        }):
            raise FloatingPointError(f"non-finite selected post-update gate at step {step}: {summary}")
        for label, stats in (
            ("pre-step bfloat16 shadow", pre_step_kl_bfloat16_shadow),
            ("pre-step fp32 no-autocast", pre_step_kl_fp32_no_autocast),
        ):
            if stats is not None and not finite_diagnostics({key: float(value) for key, value in stats.items()}):
                raise FloatingPointError(f"non-finite {label} summary at step {step}: {summary}")
        if kl_guard_enabled:
            telemetry_values = {}
            for name, values in (("final", final_parameter_delta), ("rollback", rollback_parameter_delta)):
                if values is not None:
                    telemetry_values.update({f"{name}_{key}": float(value) for key, value in values.items()})
            if telemetry_values and not finite_diagnostics(telemetry_values):
                raise FloatingPointError(f"non-finite KL guard telemetry at step {step}: {summary}")
        if summary["kl_mean"] > args.kl_threshold:
            kl_consecutive += 1
        else:
            kl_consecutive = 0
        summary["kl_threshold"] = args.kl_threshold
        summary["kl_consecutive"] = kl_consecutive
        summary["kl_triggered"] = kl_consecutive >= args.kl_patience
        step_summaries.append(summary)
        with step_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print(json.dumps(summary, ensure_ascii=False))

        if optimizer_step_rejected:
            stop_reason = (
                "kl_guard_rollback_digest_mismatch"
                if rollback_verification_failed
                else "post_step_kl_guard_unresolved"
            )
            break

        should_checkpoint = step % args.checkpoint_every == 0
        should_evaluate = validation_rows is not None and (step % args.eval_every == 0 or should_checkpoint)
        checkpoint_path = None
        if should_checkpoint:
            checkpoint_path = checkpoints_dir / f"{args.save_weight}_step_{step:04d}_768.pth"
            save_checkpoint(policy, checkpoint_path)
        if should_evaluate:
            evaluation_model = policy
            evaluation_source = "in_memory_policy"
            if checkpoint_path is not None:
                evaluation_model = load_checkpoint_for_evaluation(config, checkpoint_path, device)
                evaluation_source = "reloaded_checkpoint"
            try:
                metrics, details = evaluate_policy(
                    evaluation_model,
                    tokenizer,
                    validation_rows,
                    device,
                    max_new_tokens=args.validation_max_new_tokens,
                )
            finally:
                if evaluation_model is not policy:
                    del evaluation_model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            quality = quality_triggered(metrics, baseline_validation["metrics"], args.quality_drop_points)
            record = {
                "step": step,
                "checkpoint": str(checkpoint_path) if checkpoint_path else None,
                "evaluation_source": evaluation_source,
                "metrics": metrics,
                "quality_checks": quality,
                "quality_triggered": quality["triggered"],
                "kl_mean": summary["kl_mean"],
                "kl_consecutive": kl_consecutive,
                "kl_triggered": bool(summary["kl_triggered"]),
                "details": details,
            }
            validation_records.append(record)
            with validation_history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps({"validation": step, **metrics, "quality_triggered": quality["triggered"]}, ensure_ascii=False))

        if summary["kl_triggered"]:
            stop_reason = f"reference_kl_above_{args.kl_threshold}_for_{args.kl_patience}_steps"
            break
        if validation_records and validation_records[-1]["quality_triggered"]:
            stop_reason = "validation_safety_or_termination_drop_exceeded_quality_gate"
            break

    selected = select_best_checkpoint(validation_records)
    selection = {
        "status": "selected_rl_checkpoint" if selected else "baseline_retained",
        "selected_checkpoint": selected.get("checkpoint") if selected else None,
        "selected_step": selected.get("step") if selected else None,
        "stop_reason": stop_reason,
        "baseline": baseline_validation["metrics"] if baseline_validation else None,
        "checkpoints": validation_records,
    }
    (output_dir / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_path = output_dir / f"{args.save_weight}_768.pth"
    if selected:
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite {output_path}")
        shutil.copy2(selected["checkpoint"], output_path)
    elif validation_rows is None:
        save_weight(policy, output_path)
    result = {
        "status": "PASS" if selected or validation_rows is None else ("EARLY_STOP_BASELINE_RETAINED" if stop_reason else "BASELINE_RETAINED_NO_ELIGIBLE_CHECKPOINT"),
        "mode": args.mode,
        "steps_requested": args.max_steps,
        "steps_completed": len(step_summaries),
        "config": vars(args) | {"manifest": str(args.manifest), "validation_manifest": str(args.validation_manifest) if args.validation_manifest else None},
        "summaries": step_summaries,
        "accepted_optimizer_steps": sum(bool(summary.get("optimizer_step_applied")) for summary in step_summaries),
        "selection": {key: selection[key] for key in ("status", "selected_checkpoint", "selected_step", "stop_reason")},
        "output": str(output_path) if selected or validation_rows is None else None,
    }
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
