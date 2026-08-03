"""Lightweight MiniMind reward model and preference dataset.

The reward is read from the last valid assistant-response token.  The
implementation explicitly uses both attention and response masks so padding,
EOS presence, and truncated responses do not change the selected position
silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def pairwise_reward_loss(chosen: Tensor, rejected: Tensor) -> Tensor:
    """Bradley--Terry pairwise loss for chosen and rejected rewards."""
    if chosen.shape != rejected.shape:
        raise ValueError(f"reward shape mismatch: {chosen.shape} vs {rejected.shape}")
    return -F.logsigmoid(chosen.float() - rejected.float()).mean()


def pairwise_preference_accuracy(chosen: Tensor, rejected: Tensor) -> Tensor:
    """Return the fraction of pairs for which chosen reward is larger."""
    if chosen.shape != rejected.shape:
        raise ValueError(f"reward shape mismatch: {chosen.shape} vs {rejected.shape}")
    return (chosen > rejected).float().mean()


def last_response_positions(
    attention_mask: Tensor,
    response_mask: Tensor | None = None,
) -> Tensor:
    """Find one pooling position per sequence with a safe padding fallback.

    ``response_mask`` should include the response content and EOS when EOS is
    present.  If it is empty (for example because truncation removed the
    assistant marker), the last attention-valid token is used instead.
    """
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    valid = attention_mask.bool()
    fallback = valid.long().sum(dim=1).clamp_min(1) - 1
    if response_mask is None:
        return fallback
    if response_mask.shape != attention_mask.shape:
        raise ValueError("response_mask must have the same shape as attention_mask")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    positions = positions.unsqueeze(0).expand_as(attention_mask)
    response_positions = torch.where(response_mask.bool() & valid, positions, -1).max(dim=1).values
    return torch.where(response_positions >= 0, response_positions, fallback)


class MiniMindRewardModel(nn.Module):
    """MiniMind causal backbone with a scalar reward head."""

    def __init__(self, config: MiniMindConfig | None = None) -> None:
        super().__init__()
        self.config = config or MiniMindConfig(hidden_size=768, num_hidden_layers=8)
        self.backbone = MiniMindForCausalLM(self.config)
        self.reward_head = nn.Linear(self.config.hidden_size, 1)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        response_mask: Tensor | None = None,
    ) -> Tensor:
        """Return one float32 reward per sequence."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        positions = last_response_positions(attention_mask, response_mask)
        batch = torch.arange(input_ids.shape[0], device=input_ids.device)
        pooled = outputs.hidden_states[batch, positions]
        return self.reward_head(pooled.float()).squeeze(-1)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _assistant_markers(tokenizer: Any) -> tuple[list[int], list[int]]:
    bos = tokenizer.bos_token or ""
    eos = tokenizer.eos_token or ""
    assistant = tokenizer(f"{bos}assistant\n", add_special_tokens=False).input_ids
    close = tokenizer(f"{eos}\n", add_special_tokens=False).input_ids
    return list(assistant), list(close)


def build_response_mask(
    input_ids: Sequence[int],
    attention_mask: Sequence[int],
    assistant_marker: Sequence[int],
    eos_marker: Sequence[int],
) -> list[int]:
    """Build a response-token mask from rendered chat-template token IDs."""
    valid_length = sum(int(value) for value in attention_mask)
    ids = list(input_ids[:valid_length])
    mask = [0] * len(input_ids)
    if not ids or not assistant_marker:
        if valid_length:
            mask[valid_length - 1] = 1
        return mask
    cursor = 0
    matched = False
    while cursor <= len(ids) - len(assistant_marker):
        if ids[cursor : cursor + len(assistant_marker)] != list(assistant_marker):
            cursor += 1
            continue
        start = cursor + len(assistant_marker)
        end = len(ids)
        if eos_marker:
            for candidate in range(start, len(ids) - len(eos_marker) + 1):
                if ids[candidate : candidate + len(eos_marker)] == list(eos_marker):
                    end = min(len(ids), candidate + len(eos_marker))
                    break
        for position in range(start, end):
            mask[position] = 1
        matched = True
        cursor = end if end > cursor else cursor + 1
    if not matched and valid_length:
        mask[valid_length - 1] = 1
    return mask


class PreferenceRewardDataset(Dataset[dict[str, Tensor | str]]):
    """Tokenize chosen/rejected chat pairs with explicit response masks."""

    def __init__(self, path: str | Path, tokenizer: Any, max_length: int = 512) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = _jsonl(self.path)
        self.assistant_marker, self.eos_marker = _assistant_markers(tokenizer)

    def __len__(self) -> int:
        return len(self.samples)

    def _encode(self, messages: list[dict[str, Any]]) -> dict[str, Tensor]:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = self.tokenizer(
            rendered,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_attention_mask=True,
        )
        input_ids = list(encoded["input_ids"])
        attention = list(encoded["attention_mask"])
        response = build_response_mask(
            input_ids,
            attention,
            self.assistant_marker,
            self.eos_marker,
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "response_mask": torch.tensor(response, dtype=torch.long),
        }

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        chosen = self._encode(sample["chosen"])
        rejected = self._encode(sample["rejected"])
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_response_mask": chosen["response_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_response_mask": rejected["response_mask"],
            "id": str(sample.get("id", index)),
            "category": str(sample.get("category", "unknown")),
            "family": str(sample.get("family", "unknown")),
        }


def load_reward_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> MiniMindRewardModel:
    """Load a saved reward model checkpoint without changing the base model."""
    payload = torch.load(Path(path), map_location=device)
    config_data = payload.get("config", {}) if isinstance(payload, dict) else {}
    config = MiniMindConfig(
        hidden_size=int(config_data.get("hidden_size", 768)),
        num_hidden_layers=int(config_data.get("num_hidden_layers", 8)),
    )
    model = MiniMindRewardModel(config)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)
    return model


class RewardAPI:
    """Small inference API exposing ``reward`` and ``rank``."""

    def __init__(self, model: MiniMindRewardModel, tokenizer: Any, max_length: int = 512, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.dataset = PreferenceRewardDataset.__new__(PreferenceRewardDataset)
        self.dataset.tokenizer = tokenizer
        self.dataset.max_length = max_length
        self.dataset.assistant_marker, self.dataset.eos_marker = _assistant_markers(tokenizer)

    def _encode(self, prompt: str, response: str) -> dict[str, Tensor]:
        return self.dataset._encode([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ])

    @torch.inference_mode()
    def reward(self, prompt: str, response: str) -> float:
        """Score one prompt/response pair."""
        batch = self._encode(prompt, response)
        value = self.model(
            batch["input_ids"].unsqueeze(0).to(self.device),
            batch["attention_mask"].unsqueeze(0).to(self.device),
            batch["response_mask"].unsqueeze(0).to(self.device),
        )
        return float(value.item())

    def rank(self, prompt: str, responses: Iterable[str]) -> list[dict[str, Any]]:
        """Return responses sorted by descending reward."""
        ranked = [{"response": text, "reward": self.reward(prompt, text)} for text in responses]
        return sorted(ranked, key=lambda row: row["reward"], reverse=True)
