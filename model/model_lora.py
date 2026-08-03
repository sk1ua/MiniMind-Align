"""Small, explicit LoRA adapter support for MiniMind."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn


class LoRA(nn.Module):
    """Low-rank residual with explicit alpha scaling and dropout."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(self.dropout(x))) * self.scaling


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
    target_modules: Iterable[str] = ("q_proj", "v_proj"),
) -> list[str]:
    """Attach adapters to named attention projections and return module names."""
    target = set(target_modules)
    device = next(model.parameters()).device
    attached: list[str] = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in target:
            lora = LoRA(module.in_features, module.out_features, rank, alpha, dropout).to(device)
            module.add_module("lora", lora)
            original_forward = module.forward

            def forward_with_lora(x: torch.Tensor, layer=original_forward, adapter=lora) -> torch.Tensor:
                return layer(x) + adapter(x)

            module.forward = forward_with_lora
            attached.append(name)
    if not attached:
        raise ValueError(f"no target modules found: {sorted(target)}")
    return attached


def load_lora(model: nn.Module, path: str | Path) -> None:
    """Load an adapter state dict into an already adapted model."""
    state_dict = torch.load(path, map_location=next(model.parameters()).device)
    state_dict = {(key[7:] if key.startswith("module.") else key): value for key, value in state_dict.items()}
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            prefix = f"{name}.lora."
            values = {key.replace(prefix, ""): value for key, value in state_dict.items() if key.startswith(prefix)}
            if not values:
                raise KeyError(f"missing adapter weights for {name}")
            module.lora.load_state_dict(values)


def save_lora(model: nn.Module, path: str | Path) -> None:
    """Save only LoRA parameters and their module configuration."""
    raw_model = getattr(model, "_orig_mod", model)
    state_dict: dict[str, torch.Tensor] = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            for key, value in module.lora.state_dict().items():
                state_dict[f"{name}.lora.{key}"] = value.cpu().half()
    if not state_dict:
        raise ValueError("model has no LoRA modules")
    torch.save(state_dict, path)


def merge_lora(model: nn.Module, lora_path: str | Path, save_path: str | Path) -> None:
    """Merge trained adapter residuals into a standalone half-precision state dict."""
    load_lora(model, lora_path)
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {key: value.cpu().half() for key, value in raw_model.state_dict().items() if ".lora." not in key}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "lora"):
            delta = (module.lora.B.weight @ module.lora.A.weight) * module.lora.scaling
            state_dict[f"{name}.weight"] = (module.weight.data + delta).cpu().half()
    torch.save(state_dict, save_path)
