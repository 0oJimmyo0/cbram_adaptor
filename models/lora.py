"""Independent LoRA control for the original CBraMod attention QKV weights."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.nn.utils import parametrize


class LoRAWeight(nn.Module):
    """Low-rank additive parametrization for one frozen weight matrix."""

    def __init__(self, out_features: int, in_features: int, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.lora_A = nn.Parameter(torch.empty(self.rank, int(in_features)))
        self.lora_B = nn.Parameter(torch.zeros(int(out_features), self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        return base + self.scaling * (self.lora_B @ self.lora_A)


def inject_qkv_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0) -> List[str]:
    """Attach independent rank-r QKV updates to every CBraMod encoder layer.

    The pretrained module is expected to have been loaded before this function
    is called.  The base ``in_proj_weight`` remains frozen; only the fresh LoRA
    factors are trainable under the separate ``lora`` method.
    """
    replaced: List[str] = []
    layers = getattr(getattr(model, "encoder", None), "layers", None)
    if layers is None:
        raise TypeError("CBraMod model does not expose encoder layers")
    for layer_index, layer in enumerate(layers):
        for branch_name in ("self_attn_s", "self_attn_t"):
            attention = getattr(layer, branch_name)
            if attention.in_proj_weight is None:
                raise ValueError(f"Missing in_proj_weight for encoder layer {layer_index} {branch_name}")
            if parametrize.is_parametrized(attention, "in_proj_weight"):
                raise RuntimeError(f"LoRA already injected into layer {layer_index} {branch_name}")
            out_features, in_features = attention.in_proj_weight.shape
            parametrize.register_parametrization(
                attention,
                "in_proj_weight",
                LoRAWeight(out_features, in_features, rank=rank, alpha=alpha),
            )
            replaced.append(f"encoder.layers.{layer_index}.{branch_name}.in_proj_weight")
    if not replaced:
        raise RuntimeError("No CBraMod QKV projections were found for LoRA injection")
    return replaced


def lora_diagnostics(model: nn.Module) -> Dict[str, float]:
    """Return aggregate LoRA gradient diagnostics for epoch artifacts."""
    values = {"lora_a_grad_norm": [], "lora_b_grad_norm": []}
    counts = {"lora_a_parameter_count": 0, "lora_b_parameter_count": 0}
    for name, parameter in model.named_parameters():
        if name.endswith("lora_A"):
            counts["lora_a_parameter_count"] += parameter.numel()
            if parameter.grad is not None:
                values["lora_a_grad_norm"].append(parameter.grad.detach().cpu())
        elif name.endswith("lora_B"):
            counts["lora_b_parameter_count"] += parameter.numel()
            if parameter.grad is not None:
                values["lora_b_grad_norm"].append(parameter.grad.detach().cpu())

    def aggregate(tensors: Iterable[torch.Tensor]) -> float:
        tensors = list(tensors)
        if not tensors:
            return 0.0
        return float(sum(value.float().pow(2).sum() for value in tensors).sqrt())

    return {
        **counts,
        "lora_a_grad_norm": aggregate(values["lora_a_grad_norm"]),
        "lora_b_grad_norm": aggregate(values["lora_b_grad_norm"]),
    }
