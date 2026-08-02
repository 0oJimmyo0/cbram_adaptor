"""Backbone-agnostic residual controls for the CBraMod TMLR matrix."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn


class GenericBottleneckResidual(nn.Module):
    """Token-wise Down -> nonlinearity -> Up residual control.

    This module deliberately does not mix over channels or temporal patches.
    It is therefore a generic bottleneck control, while the axis-blind variant
    uses a width selected to match the two-branch native adapter parameter
    count.
    """

    def __init__(
        self,
        d_model: int = 200,
        bottleneck: int = 64,
        init_alpha: float = 0.01,
        gamma: float = 1.0,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0 or bottleneck <= 0:
            raise ValueError("d_model and bottleneck must be positive")
        self.d_model = int(d_model)
        self.bottleneck = int(bottleneck)
        self.gamma = float(gamma)
        self.norm = nn.LayerNorm(self.d_model)
        self.down = nn.Linear(self.d_model, self.bottleneck)
        self.activation = nn.GELU()
        self.up = nn.Linear(self.bottleneck, self.d_model)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        if zero_init_output:
            nn.init.zeros_(self.up.weight)
            if self.up.bias is not None:
                nn.init.zeros_(self.up.bias)
        self._last_raw_ratio: Optional[float] = None
        self._last_delta_ratio: Optional[float] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[-1] != self.d_model:
            raise ValueError(f"Expected [B,C,S,{self.d_model}], got {tuple(x.shape)}")
        raw = self.up(self.activation(self.down(self.norm(x))))
        self._last_raw_ratio = float(
            raw.detach().float().norm()
            .div(x.detach().float().norm().clamp_min(1e-12))
            .cpu()
        )
        delta = self.gamma * self.alpha * raw
        self._last_delta_ratio = float(
            delta.detach().float().norm()
            .div(x.detach().float().norm().clamp_min(1e-12))
            .cpu()
        )
        return delta

    def get_diagnostics(self) -> Dict[str, float]:
        result: Dict[str, float] = {
            "generic_adapter_bottleneck": self.bottleneck,
            "generic_adapter_gamma": self.gamma,
            "generic_adapter_alpha": float(self.alpha.detach().cpu()),
        }
        if self._last_raw_ratio is not None:
            result["generic_raw_ratio"] = self._last_raw_ratio
        if self._last_delta_ratio is not None:
            result["generic_delta_ratio"] = self._last_delta_ratio
        return result

    def get_gradient_diagnostics(self) -> Dict[str, float]:
        def norm(value: Optional[torch.Tensor]) -> float:
            return 0.0 if value is None else float(value.detach().float().norm().cpu())

        return {
            "generic_down_grad_norm": norm(self.down.weight.grad),
            "generic_up_grad_norm": norm(self.up.weight.grad),
            "generic_alpha_grad_norm": norm(self.alpha.grad),
        }
