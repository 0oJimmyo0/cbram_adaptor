"""Interaction-aligned residual adapters for the original CBraMod backbone.

CBraMod represents an example as ``[B, C, S, D]``.  Its native criss-cross
attention assigns the first ``D // 2`` features to spatial/channel attention
and the second ``D // 2`` features to temporal/patch attention.  This module
uses that existing semantic split; it does not import or reimplement LaBraM.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn


class _NativeBranchResidual(nn.Module):
    """Down -> attention mixer -> Up on one native CBraMod branch."""

    def __init__(
        self,
        branch_dim: int,
        bottleneck: int,
        num_heads: int,
        dropout: float,
        init_alpha: float,
        zero_init_output: bool,
    ) -> None:
        super().__init__()
        if branch_dim <= 0:
            raise ValueError(f"branch_dim must be positive, got {branch_dim}")
        if bottleneck <= 0 or bottleneck % num_heads != 0:
            raise ValueError(
                "bottleneck must be positive and divisible by num_heads; "
                f"got bottleneck={bottleneck}, num_heads={num_heads}"
            )
        self.branch_dim = int(branch_dim)
        self.bottleneck = int(bottleneck)
        self.num_heads = int(num_heads)
        self.norm = nn.LayerNorm(self.branch_dim)
        self.down = nn.Linear(self.branch_dim, self.bottleneck)
        self.mixer = nn.MultiheadAttention(
            self.bottleneck,
            self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.up = nn.Linear(self.bottleneck, self.branch_dim)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self._last_raw_ratio: Optional[float] = None

        if zero_init_output:
            nn.init.zeros_(self.up.weight)
            if self.up.bias is not None:
                nn.init.zeros_(self.up.bias)

    def forward(self, branch: torch.Tensor, axis: str) -> torch.Tensor:
        if branch.ndim != 4:
            raise ValueError(f"Expected branch [B,C,S,D], got {tuple(branch.shape)}")
        batch, channels, patches, dim = branch.shape
        if dim != self.branch_dim:
            raise ValueError(
                f"Expected branch dim {self.branch_dim}, got {dim} for axis={axis}"
            )
        if axis == "channel":
            sequence = branch.permute(0, 2, 1, 3).reshape(batch * patches, channels, dim)
        elif axis == "patch":
            sequence = branch.reshape(batch * channels, patches, dim)
        else:
            raise ValueError(f"Unknown native axis {axis!r}")

        projected = self.down(self.norm(sequence))
        mixed = self.mixer(projected, projected, projected, need_weights=False)[0]
        mixed = self.up(mixed)
        if axis == "channel":
            mixed = mixed.reshape(batch, patches, channels, dim).permute(0, 2, 1, 3)
        else:
            mixed = mixed.reshape(batch, channels, patches, dim)
        self._last_raw_ratio = float(
            mixed.detach().float().norm()
            .div(branch.detach().float().norm().clamp_min(1e-12))
            .cpu()
        )
        return mixed

    def gradient_diagnostics(self, prefix: str) -> Dict[str, float]:
        diagnostics: Dict[str, float] = {}
        weight = self.mixer.in_proj_weight
        bias = self.mixer.in_proj_bias
        dim = self.mixer.embed_dim

        def norm(value: Optional[torch.Tensor]) -> float:
            if value is None:
                return 0.0
            return float(value.detach().float().norm().cpu())

        diagnostics[f"{prefix}_q_grad_norm"] = norm(None if weight.grad is None else weight.grad[:dim])
        diagnostics[f"{prefix}_k_grad_norm"] = norm(
            None if weight.grad is None else weight.grad[dim:2 * dim]
        )
        diagnostics[f"{prefix}_v_grad_norm"] = norm(
            None if weight.grad is None else weight.grad[2 * dim:]
        )
        diagnostics[f"{prefix}_output_projection_grad_norm"] = norm(self.mixer.out_proj.weight.grad)
        diagnostics[f"{prefix}_down_grad_norm"] = norm(self.down.weight.grad)
        diagnostics[f"{prefix}_up_grad_norm"] = norm(self.up.weight.grad)
        diagnostics[f"{prefix}_alpha_grad_norm"] = norm(self.alpha.grad)
        return diagnostics


class CBraModInteractionAdapter(nn.Module):
    """Common interaction-aligned residual family for CBraMod.

    ``channel`` acts only on CBraMod's spatial/channel half and mixes over the
    channel sequence.  ``patch`` acts only on its temporal half and mixes over
    the temporal-patch sequence.  ``channel_patch`` enables both independent
    branches.  The output is a correction with the same shape as the input.
    """

    VALID_TYPES = {"channel", "patch", "channel_patch"}

    def __init__(
        self,
        d_model: int = 200,
        bottleneck: int = 64,
        num_heads: int = 4,
        dropout: float = 0.0,
        init_alpha: float = 0.01,
        gamma: float = 1.0,
        adapter_type: str = "channel_patch",
        zero_init_output: bool = True,
        allow_singleton_patch: bool = False,
    ) -> None:
        super().__init__()
        adapter_type = str(adapter_type).strip().lower()
        if adapter_type not in self.VALID_TYPES:
            raise ValueError(
                f"adapter_type must be one of {sorted(self.VALID_TYPES)}, got {adapter_type!r}"
            )
        if d_model % 2 != 0:
            raise ValueError(f"CBraMod d_model must be even, got {d_model}")
        self.d_model = int(d_model)
        self.branch_dim = self.d_model // 2
        self.adapter_type = adapter_type
        self.gamma = float(gamma)
        self.zero_init_output = bool(zero_init_output)
        self.allow_singleton_patch = bool(allow_singleton_patch)

        if adapter_type in {"channel", "channel_patch"}:
            self.channel_branch = _NativeBranchResidual(
                self.branch_dim, bottleneck, num_heads, dropout,
                init_alpha, zero_init_output,
            )
        if adapter_type in {"patch", "channel_patch"}:
            self.patch_branch = _NativeBranchResidual(
                self.branch_dim, bottleneck, num_heads, dropout,
                init_alpha, zero_init_output,
            )
        self._last_geometry: Optional[Dict[str, int]] = None
        self._last_raw_channel_ratio: Optional[float] = None
        self._last_raw_patch_ratio: Optional[float] = None
        self._last_delta_ratio: Optional[float] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected CBraMod grid [B,C,S,D], got {tuple(x.shape)}")
        batch, channels, patches, dim = x.shape
        if dim != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {dim}")
        if hasattr(self, "channel_branch") and channels <= 1:
            raise ValueError("CBraMod channel interaction requires C > 1")
        if hasattr(self, "patch_branch") and patches <= 1 and not self.allow_singleton_patch:
            raise ValueError("CBraMod patch interaction requires S > 1")

        self._last_geometry = {
            "adapter_batch_size": int(batch),
            "adapter_channel_count": int(channels),
            "adapter_patch_count": int(patches),
            "adapter_embed_dim": int(dim),
            "channel_attention_sequence_length": int(channels)
            if hasattr(self, "channel_branch") else 0,
            "patch_attention_sequence_length": int(patches)
            if hasattr(self, "patch_branch") else 0,
            "channel_spatial_interactions_active": int(
                hasattr(self, "channel_branch") and channels > 1
            ),
            "patch_temporal_interactions_active": int(
                hasattr(self, "patch_branch") and patches > 1
            ),
            "singleton_patch_control": int(
                hasattr(self, "patch_branch") and patches <= 1 and self.allow_singleton_patch
            ),
        }

        delta = torch.zeros_like(x)
        self._last_raw_channel_ratio = None
        self._last_raw_patch_ratio = None
        spatial = x[..., :self.branch_dim]
        temporal = x[..., self.branch_dim:]

        if hasattr(self, "channel_branch"):
            channel_delta = self.channel_branch(spatial, axis="channel")
            delta[..., :self.branch_dim] += self.channel_branch.alpha * channel_delta
            self._last_raw_channel_ratio = self.channel_branch._last_raw_ratio
        if hasattr(self, "patch_branch"):
            patch_delta = self.patch_branch(temporal, axis="patch")
            delta[..., self.branch_dim:] += self.patch_branch.alpha * patch_delta
            self._last_raw_patch_ratio = self.patch_branch._last_raw_ratio

        delta = self.gamma * delta
        self._last_delta_ratio = float(
            delta.detach().float().norm()
            .div(x.detach().float().norm().clamp_min(1e-12))
            .cpu()
        )
        return delta

    def get_diagnostics(self) -> Dict[str, float]:
        diagnostics: Dict[str, float] = {
            "adapter_gamma": self.gamma,
            "adapter_type": self.adapter_type,
        }
        if self._last_geometry is not None:
            diagnostics.update(self._last_geometry)
        for name, branch in (("channel", getattr(self, "channel_branch", None)),
                             ("patch", getattr(self, "patch_branch", None))):
            if branch is not None:
                diagnostics[f"alpha_{name}"] = float(branch.alpha.detach().cpu())
        if self._last_raw_channel_ratio is not None:
            diagnostics["raw_channel_ratio"] = self._last_raw_channel_ratio
        if self._last_raw_patch_ratio is not None:
            diagnostics["raw_patch_ratio"] = self._last_raw_patch_ratio
        if self._last_delta_ratio is not None:
            diagnostics["adapter_delta_ratio"] = self._last_delta_ratio
        return diagnostics

    def get_gradient_diagnostics(self) -> Dict[str, float]:
        diagnostics: Dict[str, float] = {}
        for name, branch in (("channel", getattr(self, "channel_branch", None)),
                             ("patch", getattr(self, "patch_branch", None))):
            if branch is not None:
                diagnostics.update(branch.gradient_diagnostics(name))
        return diagnostics
