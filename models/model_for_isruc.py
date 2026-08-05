"""CBraMod ISRUC sequence model.

Each 20-epoch ISRUC item is encoded epoch-by-epoch with the CBraMod grid
``[6,30,200]``.  A one-layer Transformer then models the explicit sequence of
20 epoch representations and emits one sleep-stage prediction per epoch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbramod import CBraMod
from .generic_adapter import GenericBottleneckResidual
from .lora import inject_qkv_lora


class ISRUCSequenceHead(nn.Module):
    """LaBraM-compatible sequence head, kept under ``classifier`` for audits."""

    def __init__(self, num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.epoch_projection = nn.Sequential(
            nn.Linear(6 * 30 * 200, 512),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=4,
            dim_feedforward=2048,
            dropout=float(dropout),
            batch_first=True,
            activation=F.gelu,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.output = nn.Linear(512, int(num_classes))

    def forward(self, epoch_features: torch.Tensor) -> torch.Tensor:
        if epoch_features.ndim != 5 or tuple(epoch_features.shape[2:]) != (6, 30, 200):
            raise ValueError(
                "ISRUC sequence head expects [B,20,6,30,200] epoch features, "
                f"got {tuple(epoch_features.shape)}"
            )
        batch, sequence, channels, patches, dim = epoch_features.shape
        projected = self.epoch_projection(epoch_features.reshape(batch, sequence, channels * patches * dim))
        contextual = self.sequence_encoder(projected)
        return self.output(contextual)


class Model(nn.Module):
    """Compatibility model for the legacy downstream entrypoint."""

    def __init__(self, param):
        super().__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30, n_layer=12, nhead=8,
        )
        if bool(getattr(param, "use_pretrained_weights", True)):
            checkpoint = torch.load(param.foundation_dir, map_location="cpu")
            load_report = self.backbone.load_state_dict(checkpoint, strict=True)
            if load_report.missing_keys or load_report.unexpected_keys:
                raise RuntimeError(
                    f"ISRUC checkpoint mismatch: missing={load_report.missing_keys} "
                    f"unexpected={load_report.unexpected_keys}"
                )
            self.checkpoint_report = {
                "strict_loading_status": True,
                "missing_keys": list(load_report.missing_keys),
                "unexpected_keys": list(load_report.unexpected_keys),
            }
        else:
            self.checkpoint_report = {
                "strict_loading_status": False,
                "reason": "use_pretrained_weights=false",
            }
        self.backbone.proj_out = nn.Identity()

        adapter_type = str(getattr(param, "adapter_type", "none")).strip().lower()
        method = str(getattr(param, "method", "full_finetune")).strip().lower()
        if method in {"interaction_aligned", "native_full_finetune"} and adapter_type != "none":
            self.backbone.enable_interaction_adapter(
                adapter_type=adapter_type,
                bottleneck=int(getattr(param, "adapter_bottleneck", 64)),
                num_heads=int(getattr(param, "adapter_heads", 4)),
                dropout=float(getattr(param, "adapter_dropout", 0.0)),
                init_alpha=float(getattr(param, "adapter_init_alpha", 0.01)),
                gamma=float(getattr(param, "adapter_gamma", 1.0)),
                zero_init_output=bool(getattr(param, "adapter_zero_init_output", True)),
                seed=int(getattr(param, "adapter_seed", 12345)),
            )
        elif method == "generic_bottleneck":
            self.backbone.enable_generic_adapter(
                bottleneck=int(getattr(param, "generic_bottleneck", 64)),
                init_alpha=float(getattr(param, "adapter_init_alpha", 0.01)),
                gamma=float(getattr(param, "adapter_gamma", 1.0)),
                zero_init_output=bool(getattr(param, "adapter_zero_init_output", True)),
                seed=int(getattr(param, "adapter_seed", 12345)),
                adapter_type="generic_bottleneck",
            )
        elif method == "axis_blind":
            self.backbone.enable_generic_adapter(
                bottleneck=int(getattr(param, "axis_blind_bottleneck", 148)),
                init_alpha=float(getattr(param, "adapter_init_alpha", 0.01)),
                gamma=float(getattr(param, "adapter_gamma", 1.0)),
                zero_init_output=bool(getattr(param, "adapter_zero_init_output", True)),
                seed=int(getattr(param, "adapter_seed", 12345)),
                adapter_type="axis_blind",
            )
        elif method == "lora":
            inject_qkv_lora(
                self.backbone,
                rank=int(getattr(param, "lora_rank", 8)),
                alpha=float(getattr(param, "lora_alpha", 16.0)),
            )
        elif method not in {"full_finetune", "frozen_probe", "upper_k_finetune"}:
            raise ValueError(f"Unsupported ISRUC method {method!r}")
        self.classifier = ISRUCSequenceHead(
            num_classes=int(getattr(param, "num_of_classes", 5)),
            dropout=float(getattr(param, "dropout", 0.1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or tuple(x.shape[2:]) != (6, 30, 200):
            raise ValueError(f"ISRUC model expects [B,20,6,30,200], got {tuple(x.shape)}")
        batch, sequence, channels, patches, patch_size = x.shape
        epoch_features = self.backbone(x.reshape(batch * sequence, channels, patches, patch_size))
        epoch_features = epoch_features.reshape(batch, sequence, channels, patches, 200)
        return self.classifier(epoch_features)
