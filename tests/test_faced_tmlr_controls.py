"""Construction, strict-load order, and trainability tests for TMLR controls."""

from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.cbramod import CBraMod
from models.lora import inject_qkv_lora
from tmlr.trainability import apply_trainability_contract


class TinyClassifier(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200, dim_feedforward=64,
            seq_len=10, n_layer=12, nhead=8,
        )
        if mode == "generic_bottleneck":
            self.backbone.enable_generic_adapter(bottleneck=8, seed=7)
        elif mode == "axis_blind":
            self.backbone.enable_generic_adapter(bottleneck=8, seed=7, adapter_type="axis_blind")
        elif mode in {"interaction_aligned", "native_full_finetune"}:
            self.backbone.enable_interaction_adapter(
                adapter_type="channel_patch", bottleneck=8, num_heads=2, seed=7,
            )
        elif mode == "lora":
            inject_qkv_lora(self.backbone, rank=2, alpha=4.0)
        self.backbone.proj_out = nn.Identity()
        self.classifier = nn.Linear(200, 9)

    def forward(self, x):
        return self.classifier(self.backbone(x).mean(dim=(1, 2)))


@pytest.mark.parametrize("method,mode", [
    ("generic_bottleneck", "generic_bottleneck"),
    ("axis_blind", "axis_blind"),
    ("interaction_aligned", "interaction_aligned"),
    ("native_full_finetune", "native_full_finetune"),
    ("lora", "lora"),
    ("upper_k_finetune", "none"),
])
def test_control_trainability_is_explicit(method, mode):
    model = TinyClassifier(mode)
    groups, report = apply_trainability_contract(
        model, method, "channel_patch" if method in {"interaction_aligned", "native_full_finetune"} else None,
    )
    assert groups
    assert report["trainable_parameter_count"] > 0
    if method == "lora":
        assert report["component_trainable_parameter_counts"]["lora"] > 0
        assert report["component_trainable_parameter_counts"]["adapter"] == 0
    elif method == "upper_k_finetune":
        assert report["component_trainable_parameter_counts"]["upper"] > 0
        assert report["component_trainable_parameter_counts"]["backbone"] == 0
    elif method == "interaction_aligned":
        assert report["component_trainable_parameter_counts"]["adapter"] > 0
    else:
        assert report["component_trainable_parameter_counts"]["adapter"] > 0
        assert report["component_trainable_parameter_counts"]["backbone"] > 0


def test_lora_forward_and_gradients():
    model = TinyClassifier("lora")
    samples = torch.randn(2, 4, 3, 200)
    output = model(samples)
    assert output.shape == (2, 9)
    output.sum().backward()
    lora_grads = [
        parameter.grad for name, parameter in model.named_parameters()
        if ("lora_A" in name or "lora_B" in name) and parameter.grad is not None
    ]
    assert lora_grads
    assert all(torch.isfinite(value).all() for value in lora_grads)
