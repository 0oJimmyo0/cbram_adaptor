"""Regression tests for deterministic frozen-backbone training modes."""

from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.cbramod import CBraMod
from tmlr.trainability import configure_training_modes


class TinyClassifier(nn.Module):
    def __init__(self, adapter_type=None, n_layer=4):
        super().__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200, dim_feedforward=400,
            seq_len=10, n_layer=n_layer, nhead=4,
        )
        if adapter_type is not None:
            self.backbone.enable_interaction_adapter(
                adapter_type=adapter_type, bottleneck=8, num_heads=2,
                zero_init_output=True, seed=12345,
            )
        self.backbone.proj_out = nn.Identity()
        self.classifier = nn.Linear(200, 9)

    def forward(self, x):
        return self.classifier(self.backbone(x).mean(dim=(1, 2)))


def test_frozen_native_keeps_base_deterministic_and_adapter_trainable():
    model = TinyClassifier("channel")
    report = configure_training_modes(model, "interaction_aligned")
    assert report["frozen_backbone_eval_mode"] is True
    assert model.training and model.classifier.training
    assert not model.backbone.training
    assert model.backbone.native_axis_adapter.training
    assert all(not module.training for module in model.backbone.modules()
               if isinstance(module, nn.Dropout))

    samples = torch.randn(1, 4, 2, 200)
    with torch.no_grad():
        first = model.backbone(samples)
        second = model.backbone(samples)
    assert torch.allclose(first, second, atol=0.0, rtol=0.0)


def test_full_native_keeps_backbone_trainable_mode():
    model = TinyClassifier("channel")
    report = configure_training_modes(model, "native_full_finetune")
    assert report["frozen_backbone_eval_mode"] is False
    assert model.training and model.backbone.training
    assert model.classifier.training and model.backbone.native_axis_adapter.training


def test_upper_control_only_reenables_upper_layers():
    model = TinyClassifier(None, n_layer=4)
    report = configure_training_modes(model, "upper_k_finetune", upper_k=2)
    assert report["frozen_backbone_eval_mode"] is True
    assert not model.backbone.training
    assert all(not layer.training for layer in model.backbone.encoder.layers[:2])
    assert all(layer.training for layer in model.backbone.encoder.layers[2:])
