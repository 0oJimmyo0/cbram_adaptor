import torch
import torch.nn as nn
import pytest

from models.cbramod import CBraMod
from tmlr.trainability import apply_trainability_contract


class TinyClassifier(nn.Module):
    def __init__(self, adapter_type=None):
        super().__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200, dim_feedforward=400,
            seq_len=10, n_layer=1, nhead=4,
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


@pytest.mark.parametrize("method,adapter_type", [
    ("full_finetune", None), ("frozen_probe", None),
    ("interaction_aligned", "channel"),
    ("interaction_aligned", "patch"),
    ("interaction_aligned", "channel_patch"),
])
def test_trainability_contract(method, adapter_type):
    model = TinyClassifier(adapter_type)
    groups, report = apply_trainability_contract(model, method, adapter_type)
    assert groups
    assert report["trainable_parameter_count"] > 0
    if method == "full_finetune":
        assert report["component_trainable_parameter_counts"]["backbone"] > 0
        assert report["component_trainable_parameter_counts"]["adapter"] == 0
    elif method == "frozen_probe":
        assert report["component_trainable_parameter_counts"]["backbone"] == 0
        assert report["component_trainable_parameter_counts"]["classifier"] > 0
    else:
        assert report["component_trainable_parameter_counts"]["backbone"] == 0
        assert report["component_trainable_parameter_counts"]["adapter"] > 0


def test_adapter_method_cannot_fallback_to_missing_adapter():
    with pytest.raises(AssertionError):
        apply_trainability_contract(TinyClassifier(None), "interaction_aligned", "channel")
