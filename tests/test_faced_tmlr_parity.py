from pathlib import Path

import torch
import torch.nn as nn
import pytest

from models.cbramod import CBraMod
from tmlr.trainability import apply_trainability_contract


class TinyClassifier(nn.Module):
    def __init__(self, adapter_type=None, zero_init_output=True):
        super().__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200, dim_feedforward=400,
            seq_len=10, n_layer=1, nhead=4,
        )
        if adapter_type is not None:
            self.backbone.enable_interaction_adapter(
                adapter_type=adapter_type, bottleneck=8, num_heads=2,
                zero_init_output=zero_init_output, seed=12345,
            )
        self.backbone.proj_out = nn.Identity()
        self.classifier = nn.Linear(200, 9)

    def forward(self, x):
        return self.classifier(self.backbone(x).mean(dim=(1, 2)))


def test_zero_init_feature_and_logit_parity():
    torch.manual_seed(17)
    dense = TinyClassifier(None).eval()
    torch.manual_seed(17)
    adapted = TinyClassifier("channel_patch", zero_init_output=True).eval()
    missing, unexpected = adapted.load_state_dict(dense.state_dict(), strict=False)
    assert not unexpected
    assert missing and all(name.startswith("backbone.native_axis_adapter.") for name in missing)
    inputs = torch.randn(1, 4, 3, 200)
    with torch.no_grad():
        dense_features = dense.backbone(inputs)
        adapted_features = adapted.backbone(inputs)
        dense_logits = dense(inputs)
        adapted_logits = adapted(inputs)
    feature_abs = (dense_features - adapted_features).abs().max().item()
    logit_abs = (dense_logits - adapted_logits).abs().max().item()
    assert feature_abs <= 1e-7
    assert logit_abs <= 1e-7


@pytest.mark.parametrize("adapter_type", ["channel", "patch", "channel_patch"])
def test_first_and_second_backward_contract(adapter_type):
    torch.manual_seed(3)
    model = TinyClassifier(adapter_type, zero_init_output=True)
    groups, _ = apply_trainability_contract(model, "interaction_aligned", adapter_type)
    optimizer = torch.optim.AdamW(groups)
    inputs = torch.randn(1, 4, 3, 200)
    labels = torch.tensor([2])

    optimizer.zero_grad(set_to_none=True)
    loss = nn.CrossEntropyLoss()(model(inputs), labels)
    loss.backward()
    first = model.backbone.get_adapter_diagnostics()
    if adapter_type in {"channel", "channel_patch"}:
        assert first["channel_up_grad_norm"] > 0.0
        assert first["channel_q_grad_norm"] <= 1e-12
        assert first["channel_k_grad_norm"] <= 1e-12
        assert first["channel_v_grad_norm"] <= 1e-12
        assert first["channel_down_grad_norm"] <= 1e-12
    if adapter_type in {"patch", "channel_patch"}:
        assert first["patch_up_grad_norm"] > 0.0
        assert first["patch_q_grad_norm"] <= 1e-12
        assert first["patch_k_grad_norm"] <= 1e-12
        assert first["patch_v_grad_norm"] <= 1e-12
        assert first["patch_down_grad_norm"] <= 1e-12
    if adapter_type == "channel":
        assert "patch_q_grad_norm" not in first
    if adapter_type == "patch":
        assert "channel_q_grad_norm" not in first
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    optimizer.step()

    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.") and not name.startswith("backbone.native_axis_adapter.")
    }
    optimizer.zero_grad(set_to_none=True)
    nn.CrossEntropyLoss()(model(inputs), labels).backward()
    second = model.backbone.get_adapter_diagnostics()
    if adapter_type in {"channel", "channel_patch"}:
        assert second["channel_q_grad_norm"] > 0.0
        assert second["channel_k_grad_norm"] > 0.0
        assert second["channel_v_grad_norm"] > 0.0
        assert second["channel_down_grad_norm"] > 0.0
    if adapter_type in {"patch", "channel_patch"}:
        assert second["patch_q_grad_norm"] > 0.0
        assert second["patch_k_grad_norm"] > 0.0
        assert second["patch_v_grad_norm"] > 0.0
        assert second["patch_down_grad_norm"] > 0.0
    optimizer.step()
    for name, previous in frozen_before.items():
        assert torch.equal(previous, dict(model.named_parameters())[name].detach())


def test_strict_checkpoint_load_precedes_adapter_attachment(tmp_path: Path):
    dense = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=400, seq_len=10, n_layer=1, nhead=4)
    path = tmp_path / "base.pt"
    torch.save(dense.state_dict(), path)
    loaded = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=400, seq_len=10, n_layer=1, nhead=4)
    report = loaded.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    assert report.missing_keys == []
    assert report.unexpected_keys == []
    loaded.enable_interaction_adapter("channel_patch", bottleneck=8, num_heads=2)
    assert loaded.native_axis_adapter is not None


def test_adapter_attachment_is_device_safe_and_not_repeatable():
    model = CBraMod(
        in_dim=200, out_dim=200, d_model=200, dim_feedforward=400,
        seq_len=10, n_layer=1, nhead=4,
    ).to(dtype=torch.float64)
    model.enable_interaction_adapter("channel", bottleneck=8, num_heads=2)
    assert next(model.native_axis_adapter.parameters()).dtype == torch.float64
    with pytest.raises(RuntimeError, match="Only one CBraMod adaptation family"):
        model.enable_interaction_adapter("patch", bottleneck=8, num_heads=2)
