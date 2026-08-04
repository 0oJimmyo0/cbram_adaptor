"""Geometry and dense-parity checks for the CBraMod TMLR adapter."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.cbramod import CBraMod


def build(adapter_type="channel_patch", zero_init_output=True, gamma=1.0):
    return CBraMod(
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=400,
        seq_len=10,
        n_layer=2,
        nhead=4,
        adapter_type=adapter_type,
        adapter_bottleneck=8,
        adapter_heads=2,
        adapter_init_alpha=0.01,
        adapter_gamma=gamma,
        adapter_zero_init_output=zero_init_output,
        adapter_seed=12345,
    )


def test_zero_init_dense_parity():
    torch.manual_seed(7)
    dense = build(adapter_type="none").eval()
    torch.manual_seed(7)
    adapted = build(adapter_type="channel_patch", zero_init_output=True).eval()
    missing, unexpected = adapted.load_state_dict(dense.state_dict(), strict=False)
    assert not unexpected
    assert missing and all(key.startswith("native_axis_adapter.") for key in missing)

    samples = torch.randn(2, 4, 3, 200)
    with torch.no_grad():
        dense_out = dense(samples)
        adapted_out = adapted(samples)
    assert torch.allclose(dense_out, adapted_out, atol=1e-6, rtol=0.0)


def test_native_geometry_and_gradients():
    model = build(adapter_type="channel_patch", zero_init_output=False)
    samples = torch.randn(2, 4, 3, 200)
    output = model(samples)
    assert output.shape == (2, 4, 3, 200)
    diagnostics = model.get_adapter_diagnostics()
    assert diagnostics["adapter_channel_count"] == 4
    assert diagnostics["adapter_patch_count"] == 3
    assert diagnostics["channel_attention_sequence_length"] == 4
    assert diagnostics["patch_attention_sequence_length"] == 3
    assert diagnostics["channel_spatial_interactions_active"] == 1
    assert diagnostics["patch_temporal_interactions_active"] == 1

    output.sum().backward()
    gradients = model.get_adapter_diagnostics()
    assert gradients["channel_q_grad_norm"] > 0.0
    assert gradients["patch_q_grad_norm"] > 0.0
    assert gradients["channel_v_grad_norm"] > 0.0
    assert gradients["patch_v_grad_norm"] > 0.0


def test_ineligible_axis_is_rejected():
    channel = build(adapter_type="channel")
    with torch.no_grad():
        channel(torch.randn(2, 4, 1, 200))
    patch = build(adapter_type="patch")
    try:
        patch(torch.randn(2, 1, 1, 200))
    except ValueError as exc:
        assert "patch interaction requires S > 1" in str(exc)
    else:
        raise AssertionError("degenerate patch axis was accepted")


@torch.no_grad()
def test_native_branches_are_isolated_and_axis_equivariant():
    samples = torch.randn(2, 4, 3, 200)
    channel = build(adapter_type="channel", zero_init_output=False).eval()
    patch = build(adapter_type="patch", zero_init_output=False).eval()

    channel_delta = channel.native_axis_adapter(samples)
    patch_delta = patch.native_axis_adapter(samples)
    assert torch.equal(channel_delta[..., 100:], torch.zeros_like(channel_delta[..., 100:]))
    assert torch.equal(patch_delta[..., :100], torch.zeros_like(patch_delta[..., :100]))

    channel_permutation = torch.tensor([2, 0, 3, 1])
    channel_permuted = channel.native_axis_adapter(samples[:, channel_permutation])
    assert torch.allclose(channel_permuted, channel_delta[:, channel_permutation], atol=1e-6, rtol=0.0)

    patch_permutation = torch.tensor([2, 0, 1])
    patch_permuted = patch.native_axis_adapter(samples[:, :, patch_permutation])
    assert torch.allclose(patch_permuted, patch_delta[:, :, patch_permutation], atol=1e-6, rtol=0.0)

    opposite_channel = samples.clone()
    opposite_channel[..., 100:] = torch.randn_like(opposite_channel[..., 100:])
    assert torch.allclose(
        channel.native_axis_adapter(opposite_channel)[..., :100],
        channel_delta[..., :100], atol=1e-6, rtol=0.0,
    )
    opposite_patch = samples.clone()
    opposite_patch[..., :100] = torch.randn_like(opposite_patch[..., :100])
    assert torch.allclose(
        patch.native_axis_adapter(opposite_patch)[..., 100:],
        patch_delta[..., 100:], atol=1e-6, rtol=0.0,
    )


if __name__ == "__main__":
    test_zero_init_dense_parity()
    test_native_geometry_and_gradients()
    test_ineligible_axis_is_rejected()
    print("CBraMod interaction adapter tests: PASS")
