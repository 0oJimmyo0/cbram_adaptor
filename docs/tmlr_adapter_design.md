# CBraMod TMLR adapter design

## Scientific role

The TMLR method is one interaction-aligned residual adaptation rule applied
to two different EEG backbones. The backbones are not given unrelated custom
adapters. The invariant method is:

```text
z     = Down(LayerNorm(h))
m     = native interaction mixer(z)
delta = Up(m)
h'    = h + alpha * gamma * delta
```

The only backbone-specific decision is which native interaction axis receives
the mixer. LoRA, generic bottleneck, upper-block-only, frozen-backbone, and
later parameter-matched axis-blind controls remain independent comparison
conditions; they are never combined with the native adapter.

## CBraMod mapping

The original CBraMod implementation emits `h` with shape `[B,C,S,D]`. In
`models/criss_cross_transformer.py`, the native attention explicitly splits
the embedding dimension into:

- `h[..., :D/2]`: spatial/channel branch, mixed over `C` for each temporal
  patch;
- `h[..., D/2:]`: temporal/patch branch, mixed over `S` for each channel.

The adapter therefore has three prespecified variants:

| Variant | Modified native subspace | Mixer sequence |
| --- | --- | --- |
| `channel` | first `D/2` features | `C` channels |
| `patch` | second `D/2` features | `S` temporal patches |
| `channel_patch` | both independent halves | `C` and `S` |

For FACED, the realized geometry is `[B,32,10,200]`, so both axes are
eligible. The adapter is attached once after the pretrained encoder output,
matching the current LaBraM residual placement while leaving all pretrained
CBraMod blocks and their criss-cross semantics unchanged.

The Up projections are zero-initialized by default. This gives exact dense
parity at initialization while preserving gradients into the adapter core.
The branch alpha parameters and residual/update diagnostics are recorded for
every run. A branch is rejected as inapplicable when its runtime sequence
length is one; no degenerate axis is silently reported as a positive result.

## FACED implementation gate

Before any paper run:

1. Load the original CBraMod checkpoint strictly before attaching adapter
   parameters.
2. Verify the local FACED LMDB contract: 32 channels in the persisted manifest,
   shape `(32,10,200)`, `/100` scaling, and the existing subject-disjoint
   train/validation/test split.
3. Run dense-versus-zero-initialized-adapter parity on the same input tensor.
4. Verify non-degenerate channel and patch sequence lengths and Q/K/V
   gradients with a nonzero residual probe.
5. Lock the dense baseline recipe by validation performance before selecting
   any adapter condition.

Only after these gates should the FACED registry run dense, frozen controls,
native channel/patch/channel+patch, native full-backbone-plus-adapter
conditions, generic, independent LoRA, and upper-block controls with seeds
`{42,1024,3407}`. Test performance is retained for reporting but never used to
select the recipe or checkpoint.

The native adapter has two distinct trainability regimes:

- `interaction_aligned`: frozen CBraMod plus native adapter and classifier;
- `native_full_finetune`: trainable CBraMod plus the same native adapter and
  classifier.

The second regime is a complement to, not a replacement for, dense
fine-tuning. It tests whether the native residual adds value when the backbone
is allowed to move. The two regimes must not be pooled into one headline.

This repository is the dedicated CBraMod TMLR clone. No LaBraM import,
EEGxPlore depth/router code, or ICASSP experiment is part of this implementation.
