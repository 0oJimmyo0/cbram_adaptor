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

- `h[..., :D/2]`: spatial/channel branch of the criss-cross attention, mixed
  over `C` for each temporal patch;
- `h[..., D/2:]`: temporal/patch branch of the criss-cross attention, mixed
  over `S` for each channel.

These are architecturally defined feature partitions, not claims that the
final halves are semantically pure. CBraMod's full-width feed-forward blocks
can mix information across the two halves after the native attention. The
paper should therefore use “native spatial/channel and temporal/patch
subspaces” or “architecturally defined branches,” not “pure spatial features”
or “pure temporal features.”

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

For SEED-V, the realized geometry is `[B,62,1,200]`. The channel branch is
therefore a genuine native-axis adapter over 62 electrodes. The patch branch
is retained only as an explicitly enabled singleton-axis capacity control;
with one temporal patch it cannot perform temporal patch-to-patch interaction.
This geometry distinction is part of the result interpretation, not a
post-hoc exclusion.

The Up projections are zero-initialized by default. This gives exact dense
parity at initialization. The first backward pass consequently gives the Up
projection a gradient while the Down, Q/K/V mixer, and alpha gradients are
zero (up to numerical tolerance); after the first optimizer step, the core
receives gradients. The two-step behavior is tested explicitly and must not be
described as “preserving first-step gradients into the core.” The branch alpha
parameters and residual/update diagnostics are recorded for every run. A
branch is rejected as inapplicable when its runtime sequence length is one; no
degenerate axis is silently reported as a positive result.

## Initialization and diagnostic contracts

The current operational packet uses the following policy: LayerNorm uses its
standard affine identity/zero initialization, Down and the attention mixer use
PyTorch's standard Linear/MultiheadAttention initialization, Up is
zero-initialized when `adapter_zero_init_output=true`, and alpha is initialized
to the configured scalar. This policy is behaviorally frozen for the corrected
2026-08-04 packet. Before final cross-backbone aggregation, CBraMod and LaBraM
need a seed-42 initialization gate with a shared named policy covering the
exact Down, Q/K/V, output-projection, Up, norm, and alpha rules. The two
repositories remain separate and are not wired together.

The current CBraMod logs retain historical diagnostic names. Their definitions
are: `raw_channel_ratio` and `raw_patch_ratio` divide the raw branch correction
by the corresponding branch input norm, while `adapter_delta_ratio` divides
the scaled total correction by the full feature-grid norm. Final reporting
should additionally expose the unambiguous names
`raw_channel_to_channel_input_ratio`, `raw_patch_to_patch_input_ratio`,
`scaled_channel_to_full_backbone_ratio`,
`scaled_patch_to_full_backbone_ratio`, and
`total_delta_to_full_backbone_ratio`, with one documented treatment of Q/K/V
weights and biases across backbones. Old and new names must not be pooled
without recording the schema version.

## Bottleneck-ratio gate

The default CBraMod native branch has width `D/2=100` and currently uses
`r=64`, so `r/(D/2)=0.64`. That is a valid residual adapter, but it is not a
strong low-rank claim. The current corrected packet must finish unchanged so
its results remain internally comparable. The next controlled seed-42 gate
should compare operational `r=64` against `r=32` (divisible by four heads),
while LaBraM's corresponding gate should use `r=64` for `D=200`. Only a
predeclared ratio rule that improves or preserves the relevant validation/test
behavior should replace the operational setting. Until that gate is complete,
describe the module as a bottlenecked native residual adapter, not as
universally low-rank.

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

For every frozen-base regime, the training-mode contract is distinct from the
parameter contract: the base CBraMod modules remain in `.eval()` mode so their
dropout is disabled, while the classifier and trainable adapter remain in
`.train()` mode. This is recorded in `training_mode_report.json` and is
required before interpreting frozen-backbone results.

The second regime is a complement to, not a replacement for, dense
fine-tuning. It tests whether the native residual adds value when the backbone
is allowed to move. The two regimes must not be pooled into one headline.

This repository is the dedicated CBraMod TMLR clone. No LaBraM import,
EEGxPlore depth/router code, or ICASSP experiment is part of this implementation.
