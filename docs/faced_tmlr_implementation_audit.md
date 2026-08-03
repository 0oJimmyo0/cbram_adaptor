# CBraMod TMLR FACED implementation audit

This document records the isolated FACED implementation gate for the TMLR
study.  The CBraMod repository is the only codebase used by this pipeline;
the LaBraM repository and EEGxPlore training modules are not imported.

## Decision at the current gate

**FACED production packet complete.** The data, code, checkpoint, and
three-seed condition contracts pass. The checkpoint was
found at `/data/neurogroup/mingyangjiang/data/weights/pretrained_weights.pth`
and strict loading matched all 211 expected keys with no missing or unexpected
keys. Its recorded SHA-256 is
`0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`.

The audit-only artifact is:

`results/faced/faced_audit_gate_20260731/`

The operational config now points directly to that external checkpoint path;
the weights are not copied into the Git repository.

## Exact implementation entry points

| Concern | Entry point | Contract |
| --- | --- | --- |
| Foundation model | `models/cbramod.py:9-98` | Original CBraMod encoder and projection; no LaBraM dependency. |
| Existing FACED wrapper | `models/model_for_faced.py:8-77` | Upstream-style wrapper retained for reference; its strict base load is at lines 17-24. |
| Native adapter | `models/interaction_adapter.py:111-238` | Channel, patch, or channel+patch residual branches. |
| Adapter attachment | `models/cbramod.py:30-81,90-98` | One post-encoder insertion at line 94, before `proj_out`; never inside a transformer block. |
| FACED loader | `datasets/faced_dataset.py:13-73` | LMDB split loading, 32-channel validation, `(32,10,200)` validation, finite-value check, `/100` scaling. |
| Provenance audit | `tmlr/provenance.py:126-256` | Full LMDB scan, manifest/channel-order check, hashes, shape/dtype/scaling and overlap checks. |
| Dedicated runner | `run_faced_tmlr.py:1-10` and `tmlr/faced_runner.py:274-446` | Resolved configuration, strict load, geometry gate, training, validation selection, test evaluation and artifacts. |
| Classifier head | `tmlr/faced_runner.py:53-124` | Source-style `all_patch_reps` head: flatten `[B,32,10,200]`, then `Linear(64000,200) -> Linear(200,200) -> Linear(200,9)` with ELU/dropout; identical across methods. |
| Metrics | `tmlr/metrics.py:26-80` | Balanced accuracy and macro-F1 primary; kappa, weighted-F1, accuracy, loss, per-class metrics and confusion matrix secondary. |
| Trainability registry | `tmlr/trainability.py:26-157` | Explicit full-finetune, frozen-probe and interaction-aligned parameter groups/assertions. |
| Artifacts | `tmlr/artifact_writer.py:14-39` | Immutable machine-readable run directory writer. |

## Adapter structure

CBraMod emits `[B,C,S,D]`, with `D=200`. The implemented native adapter
partitions the representation into the first `D/2=100` channel/spatial
features and the second `D/2=100` temporal/patch features. Each enabled branch
uses layer normalization, a `100 -> bottleneck` projection, multi-head
self-attention along its native axis, an up-projection, and a scalar residual
gate. The channel branch attends over `C=32`; the patch branch attends over
`S=10`. `channel_patch` is the sum of two independent residual branches.

The residual is attached once after the encoder, so the backbone remains the
original CBraMod backbone plus a clearly isolated adaptation module. The
default output projection is replaced only after strict foundation loading,
because the downstream FACED head consumes the encoder grid.

Supported methods in this gate are:

- `full_finetune`: original CBraMod backbone plus the shared classifier head;
- `frozen_probe`: frozen CBraMod plus the shared classifier head;
- `interaction_aligned`: frozen CBraMod plus `channel`, `patch`, or
  `channel_patch` adapter and the shared classifier head;
- `native_full_finetune`: trainable CBraMod plus the same native adapter and
  shared classifier head.

`upper_k_finetune`, `lora`, `generic_bottleneck`, and `axis_blind` are
independent comparison methods. They are never silently substituted for a
native adapter. The native adapter is not combined with LoRA or generic
controls.

The locked FACED classifier is **not** a mean-pooled linear probe. It uses the
source-style `all_patch_reps` head so that the dense baseline and every adapter
condition share the same downstream capacity. This is why the classifier
accounts for most trainable parameters in frozen-backbone runs; the CBraMod
backbone itself remains frozen under `interaction_aligned`.

## FACED provenance result

The audit scanned the local LMDB at
`/data/neurogroup/mingyangjiang/data/FACED` and found:

- train/validation/test: **6,720 / 1,680 / 1,932** samples;
- nine labels, `0..8`, with the expected class-count pattern;
- every sample shape `(32,10,200)` and stored dtype `float64`, converted to
  `float32` after division by 100;
- the canonical 32-channel order from
  `configs/faced_channel_manifest.json`;
- no duplicate keys, split-key overlap, subject overlap, or session/trial
  overlap;
- all scanned values finite.

The dataset audit also records the split-manifest SHA-256 and channel-manifest
SHA-256. The exact values are in `dataset_audit.json`, rather than copied into
this narrative so the artifact remains the source of truth.

## Construction smoke and production evidence

The checkpoint-dependent seed-42 smoke sequence completed successfully on the
local CPU environment for all required method/branch constructions:

- dense `full_finetune`;
- `frozen_probe`;
- frozen zero-initialized `interaction_aligned` with `channel`, `patch`, and
  `channel_patch` branches.
- trainable-backbone zero-initialized `native_full_finetune` construction for
  `channel`, `patch`, and `channel_patch`.

The resulting run directories are under `results/faced/`. The reports confirm
the expected trainable counts: 1,809 for the head-only probe, 31,614 for each
single branch plus head, and 61,419 for both branches plus head. The channel
and patch sequences were recorded as 32 and 10 respectively. With zero-init,
the first-step Q/K/V gradients were zero while the up-projection gradients
were nonzero, matching the intended gate behavior.

These construction checks are historical smoke evidence, not manuscript
performance numbers. The production runner subsequently used the locked
50-epoch schedule, full train/validation/test splits, validation-kappa
selection, strict checkpoint loading, and the same `all_patch_reps` head for
every condition.

The dense baseline and independent controls (generic bottleneck, LoRA QKV-r8,
upper-2, axis-blind, and frozen probe) have complete three-seed packets. The
native full-backbone channel, patch, and channel+patch conditions also have
complete three-seed packets. Their artifact directories contain per-epoch
metrics, strict-load reports, trainability contracts, adapter diagnostics, and
test metrics.

## Production checklist status

The following production conditions use the same resolved dataset manifest,
scaling, classifier, split, batch size, and 50-epoch budget:

- `full_finetune` dense: complete for seeds `42,1024,3407`;
- `frozen_probe`: complete for seeds `42,1024,3407`;
- generic bottleneck, LoRA QKV-r8, upper-2, and axis-blind controls: complete
  for seeds `42,1024,3407`;
- `native_full_finetune` channel, patch, and channel+patch: complete for
  seeds `42,1024,3407`;
- `interaction_aligned` channel, patch, and channel+patch: complete for seeds
  `42,1024,3407` using the promoted `adapter_lr=5e-4` contract. The exploratory
  seed-42 `adapter_lr=1e-4` packet is retained separately.

The final frozen-native test BA means are channel `0.2730`, patch `0.3141`,
and channel+patch `0.3303`. These are conditional frozen-backbone results,
not replacements for dense fine-tuning.

The runner must produce strict checkpoint-load evidence (path, SHA-256,
missing/unexpected keys and parameter counts), runtime geometry, trainability
groups, per-epoch metrics, adapter gradients/update norms, validation-selected
test metrics, per-class metrics, confusion matrix, timing, memory and a best
model. Test metrics are retained for reporting but are never used to choose
the recipe or checkpoint. No production FACED run remains.

## Test evidence

The isolated unit suite currently passes **23 tests** with

`/tmp/cbramod-tmlr-test-venv/bin/python -m pytest -q`

The tests cover adapter geometry/branch eligibility, zero-init behavior and
gradients, configuration/reserved-method rejection, trainability contracts,
metrics shape safety, data-contract failures, artifact immutability and
parity-related controls. The remaining unexecuted evidence is necessarily the
checkpoint-dependent model smoke and training/test evaluation.

## Known limitations at this gate

- The frozen-native seed-42 condition is weak under the lower exploratory
  `adapter_lr=1e-4` setting. The completed multiseed packet uses the stronger
  tested `adapter_lr=5e-4` setting and is interpreted separately from the
  native full-backbone regime.
- Native full-backbone adapters are a lightweight residual augmentation, not a
  replacement for dense full fine-tuning; current FACED results are close to
  but do not consistently exceed dense performance.
- The TMLR protocol uses exactly three seeds: `42,1024,3407`.
- This pipeline does not wire CBraMod to any LaBraM data/model code.
