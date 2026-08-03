# CBraMod TMLR execution status

This is the CBraMod-side status table for the TMLR paper. The LaBraM
experiments remain in the separate LaBraM repository. The two repositories
share the scientific protocol and three-seed rule, but never share backbone
implementation or data-loader code.

## Dataset status

| Dataset | Backbone | Data/provenance gate | Baseline | Native adapter packet | Controls | Status |
| --- | --- | --- | --- | --- | --- | --- |
| FACED | CBraMod | Complete; LMDB, channel order, `/100`, split overlap and strict checkpoint audited | Dense full fine-tuning, 3 seeds | Frozen and full-backbone channel, patch, channel+patch; 3 seeds | Frozen probe, generic bottleneck, LoRA QKV-r8, upper-2, axis-blind; 3 seeds | **Complete** |
| SEED-V | CBraMod | Audit complete; `(62,1,200)`, `/100`, 62-channel manifest, split hashes and no overlap verified | Production matrix queued | Frozen channel is primary; frozen patch is singleton-axis capacity control; full-backbone channel is secondary | Frozen generic, LoRA QKV-r8, upper-2, axis-blind | **Smoke gates passed; 27 production runs queued** |

## SEED-V locked contract

- Source LMDB: `/data/neurogroup/mingyangjiang/data/SEED-V_processed_lmdb`.
- Persisted geometry: `[B,62,1,200]`; labels are `0..4`; all input values are
  divided by `100` in the CBraMod-only loader.
- Split counts: train `34,432`, validation `42,960`, test `40,352`; split key
  hashes and sampled finite/schema checks are in the audit artifact.
- Channel order is recorded in `configs/seedv_channel_manifest.json`; the
  external source hashes and legacy row-order limitation are preserved there.
- Source-style classifier is `all_patch_reps`: flatten `62*1*200`, then
  `Linear(800) -> ELU/dropout -> Linear(200) -> ELU/dropout -> Linear(5)`.
- Default training contract is batch `32`, `40` epochs, validation-kappa
  selection, test evaluation after selection, and seeds `{42,1024,3407}`.

The one-GPU frozen-dense and frozen-channel smoke jobs completed with strict
checkpoint reports and test artifacts. The production packet is queued as two
serial `afterok` lanes, tails `12948315` and `12948330`, so no more than two
SEED-V jobs can run concurrently.

## SEED-V interpretation boundary

The channel axis has sequence length `62` and supports genuine native channel
interaction. The temporal patch axis has sequence length `1`, so it cannot
support patch-to-patch temporal interaction. Any patch result is therefore a
prespecified singleton-axis capacity control, not evidence for temporal
interaction. The primary CBraMod SEED-V claim is frozen dense versus frozen
channel adapter under the same classifier, split, scaling, checkpoint, seed,
selection, and test protocol used for the matched LaBraM study.

## Production checklist

1. Complete one-GPU frozen-dense and frozen-channel smoke checks with strict
   checkpoint and trainability reports.
2. Run frozen dense and frozen channel for seeds `42,1024,3407`.
3. Run the prespecified frozen patch singleton control for the same three
   seeds, clearly labeled as a capacity control.
4. Run native full-backbone channel for the same three seeds as the secondary
   trainable-backbone regime.
5. Run the independent frozen controls (generic bottleneck, LoRA QKV-r8,
   upper-2, axis-blind) for the same three seeds.
6. Confirm every successful run has per-epoch validation metrics, selected
   test BA/kappa/F1, strict checkpoint report, trainability report, geometry,
   adapter/update diagnostics, and no test-based selection.

No SEED-V multiseed result is interpreted until the single-seed production
gate is technically clean. Exactly three seeds are used; five-seed packets
are not part of this study.
