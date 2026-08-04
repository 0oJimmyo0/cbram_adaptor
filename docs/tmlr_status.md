# CBraMod TMLR execution status

This is the CBraMod-side status table for the TMLR paper. The LaBraM
experiments remain in the separate LaBraM repository. The two repositories
share the scientific protocol and three-seed rule, but never share backbone
implementation or data-loader code.

## Dataset status

| Dataset | Backbone | Data/provenance gate | Baseline | Native adapter packet | Controls | Status |
| --- | --- | --- | --- | --- | --- | --- |
| FACED | CBraMod | Complete; LMDB, channel order, `/100`, split overlap and strict checkpoint audited | Dense full fine-tuning, 3 seeds | Full-backbone packet remains valid; corrected frozen packet 24/24 complete | Corrected frozen controls 24/24 complete | **Corrected packet complete; final artifact audit remains** |
| SEED-V | CBraMod | Audit complete; `(62,1,200)`, `/100`, 62-channel manifest, split hashes and no overlap verified | Dense full fine-tuning, 3 seeds remains valid | Corrected frozen channel/dense/patch packet 14/21 complete | Corrected frozen controls 14/21 complete | **Corrected packet in progress; 7 jobs remain (1 running, 6 pending)** |

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

The original frozen production packet is not manuscript-eligible because the
pre-correction loop enabled dropout in the parameter-frozen backbone. Corrected
jobs write `training_mode_report.json` and are queued in serial lanes with no
more than two jobs concurrently. The corrected packet heads are SEED-V
`12969503` and FACED `12969524`; tails are `12969523` and `12969547`.

## SEED-V interpretation boundary

The channel axis has sequence length `62` and supports genuine native channel
interaction. The temporal patch axis has sequence length `1`, so it cannot
support patch-to-patch temporal interaction. Any patch result is therefore a
prespecified singleton-axis capacity control, not evidence for temporal
interaction. The primary CBraMod SEED-V claim is frozen dense versus frozen
channel adapter under the same classifier, split, scaling, checkpoint, seed,
selection, and test protocol used for the matched LaBraM study.

The corrected contract also requires the frozen CBraMod base modules to remain
in `.eval()` mode while the classifier and any trainable adapter remain in
`.train()` mode.

## Frozen-mode correction (2026-08-04)

The earlier CBraMod frozen runs on FACED and SEED-V called `model.train()` at
the start of each epoch. CBraMod has dropout in its spectral projection and
encoder blocks, so this made the supposedly frozen representation stochastic.
The parameter `requires_grad=False` reports were correct, but the module-mode
contract was not. Those frozen results are retained as diagnostics only and
must not be used as final manuscript evidence.

`tmlr.trainability.configure_training_modes` now enforces the intended modes:
the frozen base is in evaluation mode, while the classifier and trainable
native adapter, generic adapter, LoRA factors, or selected upper layers are in
training mode. Every corrected run records this in
`training_mode_report.json`. Dense full-finetuning and native full-backbone
runs are unaffected because their backbones should remain in training mode.

## Production checklist

1. Complete corrected seed-42 frozen-dense and frozen-channel gates with strict
   checkpoint, trainability, and training-mode reports.
2. Run corrected frozen dense and frozen channel for seeds `42,1024,3407`.
3. Run the corrected prespecified frozen patch singleton control for the same three
   seeds, clearly labeled as a capacity control.
4. Run native full-backbone channel for the same three seeds as the secondary
   trainable-backbone regime.
5. Run the independent frozen controls (generic bottleneck, LoRA QKV-r8,
   upper-2, axis-blind) for the same three seeds.
6. Confirm every successful run has per-epoch validation metrics, selected
   test BA/kappa/F1, strict checkpoint report, trainability report, geometry,
   adapter/update diagnostics, and no test-based selection.

No corrected SEED-V multiseed result is interpreted until the seed-42 mode gate
is technically clean. Exactly three seeds are used; five-seed packets are not
part of this study. Historical pre-correction frozen artifacts must not be
substituted for the corrected runs when the datasets are revisited.

## Dataset-transition gate

Before moving to another dataset, the current dataset must satisfy all of the
following:

1. Every frozen-base method uses the shared mode controller and writes
   `training_mode_report.json` with `frozen_backbone_eval_mode=true`, base
   backbone training mode false, and trainable head/adapter mode true.
2. Every required condition has exactly seeds `42,1024,3407`; no historical
   pre-correction artifact substitutes for a corrected result.
3. Every completed artifact has strict checkpoint loading, correct geometry,
   trainability/optimizer reports, all expected epoch records, validation
   selection, test metrics, and adapter/update diagnostics.
4. `sacct` shows no failed or `DependencyNeverSatisfied` production job, and
   no condition is silently missing from the checklist.
5. Only after this audit is the dataset marked complete and the next dataset
   launched.

## Adapter audit gate before final cross-backbone aggregation

The current corrected packet is an operational `r=64` CBraMod bottleneck
packet. Before final TMLR aggregation, run and archive seed-42 gates for:

- CBraMod `r=64` versus `r=32` native branches;
- an explicit shared initialization policy, mirrored independently in the
  LaBraM repository;
- schema-v2 branch/full-grid residual and Q/K/V diagnostics.

These are refinement/fidelity gates, not reasons to reinterpret the current
completed three-seed packet as a different architecture. The final paper must
also describe the CBraMod halves as architecturally defined native branches,
not pure semantic feature halves.

The final corrected seed-3407 controls were split into jobs `12982900` and
`12982901`; `12982900` is now dependency-free and `12982901` remains after
the currently running `12969521`. The seed-42 `r=32`
gate is then queued as `12982902–12982904`, after both controls succeed. It
contains FACED channel, FACED channel+patch, and SEED-V channel; it is
intentionally single-seed and must be interpreted against the existing
corrected `r=64` logs before any multiseed decision.
