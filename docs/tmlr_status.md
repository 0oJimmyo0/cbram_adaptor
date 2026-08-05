# CBraMod TMLR execution status

This is the CBraMod-side status table for the TMLR paper. The LaBraM
experiments remain in the separate LaBraM repository. The two repositories
share the scientific protocol and three-seed rule, but never share backbone
implementation or data-loader code.

## Dataset status

| Dataset | Backbone | Data/provenance gate | Baseline | Native adapter packet | Controls | Status |
| --- | --- | --- | --- | --- | --- | --- |
| FACED | CBraMod | Complete; LMDB, channel order, `/100`, split overlap and strict checkpoint audited | Dense full fine-tuning, 3 seeds | Corrected frozen packet 24/24 complete; r=32 seed-42 gate complete | Corrected frozen controls 24/24 complete | **Packet complete; r=64 retained as operational setting** |
| SEED-V | CBraMod | Audit complete; `(62,1,200)`, `/100`, 62-channel manifest, split hashes and no overlap verified | Dense full fine-tuning, 3 seeds | Corrected frozen packet 21/21 complete; r=32 diagnostic gate complete | Corrected frozen controls 21/21 complete | **Complete; r=64 locked** |
| ISRUC | CBraMod | **Serialized audit passed; 3,559/468/435 sequences, no overlap, `(20,6,6000)` stored -> `(20,6,30,200)` model geometry, divisor `1.0`** | Seed-42 dense + LR gate complete; operational LR `2e-4 / 3.536e-4` | Seed-42 remaining ladder not yet submitted | Seed-42 remaining controls not yet submitted | **Single-seed adaptor/control ladder ready** |

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

For SEED-V specifically, the corrected r=64 packet is complete. The r=32
single-seed gate is retained as a diagnostic because its adapter LR differed
from the r=64 channel packet; it is not promoted to a production setting and
will not receive multiseeds.

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

The final corrected seed-3407 controls completed as jobs `12982900` and
`12982901`. The seed-42 r=32 gate then completed as `12982902–12982904`, with
FACED channel, FACED channel+patch, and SEED-V channel. It was intentionally
single-seed and is interpreted against the existing corrected r=64 logs before
any multiseed decision.

## Transition decision: FACED and SEED-V complete

The corrected CBraMod FACED and SEED-V packets are complete for the current
TMLR study scope. Each has the required r=64 conditions with exactly the
project-wide three seeds `{42,1024,3407}`, complete epoch and test artifacts,
strict checkpoint/trainability/mode reports, and no failed or dead production
jobs. The r=32 experiments are diagnostic gates only. We therefore move to
ISRUC rather than extending the FACED or SEED-V tuning queue.

## ISRUC CBraMod entry gate

ISRUC must be implemented and audited inside this CBraMod repository. LaBraM
is a protocol and geometry reference only; no LaBraM backbone, loader, or
implementation is imported into CBraMod.

The reference contract is ISRUC-Sleep Subgroup I, subjects 1--100 with
subject-wise splits 1--80 train, 81--90 validation, and 91--100 test. Each
stored item contains 20 consecutive 30-second epochs, six bipolar EEG channels
in the order `F3-A2, C3-A2, O1-A2, F4-A1, C4-A1, O2-A1`, and 30 temporal
patches of 200 samples: `[20,6,30,200]`. Labels `{0,1,2,3,5}` map to
`{0,1,2,3,4}`. Existing serialized arrays are already filtered and segmented,
so they must not be filtered again. ISRUC uses input scale divisor `1.0`; the
`/100` rule for FACED/SEED-V must not be copied to ISRUC.

The existing `datasets/isruc_dataset.py` and `models/model_for_isruc.py` are
only an incomplete prototype. They currently apply `/100`, pair files using
directory iteration order, and are not registered in `tmlr/config.py` or the
production runner. They must not be used for paper-facing runs until repaired
and covered by the audit below.

ISRUC work order:

1. Audit serialized pairing, shapes, finite values, labels, subject split
   counts/no overlap, channel metadata/order, and the scale divisor.
2. Implement a CBraMod-only sequence loader/config preserving
   `[B,20,6,30,200]`; do not flatten the 20-epoch sequence into independent
   examples.
3. Encode each epoch with CBraMod, attach native adapters at its `[6,30,200]`
   grid, then use the sequence encoder/classifier. Both native axes are
   meaningful on ISRUC: channel length 6 and patch length 30.
4. Run data/shape/checkpoint smoke tests, then one seed-42 dense baseline.
   Lock the operational recipe from its full validation trajectory before
   comparing adapters.
5. Run seed-42 native channel, patch, channel+patch, and the common frozen,
   generic, LoRA, upper-layer, and axis-blind controls only after the dense
   gate is valid.
6. Promote only technically clean conditions to exactly `{42,1024,3407}` and
   archive per-epoch validation, selected test BA/kappa/weighted-F1,
   checkpoint, trainability, mode, geometry, and adapter/update diagnostics.

## ISRUC current gate status

The serialized-only audit passed and is archived at
`results/isruc/audits/isruc_serialized_audit.json`. It found 3,559 train,
468 validation, and 435 test sequences, exact same-stem signal/label pairing,
the expected labels `0..4`, no subject overlap, and no non-finite values. The
loader/model smoke test also passed with strict checkpoint loading and a
sequence-level `[B,20,6,30,200] -> [B,20,5]` forward/backward/test path.

The first full baseline is job `12992930`, seed 42, batch 8, 20 epochs. It
completed with test BA `0.7855`, κ `0.7397`, and weighted F1 `0.8003`, with
validation-κ selection at epoch 5. The LR sweep jobs `13001412` and
`13001413` also completed cleanly. The upper pair (`lr=2e-4`,
`head_lr=3.536e-4`) is the operational choice by validation κ (`0.7381`),
and has test BA `0.7884`, κ `0.7503`, weighted F1 `0.8078`. These test values
are reported descriptively; the recipe was selected by validation κ.

The remaining seed-42 ladder is exactly: frozen probe; frozen native channel,
patch, and channel+patch; frozen generic bottleneck; frozen LoRA QKV-r8;
frozen upper-2; frozen axis-blind; and native full-backbone channel, patch,
and channel+patch. All use the locked dense LR pair, batch 8, 20 epochs,
divisor 1.0, and the same checkpoint/split/selection protocol. Native/generic
adapter and LoRA LR is `2e-5` (0.1x dense LR), while upper-2 uses `2e-4`.
They are queued in two `afterok` lanes, at most two A6000 jobs concurrently.
No three-seed promotion is allowed until these single-seed trajectories and
mode/diagnostic reports are reviewed.
