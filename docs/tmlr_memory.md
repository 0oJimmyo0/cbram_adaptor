# CBraMod TMLR working memory

## 2026-08-04 frozen-backbone correction

The original CBraMod frozen runs on FACED and SEED-V must not be used as final
manuscript evidence. The runners called `model.train()` at the start of every
epoch, which enabled dropout in the parameter-frozen CBraMod backbone. The
backbone weights were frozen correctly, but the representation was stochastic.

The fix is `tmlr.trainability.configure_training_modes`:

- frozen base CBraMod modules: `.eval()`;
- classifier: `.train()`;
- native/generic adapter: `.train()`;
- LoRA factors: `.train()` with the base backbone in `.eval()`;
- upper-k control: only selected upper layers `.train()`, lower base layers
  `.eval()`;
- dense and native full-backbone methods: complete model `.train()`.

The corrected runner writes `training_mode_report.json`. Zero-init parity,
strict checkpoint loading, geometry, trainability, optimizer groups, and
channel gradients already passed the implementation audit. The issue was
module mode, not a disconnected adaptor.

The corrected frozen packet was submitted on 2026-08-04 as two serial dataset
lanes: SEED-V jobs `12969503–12969523` and FACED jobs `12969524–12969547`.

Historical frozen artifacts are diagnostic only. When revisiting FACED or
SEED-V, use only corrected eval-mode artifacts. The corrected workflow is:

1. seed-42 frozen dense/channel gate;
2. inspect mode report and full epoch trajectory;
3. run exactly three seeds `{42,1024,3407}` for the corrected frozen packet;
4. retain native full-backbone and dense results separately, since those were
   not affected by the frozen-mode defect.

SEED-V has meaningful channel interaction (`C=62`) but singleton patch axis
(`S=1`), so patch remains a capacity control rather than temporal interaction
evidence.

## 2026-08-05 corrected packet progress

The corrected frozen queue is technically clean so far. Of 45 jobs, 38 have
completed with valid `training_mode_report.json`, strict checkpoint reports,
complete epoch/test artifacts, and no integrity failures:

- FACED: 24/24 complete; accounting shows all jobs completed with exit code
  zero.
- SEED-V: 14/21 complete; job `12969517` is running and jobs
  `12969518–12969523` remain pending in the serial dependency lane.

Early corrected results already show the mode fix matters. On SEED-V,
frozen-dense test BA is about `0.299` for seeds 42 and 1024, versus roughly
`0.239` in the old stochastic-backbone packet. Corrected frozen-channel BA is
currently `0.288` and `0.291` with the conservative adapter LR `1e-5`, so it
does not yet exceed corrected frozen dense. FACED seed-42 corrected frozen
probe/channel/patch/channel+patch test BA are approximately
`0.375/0.384/0.383/0.395`; these are not final until all three seeds finish.

The dataset-transition rule is now mandatory: do not move to another dataset
until the current dataset has all required three-seed conditions, valid mode
reports, complete artifacts, and no failed/dead dependency jobs. When
revisiting FACED or SEED-V, use only `evalmode_20260804` (or later) artifacts.

## 2026-08-05 adapter-structure audit

The attached architecture review was checked against the live CBraMod code.
The native adapter remains logically valid, with these protocol distinctions:

- CBraMod's first and second `D/2` halves are the native criss-cross
  attention branches; they are not semantically pure after full-width
  feed-forward mixing. Manuscript wording must use “architecturally defined
  branches.”
- With `D=200`, each native branch has width 100 and the current `r=64`
  setting has ratio 0.64. It is not a strong low-rank setting. Do not alter
  the corrected frozen packet in flight; run a seed-42 `r=64` versus `r=32`
  bottleneck gate after it finishes, then define the final ratio contract.
- Zero-init Up gives exact dense parity. On backward step one, Up receives the
  learning signal while Down/Q/K/V/alpha are zero; after one optimizer update,
  the core becomes active. This is now tested and is the correct diagnostic
  interpretation.
- The current non-output initialization follows PyTorch defaults in CBraMod.
  A named, exact shared initialization policy must be gated before pooling
  CBraMod and LaBraM results; no repository mixing is permitted.
- Current CBraMod raw ratios use branch-input denominators, while total delta
  uses the full-grid denominator. Final tables need explicit schema-v2 names
  and the same Q/K/V weight-and-bias convention across both repositories.

Safe implementation fixes applied in this audit:

1. Native adapter attachment now preserves the model's existing device and
   dtype, including callers that attach after `.to(device)`.
2. Repeated native attachment now raises instead of silently replacing an
   existing adapter.
3. Tests now cover branch isolation, channel/patch permutation equivariance,
   opposite-half independence, strict checkpoint-before-attachment, and the
   two-step zero-init gradient contract.

These changes do not change the architecture or optimizer numerics of the
already submitted corrected packet. The bottleneck, initialization, and
diagnostic-schema gates are deliberately deferred until that packet is
complete so completed and pending jobs are not scientifically mixed.

The first bottleneck gate was queued without multiseeds on 2026-08-05. The
initial serial chain was cancelled after inspection showed it would leave
available GPUs idle. The corrected split queue is:

- `12982900`: corrected SEED-V upper-2, seed 3407, dependency released after
  inspection so it is eligible to use the second GPU immediately;
- `12982901`: corrected SEED-V axis-blind, seed 3407, after the currently
  running `12969521`;
- `12982902 -> 12982903`: FACED native channel then channel+patch, `r=32`,
  seed 42; `12982902` was released from dependency once the independent
  upper-2 control completed, so it can use the second GPU while
  `12982901` runs;
- `12982904`: SEED-V native channel, `r=32`, seed 42.

The two corrected controls were initially split for concurrency. The FACED
r=32 channel gate is also independent of the axis-blind result and was
released once `12982900` completed, so it can run alongside `12982901`. The
two lanes permit at most two simultaneous A6000 jobs.
These are comparison gates against the existing corrected `r=64` seed-42
artifacts, not multiseed production runs. Cancelled obsolete jobs were
`12969522`, `12969523`, and `12981365–12981367`.

## 2026-08-05 r=32 gate results

The corrected queue and r=32 gate are complete. All 22 relevant SEED-V
artifacts (21 corrected r=64 packet conditions plus one r=32 seed-42 gate) and
26 FACED artifacts (24 corrected r=64 packet conditions plus two r=32 seed-42
gates) contain complete epoch logs, strict checkpoint reports,
`training_mode_report.json`, and test metrics.

The FACED r=32 gate is a fair comparison because both r=32 and corrected r=64
native runs use adapter LR `5e-4`:

- channel: r=32 test BA `0.3733` versus r=64 `0.3837`;
- channel+patch: r=32 test BA `0.3875` versus r=64 `0.3952`.

Both r=32 variants are technically valid but less effective than r=64 on this
seed. The r=32 channel+patch variant still exceeds the frozen probe on seed
42, but it does not justify replacing the completed r=64 packet.

The SEED-V r=32 gate is technically valid but not a fair architecture-only
comparison: it uses adapter LR `1e-4`, whereas the corrected r=64 channel
packet uses `1e-5`. It reaches test BA `0.2915`, slightly above r=64 channel
`0.2882`, but remains below corrected frozen dense `0.2989`; its larger
residual ratio also reflects the tenfold LR difference. No r=32 multiseed
claim may be made from this result.

The operational bottleneck is now locked at r=64. The r=32 results are
diagnostic-only and no r=32 multiseed packet will be run. Both SEED-V and
FACED have completed all required r=64 three-seed conditions; remaining work
is cross-backbone aggregation, final artifact audit, and manuscript analysis.

## 2026-08-05 transition to ISRUC

The corrected CBraMod FACED and SEED-V packets are now complete. Both datasets
have the required r=64 conditions under exactly three seeds `{42,1024,3407}`;
all production artifacts have complete epoch/test records and strict
checkpoint, trainability, and frozen-mode reports; and the final accounting
audit found no failed or `DependencyNeverSatisfied` production jobs. The r=32
jobs remain diagnostic-only. Do not add more FACED or SEED-V jobs unless a
later artifact audit finds a concrete integrity issue.

The next target is ISRUC in this CBraMod repository. The LaBraM ISRUC work is
the reference for data provenance, channel order, sequence geometry, and model
placement only. The backbone and loader implementations must remain separate.

### ISRUC reference contract

- ISRUC-Sleep Subgroup I, subjects 1--100.
- Subject-wise split: subjects 1--80 train, 81--90 validation, 91--100 test.
- Six bipolar channels, ordered `F3-A2, C3-A2, O1-A2, F4-A1, C4-A1, O2-A1`.
- 200 Hz, 30-second epochs, 30 temporal patches of 200 samples.
- Twenty consecutive epochs per stored sequence: `[20,6,30,200]`.
- Raw labels `{0,1,2,3,5}` mapped to `{0,1,2,3,4}`.
- Serialized arrays are already filtered/segmented; no second filtering pass.
- ISRUC scale divisor is `1.0`, not the `/100` used by the current FACED and
  SEED-V CBraMod loaders.
- The project-wide confirmatory seeds remain `{42,1024,3407}`. An older
  LaBraM note listing another seed set does not override this project rule.

### ISRUC implementation warning and checklist

CBraMod already contains `datasets/isruc_dataset.py` and
`models/model_for_isruc.py`, but they are an incomplete prototype, not a
faithful production path: the loader applies `/100`, pairs files by
`os.listdir` order, and ISRUC is not registered in the TMLR config/runner.
Do not submit ISRUC model jobs until these issues are repaired and tested.

Required order:

1. Build a CBraMod-only serialized-data audit for numeric pairing, shape and
   finite checks, labels, subject splits/no overlap, channel metadata/order,
   sequence counts, and scale.
2. Implement/register the CBraMod ISRUC loader and configuration while keeping
   the 20-epoch sequence explicit. Encode each epoch as `[6,30,200]`, attach
   native adapters at that grid, and classify with the shared sequence encoder.
   Both ISRUC native axes are meaningful: channel length 6 and patch length 30.
3. Pass data/shape/checkpoint smoke tests, then run one seed-42 dense baseline.
   Lock the recipe from its full validation trajectory before comparing
   adapters.
4. Run seed-42 native channel/patch/channel+patch and the common frozen and
   parameter-matched controls only after the dense gate is valid.
5. Promote only technically clean conditions to `{42,1024,3407}` and archive
   per-epoch validation, selected test BA/kappa/weighted-F1, checkpoint,
   trainability, mode, geometry, and adapter/update diagnostics.

### ISRUC implementation progress

The serialized-only ISRUC audit passed on 2026-08-05. It found 3,559 train,
468 validation, and 435 test sequences, exact same-stem signal/label pairing,
stored arrays `[20,6,6000]`, model tensors `[20,6,30,200]`, labels `0..4`, no
subject overlap, finite values, and the locked scale divisor `1.0`. The
CBraMod-only loader now uses numeric filename ordering and explicit subject
splits. The sequence model encodes each of the 20 epochs through CBraMod and
classifies the sequence with a one-layer Transformer head, returning
`[B,20,5]` logits. The strict smoke path passed checkpoint loading,
geometry, one training step, validation, selected-test artifact generation,
and training reports.

The first full ISRUC dense baseline is job `12992930` (seed 42, batch 8,
20 epochs). No ISRUC adapter/control or multiseed job may be submitted before
its full trajectory and final metrics are reviewed.

The baseline completed successfully with strict checkpoint loading and all
artifacts. It selected epoch 5 by validation κ (`0.7376`) and reached test BA
`0.7855`, test κ `0.7397`, and weighted F1 `0.8003`. The complete trajectory
shows early validation improvement followed by oscillation while train loss
continues down, so the dense recipe was not locked immediately. A controlled
seed-42 LR sweep is pending as jobs `13001412` and `13001413`: the lower run
uses backbone/head LR `5e-5/8.84e-5`, the upper run uses `2e-4/3.536e-4`, and
both retain batch 8, 20 epochs, scale 1.0, the same loader/checkpoint/split,
and validation-κ selection. No adaptor or multiseed job is allowed until this
gate is resolved.
