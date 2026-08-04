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
