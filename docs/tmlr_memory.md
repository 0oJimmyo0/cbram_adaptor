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

The corrected frozen queue is technically clean so far. Of 45 jobs, 35 have
completed with valid `training_mode_report.json`, strict checkpoint reports,
complete epoch/test artifacts, and no integrity failures:

- FACED: 22/24 complete; job `12969546` (upper-2, seed 3407) is running and
  `12969547` (axis-blind, seed 3407) is pending.
- SEED-V: 13/21 complete; job `12969516` (axis-blind, seed 1024) is running
  and jobs `12969517–12969523` (all seed-3407 frozen conditions) are pending.

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
