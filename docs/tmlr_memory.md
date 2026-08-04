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
