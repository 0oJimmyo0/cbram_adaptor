# CBraMod TMLR SEED-V implementation audit

The SEED-V runner is isolated to this CBraMod repository. It does not import
LaBraM or EEGxPlore data/model code.

## Data and provenance gate

The audit artifact is `results/seedv/seedv_audit_gate_20260802/`. It records:

- shape `(62,1,200)`, input divisor `100`, labels `0..4`;
- split counts train/validation/test `34,432 / 42,960 / 40,352`;
- no split-key overlap;
- split-key hashes, manifest SHA-256, and sampled finite/schema checks;
- the ordered 62-channel manifest and source metadata hashes.

The legacy LMDB does not embed channel names in each tensor row. The external
channel-order evidence and this limitation are recorded in
`configs/seedv_channel_manifest.json`; the loader never invents a reorder.

## Model and method gate

`tmlr/seedv_runner.py` strictly loads the original CBraMod checkpoint before
replacing its downstream projection and attaching any TMLR module. The
runtime geometry gate verifies `[B,62,1,200]`, and every run writes the strict
checkpoint report, trainability contract, optimizer groups, structure
diagnostics, per-epoch metrics, selected test metrics, and update norms.

The shared CBraMod adapter uses the backbone-native split of the embedding:
the first half is mixed over channels and the second half over temporal
patches. On SEED-V, channel attention is valid (`C=62`). A patch branch is
allowed only through the explicit `allow_singleton_patch_control` flag and is
reported as a singleton-axis capacity control (`S=1`); it is not a temporal
patch-interaction claim.

The local eager AdamW implementation is numerically equivalent to the locked
AdamW defaults (`betas=(0.9,0.999)`, `eps=1e-8`, decoupled weight decay) and
the same per-iteration cosine equation. It exists only because the cluster
PyTorch 2.12 optimizer wrapper blocks in `torch._dynamo` before training; the
runtime path and reason are written to `optimizer_groups.json`.

## Comparison boundary

The primary SEED-V comparison is frozen CBraMod plus the shared classifier
versus frozen CBraMod plus the native channel adapter. Frozen patch is a
singleton capacity control. Native full-backbone channel tests the secondary
regime in which the backbone may move. Generic bottleneck, LoRA QKV-r8,
upper-2, and axis-blind are independent controls, never combinations with
the native adapter.

The production status and exact remaining checklist are maintained in
`docs/tmlr_status.md`.

## Frozen-backbone correction

The first SEED-V frozen packet is invalid for the intended deterministic
frozen-backbone claim. The training loop called `model.train()` each epoch,
which enabled dropout in CBraMod's frozen spectral projection and encoder.
The artifact trainability reports correctly show zero base-backbone updates,
but they do not make a module with active dropout a deterministic feature
extractor. A direct repeated-forward check produced different frozen features
in train mode and identical features in eval mode.

The runner now uses `configure_training_modes`: the frozen base stays in
`.eval()`, while the classifier and trainable native adapter remain in
`.train()`. Corrected artifacts include `training_mode_report.json`. The
previous frozen dense/channel/patch/control results, including the exploratory
low-adapter-LR channel run, are retained only as failure diagnostics. They are
not final SEED-V manuscript results. The corrected seed-42 gate must complete
before the three-seed frozen packet is promoted.

The two one-GPU smoke gates completed successfully (`12948289` frozen dense
and `12948298` frozen channel). The 27-run production matrix is now queued in
two serial `afterok` lanes (`12948315` and `12948330`); no production metric is
claimed until those artifacts complete and pass the same integrity checks.
