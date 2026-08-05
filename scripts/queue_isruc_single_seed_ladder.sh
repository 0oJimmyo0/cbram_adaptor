#!/bin/bash
# Queue the remaining ISRUC seed-42 conditions in two afterok lanes.
# Each job requests one A6000; at most two jobs can run concurrently.
set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod"
SCRIPT="$REPO_DIR/scripts/submit_isruc_tmlr.slurm"
CONFIG="$REPO_DIR/configs/isruc_tmlr_locked.yaml"
COMMON="CONFIG=$CONFIG,SEED=42,LR=2e-4,HEAD_LR=3.536e-4,ADAPTER_LR=2e-5,LORA_LR=2e-5,UPPER_LR=2e-4,BATCH_SIZE=8,EPOCHS=20,ADAPTER_SEED=30042"

submit() {
  local previous="$1"
  local method="$2"
  local adapter="$3"
  local run_id="$4"
  local export_value="ALL,$COMMON,METHOD=$method,RUN_ID=$run_id"
  if [[ -n "$adapter" ]]; then
    export_value="$export_value,ADAPTER_TYPE=$adapter"
  else
    export_value="$export_value,ADAPTER_TYPE="
  fi
  local args=(--parsable --export="$export_value")
  if [[ -n "$previous" ]]; then
    args+=(--dependency="afterok:$previous")
  fi
  local job
  job=$(sbatch "${args[@]}" "$SCRIPT")
  echo "$job  $method ${adapter:-none}  $run_id" >&2
  printf '%s' "$job"
}

# Lane A: frozen native axes and generic/LoRA controls.
a=$(submit "" frozen_probe "" isruc_cbramod_frozen_probe_s42_lr2e-4_b8_e20)
a=$(submit "$a" interaction_aligned channel isruc_cbramod_frozen_channel_s42_lr2e-4_b8_e20)
a=$(submit "$a" interaction_aligned patch isruc_cbramod_frozen_patch_s42_lr2e-4_b8_e20)
a=$(submit "$a" interaction_aligned channel_patch isruc_cbramod_frozen_channel_patch_s42_lr2e-4_b8_e20)
a=$(submit "$a" generic_bottleneck "" isruc_cbramod_frozen_generic_s42_lr2e-4_b8_e20)
a=$(submit "$a" lora "" isruc_cbramod_lora_qkv_r8_s42_lr2e-4_b8_e20)

# Lane B: upper/axis-blind controls and trainable-backbone native variants.
b=$(submit "" upper_k_finetune "" isruc_cbramod_upper2_s42_lr2e-4_b8_e20)
b=$(submit "$b" axis_blind "" isruc_cbramod_axisblind_s42_lr2e-4_b8_e20)
b=$(submit "$b" native_full_finetune channel isruc_cbramod_native_full_channel_s42_lr2e-4_b8_e20)
b=$(submit "$b" native_full_finetune patch isruc_cbramod_native_full_patch_s42_lr2e-4_b8_e20)
b=$(submit "$b" native_full_finetune channel_patch isruc_cbramod_native_full_channel_patch_s42_lr2e-4_b8_e20)

echo "Lane A tail: $a" >&2
echo "Lane B tail: $b" >&2
