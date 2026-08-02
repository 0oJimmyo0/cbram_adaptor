#!/bin/bash
# Queue seed-42 native full-backbone-plus-adapter screens.
# One sequential chain: at most one GPU is used by this packet.
set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod"
CONFIG="$REPO_DIR/configs/faced_tmlr_locked.yaml"
SEED=42
ADAPTER_SEED=30042

submit_next() {
  local previous="$1" adapter="$2" run_id="$3"
  local dependency=()
  if [[ -n "$previous" ]]; then
    dependency=(--dependency="afterok:${previous}")
  fi
  sbatch "${dependency[@]}" \
    --export="ALL,CONFIG=${CONFIG},SEED=${SEED},ADAPTER_SEED=${ADAPTER_SEED},METHOD=native_full_finetune,ADAPTER_TYPE=${adapter},RUN_ID=${run_id}" \
    "$REPO_DIR/scripts/submit_faced_tmlr.slurm" | awk '{print $NF}'
}

previous=""
previous=$(submit_next "$previous" channel faced_native_full_channel_s42_lr1e-4_b64_e50)
previous=$(submit_next "$previous" patch faced_native_full_patch_s42_lr1e-4_b64_e50)
previous=$(submit_next "$previous" channel_patch faced_native_full_channel_patch_s42_lr1e-4_b64_e50)
echo "native_full_screen_tail=${previous}"
