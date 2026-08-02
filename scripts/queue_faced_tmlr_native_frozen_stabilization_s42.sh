#!/bin/bash
# One seed-42 stabilization packet for frozen native adapters.
# Frozen-native multiseeds remain gated until this packet is inspected.
set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod"
CONFIG="$REPO_DIR/configs/faced_tmlr_locked.yaml"
SEED=42
ADAPTER_SEED=30042
ADAPTER_LR=1e-4

submit_next() {
  local previous="$1" adapter="$2" run_id="$3"
  local dependency=()
  if [[ -n "$previous" ]]; then
    dependency=(--dependency="afterok:${previous}")
  fi
  sbatch "${dependency[@]}" \
    --export="ALL,CONFIG=${CONFIG},SEED=${SEED},ADAPTER_SEED=${ADAPTER_SEED},ADAPTER_LR=${ADAPTER_LR},METHOD=interaction_aligned,ADAPTER_TYPE=${adapter},RUN_ID=${run_id}" \
    "$REPO_DIR/scripts/submit_faced_tmlr.slurm" | awk '{print $NF}'
}

previous=""
previous=$(submit_next "$previous" channel faced_native_channel_s42_adapterlr1e-4_b64_e50)
previous=$(submit_next "$previous" patch faced_native_patch_s42_adapterlr1e-4_b64_e50)
previous=$(submit_next "$previous" channel_patch faced_native_channel_patch_s42_adapterlr1e-4_b64_e50)
echo "native_frozen_stabilization_tail=${previous}"
