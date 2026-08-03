#!/bin/bash
# Queue the remaining frozen native FACED multiseed packet.
# Two independent per-seed chains: at most two GPUs are used by this packet.
set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod"
CONFIG="$REPO_DIR/configs/faced_tmlr_locked.yaml"
ADAPTER_LR=5e-4

submit_next() {
  local previous="$1" adapter="$2" seed="$3" run_id="$4"
  local dependency=()
  if [[ -n "$previous" ]]; then
    dependency=(--dependency="afterok:${previous}")
  fi
  local adapter_seed=$((30000 + seed))
  sbatch "${dependency[@]}" \
    --export="ALL,CONFIG=${CONFIG},SEED=${seed},ADAPTER_SEED=${adapter_seed},ADAPTER_LR=${ADAPTER_LR},METHOD=interaction_aligned,ADAPTER_TYPE=${adapter},RUN_ID=${run_id}" \
    "$REPO_DIR/scripts/submit_faced_tmlr.slurm" | awk '{print $NF}'
}

queue_seed_chain() {
  local seed="$1"
  local previous=""
  previous=$(submit_next "$previous" channel "$seed" "faced_native_channel_s${seed}_adapterlr5e-4_b64_e50")
  previous=$(submit_next "$previous" patch "$seed" "faced_native_patch_s${seed}_adapterlr5e-4_b64_e50")
  previous=$(submit_next "$previous" channel_patch "$seed" "faced_native_channel_patch_s${seed}_adapterlr5e-4_b64_e50")
  echo "native_frozen_multiseed_s${seed}_tail=${previous}"
}

# Each seed starts independently, so two GPUs can execute the two heads while
# preserving channel -> patch -> channel_patch order within each seed.
queue_seed_chain 1024
queue_seed_chain 3407
