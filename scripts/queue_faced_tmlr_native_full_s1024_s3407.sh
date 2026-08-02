#!/bin/bash
# Queue the remaining native full-backbone-plus-adapter seeds.
# One sequential chain: at most one GPU is used by this packet.
set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod"
CONFIG="$REPO_DIR/configs/faced_tmlr_locked.yaml"

submit_next() {
  local previous="$1" adapter="$2" seed="$3" run_id="$4"
  local dependency=()
  if [[ -n "$previous" ]]; then
    dependency=(--dependency="afterok:${previous}")
  fi
  local adapter_seed=$((30000 + seed))
  sbatch "${dependency[@]}" \
    --export="ALL,CONFIG=${CONFIG},SEED=${seed},ADAPTER_SEED=${adapter_seed},METHOD=native_full_finetune,ADAPTER_TYPE=${adapter},RUN_ID=${run_id}" \
    "$REPO_DIR/scripts/submit_faced_tmlr.slurm" | awk '{print $NF}'
}

previous=""
for seed in 1024 3407; do
  previous=$(submit_next "$previous" channel "$seed" "faced_native_full_channel_s${seed}_lr1e-4_b64_e50")
  previous=$(submit_next "$previous" patch "$seed" "faced_native_full_patch_s${seed}_lr1e-4_b64_e50")
  previous=$(submit_next "$previous" channel_patch "$seed" "faced_native_full_channel_patch_s${seed}_lr1e-4_b64_e50")
done
echo "native_full_multiseed_tail=${previous}"
