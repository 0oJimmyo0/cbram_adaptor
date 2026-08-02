#!/bin/bash
# Queue missing seeds for controls whose seed-42 screens have no clear issue.
# One sequential chain: at most one GPU is used by this packet.
set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod"
CONFIG="$REPO_DIR/configs/faced_tmlr_locked.yaml"

submit_next() {
  local previous="$1" method="$2" seed="$3" run_id="$4"
  local dependency=()
  if [[ -n "$previous" ]]; then
    dependency=(--dependency="afterok:${previous}")
  fi
  local adapter_seed=$((30000 + seed))
  sbatch "${dependency[@]}" \
    --export="ALL,CONFIG=${CONFIG},SEED=${seed},ADAPTER_SEED=${adapter_seed},METHOD=${method},RUN_ID=${run_id}" \
    "$REPO_DIR/scripts/submit_faced_tmlr.slurm" | awk '{print $NF}'
}

previous=""
for seed in 1024 3407; do
  previous=$(submit_next "$previous" frozen_probe "$seed" "faced_frozen_probe_s${seed}_lr1e-4_b64_e50")
  previous=$(submit_next "$previous" generic_bottleneck "$seed" "faced_generic_bottleneck_s${seed}_lr1e-4_b64_e50")
  previous=$(submit_next "$previous" lora "$seed" "faced_lora_qkv_r8_s${seed}_lr1e-4_b64_e50")
  previous=$(submit_next "$previous" upper_k_finetune "$seed" "faced_upper2_s${seed}_lr1e-4_b64_e50")
  previous=$(submit_next "$previous" axis_blind "$seed" "faced_axis_blind_s${seed}_lr1e-4_b64_e50")
done
echo "control_multiseed_tail=${previous}"
