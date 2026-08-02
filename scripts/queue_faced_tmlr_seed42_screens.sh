#!/bin/bash
# Queue the first paper-grade CBraMod FACED screen packet.
# Each job uses one GPU.  Two independent afterok chains cap concurrency at 2.
set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod"
CONFIG="$REPO_DIR/configs/faced_tmlr_locked.yaml"
SEED=42
ADAPTER_SEED=30042

submit_next() {
  local previous="$1" method="$2" adapter="$3" run_id="$4"
  local dependency=()
  if [[ -n "$previous" ]]; then
    dependency=(--dependency="afterok:${previous}")
  fi
  local export_value="ALL,CONFIG=${CONFIG},SEED=${SEED},ADAPTER_SEED=${ADAPTER_SEED},METHOD=${method},RUN_ID=${run_id}"
  if [[ -n "$adapter" ]]; then
    export_value+=",ADAPTER_TYPE=${adapter}"
  fi
  sbatch "${dependency[@]}" --export="$export_value" \
    "$REPO_DIR/scripts/submit_faced_tmlr.slurm" | awk '{print $NF}'
}

lane_a_prev=""
lane_a_prev=$(submit_next "$lane_a_prev" frozen_probe "" faced_frozen_probe_s42_lr1e-4_b64_e50)
lane_a_prev=$(submit_next "$lane_a_prev" interaction_aligned channel faced_native_channel_s42_lr1e-4_b64_e50)
lane_a_prev=$(submit_next "$lane_a_prev" interaction_aligned patch faced_native_patch_s42_lr1e-4_b64_e50)
lane_a_prev=$(submit_next "$lane_a_prev" interaction_aligned channel_patch faced_native_channel_patch_s42_lr1e-4_b64_e50)

lane_b_prev=""
lane_b_prev=$(submit_next "$lane_b_prev" generic_bottleneck "" faced_generic_bottleneck_s42_lr1e-4_b64_e50)
lane_b_prev=$(submit_next "$lane_b_prev" lora "" faced_lora_qkv_r8_s42_lr1e-4_b64_e50)
lane_b_prev=$(submit_next "$lane_b_prev" upper_k_finetune "" faced_upper2_s42_lr1e-4_b64_e50)
lane_b_prev=$(submit_next "$lane_b_prev" axis_blind "" faced_axis_blind_s42_lr1e-4_b64_e50)

echo "lane_a_tail=${lane_a_prev}"
echo "lane_b_tail=${lane_b_prev}"
