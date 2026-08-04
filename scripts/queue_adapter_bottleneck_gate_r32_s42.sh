#!/bin/bash
# Queue the single-seed r=32 native-adapter gate after the corrected frozen
# packet. Two serial lanes are used; at most two one-GPU jobs can run.
set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod"
FACED_CONFIG="$REPO_DIR/configs/faced_tmlr_locked.yaml"
SEEDV_CONFIG="$REPO_DIR/configs/seedv_tmlr_locked.yaml"
SEED=42
ADAPTER_SEED=30042
BOTTLENECK=32
PREVIOUS="${1:-12969523}"

submit_faced() {
  local previous="$1" adapter="$2" run_id="$3"
  local dependency=()
  if [[ -n "$previous" ]]; then dependency=(--dependency="afterok:${previous}"); fi
  sbatch "${dependency[@]}" \
    --export="ALL,CONFIG=${FACED_CONFIG},SEED=${SEED},ADAPTER_SEED=${ADAPTER_SEED},ADAPTER_BOTTLENECK=${BOTTLENECK},METHOD=interaction_aligned,ADAPTER_TYPE=${adapter},RUN_ID=${run_id}" \
    "$REPO_DIR/scripts/submit_faced_tmlr.slurm" | awk '{print $NF}'
}

submit_seedv() {
  local previous="$1" adapter="$2" run_id="$3"
  local dependency=()
  if [[ -n "$previous" ]]; then dependency=(--dependency="afterok:${previous}"); fi
  sbatch "${dependency[@]}" \
    --export="ALL,CONFIG=${SEEDV_CONFIG},SEED=${SEED},LOADER_SEED=20042,ADAPTER_SEED=${ADAPTER_SEED},ADAPTER_BOTTLENECK=${BOTTLENECK},METHOD=interaction_aligned,ADAPTER_TYPE=${adapter},RUN_ID=${run_id}" \
    "$REPO_DIR/scripts/submit_seedv_tmlr.slurm" | awk '{print $NF}'
}

# Lane A: FACED native variants.
lane_a_prev=$(submit_faced "$PREVIOUS" channel faced_native_channel_r32_s42_gate_20260805)
lane_a_prev=$(submit_faced "$lane_a_prev" channel_patch faced_native_channel_patch_r32_s42_gate_20260805)

# Lane B: SEED-V native channel. Its patch axis is singleton and is not part
# of this bottleneck gate.
lane_b_prev=$(submit_seedv "$PREVIOUS" channel seedv_native_channel_r32_s42_gate_20260805)

echo "lane_a_tail=${lane_a_prev}"
echo "lane_b_tail=${lane_b_prev}"
