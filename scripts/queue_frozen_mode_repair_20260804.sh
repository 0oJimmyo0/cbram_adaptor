#!/usr/bin/env bash
# Queue corrected CBraMod frozen-base packets for FACED and SEED-V.
# There is one serial lane per dataset, so at most two A6000 jobs run at once.
set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod"
SEEDV_LAUNCHER="$REPO_DIR/scripts/submit_seedv_tmlr.slurm"
FACED_LAUNCHER="$REPO_DIR/scripts/submit_faced_tmlr.slurm"
SEEDV_CONFIG="$REPO_DIR/configs/seedv_tmlr_locked.yaml"
FACED_CONFIG="$REPO_DIR/configs/faced_tmlr_locked.yaml"

submit_seedv() {
  local previous="$1" method="$2" adapter="$3" seed="$4" run_id="$5" adapter_lr="$6"
  local dependency=()
  if [[ -n "$previous" ]]; then dependency=(--dependency="afterok:${previous}"); fi
  local export_value="ALL,SEED=${seed},LOADER_SEED=$((20000 + seed)),ADAPTER_SEED=$((30000 + seed)),METHOD=${method},RUN_ID=${run_id},ADAPTER_LR=${adapter_lr}"
  if [[ -n "$adapter" ]]; then export_value+=",ADAPTER_TYPE=${adapter}"; fi
  sbatch --parsable "${dependency[@]}" --export="$export_value" "$SEEDV_LAUNCHER"
}

submit_faced() {
  local previous="$1" method="$2" adapter="$3" seed="$4" run_id="$5" adapter_lr="$6"
  local dependency=()
  if [[ -n "$previous" ]]; then dependency=(--dependency="afterok:${previous}"); fi
  local adapter_seed=$((30000 + seed))
  local export_value="ALL,CONFIG=${FACED_CONFIG},SEED=${seed},ADAPTER_SEED=${adapter_seed},METHOD=${method},RUN_ID=${run_id},ADAPTER_LR=${adapter_lr}"
  if [[ -n "$adapter" ]]; then export_value+=",ADAPTER_TYPE=${adapter}"; fi
  sbatch --parsable "${dependency[@]}" --export="$export_value" "$FACED_LAUNCHER"
}

seedv_prev=""
for seed in 42 1024 3407; do
  seedv_prev=$(submit_seedv "$seedv_prev" frozen_probe "" "$seed" "seedv_evalmode_frozen_dense_s${seed}_20260804" 1e-4)
  # The previous SEED-V channel audit showed residual amplification at 1e-4;
  # retain the already tested conservative 1e-5 channel LR for the corrected
  # frozen packet. The mode correction is common to every frozen condition.
  seedv_prev=$(submit_seedv "$seedv_prev" interaction_aligned channel "$seed" "seedv_evalmode_frozen_channel_s${seed}_lr1e-5_20260804" 1e-5)
  seedv_prev=$(submit_seedv "$seedv_prev" interaction_aligned patch "$seed" "seedv_evalmode_frozen_patch_singleton_s${seed}_20260804" 1e-4)
  seedv_prev=$(submit_seedv "$seedv_prev" generic_bottleneck "" "$seed" "seedv_evalmode_frozen_generic_s${seed}_20260804" 1e-4)
  seedv_prev=$(submit_seedv "$seedv_prev" lora "" "$seed" "seedv_evalmode_frozen_lora_qkv_r8_s${seed}_20260804" 1e-4)
  seedv_prev=$(submit_seedv "$seedv_prev" upper_k_finetune "" "$seed" "seedv_evalmode_frozen_upper2_s${seed}_20260804" 1e-4)
  seedv_prev=$(submit_seedv "$seedv_prev" axis_blind "" "$seed" "seedv_evalmode_frozen_axis_blind_s${seed}_20260804" 1e-4)
done

faced_prev=""
for seed in 42 1024 3407; do
  faced_prev=$(submit_faced "$faced_prev" frozen_probe "" "$seed" "faced_evalmode_frozen_probe_s${seed}_20260804" 5e-4)
  faced_prev=$(submit_faced "$faced_prev" interaction_aligned channel "$seed" "faced_evalmode_frozen_channel_s${seed}_lr5e-4_20260804" 5e-4)
  faced_prev=$(submit_faced "$faced_prev" interaction_aligned patch "$seed" "faced_evalmode_frozen_patch_s${seed}_lr5e-4_20260804" 5e-4)
  faced_prev=$(submit_faced "$faced_prev" interaction_aligned channel_patch "$seed" "faced_evalmode_frozen_channel_patch_s${seed}_lr5e-4_20260804" 5e-4)
  faced_prev=$(submit_faced "$faced_prev" generic_bottleneck "" "$seed" "faced_evalmode_frozen_generic_s${seed}_20260804" 5e-4)
  faced_prev=$(submit_faced "$faced_prev" lora "" "$seed" "faced_evalmode_frozen_lora_qkv_r8_s${seed}_20260804" 5e-4)
  faced_prev=$(submit_faced "$faced_prev" upper_k_finetune "" "$seed" "faced_evalmode_frozen_upper2_s${seed}_20260804" 5e-4)
  faced_prev=$(submit_faced "$faced_prev" axis_blind "" "$seed" "faced_evalmode_frozen_axis_blind_s${seed}_20260804" 5e-4)
done

echo "seedv_frozen_mode_repair_tail=${seedv_prev}"
echo "faced_frozen_mode_repair_tail=${faced_prev}"
