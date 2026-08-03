#!/bin/bash
# Queue the complete CBraMod SEED-V TMLR packet in two serial lanes.
# Each run requests one A6000; the two lanes cap simultaneous GPU use at two.
set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod"
LAUNCHER="$REPO_DIR/scripts/submit_seedv_tmlr.slurm"
SMOKE_JOB="${SMOKE_JOB:?Set SMOKE_JOB to the successful native-channel smoke job}"

submit_next() {
  local previous="$1" method="$2" adapter="$3" seed="$4" run_id="$5"
  local adapter_seed=$((30000 + seed))
  local dependency=(--dependency="afterok:${previous}")
  local export_value="ALL,SEED=${seed},LOADER_SEED=$((20000 + seed)),ADAPTER_SEED=${adapter_seed},METHOD=${method},RUN_ID=${run_id}"
  if [[ -n "$adapter" ]]; then export_value+=",ADAPTER_TYPE=${adapter}"; fi
  sbatch --parsable "${dependency[@]}" --export="$export_value" "$LAUNCHER"
}

# Primary dense/frozen/native-channel regimes are first in lane A.  The
# singleton patch control and independent controls follow in lane B.
lane_a="$SMOKE_JOB"
lane_b="$SMOKE_JOB"
for seed in 42 1024 3407; do
  lane_a=$(submit_next "$lane_a" full_finetune "" "$seed" "seedv_dense_full_s${seed}")
  lane_a=$(submit_next "$lane_a" frozen_probe "" "$seed" "seedv_frozen_dense_s${seed}")
  lane_a=$(submit_next "$lane_a" interaction_aligned channel "$seed" "seedv_frozen_channel_s${seed}")
  lane_a=$(submit_next "$lane_a" native_full_finetune channel "$seed" "seedv_native_full_channel_s${seed}")
done
for seed in 42 1024 3407; do
  lane_b=$(submit_next "$lane_b" interaction_aligned patch "$seed" "seedv_frozen_patch_singleton_control_s${seed}")
  lane_b=$(submit_next "$lane_b" generic_bottleneck "" "$seed" "seedv_frozen_generic_s${seed}")
  lane_b=$(submit_next "$lane_b" lora "" "$seed" "seedv_frozen_lora_qkv_r8_s${seed}")
  lane_b=$(submit_next "$lane_b" upper_k_finetune "" "$seed" "seedv_frozen_upper2_s${seed}")
  lane_b=$(submit_next "$lane_b" axis_blind "" "$seed" "seedv_frozen_axis_blind_s${seed}")
done

echo "lane_a_tail=$lane_a"
echo "lane_b_tail=$lane_b"
