#!/bin/bash
# Split the final SEED-V corrected seed-3407 controls after the currently
# running parent, then queue the r=32 gate after both controls succeed.
set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod"
LAUNCHER="$REPO_DIR/scripts/submit_seedv_tmlr.slurm"
CONFIG="$REPO_DIR/configs/seedv_tmlr_locked.yaml"
PREVIOUS="${1:-12969521}"

submit() {
  local method="$1" run_id="$2"
  sbatch --dependency="afterok:${PREVIOUS}" \
    --export="ALL,CONFIG=${CONFIG},SEED=3407,LOADER_SEED=23407,ADAPTER_SEED=33407,METHOD=${method},RUN_ID=${run_id}" \
    "$LAUNCHER" | awk '{print $NF}'
}

upper_job=$(submit upper_k_finetune seedv_evalmode_frozen_upper2_s3407_requeue_20260805)
axis_job=$(submit axis_blind seedv_evalmode_frozen_axis_blind_s3407_requeue_20260805)

# The gate script accepts a colon-separated afterok dependency, which means
# both independent controls must complete successfully before r=32 starts.
r32_gate=$(
  "$REPO_DIR/scripts/queue_adapter_bottleneck_gate_r32_s42.sh" \
    "${upper_job}:${axis_job}"
)

echo "upper_job=${upper_job}"
echo "axis_job=${axis_job}"
echo "$r32_gate"
