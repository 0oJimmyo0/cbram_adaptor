#!/bin/bash
#SBATCH --job-name=cbram_faced
#SBATCH --account=neurogroup_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod/logs/slurm/%x/%j.out
#SBATCH --error=/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod/logs/slurm/%x/%j.err

set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod"
ENV_DIR="${ENV_DIR:-/data/neurogroup/mingyangjiang/venvs/cbramod-accre}"
CONFIG="${CONFIG:-$REPO_DIR/configs/faced_tmlr_locked.yaml}"
SEED="${SEED:-42}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-50}"
METHOD="${METHOD:-}"
ADAPTER_TYPE="${ADAPTER_TYPE:-}"
ADAPTER_SEED="${ADAPTER_SEED:-}"
ADAPTER_LR="${ADAPTER_LR:-}"
RUN_ID="${RUN_ID:-faced_cbramod_${METHOD:-full_finetune}_${ADAPTER_TYPE:-none}_s${SEED}_lr${LR}_b${BATCH_SIZE}_e${EPOCHS}}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$REPO_DIR/logs/slurm/cbram_faced" "$REPO_DIR/results/faced"

unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
module --force purge >/dev/null 2>&1 || true
module load gentoo/2023 gcc/12.3 python/3.11.5 scipy-stack/2026a >/dev/null 2>&1 || true
export PYTHONNOUSERSITE=1
source "$ENV_DIR/bin/activate"
cd "$REPO_DIR"

echo "CBraMod FACED launch: config=$CONFIG method=${METHOD:-from_config} adapter=${ADAPTER_TYPE:-none} seed=$SEED lr=$LR batch=$BATCH_SIZE epochs=$EPOCHS run_id=$RUN_ID" >&2
echo "git_commit: $(git rev-parse HEAD)" >&2
echo "checkpoint_sha256: $(sha256sum /data/neurogroup/mingyangjiang/data/weights/pretrained_weights.pth | awk '{print $1}')" >&2
python - <<'PY'
import os, sys, torch, lmdb
import einops, yaml
expected = os.path.realpath(os.environ["VIRTUAL_ENV"])
actual = os.path.realpath(sys.prefix)
if actual != expected:
    raise RuntimeError(f"Activated environment mismatch: VIRTUAL_ENV={expected} sys.prefix={actual}")
print("python:", sys.executable)
print("sys.prefix:", sys.prefix)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
print("lmdb:", lmdb.__file__)
print("einops:", einops.__version__)
print("pyyaml:", yaml.__version__)
PY
nvidia-smi || true

RUN_ARGS=(
  --config "$CONFIG"
  --seed "$SEED"
  --lr "$LR"
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --device "$DEVICE"
  --run-id "$RUN_ID"
)
if [[ -n "$METHOD" ]]; then RUN_ARGS+=(--method "$METHOD"); fi
if [[ -n "$ADAPTER_TYPE" ]]; then RUN_ARGS+=(--adapter-type "$ADAPTER_TYPE"); fi
if [[ -n "$ADAPTER_SEED" ]]; then RUN_ARGS+=(--adapter-seed "$ADAPTER_SEED"); fi
if [[ -n "$ADAPTER_LR" ]]; then RUN_ARGS+=(--adapter-lr "$ADAPTER_LR"); fi

OMP_NUM_THREADS=1 python -u run_faced_tmlr.py \
  "${RUN_ARGS[@]}"
