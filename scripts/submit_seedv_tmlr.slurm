#!/bin/bash
# Isolated CBraMod TMLR SEED-V runner.  One job consumes one A6000.
# The launcher intentionally does not purge/load cluster modules: the
# dedicated environment is the reproducibility boundary for this study.
#SBATCH --job-name=cbram_seedv
#SBATCH --account=neurogroup_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod/logs/slurm/%x/%j.out
#SBATCH --error=/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod/logs/slurm/%x/%j.err

set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod"
ENV_DIR="${ENV_DIR:-/data/neurogroup/mingyangjiang/venvs/cbramod-accre}"
CONFIG="${CONFIG:-$REPO_DIR/configs/seedv_tmlr_locked.yaml}"
SEED="${SEED:-42}"
ADAPTER_SEED="${ADAPTER_SEED:-30042}"
LOADER_SEED="${LOADER_SEED:-$((20000 + SEED))}"
METHOD="${METHOD:-frozen_probe}"
ADAPTER_TYPE="${ADAPTER_TYPE:-}"
RUN_ID="${RUN_ID:-seedv_cbramod_${METHOD}_${ADAPTER_TYPE:-none}_s${SEED}}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
ADAPTER_LR="${ADAPTER_LR:-1e-4}"
DEVICE="${DEVICE:-cuda}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-}"
MAX_TEST_BATCHES="${MAX_TEST_BATCHES:-}"

mkdir -p "$REPO_DIR/logs/slurm/cbram_seedv" "$REPO_DIR/results/seedv"
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
export PYTHONNOUSERSITE=1
source "$ENV_DIR/bin/activate"
cd "$REPO_DIR"

echo "CBraMod SEED-V launch: method=$METHOD adapter=${ADAPTER_TYPE:-none} seed=$SEED batch=$BATCH_SIZE epochs=$EPOCHS run_id=$RUN_ID" >&2
echo "python: $(command -v python)" >&2
echo "git_commit: $(git rev-parse HEAD)" >&2
echo "checkpoint_sha256: $(sha256sum /data/neurogroup/mingyangjiang/data/weights/pretrained_weights.pth | awk '{print $1}')" >&2
python - <<'PY'
import os, sys, torch, lmdb, einops
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
PY
nvidia-smi || true

RUN_ARGS=(
  --config "$CONFIG" --method "$METHOD" --seed "$SEED"
  --adapter-seed "$ADAPTER_SEED" --lr "$LR" --adapter-lr "$ADAPTER_LR"
  --loader-seed "$LOADER_SEED"
  --batch-size "$BATCH_SIZE" --epochs "$EPOCHS" --device "$DEVICE"
  --run-id "$RUN_ID"
)
if [[ -n "$ADAPTER_TYPE" ]]; then RUN_ARGS+=(--adapter-type "$ADAPTER_TYPE"); fi
if [[ -n "$MAX_TRAIN_BATCHES" ]]; then RUN_ARGS+=(--max-train-batches "$MAX_TRAIN_BATCHES"); fi
if [[ -n "$MAX_VAL_BATCHES" ]]; then RUN_ARGS+=(--max-val-batches "$MAX_VAL_BATCHES"); fi
if [[ -n "$MAX_TEST_BATCHES" ]]; then RUN_ARGS+=(--max-test-batches "$MAX_TEST_BATCHES"); fi

OMP_NUM_THREADS=1 python -u run_seedv_tmlr.py "${RUN_ARGS[@]}"
