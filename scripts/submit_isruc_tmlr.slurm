#!/bin/bash
# One isolated CBraMod ISRUC TMLR job; one A6000 per job.
# The baseline is submitted alone before any adaptor/control jobs.
#SBATCH --job-name=cbram_isruc
#SBATCH --account=neurogroup_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod/logs/slurm/%x/%j.out
#SBATCH --error=/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod/logs/slurm/%x/%j.err

set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod"
ENV_DIR="${ENV_DIR:-/data/neurogroup/mingyangjiang/venvs/cbramod-accre}"
CONFIG="${CONFIG:-$REPO_DIR/configs/isruc_tmlr_locked.yaml}"
SEED="${SEED:-42}"
METHOD="${METHOD:-full_finetune}"
ADAPTER_TYPE="${ADAPTER_TYPE:-}"
ADAPTER_SEED="${ADAPTER_SEED:-30042}"
RUN_ID="${RUN_ID:-isruc_cbramod_${METHOD}_${ADAPTER_TYPE:-none}_s${SEED}}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
HEAD_LR="${HEAD_LR:-1.768e-4}"
ADAPTER_LR="${ADAPTER_LR:-2e-5}"
ADAPTER_BOTTLENECK="${ADAPTER_BOTTLENECK:-}"
ADAPTER_GAMMA="${ADAPTER_GAMMA:-}"
ADAPTER_INIT_ALPHA="${ADAPTER_INIT_ALPHA:-}"
ADAPTER_WEIGHT_DECAY="${ADAPTER_WEIGHT_DECAY:-}"
ADAPTER_ALPHA_WEIGHT_DECAY="${ADAPTER_ALPHA_WEIGHT_DECAY:-}"
LORA_LR="${LORA_LR:-2e-5}"
UPPER_LR="${UPPER_LR:-2e-4}"
DEVICE="${DEVICE:-cuda}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-}"
MAX_TEST_BATCHES="${MAX_TEST_BATCHES:-}"

mkdir -p "$REPO_DIR/logs/slurm/cbram_isruc" "$REPO_DIR/results/isruc"
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
export PYTHONNOUSERSITE=1
source "$ENV_DIR/bin/activate"
cd "$REPO_DIR"

echo "CBraMod ISRUC launch: method=$METHOD adapter=${ADAPTER_TYPE:-none} seed=$SEED batch=$BATCH_SIZE epochs=$EPOCHS run_id=$RUN_ID" >&2
echo "adapter_lr=$ADAPTER_LR bottleneck=${ADAPTER_BOTTLENECK:-from_config} gamma=${ADAPTER_GAMMA:-from_config} init_alpha=${ADAPTER_INIT_ALPHA:-from_config} adapter_weight_decay=${ADAPTER_WEIGHT_DECAY:-from_config} alpha_weight_decay=${ADAPTER_ALPHA_WEIGHT_DECAY:-from_config}" >&2
echo "python: $(command -v python)" >&2
echo "git_commit: $(git rev-parse HEAD)" >&2
echo "checkpoint_sha256: $(sha256sum /data/neurogroup/mingyangjiang/data/weights/pretrained_weights.pth | awk '{print $1}')" >&2
python - <<'PY'
import os, sys, torch, einops, yaml
expected = os.path.realpath(os.environ["VIRTUAL_ENV"])
actual = os.path.realpath(sys.prefix)
if actual != expected:
    raise RuntimeError(f"Activated environment mismatch: VIRTUAL_ENV={expected} sys.prefix={actual}")
print("python:", sys.executable)
print("sys.prefix:", sys.prefix)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
print("einops:", einops.__version__)
print("pyyaml:", yaml.__version__)
PY
nvidia-smi || true

RUN_ARGS=(
  --config "$CONFIG" --method "$METHOD" --seed "$SEED"
  --adapter-seed "$ADAPTER_SEED" --lr "$LR" --head-lr "$HEAD_LR"
  --adapter-lr "$ADAPTER_LR" --lora-lr "$LORA_LR" --upper-lr "$UPPER_LR"
  --batch-size "$BATCH_SIZE" --epochs "$EPOCHS"
  --device "$DEVICE" --run-id "$RUN_ID"
)
if [[ -n "$ADAPTER_TYPE" ]]; then RUN_ARGS+=(--adapter-type "$ADAPTER_TYPE"); fi
if [[ -n "$ADAPTER_BOTTLENECK" ]]; then RUN_ARGS+=(--adapter-bottleneck "$ADAPTER_BOTTLENECK"); fi
if [[ -n "$ADAPTER_GAMMA" ]]; then RUN_ARGS+=(--adapter-gamma "$ADAPTER_GAMMA"); fi
if [[ -n "$ADAPTER_INIT_ALPHA" ]]; then RUN_ARGS+=(--adapter-init-alpha "$ADAPTER_INIT_ALPHA"); fi
if [[ -n "$ADAPTER_WEIGHT_DECAY" ]]; then RUN_ARGS+=(--adapter-weight-decay "$ADAPTER_WEIGHT_DECAY"); fi
if [[ -n "$ADAPTER_ALPHA_WEIGHT_DECAY" ]]; then RUN_ARGS+=(--adapter-alpha-weight-decay "$ADAPTER_ALPHA_WEIGHT_DECAY"); fi
if [[ -n "$MAX_TRAIN_BATCHES" ]]; then RUN_ARGS+=(--max-train-batches "$MAX_TRAIN_BATCHES"); fi
if [[ -n "$MAX_VAL_BATCHES" ]]; then RUN_ARGS+=(--max-val-batches "$MAX_VAL_BATCHES"); fi
if [[ -n "$MAX_TEST_BATCHES" ]]; then RUN_ARGS+=(--max-test-batches "$MAX_TEST_BATCHES"); fi

OMP_NUM_THREADS=1 python -u run_isruc_tmlr.py "${RUN_ARGS[@]}"
