#!/usr/bin/env bash
# r740 GPU 訓練 + 評估。由 submit_exp.sh 提交，勿直接在 head node 執行。
#SBATCH --job-name=pinn-cavity
#SBATCH --nodelist=r740
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm-%x-%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
# compute node 離線：依賴須在 head node 先 `uv sync --extra cuda`
export UV_OFFLINE=1

CONFIG="${CONFIG:-configs/re1000.yaml}"
OUT="${OUT:-results/re1000}"
mkdir -p "${OUT}"

# 若已有 checkpoint 則自動 resume（長訓練中斷續跑）
RESUME=""
if [ -f "${OUT}/state.pkl" ]; then
  RESUME="--resume ${OUT}/state.pkl"
  echo "resuming from ${OUT}/state.pkl"
fi

echo "=== train: config=${CONFIG} out=${OUT} ==="
uv run python scripts/train.py --config "${CONFIG}" --out "${OUT}" ${RESUME}

echo "=== evaluate ==="
uv run python scripts/evaluate.py --config "${CONFIG}" --state "${OUT}/state.pkl" --out "${OUT}"
