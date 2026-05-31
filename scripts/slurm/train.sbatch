#!/bin/bash
# r740 GPU 訓練 + 評估。由 submit_exp.sh 提交，勿直接在 head node 執行。
# CONFIG / OUT 由 submit_exp.sh 經 --export 傳入；job-name/output 由 CLI 覆寫。
#SBATCH --job-name=pinn-cavity
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/sci-algorithm}"

# 環境：unbuffered（避免 stdout block-buffer 看似 hung）+ uv on PATH + compute node 離線
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_OFFLINE=1
command -v uv >/dev/null 2>&1 || { echo "[ERR] uv not in PATH" >&2; exit 2; }
[ -x .venv/bin/python ] || { echo "[ERR] .venv 不存在；先在 head node 跑 'uv sync --extra cuda --python 3.12'" >&2; exit 3; }

CONFIG="${CONFIG:-configs/re1000.yaml}"
OUT="${OUT:-results/re1000}"
mkdir -p "${OUT}" logs

echo "======================================================================"
echo "[pinn-cavity] $(date '+%F %T') job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
echo "  config=${CONFIG}  out=${OUT}  cuda=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "======================================================================"
nvidia-smi -L 2>/dev/null || echo "[WARN] nvidia-smi 不可用"
uv --version

# 若已有 checkpoint 則自動 resume（長訓練中斷續跑）
RESUME=""
if [ -f "${OUT}/state.pkl" ]; then
  RESUME="--resume ${OUT}/state.pkl"
  echo "resuming from ${OUT}/state.pkl"
fi

echo "=== train ==="
uv run python -u scripts/train.py --config "${CONFIG}" --out "${OUT}" ${RESUME}

echo "=== evaluate ==="
uv run python -u scripts/evaluate.py --config "${CONFIG}" --state "${OUT}/state.pkl" --out "${OUT}"

# SLURM 層級資源用量（host MaxRSS / Elapsed / NodeList），補 summary.json 的 GPU 記憶體
# 用 || true：compute node 上 sacct 可能回非零，勿讓它在 set -e 下誤判整個 job 失敗
echo "=== slurm resource usage ==="
{ sacct -j "${SLURM_JOB_ID}" \
  --format=JobID,JobName,State,Elapsed,MaxRSS,ReqMem,NNodes,NodeList,AllocGRES%20 \
  2>/dev/null | tee "${OUT}/slurm_metrics.txt"; } || true

echo "[$(date '+%F %T')] pinn-cavity done"
