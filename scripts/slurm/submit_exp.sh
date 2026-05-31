#!/usr/bin/env bash
# 提交一個 PINN 訓練實驗到 SLURM partition r740（compute node）。
# 用法：scripts/slurm/submit_exp.sh <EXP_ID> [CONFIG]
#   <EXP_ID>  實驗識別碼，輸出落在 results/<EXP_ID>/
#   [CONFIG]  設定檔，預設 configs/re1000.yaml
#   DRY=1     只印將提交的指令，不實際 sbatch
#
# 前提：已在 head node 執行 `uv sync --extra cuda --python 3.12`（compute node 離線）。
set -euo pipefail

EXP_ID="${1:?usage: submit_exp.sh <EXP_ID> [CONFIG]}"
CONFIG="${2:-configs/re1000.yaml}"
OUT="results/${EXP_ID}"
mkdir -p "${OUT}" logs

CMD=(sbatch
  --job-name="pinn-${EXP_ID}"
  --export=ALL,CONFIG="${CONFIG}",OUT="${OUT}"
  scripts/slurm/train.sbatch)

if [ "${DRY:-0}" = "1" ]; then
  printf '%q ' "${CMD[@]}"; echo
else
  "${CMD[@]}"
fi
