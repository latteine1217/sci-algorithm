"""CLI：載入 config → 訓練 → 存完整狀態（results/<out>/state.pkl）。
用法：
  uv run python scripts/train.py --config configs/smoke.yaml --out results/smoke
  uv run python scripts/train.py --config configs/re1000.yaml --out results/re1000 --resume results/re1000/state.pkl
"""
import argparse
from pinn_cavity.config import load_config
from pinn_cavity.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--resume", default=None, help="state.pkl 路徑；存在則續訓")
    args = ap.parse_args()
    cfg = load_config(args.config)
    train(cfg, out_dir=args.out, resume_path=args.resume)
    print(f"done -> {args.out}/state.pkl")


if __name__ == "__main__":
    main()
