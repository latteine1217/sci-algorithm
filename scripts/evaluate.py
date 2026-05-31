"""CLI：載入 state.pkl → 完整評估（Ghia + 場診斷 + layered 報告 + 圖）。
用法：uv run python scripts/evaluate.py --config configs/re1000.yaml --state results/re1000/state.pkl --out results/re1000
"""
import argparse
from pinn_cavity.config import load_config, apply_runtime
from pinn_cavity.checkpoint import load_state
from pinn_cavity.evaluate import evaluate, static_from_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    cfg = load_config(args.config)
    apply_runtime(cfg)
    st = load_state(args.state)
    static = static_from_state(st, cfg.lid_r)
    metrics = evaluate(st["params"], static, re=cfg.re, out_dir=args.out)
    print(metrics)


if __name__ == "__main__":
    main()
