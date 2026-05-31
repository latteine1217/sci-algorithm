"""跨實驗比較：讀多個 results/<exp>/summary.json + history.csv，
輸出比較表（markdown + csv）與疊加收斂曲線。服務演算法比較研究。

用法：
  uv run python scripts/compare.py results/re1000-fp32 results/re1000-rwf-sdf-gn --out results/compare
"""
import os
import csv
import json
import argparse
import numpy as np


def _load(exp_dir):
    s = {}
    sp = os.path.join(exp_dir, "summary.json")
    if os.path.exists(sp):
        with open(sp) as f:
            s = json.load(f)
    hist = None
    hp = os.path.join(exp_dir, "history.csv")
    if os.path.exists(hp):
        step, loss = [], []
        with open(hp) as f:
            for r in csv.DictReader(f):
                step.append(float(r["global_step"])); loss.append(float(r["loss"]))
        hist = (np.array(step), np.array(loss))
    return s, hist


_COLS = [
    ("exp", lambda s: s.get("_name", "?")),
    ("optimizer", lambda s: s.get("config", {}).get("optimizer", "?")),
    ("weighting", lambda s: s.get("config", {}).get("weighting", "?")),
    ("rwf", lambda s: s.get("config", {}).get("rwf", "?")),
    ("net", lambda s: f"{s.get('config',{}).get('network',{}).get('width','?')}x{s.get('config',{}).get('network',{}).get('depth','?')}"),
    ("wall_s", lambda s: s.get("wall_seconds_total", "?")),
    ("steps/s", lambda s: s.get("steps_per_sec", "?")),
    ("peak_mb", lambda s: s.get("peak_memory_mb", {}).get("peak_mb", "?")),
    ("L2_u", lambda s: round(s.get("accuracy", {}).get("rel_l2_u", float("nan")), 4)),
    ("L2_v", lambda s: round(s.get("accuracy", {}).get("rel_l2_v", float("nan")), 4)),
    ("div_max", lambda s: f"{s.get('accuracy',{}).get('divergence_max',float('nan')):.2e}"),
    ("cont_rms", lambda s: f"{s.get('accuracy',{}).get('residual_rms_cont',float('nan')):.2e}"),
    ("verdict", lambda s: s.get("assessment", {}).get("verdict", "?")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+", help="results/<exp> 目錄")
    ap.add_argument("--out", default="results/compare")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summaries, hists = [], []
    for d in args.exps:
        s, h = _load(d)
        s["_name"] = os.path.basename(d.rstrip("/"))
        summaries.append(s); hists.append((s["_name"], h))

    header = [c[0] for c in _COLS]
    rows = [[str(fn(s)) for _, fn in _COLS] for s in summaries]

    # markdown 表
    md = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md_str = "\n".join(md)
    with open(os.path.join(args.out, "comparison.md"), "w") as f:
        f.write(md_str + "\n")
    with open(os.path.join(args.out, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(md_str)

    # 疊加收斂曲線
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, h in hists:
        if h is not None:
            ax.semilogy(h[0], h[1], label=name)
    ax.set_xlabel("step"); ax.set_ylabel("total loss"); ax.set_title("Convergence comparison")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(args.out, "convergence.png"), dpi=150)
    print(f"\nsaved -> {args.out}/comparison.md, comparison.csv, convergence.png")


if __name__ == "__main__":
    main()
