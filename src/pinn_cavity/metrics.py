"""實驗指標蒐集：wall time、GPU 峰值記憶體、準度、設定快照。

為演算法比較研究而設：每個實驗輸出結構化 summary.json，可直接跨實驗 diff
（wall_seconds、steps_per_sec、peak_memory_mb、accuracy）。
"""
import os
import json
import jax


def device_memory_mb():
    """回傳當前裝置記憶體統計（MB）。CPU backend 無統計時回 {}。"""
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:
        stats = None
    if not stats:
        return {}
    to_mb = lambda b: round(b / (1024 ** 2), 1)
    out = {}
    if "peak_bytes_in_use" in stats:
        out["peak_mb"] = to_mb(stats["peak_bytes_in_use"])
    if "bytes_in_use" in stats:
        out["current_mb"] = to_mb(stats["bytes_in_use"])
    if "bytes_limit" in stats:
        out["limit_mb"] = to_mb(stats["bytes_limit"])
    return out


def config_snapshot(cfg):
    """擷取影響效能/準度比較的關鍵設定。"""
    return {
        "re": cfg.re,
        "x64": bool(cfg.x64),
        "weighting": cfg.weighting,
        "weight_ema": cfg.weight_ema,
        "autodiff": cfg.autodiff,
        "rwf": getattr(cfg.network, "rwf", False),
        "optimizer": cfg.optimizer.name,
        "sampler": cfg.sampler.name,
        "lid_r": cfg.lid_r,
        "network": {
            "width": cfg.network.width,
            "depth": cfg.network.depth,
            "n_fourier": cfg.network.n_fourier,
            "fourier_sigma": cfg.network.fourier_sigma,
        },
        "n_collocation": cfg.train.n_collocation,
    }


def write_summary(path, summary: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=float, ensure_ascii=False)


def load_summary(path) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def update_summary(path, patch: dict):
    """讀-改-寫 summary.json（evaluate 端併入準度用）。"""
    s = load_summary(path)
    s.update(patch)
    write_summary(path, s)
