"""深度剖析 SOAP optimizer 成本：掃 precision × precondition_frequency。

隔離 SOAP 的 opt.update 成本（給定固定梯度），找出：
- precision=HIGHEST（預設）vs DEFAULT(TF32) vs HIGH 在 3090 的差異
- precondition_frequency 10/25/50/10000(等同不 refresh) → 攤提 QR 成本
目標：分離「每步投影 matmul」與「週期性 QR」各佔多少，據此深度優化。

用法：uv run python scripts/bench_soap.py --config configs/re1000.yaml
"""
import time
import argparse
import jax
import jax.numpy as jnp
import optax
from soap_jax import soap
from pinn_cavity.config import load_config, apply_runtime
from pinn_cavity.networks import build_model
from pinn_cavity.losses import total_loss, init_weights
from pinn_cavity.geometry import make_sampler
from pinn_cavity.metrics import device_memory_mb

_PREC = {"highest": jax.lax.Precision.HIGHEST,
         "high": jax.lax.Precision.HIGH,
         "default": jax.lax.Precision.DEFAULT}


def timed(fn, warmup=5, steps=30, label=""):
    for _ in range(warmup):
        out = fn(); jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(steps):
        out = fn(); jax.block_until_ready(out)
    dt = (time.perf_counter() - t0) / steps * 1000
    print(f"  {label:<44} {dt:7.2f} ms | mem {device_memory_mb().get('peak_mb','?')} MB")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config); apply_runtime(cfg)
    print(f"=== SOAP deep-dive: backend={jax.default_backend()} net={cfg.network.width}x{cfg.network.depth} ===")

    p, s = build_model(jax.random.PRNGKey(0), cfg.network, cfg.lid_r)
    xy = make_sampler(cfg.sampler)(jax.random.PRNGKey(1), cfg.train.n_collocation)
    re = float(cfg.re); mode = cfg.autodiff; w = init_weights()
    # 固定梯度（隔離 optimizer，不含 backward 成本）
    grad = jax.jit(jax.grad(lambda q: total_loss(q, s, xy, w, re, mode)))(p)
    jax.block_until_ready(grad)

    def bench_variant(precision, freq, label):
        opt = soap(learning_rate=1e-3, b1=0.95, b2=0.95, precondition_frequency=freq,
                   precision=_PREC[precision], qr_dtype=jnp.float32)
        st = opt.init(p)
        @jax.jit
        def upd(p, st, g):
            u, st = opt.update(g, st, p)
            return optax.apply_updates(p, u), st
        timed(lambda: upd(p, st, grad), label=label)

    print("\n--- precision 掃描（precondition_frequency=10）---")
    for prec in ("highest", "high", "default"):
        bench_variant(prec, 10, f"precision={prec}")

    print("\n--- precondition_frequency 掃描（precision=highest）---")
    for f in (10, 25, 50, 100000):
        bench_variant("highest", f, f"freq={f}" + (" (no refresh)" if f > 50000 else ""))

    print("\n--- 最佳組合候選 ---")
    bench_variant("default", 50, "precision=default + freq=50")
    bench_variant("default", 100000, "precision=default + no-refresh (lower bound)")


if __name__ == "__main__":
    main()
