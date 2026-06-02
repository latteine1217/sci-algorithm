"""量測 matrix-free Gauss-Newton step 的速度與記憶體。

對比維度：
- linearize True/False（前向只算一次 vs 每 CG 迭代重算）
- cg_iters（CG 迭代數 → 每步成本線性增長）
- 殘差 mode（taylor vs hessian）
與 SOAP 單步對照（per-step 成本基準）。

用法：
  uv run python scripts/bench_gn.py --config configs/re1000.yaml
"""
import time
import argparse
import jax
import optax
from pinn_cavity.config import load_config, apply_runtime
from pinn_cavity.networks import build_model
from pinn_cavity.natural_gradient import gn_step
from pinn_cavity.losses import total_loss, init_weights
from pinn_cavity.optimizers import build_optimizer
from pinn_cavity.geometry import make_sampler
from pinn_cavity.metrics import device_memory_mb


def timed(fn, warmup=3, steps=15, label=""):
    for _ in range(warmup):
        out = fn(); jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(steps):
        out = fn(); jax.block_until_ready(out)
    dt = (time.perf_counter() - t0) / steps * 1000
    print(f"  {label:<40} {dt:8.2f} ms/step | mem {device_memory_mb().get('peak_mb','?')} MB")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config); apply_runtime(cfg)
    print(f"=== bench GN: backend={jax.default_backend()} net={cfg.network.width}x{cfg.network.depth} "
          f"n_coll={cfg.train.n_collocation} mode={cfg.autodiff} ===")

    p, s = build_model(jax.random.PRNGKey(0), cfg.network, cfg.lid_r)
    xy = make_sampler(cfg.sampler)(jax.random.PRNGKey(1), cfg.train.n_collocation)
    re = float(cfg.re); mode = cfg.autodiff

    # SOAP 單步基準
    opt = build_optimizer(cfg.optimizer); st = opt.init(p); w = init_weights()
    @jax.jit
    def soap_step(p, st):
        L, g = jax.value_and_grad(lambda q: total_loss(q, s, xy, w, re, mode))(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, L
    print("\n--- SOAP 基準 ---")
    timed(lambda: soap_step(p, st), label="SOAP step")

    print("\n--- Gauss-Newton（linearize=True，快速路徑）---")
    for k in (5, 10, 20, 40):
        timed(lambda: gn_step(p, s, xy, re, mode, 1.0, k, True, 1e-3),
              label=f"GN cg_iters={k}")

    print("\n--- Gauss-Newton（linearize=False，每迭代重算前向）---")
    for k in (10, 20):
        timed(lambda: gn_step(p, s, xy, re, mode, 1.0, k, False, 1e-3),
              label=f"GN cg_iters={k} (recompute)")

    print("\n--- mode 對比（GN cg_iters=20, linearize=True）---")
    for m in ("taylor", "hessian"):
        timed(lambda: gn_step(p, s, xy, re, m, 1.0, 20, True, 1e-3),
              label=f"GN mode={m}")


if __name__ == "__main__":
    main()
