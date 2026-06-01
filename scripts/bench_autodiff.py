"""量測 physics 二階導模式（fwd_over_rev vs hessian）的單步耗時與 GPU 峰值記憶體。

分別跑兩次比較（peak memory 為 process 累積，故一次一模式）：
  uv run python scripts/bench_autodiff.py --config configs/re1000.yaml --mode fwd_over_rev
  uv run python scripts/bench_autodiff.py --config configs/re1000.yaml --mode hessian
"""
import time
import argparse
import jax
import optax
from pinn_cavity.config import load_config, apply_runtime
from pinn_cavity.networks import build_model
from pinn_cavity.losses import total_loss, init_weights
from pinn_cavity.optimizers import build_optimizer
from pinn_cavity.geometry import make_sampler
from pinn_cavity.metrics import device_memory_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", required=True, choices=["fwd_over_rev", "hessian", "taylor"])
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()
    cfg = load_config(args.config); apply_runtime(cfg)
    print(f"backend={jax.default_backend()} x64={jax.config.read('jax_enable_x64')} mode={args.mode}")

    p, s = build_model(jax.random.PRNGKey(0), cfg.network, cfg.lid_r)
    opt = build_optimizer(cfg.optimizer); st = opt.init(p); w = init_weights()
    xy = make_sampler(cfg.sampler)(jax.random.PRNGKey(1), cfg.train.n_collocation)

    @jax.jit
    def step(p, st, xy):
        L, g = jax.value_and_grad(lambda q: total_loss(q, s, xy, w, cfg.re, args.mode))(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, L

    p, st, L = step(p, st, xy); L.block_until_ready()  # warmup/compile
    t0 = time.time()
    for _ in range(args.steps):
        p, st, L = step(p, st, xy)
    L.block_until_ready()
    dt = (time.time() - t0) / args.steps
    print(f"per-step = {dt*1000:.1f} ms | peak_memory = {device_memory_mb()} | final_loss={float(L):.3e}")


if __name__ == "__main__":
    main()
