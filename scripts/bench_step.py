"""量測單一訓練 step 耗時（用以比較 x32/x64 在特定 GPU 的吞吐）。

用法：
  uv run python scripts/bench_step.py --config configs/re1000.yaml --steps 20
  PINN_DISABLE_X64=1 uv run python scripts/bench_step.py --config configs/re1000.yaml --steps 20
x64 狀態由環境變數 PINN_DISABLE_X64 控制（見 pinn_cavity/__init__.py）。
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    cfg = load_config(args.config)
    apply_runtime(cfg)  # 依 config 設 x64（須在建任何陣列前）

    x64 = jax.config.read("jax_enable_x64")
    print(f"backend={jax.default_backend()} x64={x64} devices={jax.devices()}")

    p, s = build_model(jax.random.PRNGKey(0), cfg.network, cfg.lid_r)
    opt = build_optimizer(cfg.optimizer)
    st = opt.init(p)
    w = init_weights()
    sampler = make_sampler(cfg.sampler)
    xy = sampler(jax.random.PRNGKey(1), cfg.train.n_collocation)

    @jax.jit
    def step(p, st, xy, w):
        L, g = jax.value_and_grad(lambda q: total_loss(q, s, xy, w, cfg.re))(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, L

    p, st, L = step(p, st, xy, w); L.block_until_ready()  # warmup/compile
    t0 = time.time()
    for _ in range(args.steps):
        p, st, L = step(p, st, xy, w)
    L.block_until_ready()
    dt = (time.time() - t0) / args.steps
    print(f"per-step = {dt*1000:.1f} ms  ->  1000 steps ~ {dt*1000/60:.1f} min")


if __name__ == "__main__":
    main()
