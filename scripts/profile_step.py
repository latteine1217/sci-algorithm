"""效能剖析：分解訓練 step 各元件的耗時。

量測項目：
  1. forward（predict 一批點）
  2. residuals（NS 殘差，含二階導）
  3. loss（殘差 MSE）
  4. grad（loss 對 params 的梯度 = value_and_grad）
  5. optimizer update（SOAP update + apply_updates）
  6. full step（grad + update，即訓練 step 全程）

用法：
  uv run python scripts/profile_step.py --config configs/re1000.yaml
  uv run python scripts/profile_step.py --config configs/re1000.yaml --trace  # 額外存 XLA trace
"""
import time
import argparse
import jax
import jax.numpy as jnp
import optax
from pinn_cavity.config import load_config, apply_runtime
from pinn_cavity.networks import build_model, predict
from pinn_cavity.physics import ns_residuals
from pinn_cavity.losses import loss_terms, total_loss, init_weights
from pinn_cavity.optimizers import build_optimizer
from pinn_cavity.geometry import make_sampler
from pinn_cavity.metrics import device_memory_mb


def timed(fn, warmup=3, steps=20, label=""):
    """JIT 後計時，每次 block_until_ready 確保 GPU 完成。"""
    for _ in range(warmup):
        out = fn()
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, out)
    t0 = time.perf_counter()
    for _ in range(steps):
        out = fn()
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, out)
    dt = (time.perf_counter() - t0) / steps * 1000
    print(f"  {label:<28} {dt:7.2f} ms/step")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--trace", action="store_true", help="存 XLA profiler trace 到 /tmp/jax_profile")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_runtime(cfg)
    print(f"=== Profile: backend={jax.default_backend()} x64={jax.config.read('jax_enable_x64')} "
          f"autodiff={cfg.autodiff} net={cfg.network.width}x{cfg.network.depth} "
          f"n_coll={cfg.train.n_collocation} ===")

    key = jax.random.PRNGKey(0)
    params, static = build_model(key, cfg.network, cfg.lid_r)
    opt = build_optimizer(cfg.optimizer)
    opt_state = opt.init(params)
    weights = init_weights()
    sampler = make_sampler(cfg.sampler)
    xy = sampler(jax.random.PRNGKey(1), cfg.train.n_collocation)
    re = float(cfg.re)
    mode = cfg.autodiff

    print("\n--- 分解計時（各元件獨立 JIT）---")

    # 1. Forward pass
    @jax.jit
    def run_forward(p):
        return predict(p, static, xy)
    t_fwd = timed(lambda: run_forward(params), args.warmup, args.steps, "1. forward (predict)")

    # 2. Residuals only
    @jax.jit
    def run_residuals(p):
        return ns_residuals(p, static, xy, re, mode=mode)
    t_res = timed(lambda: run_residuals(params), args.warmup, args.steps, "2. ns_residuals")

    # 3. Loss (residuals + MSE)
    @jax.jit
    def run_loss(p):
        return total_loss(p, static, xy, weights, re, mode)
    t_loss = timed(lambda: run_loss(params), args.warmup, args.steps, "3. total_loss")

    # 4. Grad（value_and_grad，即反向傳播）
    @jax.jit
    def run_grad(p):
        L, g = jax.value_and_grad(lambda q: total_loss(q, static, xy, weights, re, mode))(p)
        return L, g
    t_grad = timed(lambda: run_grad(params), args.warmup, args.steps, "4. value_and_grad (backward)")

    # 5. Optimizer update only（從 dummy grad）
    dummy_grad = jax.grad(lambda p: total_loss(p, static, xy, weights, re, mode))(params)
    @jax.jit
    def run_opt(p, st, g):
        updates, new_st = opt.update(g, st, p)
        return optax.apply_updates(p, updates), new_st
    t_opt = timed(lambda: run_opt(params, opt_state, dummy_grad), args.warmup, args.steps, "5. optimizer update")

    # 6. Full step（grad + opt，= 實際訓練 step）
    @jax.jit
    def run_full(p, st):
        L, g = jax.value_and_grad(lambda q: total_loss(q, static, xy, weights, re, mode))(p)
        updates, new_st = opt.update(g, st, p)
        return optax.apply_updates(p, updates), new_st, L
    t_full = timed(lambda: run_full(params, opt_state), args.warmup, args.steps, "6. full step (grad+opt)")

    print(f"\n--- 分解佔比（相對 full step）---")
    for label, t in [("forward", t_fwd), ("residuals", t_res),
                     ("loss", t_loss), ("backward (grad)", t_grad), ("optimizer", t_opt)]:
        print(f"  {label:<22} {t/t_full*100:5.1f}%  ({t:.2f} ms)")

    print(f"\n--- 記憶體 ---")
    print(f"  {device_memory_mb()}")

    # 推算各項在 full step 中的邊際成本
    t_backward_only = t_grad - t_loss   # grad = forward + backward，loss ≈ forward
    t_opt_only = t_full - t_grad        # full - grad ≈ optimizer
    print(f"\n--- 估算：full step 拆解 ---")
    print(f"  forward:          {t_loss:7.2f} ms  ({t_loss/t_full*100:.1f}%)")
    print(f"  backward only:    {t_backward_only:7.2f} ms  ({t_backward_only/t_full*100:.1f}%)")
    print(f"  optimizer only:   {t_opt_only:7.2f} ms  ({t_opt_only/t_full*100:.1f}%)")

    if args.trace:
        import os
        out = "/tmp/jax_profile"
        os.makedirs(out, exist_ok=True)
        print(f"\n=== 存 XLA profiler trace -> {out} ===")
        with jax.profiler.trace(out):
            run_full(params, opt_state)[2].block_until_ready()
        print(f"  用 tensorboard --logdir {out} 檢視")


if __name__ == "__main__":
    main()
