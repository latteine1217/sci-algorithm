"""Gauss-Newton refinement：載入已收斂的 SOAP checkpoint，最後跑 N 步 GN 精修。

正確的二階用法：一階（SOAP）先進到正確 basin，GN 在好 basin 內快速精修
（對應經典 Adam→L-BFGS）。從零開 GN 會掉進 spurious 解，故只在收斂後 refine。

用法：
  uv run python scripts/refine_gn.py --config configs/re1000.yaml \
      --state results/re1000-taylor/state.pkl --steps 5000 --out results/re1000-gn-refine
"""
import time
import argparse
import jax
import jax.numpy as jnp
from pinn_cavity.config import load_config, apply_runtime
from pinn_cavity.checkpoint import load_state, save_state
from pinn_cavity.networks import NetStatic
from pinn_cavity.natural_gradient import gn_step
from pinn_cavity.geometry import make_sampler
from pinn_cavity.evaluate import evaluate
from pinn_cavity.metrics import device_memory_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--state", required=True, help="已收斂的 SOAP checkpoint")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--cg", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--damping", type=float, default=1e-3)
    ap.add_argument("--resample_every", type=int, default=1000,
                    help="GN refine 期間重抽頻率（0=固定不抽，二階更穩）")
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--out", default="results/gn-refine")
    args = ap.parse_args()

    cfg = load_config(args.config); apply_runtime(cfg)
    st = load_state(args.state)
    params = st["params"]
    static = NetStatic(B=st["fourier_B"], lid_r=cfg.lid_r)
    re = float(cfg.re); mode = cfg.autodiff
    sampler = make_sampler(cfg.sampler)
    print(f"=== GN refine: backend={jax.default_backend()} from {args.state} "
          f"steps={args.steps} cg={args.cg} lr={args.lr} damping={args.damping} ===")

    # 精修前先評估基準
    print("--- before refine ---")
    m0 = evaluate(params, static, re=re, out_dir=args.out + "_before")
    print(f"  field-L2 u/v = {m0['field_rel_l2_u']:.4f} / {m0['field_rel_l2_v']:.4f}")

    key = jax.random.PRNGKey(cfg.seed + 999)
    key, sk = jax.random.split(key)
    xy = sampler(sk, cfg.train.n_collocation)
    t0 = time.time()
    for it in range(args.steps):
        if args.resample_every and it % args.resample_every == 0 and it > 0:
            key, sk = jax.random.split(key)
            xy = sampler(sk, cfg.train.n_collocation)
        params, L = gn_step(params, static, xy, re, mode, args.lr, args.cg, True, args.damping)
        if it % args.log_every == 0 or it == args.steps - 1:
            print(f"  GN it={it} loss={float(L):.3e}")
    wall = time.time() - t0
    print(f"=== refine wall={wall:.1f}s ({args.steps/wall:.2f} steps/s) "
          f"peak_mem={device_memory_mb()} ===")

    save_state(args.out + "/state.pkl", {**st, "params": params})
    print("--- after refine ---")
    m1 = evaluate(params, static, re=re, out_dir=args.out)
    print(f"  field-L2 u/v: {m0['field_rel_l2_u']:.4f}/{m0['field_rel_l2_v']:.4f} "
          f"-> {m1['field_rel_l2_u']:.4f}/{m1['field_rel_l2_v']:.4f}")


if __name__ == "__main__":
    main()
