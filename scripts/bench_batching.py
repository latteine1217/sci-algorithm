"""量測 SOAP batching 的實際加速潛力。

關鍵假設：soap-jax 用 jtu.tree_map 對每個 leaf 獨立做 matmul/QR，
同形狀的 hidden layer (128×128 × 3) 若改成 batched op 可省 kernel launch 開銷。

本腳本量測：
  A. N 個獨立 (m,m) matmul（現況）
  B. 1 個 (N,m,m) batched matmul（改後）
  C. N 個獨立 (m,m) QR
  D. jax.vmap over N (m,m) QR

對應我們 MLP 的實際 SOAP 熱路徑：
  - hidden 128×128 × 3 层：update_preconditioner / project / project_back / QR refresh

結論指標：若 batched/sequential speedup < 1.3×，batching ROI 不足，不值得實作。

用法：
  uv run python scripts/bench_batching.py
"""
import time
import jax
import jax.numpy as jnp

N = 3     # 三個 hidden 128×128 層
M = 128   # hidden width

def timed(fn, warmup=10, reps=100, label=""):
    for _ in range(warmup):
        out = fn(); jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(); jax.block_until_ready(out)
    ms = (time.perf_counter() - t0) / reps * 1000
    print(f"  {label:<52} {ms:7.3f} ms")
    return ms


def main():
    key = jax.random.PRNGKey(0)
    As = [jax.random.normal(key, (M, M)) for _ in range(N)]
    Bs = [jax.random.normal(key, (M, M)) for _ in range(N)]
    A_bat = jnp.stack(As)   # (N, M, M)
    B_bat = jnp.stack(Bs)   # (N, M, M)

    print(f"=== Batching benchmark: backend={jax.default_backend()} "
          f"N={N} M={M} ===")

    # ── matmul：SOAP update_preconditioner / project 的核心 ──────────────
    print("\n--- matmul (A @ B), 代表 outer product / projection ---")

    @jax.jit
    def seq_mm():
        return [A @ B for A, B in zip(As, Bs)]

    @jax.jit
    def bat_mm():
        return jnp.einsum('bij,bjk->bik', A_bat, B_bat)

    @jax.jit
    def vmap_mm():
        return jax.vmap(jnp.matmul)(A_bat, B_bat)

    t_seq = timed(seq_mm, label=f"sequential: {N}x ({M},{M}) matmul")
    t_bat = timed(bat_mm, label=f"batched einsum: ({N},{M},{M}) matmul")
    t_vmap = timed(vmap_mm, label=f"vmap matmul: ({N},{M},{M})")
    print(f"  speedup batched/seq = {t_seq/t_bat:.2f}×   vmap/seq = {t_seq/t_vmap:.2f}×")

    # ── QR：SOAP refresh_preconditioner 的核心 ──────────────────────────
    print("\n--- QR decomposition, 代表 preconditioner refresh ---")

    @jax.jit
    def seq_qr():
        return [jnp.linalg.qr(A)[0] for A in As]

    @jax.jit
    def bat_qr():
        return jnp.linalg.qr(A_bat)[0]

    @jax.jit
    def vmap_qr():
        return jax.vmap(lambda A: jnp.linalg.qr(A)[0])(A_bat)

    t_seq = timed(seq_qr, label=f"sequential: {N}x ({M},{M}) QR")
    t_bat = timed(bat_qr, label=f"batched jnp.linalg.qr: ({N},{M},{M})")
    t_vmap = timed(vmap_qr, label=f"vmap QR: ({N},{M},{M})")
    print(f"  speedup batched/seq = {t_seq/t_bat:.2f}×   vmap/seq = {t_seq/t_vmap:.2f}×")

    # ── eigh（SOAP 的 init 用 eigh，QR refresh 後主要是 QR）──────────────
    print("\n--- eigh（init_conditioner 用） ---")
    # GG matrices (Gram matrices，symmetric positive semi-definite)
    GGs = [A @ A.T + 1e-6 * jnp.eye(M) for A in As]
    GG_bat = jnp.stack(GGs)

    @jax.jit
    def seq_eigh():
        return [jnp.linalg.eigh(G) for G in GGs]

    @jax.jit
    def bat_eigh():
        return jnp.linalg.eigh(GG_bat)

    @jax.jit
    def vmap_eigh():
        return jax.vmap(jnp.linalg.eigh)(GG_bat)

    t_seq = timed(seq_eigh, label=f"sequential: {N}x ({M},{M}) eigh")
    t_bat = timed(bat_eigh, label=f"batched jnp.linalg.eigh: ({N},{M},{M})")
    t_vmap = timed(vmap_eigh, label=f"vmap eigh: ({N},{M},{M})")
    print(f"  speedup batched/seq = {t_seq/t_bat:.2f}×   vmap/seq = {t_seq/t_vmap:.2f}×")

    # ── 全套 SOAP per-step 動作（N 個 128×128 層的熱路徑）────────────────
    print("\n--- SOAP per-step hot-path（6× matmul + 2× outer product）---")
    Qs = [jax.random.normal(key, (M, M)) for _ in range(N)]
    Gs = [jax.random.normal(key, (M, M)) for _ in range(N)]
    GGs2 = [A @ A.T for A in As]
    Q_bat = jnp.stack(Qs)
    G_bat = jnp.stack(Gs)
    GG_bat2 = jnp.stack(GGs2)

    @jax.jit
    def seq_full():
        out = []
        for Q, G, GG in zip(Qs, Gs, GGs2):
            # project: Q.T @ G @ Q
            G_proj = Q.T @ G @ Q
            # update_preconditioner outer products (2×, one per dim)
            outer_L = G @ G.T
            outer_R = G.T @ G
            # GG update
            new_GG_L = 0.95 * GG + 0.05 * outer_L
            new_GG_R = 0.95 * GG + 0.05 * outer_R
            out.append((G_proj, new_GG_L, new_GG_R))
        return out

    @jax.jit
    def vmap_full():
        def single(Q, G, GG):
            G_proj = Q.T @ G @ Q
            outer_L = G @ G.T
            outer_R = G.T @ G
            new_GG_L = 0.95 * GG + 0.05 * outer_L
            new_GG_R = 0.95 * GG + 0.05 * outer_R
            return G_proj, new_GG_L, new_GG_R
        return jax.vmap(single)(Q_bat, G_bat, GG_bat2)

    t_seq = timed(seq_full, label=f"sequential {N}× full SOAP step ops")
    t_vmap = timed(vmap_full, label=f"vmap {N}× full SOAP step ops")
    print(f"  speedup vmap/seq = {t_seq/t_vmap:.2f}×")

    # ── 結論 ─────────────────────────────────────────────────────────────
    print("\n=== 結論 ===")
    if t_seq / t_vmap > 1.3:
        print(f"  vmap batching 有效：{t_seq/t_vmap:.2f}× speedup → 值得實作")
    else:
        print(f"  vmap batching 收益有限：{t_seq/t_vmap:.2f}× → 考慮其他方向")
    print("  (hidden layers 占 SOAP 約 40-60%，全局提升約 speedup×0.5)")


if __name__ == "__main__":
    main()
