"""最佳化器組裝：以 OPTIMIZERS registry 設定驅動，便於比較研究。

預設 SOAP（haydn-jones/SOAP_JAX, optax 相容）+ cosine 學習率排程。
可選 Muon（Nesterov momentum + Newton-Schulz 正交化，Keller Jordan et al. 2024）。
"""
from typing import NamedTuple, Any
import jax
import jax.numpy as jnp
import optax
from soap_jax import soap
from .soap_batched import soap_batched


# ---------------------------------------------------------------------------
# SOAP
# ---------------------------------------------------------------------------

def _build_soap(opt_cfg):
    schedule = optax.cosine_decay_schedule(
        init_value=opt_cfg.learning_rate,
        decay_steps=max(1, opt_cfg.decay_steps),
        alpha=0.01,
    )
    # SOAP 內部 QR 預設 float32；x64 下需對齊 float64，否則 lax.cond
    # 兩分支（refresh vs keep preconditioner）dtype 不一致而報錯。
    qr_dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    return soap(
        learning_rate=schedule,
        b1=opt_cfg.b1,
        b2=opt_cfg.b2,
        weight_decay=opt_cfg.weight_decay,
        precondition_frequency=opt_cfg.precondition_frequency,
        qr_dtype=qr_dtype,
    )


# ---------------------------------------------------------------------------
# Muon
# ---------------------------------------------------------------------------

def _ns_ortho(G: jnp.ndarray, steps: int = 5) -> jnp.ndarray:
    """Newton-Schulz 正交化：以五次多項式迭代逼近 G 的 polar factor。

    輸入 G: (m, n)；輸出近似 U（G = U Σ Vᵀ 的左奇異向量矩陣側）。
    Frobenius norm ≈ sqrt(min(m, n))。

    係數 (a=3.4445, b=-4.7750, c=2.0315) 來自 Keller Jordan et al. 2024，
    針對奇異值範圍 [0.1, 1] 最小化多項式近似誤差的最優解。

    注意：shapes 在 JIT 編譯期為靜態值，`if transposed` 是 Python 分支（合法）。
    """
    transposed = G.shape[0] > G.shape[1]
    X = G.T if transposed else G            # 保證 m <= n
    X = X / (jnp.linalg.norm(X) + 1e-7)   # 正規化使最大奇異值 ≤ 1
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.T                         # (m, m)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


class MuonState(NamedTuple):
    """Muon 狀態：step count + Nesterov buffer + Adam fallback（1D params 用）。"""
    count: jnp.ndarray    # int32 step counter
    momentum: Any         # Nesterov momentum buffer，與 params 同結構
    adam_m: Any           # Adam first moment（1D fallback）
    adam_v: Any           # Adam second moment（1D fallback）


def _build_muon(opt_cfg) -> optax.GradientTransformation:
    """Muon optimizer（Keller Jordan et al. 2024）。

    演算法：
      1. Nesterov momentum: m_t = β·m_{t-1} + g_t; ĝ_t = g_t + β·m_t
      2. 2D params: update = -lr · NS(ĝ_t)
         （NS 正交化使奇異值 ≈ 1；不加 sqrt(max_dim)，小網路已 overscale）
      3. 1D params（bias 等）: Adam fallback（相同 lr）

    主要優勢相對 SOAP：
      - NS 迭代全為 matmul，無 QR 三角求解，可跨同形狀層 vmap（減少 kernel 數）
      - 理論上 kernel launch overhead 顯著低於 SOAP 逐層 QR
    """
    schedule = optax.cosine_decay_schedule(
        init_value=opt_cfg.learning_rate,
        decay_steps=max(1, opt_cfg.decay_steps),
        alpha=0.01,
    )
    ns = getattr(opt_cfg, "muon_ns_steps", 5)
    mom = getattr(opt_cfg, "muon_momentum", 0.95)
    adam_b1 = opt_cfg.b1
    adam_b2 = opt_cfg.b2
    adam_eps = 1e-8

    def init_fn(params):
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        return MuonState(
            count=jnp.zeros([], jnp.int32),
            momentum=jax.tree_util.tree_map(jnp.zeros_like, params),
            adam_m=zeros,
            adam_v=jax.tree_util.tree_map(jnp.zeros_like, params),
        )

    def update_fn(grads, state, params=None):
        count = state.count + 1
        lr = schedule(state.count)          # cosine-decayed lr（動態值）

        # Nesterov momentum（所有層共用）
        new_buf = jax.tree_util.tree_map(
            lambda b, g: mom * b + g, state.momentum, grads
        )
        g_eff = jax.tree_util.tree_map(     # Nesterov lookahead
            lambda g, b: g + mom * b, grads, new_buf
        )

        # Adam 狀態（1D fallback 用，所有層更新以維持結構一致）
        new_am = jax.tree_util.tree_map(
            lambda m, g: adam_b1 * m + (1 - adam_b1) * g, state.adam_m, grads
        )
        new_av = jax.tree_util.tree_map(
            lambda v, g: adam_b2 * v + (1 - adam_b2) * g ** 2, state.adam_v, grads
        )
        bc1 = 1.0 - adam_b1 ** count.astype(jnp.float32)
        bc2 = 1.0 - adam_b2 ** count.astype(jnp.float32)

        def per_leaf(ge, am, av):
            # ndim 是靜態值，Python if 在 JIT 編譯期決定分支，不產生 lax.cond。
            if ge.ndim >= 2:
                # Muon path：NS 正交化，不加 sqrt(max_dim) scale。
                # 去掉原因：NS 已把所有奇異值壓成 ~1（失去 adaptive scaling），
                # 再乘 sqrt(max_dim) 給 128×5 MLP 帶來 10–20× overscaled step
                # → it=1000 起即振盪爆掉（job 3915 診斷）。
                # step size 完全由 cosine-decayed lr（1e-3→1e-5）控制。
                u = _ns_ortho(ge, ns)
                return -lr * u
            else:
                # Adam fallback（bias / 1D params）
                m_hat = am / bc1
                v_hat = av / bc2
                return -lr * m_hat / (jnp.sqrt(v_hat) + adam_eps)

        updates = jax.tree_util.tree_map(per_leaf, g_eff, new_am, new_av)
        new_state = MuonState(count=count, momentum=new_buf, adam_m=new_am, adam_v=new_av)
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _build_soap_batched(opt_cfg):
    """SOAP 批次化版本：同形狀 2D 層的 per-step matmul 用 vmap 批次化。
    QR refresh 保持 sequential（bench_batching 證實批次 QR 反而 12× 更慢）。
    """
    schedule = optax.cosine_decay_schedule(
        init_value=opt_cfg.learning_rate,
        decay_steps=max(1, opt_cfg.decay_steps),
        alpha=0.01,
    )
    qr_dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    return soap_batched(
        learning_rate=schedule,
        b1=opt_cfg.b1,
        b2=opt_cfg.b2,
        weight_decay=opt_cfg.weight_decay,
        precondition_frequency=opt_cfg.precondition_frequency,
        qr_dtype=qr_dtype,
    )


OPTIMIZERS = {
    "soap": _build_soap,
    "soap_batched": _build_soap_batched,
    "muon": _build_muon,
}


def build_optimizer(opt_cfg):
    """依 OptimizerConfig.name 回傳 optax GradientTransformation。"""
    if opt_cfg.name not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer: {opt_cfg.name} "
                         f"(available: {tuple(OPTIMIZERS)})")
    return OPTIMIZERS[opt_cfg.name](opt_cfg)
