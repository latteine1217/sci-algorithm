"""損失組裝與自適應權重。

hard-BC 已免去 BC penalty，故僅平衡三項 PDE 殘差。
權重以 jnp 陣列保存並作為 traced 參數傳入訓練 step，避免 Python float
觸發 jit 重編譯或 host 同步。權重策略以 WEIGHTERS registry 設定驅動。
"""
import jax
import jax.numpy as jnp
from .physics import ns_residuals


def loss_terms(params, static, xy, re):
    """回傳 (Lx, Ly, Lc) 三項殘差 MSE 純量。"""
    rx, ry, rc = ns_residuals(params, static, xy, re=re)
    return jnp.mean(rx ** 2), jnp.mean(ry ** 2), jnp.mean(rc ** 2)


def init_weights():
    """等權，jnp 陣列形式。"""
    return {"x": jnp.asarray(1.0), "y": jnp.asarray(1.0), "c": jnp.asarray(1.0)}


def ema_blend(old, new, alpha):
    """權重 EMA 平滑：w ← alpha·old + (1-alpha)·new。alpha 越大變化越慢。"""
    return {k: alpha * old[k] + (1.0 - alpha) * new[k] for k in old}


def total_loss(params, static, xy, weights, re):
    lx, ly, lc = loss_terms(params, static, xy, re=re)
    return weights["x"] * lx + weights["y"] * ly + weights["c"] * lc


def _grad_norm(loss_fn, params):
    g = jax.grad(loss_fn)(params)
    leaves = jax.tree_util.tree_leaves(g)
    sq = sum(jnp.sum(l ** 2) for l in leaves)
    return jnp.sqrt(sq) + 1e-12


def _balance(sx, sy, sc):
    """以 mean/scale 平衡三項，回傳 jnp 權重 dict。"""
    mean = (sx + sy + sc) / 3.0
    return {"x": mean / sx, "y": mean / sy, "c": mean / sc}


WEIGHTERS = ("fixed", "gradnorm", "ntk")


def update_weights(params, static, xy, re, method="gradnorm"):
    """回傳新權重 dict（jnp 陣列）。"""
    if method == "fixed":
        return init_weights()
    if method not in ("gradnorm", "ntk"):
        raise ValueError(f"unknown weighting method: {method} (available: {WEIGHTERS})")
    gx = _grad_norm(lambda p: loss_terms(p, static, xy, re)[0], params)
    gy = _grad_norm(lambda p: loss_terms(p, static, xy, re)[1], params)
    gc = _grad_norm(lambda p: loss_terms(p, static, xy, re)[2], params)
    if method == "gradnorm":            # learning-rate annealing（Wang et al. 2021）
        return _balance(gx, gy, gc)
    return _balance(gx ** 2, gy ** 2, gc ** 2)  # ntk 對角近似（Wang et al. 2022）
