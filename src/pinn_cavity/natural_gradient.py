"""Matrix-free Gauss-Newton / 自然梯度（ENGD 風格）for PINN。

PINN loss = mean(r²)，r 為 PDE 殘差向量、J=∂r/∂θ。Gauss-Newton 以 G=JᵀJ 近似
Hessian（對最小平方 loss，G 即 function-space Fisher → 自然梯度）。
解 (G + λI) δ = -Jᵀr，θ ← θ + lr·δ。

matrix-free（記憶體 O(P)，不 materialize J 或 G）：
- (JᵀJ)v 用 jvp（Jv，前向）+ vjp（Jᵀ·，反向）算，CG 只需此 matvec。
- 速度優化：jax.linearize 把殘差前向只算一次，CG 各迭代重用線性算子
  （linearize=True）；否則每迭代 jvp/vjp 重算前向（robust 後備）。
- damping λ：Levenberg-Marquardt 穩定化（G 半正定可能奇異）。
- 殘差用 taylor 模式 → 前向圖小 → matvec 便宜（與 jet 加速綜效）。
"""
import jax
import jax.numpy as jnp
from functools import partial
from .physics import ns_residuals


def residual_vector(params, static, xy, re, mode):
    """攤平所有 collocation 點的三項 PDE 殘差為 (3N,) 向量。"""
    rx, ry, rc = ns_residuals(params, static, xy, re=re, mode=mode)
    return jnp.concatenate([rx, ry, rc])


def _axpy(a, x, y):
    """a*x + y（pytree）。"""
    return jax.tree_util.tree_map(lambda xi, yi: a * xi + yi, x, y)


@partial(jax.jit, static_argnums=(4, 6, 7))
def gn_step(params, static, xy, re, mode="taylor",
            lr=1.0, cg_iters=20, linearize=True, damping=1e-3):
    """一步 matrix-free Gauss-Newton。回傳 (new_params, loss, cg_resid)。"""
    def rfn(theta):
        return residual_vector(theta, static, xy, re, mode)

    if linearize:
        r0, lin = jax.linearize(rfn, params)          # 前向只算一次；lin: pytree→R^M
        jt = jax.linear_transpose(lin, params)         # R^M→pytree（Jᵀ·）

        def matvec(v):
            jv = lin(v)
            jtjv = jt(jv)[0]
            return _axpy(damping, v, jtjv)
        g = jt(r0)[0]                                   # Jᵀr0
    else:
        r0, vjp_fn = jax.vjp(rfn, params)               # 後備：robust 但每迭代重算前向

        def matvec(v):
            jv = jax.jvp(rfn, (params,), (v,))[1]
            jtjv = jax.vjp(rfn, params)[1](jv)[0]
            return _axpy(damping, v, jtjv)
        g = vjp_fn(r0)[0]

    neg_g = jax.tree_util.tree_map(lambda x: -x, g)
    delta, _ = jax.scipy.sparse.linalg.cg(matvec, neg_g, maxiter=cg_iters)
    new_params = _axpy(lr, delta, params)
    return new_params, jnp.mean(r0 ** 2)
