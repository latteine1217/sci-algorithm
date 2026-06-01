"""穩態不可壓 NS 殘差（velocity-pressure formulation）。

殘差：
  mom_x = u u_x + v u_y + p_x - (1/Re)(u_xx+u_yy)
  mom_y = u v_x + v v_y + p_y - (1/Re)(v_xx+v_yy)
  cont  = u_x + v_y

二階導兩種模式（config.autodiff）：
- "fwd_over_rev"（預設）：forward-over-reverse 只取 Laplacian trace + 一階導，
  不 materialize 完整 Hessian。輸入維度小、需高階導時較省記憶體/算力，
  優勢隨維度 d 增大（2D 常數倍，3D/4D 顯著）。
- "hessian"：jax.jacfwd(jax.jacrev) 完整 Hessian（仍是 forward-over-reverse，
  但算了不需要的 p 二階與交叉項）。保留供 A/B 對照。

擴充點：若換 formulation，於此實作對應殘差。
"""
import jax
import jax.numpy as jnp
from jax.experimental import jet
from .networks import predict


def _field_fn(params, static):
    def f(pt):  # (2,) -> (u,v,p)
        return predict(params, static, pt[None, :])[0]
    return f


def _residual_fwd_over_rev(params, static, xy_pt, re):
    """forward-over-reverse：一次取 value、Jacobian、(u,v,p) 的 Laplacian。"""
    f = _field_fn(params, static)
    jac_f = jax.jacrev(f)                       # (2,) -> (3,2)
    e0 = jnp.array([1.0, 0.0], dtype=xy_pt.dtype)
    e1 = jnp.array([0.0, 1.0], dtype=xy_pt.dtype)
    # jvp over jacrev：primal 給 Jacobian，tangent 給 Jacobian 沿 e 的方向導 = Hessian 該列
    jac, h0 = jax.jvp(jac_f, (xy_pt,), (e0,))   # h0[k,i] = ∂²f_k/∂x_i∂x
    _, h1 = jax.jvp(jac_f, (xy_pt,), (e1,))      # h1[k,i] = ∂²f_k/∂x_i∂y
    val = f(xy_pt)
    lap = h0[:, 0] + h1[:, 1]                    # (3,) = ∂²/∂x² + ∂²/∂y²（只取 trace）
    return _assemble(val, jac, lap, re)


def _residual_hessian(params, static, xy_pt, re):
    """完整 Hessian 版（jacfwd∘jacrev），供對照。"""
    f = _field_fn(params, static)
    val = f(xy_pt)
    jac = jax.jacrev(f)(xy_pt)                   # (3,2)
    hess = jax.jacfwd(jax.jacrev(f))(xy_pt)      # (3,2,2)
    lap = jnp.stack([hess[0, 0, 0] + hess[0, 1, 1],
                     hess[1, 0, 0] + hess[1, 1, 1],
                     hess[2, 0, 0] + hess[2, 1, 1]])
    return _assemble(val, jac, lap, re)


def _residual_taylor(params, static, xy_pt, re):
    """Taylor-mode（Forward-Laplacian, jax.experimental.jet）：

    沿各座標單位向量 e_i 做二階 Taylor 前向，一次拿到 value、Jacobian 第 i 列、
    與二階方向導 D²f(e_i,e_i)；對 i 加總即 Laplacian。前向模式、不建二階反向圖，
    高維/高階時記憶體與算力優勢顯著（STDE / Forward Laplacian）。
    """
    f = _field_fn(params, static)
    n = xy_pt.shape[0]
    eye = jnp.eye(n, dtype=xy_pt.dtype)

    def along(e):
        f0, series = jet.jet(f, (xy_pt,), ((e, jnp.zeros_like(e)),))
        d1, d2 = series[0], series[1]   # d1=Df·e（Jacobian 第 i 列）, d2=D²f(e,e)
        return f0, d1, d2

    f0, d1, d2 = jax.vmap(along)(eye)   # (n,3) each
    val = f0[0]
    jac = d1.T                          # (3,n): jac[k,i]=Df_k·e_i
    lap = d2.sum(axis=0)               # (3,) = Σ_i D²f(e_i,e_i)
    return _assemble(val, jac, lap, re)


def _assemble(val, jac, lap, re):
    u, v = val[0], val[1]
    u_x, u_y = jac[0, 0], jac[0, 1]
    v_x, v_y = jac[1, 0], jac[1, 1]
    p_x, p_y = jac[2, 0], jac[2, 1]
    nu = 1.0 / re
    mom_x = u * u_x + v * u_y + p_x - nu * lap[0]
    mom_y = u * v_x + v * v_y + p_y - nu * lap[1]
    cont = u_x + v_y
    return mom_x, mom_y, cont


_MODES = {"fwd_over_rev": _residual_fwd_over_rev, "hessian": _residual_hessian,
          "taylor": _residual_taylor}


def ns_residuals(params, static, xy, re: float, mode: str = "fwd_over_rev"):
    """xy: (N,2) -> (rx, ry, rc) 各 (N,)。mode 見 config.autodiff。"""
    if mode not in _MODES:
        raise ValueError(f"unknown autodiff mode: {mode} (available: {tuple(_MODES)})")
    fn = _MODES[mode]
    rx, ry, rc = jax.vmap(lambda pt: fn(params, static, pt, re))(xy)
    return rx, ry, rc
