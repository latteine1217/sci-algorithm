"""穩態不可壓 NS 殘差（velocity-pressure formulation）。

逐點以 autodiff 取一階/二階導數，再 vmap 批次化。
殘差：
  mom_x = u u_x + v u_y + p_x - (1/Re)(u_xx+u_yy)
  mom_y = u v_x + v v_y + p_y - (1/Re)(v_xx+v_yy)
  cont  = u_x + v_y

擴充點（未來研究）：若要換 formulation（stream-function、加湍流模型），
在此實作對應的場預測與殘差，並讓 losses 對「殘差函式」而非具體 NS 編程。
目前僅一種 formulation，依 YAGNI 不預先抽象。
"""
import jax
from .networks import predict


def _pointwise_residual(params, static, xy_pt, re):
    """xy_pt: (2,) -> (mom_x, mom_y, cont) 純量三元組。"""
    def f(pt):  # pt:(2,) -> (3,)
        return predict(params, static, pt[None, :])[0]

    val = f(xy_pt)                            # (3,)
    jac = jax.jacrev(f)(xy_pt)                # (3,2): d(u,v,p)/d(x,y)
    hess = jax.jacfwd(jax.jacrev(f))(xy_pt)   # (3,2,2)

    u, v = val[0], val[1]
    u_x, u_y = jac[0, 0], jac[0, 1]
    v_x, v_y = jac[1, 0], jac[1, 1]
    p_x, p_y = jac[2, 0], jac[2, 1]
    u_xx, u_yy = hess[0, 0, 0], hess[0, 1, 1]
    v_xx, v_yy = hess[1, 0, 0], hess[1, 1, 1]

    nu = 1.0 / re
    mom_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    mom_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    cont = u_x + v_y
    return mom_x, mom_y, cont


def ns_residuals(params, static, xy, re: float):
    """xy: (N,2) -> (rx, ry, rc) 各 (N,)。"""
    rx, ry, rc = jax.vmap(
        lambda pt: _pointwise_residual(params, static, pt, re)
    )(xy)
    return rx, ry, rc
