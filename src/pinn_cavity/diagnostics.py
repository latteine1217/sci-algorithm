"""場診斷與指標彙整（純計算，無繪圖）。

對齊 CFD 評估 rubric：
- 連續/質量守恆：散度 ∇·u（PINN 的 mass imbalance 類比）
- 物理特徵：渦量 ω=v_x−u_y、流函數 ψ、主渦/次渦
- 數值保真：PDE 殘差場（在比訓練更密的網格上評估 → 解析度充足性類比）
- 對照：Ghia 中線相對 L2 + max error
"""
import numpy as np
import jax
import jax.numpy as jnp
from .networks import predict
from .physics import ns_residuals
from . import reference as ref


def _grid(n):
    xs = np.linspace(0.0, 1.0, n); ys = np.linspace(0.0, 1.0, n)
    XX, YY = np.meshgrid(xs, ys)
    pts = jnp.stack([jnp.array(XX.ravel()), jnp.array(YY.ravel())], axis=-1)
    return XX, YY, pts


def compute_fields(params, static, re, n=120):
    """回傳網格場 dict：U,V,P(錨定),div,vort,rx,ry,rc,psi。"""
    XX, YY, pts = _grid(n)
    vals = np.array(predict(params, static, pts))
    U = vals[:, 0].reshape(n, n); V = vals[:, 1].reshape(n, n)
    P = vals[:, 2].reshape(n, n); P = P - P.mean()

    # 一階導 → 散度、渦量
    def f(pt):
        return predict(params, static, pt[None, :])[0]
    jac = np.array(jax.vmap(jax.jacrev(f))(pts))   # (N,3,2)
    u_x, u_y = jac[:, 0, 0], jac[:, 0, 1]
    v_x, v_y = jac[:, 1, 0], jac[:, 1, 1]
    div = (u_x + v_y).reshape(n, n)
    vort = (v_x - u_y).reshape(n, n)

    # PDE 殘差場（含二階導）
    rx, ry, rc = ns_residuals(params, static, pts, re)
    rx = np.array(rx).reshape(n, n); ry = np.array(ry).reshape(n, n); rc = np.array(rc).reshape(n, n)

    psi = stream_function(U, n)
    return {"XX": XX, "YY": YY, "U": U, "V": V, "P": P,
            "div": div, "vort": vort, "rx": rx, "ry": ry, "rc": rc, "psi": psi, "n": n}


def stream_function(U, n):
    """ψ(x,y)=∫_0^y u dy'（ψ=0 於底壁）。沿 y 軸（axis 0）梯形積分。"""
    dy = 1.0 / (n - 1)
    psi = np.zeros_like(U)
    incr = 0.5 * (U[1:, :] + U[:-1, :]) * dy
    psi[1:, :] = np.cumsum(incr, axis=0)
    return psi


def _interior_argext(field, XX, YY, margin, mode):
    """於內部區（離壁 > margin）找極值點，回傳 (x,y,value)。"""
    interior = (XX > margin) & (XX < 1 - margin) & (YY > margin) & (YY < 1 - margin)
    masked = np.where(interior, field, np.nan)
    idx = np.nanargmin(masked) if mode == "min" else np.nanargmax(masked)
    j, i = np.unravel_index(idx, field.shape)
    return float(XX[j, i]), float(YY[j, i]), float(field[j, i])


def detect_vortices(fields):
    """主渦（ψ 最負）+ 底部兩角落次渦（ψ 反號局部極值）。"""
    XX, YY, psi = fields["XX"], fields["YY"], fields["psi"]
    px, py, pmin = _interior_argext(psi, XX, YY, 0.1, "min")
    # 角落次渦：與主渦反號 → 找區域內 ψ 最大
    def corner(xlo, xhi):
        reg = (XX > xlo) & (XX < xhi) & (YY > 0.0) & (YY < 0.3)
        m = np.where(reg, psi, -np.inf)
        idx = np.argmax(m); j, i = np.unravel_index(idx, psi.shape)
        return {"x": float(XX[j, i]), "y": float(YY[j, i]), "psi": float(psi[j, i])}
    bl = corner(0.0, 0.3); br = corner(0.7, 1.0)
    # 存在性：角落 ψ 與主渦反號且量值非微小
    thr = 1e-4 * abs(pmin)
    return {
        "primary": {"x": px, "y": py, "psi": pmin},
        "BL1": {**bl, "present": bool(bl["psi"] > thr)},
        "BR1": {**br, "present": bool(br["psi"] > thr)},
    }


def centerline(params, static, n=201):
    """密中線 u(y@x=0.5)、v(x@y=0.5) 的 PINN 預測 + DNS 參考（同點位）。"""
    dns = ref.dns_centerline(n)
    t = dns["y"]
    pu = jnp.stack([jnp.full_like(jnp.array(t), 0.5), jnp.array(t)], axis=-1)
    pv = jnp.stack([jnp.array(t), jnp.full_like(jnp.array(t), 0.5)], axis=-1)
    u_pred = np.array(predict(params, static, pu)[:, 0])
    v_pred = np.array(predict(params, static, pv)[:, 1])
    return {"y": t, "u_pred": u_pred, "u_ref": dns["u"],
            "x": t, "v_pred": v_pred, "v_ref": dns["v"]}


def _rel_l2(a, b):
    a = np.asarray(a); b = np.asarray(b)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def _max_err(a, b):
    a = np.asarray(a); b = np.asarray(b)
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[mask] - b[mask]))) if mask.any() else float("nan")


def aggregate_metrics(params, static, fields):
    """彙整跨演算法可比的標量指標（對照 DNS）。"""
    cl = centerline(params, static)
    l2_u = _rel_l2(cl["u_pred"], cl["u_ref"]); l2_v = _rel_l2(cl["v_pred"], cl["v_ref"])
    maxerr_u = _max_err(cl["u_pred"], cl["u_ref"]); maxerr_v = _max_err(cl["v_pred"], cl["v_ref"])

    # 全場 rel-L2 vs DNS（DNS 唯一優勢：可比整個域，非僅中線）
    XX, YY = fields["XX"], fields["YY"]
    dU, dV, _ = ref.dns_field(XX, YY)
    f_l2_u = _rel_l2(fields["U"], dU); f_l2_v = _rel_l2(fields["V"], dV)
    spd_p = np.sqrt(fields["U"] ** 2 + fields["V"] ** 2)
    spd_d = np.sqrt(np.nan_to_num(dU) ** 2 + np.nan_to_num(dV) ** 2)
    f_l2_speed = _rel_l2(spd_p, spd_d)

    div = fields["div"]
    rx, ry, rc = fields["rx"], fields["ry"], fields["rc"]
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    vor = detect_vortices(fields)
    pv = vor["primary"]
    dns_vx, dns_vy = ref.dns_primary_vortex()
    vortex_err = float(np.hypot(pv["x"] - dns_vx, pv["y"] - dns_vy))
    return {
        "rel_l2_u": l2_u, "rel_l2_v": l2_v,
        "max_err_u": maxerr_u, "max_err_v": maxerr_v,
        "field_rel_l2_u": f_l2_u, "field_rel_l2_v": f_l2_v, "field_rel_l2_speed": f_l2_speed,
        "divergence_max": float(np.max(np.abs(div))),
        "divergence_mean": float(np.mean(np.abs(div))),
        "residual_rms_momx": rms(rx), "residual_rms_momy": rms(ry), "residual_rms_cont": rms(rc),
        "residual_max_cont": float(np.max(np.abs(rc))),
        "vorticity_min": float(np.min(fields["vort"])), "vorticity_max": float(np.max(fields["vort"])),
        "primary_vortex": [pv["x"], pv["y"]], "primary_psi": pv["psi"],
        "primary_vortex_pos_err": vortex_err, "dns_primary_vortex": [dns_vx, dns_vy],
        "secondary_BL_present": vor["BL1"]["present"], "secondary_BR_present": vor["BR1"]["present"],
        "secondary_BL": [vor["BL1"]["x"], vor["BL1"]["y"]],
        "secondary_BR": [vor["BR1"]["x"], vor["BR1"]["y"]],
    }
