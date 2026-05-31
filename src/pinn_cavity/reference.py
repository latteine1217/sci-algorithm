"""DNS 參考解（Lethe steady, Re=1000, cosh r=10 lid，與 PINN 同剖面）。

來源見 reference/README.md。取代 Ghia 1982（sharp lid）以達 apples-to-apples 驗證：
DNS 用與 PINN 相同的 g(x)=1-cosh(10(x-0.5))/cosh(5) lid，消除 lid 剖面 modeling 落差。
提供密中線、全場插值、主渦心，供 evaluate/diagnostics 對照。
"""
import functools
import importlib.resources
import numpy as np
from scipy.interpolate import LinearNDInterpolator


@functools.lru_cache(maxsize=1)
def _data():
    res = importlib.resources.files("pinn_cavity.data").joinpath("dns_re1000_r10.npz")
    with res.open("rb") as f:
        d = np.load(f)
        return {k: d[k] for k in d.files}


@functools.lru_cache(maxsize=1)
def _interp():
    d = _data()
    pts = np.c_[d["x"], d["y"]]
    return (LinearNDInterpolator(pts, d["u"]),
            LinearNDInterpolator(pts, d["v"]),
            LinearNDInterpolator(pts, d["p"]))


def dns_centerline(n=201):
    """密中線：u(y@x=0.5)、v(x@y=0.5)。"""
    iu, iv, _ = _interp()
    t = np.linspace(0.0, 1.0, n)
    return {"y": t, "u": np.asarray(iu(np.c_[np.full(n, 0.5), t])),
            "x": t, "v": np.asarray(iv(np.c_[t, np.full(n, 0.5)]))}


def dns_field(XX, YY):
    """插值 DNS (u,v,p) 到給定網格；凸包外回 NaN（域為 [0,1]² 故僅邊界可能）。"""
    iu, iv, ip = _interp()
    pts = np.c_[XX.ravel(), YY.ravel()]
    return (iu(pts).reshape(XX.shape), iv(pts).reshape(XX.shape), ip(pts).reshape(XX.shape))


@functools.lru_cache(maxsize=1)
def dns_primary_vortex(n=129):
    """DNS 主渦心：stream-function 最負處（內部）。"""
    xs = np.linspace(0, 1, n); ys = np.linspace(0, 1, n)
    XX, YY = np.meshgrid(xs, ys)
    U, V, _ = dns_field(XX, YY)
    U = np.nan_to_num(U)
    dy = 1.0 / (n - 1)
    psi = np.zeros_like(U); psi[1:, :] = np.cumsum(0.5 * (U[1:, :] + U[:-1, :]) * dy, axis=0)
    interior = (XX > 0.1) & (XX < 0.9) & (YY > 0.1) & (YY < 0.9)
    m = np.where(interior, psi, np.nan)
    j, i = np.unravel_index(np.nanargmin(m), psi.shape)
    return float(XX[j, i]), float(YY[j, i])
