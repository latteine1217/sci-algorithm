"""網路：Random Fourier Features + Modified MLP（Wang et al. 2021）+ hard-BC 包裝。

可訓練參數以純 dict pytree 手寫，保持 jax.grad/jit/vmap 透明。
固定量（Fourier 矩陣 B、lid 陡峭度 r）裝進 NetStatic 與 params 分離，
避免最佳化器（如 SOAP）為凍結參數配置/預條件狀態，也讓 grad 目標乾淨。
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .geometry import distance_fn, lid_profile


class NetStatic(NamedTuple):
    """網路固定量（非訓練）。B: (2, n_fourier) Fourier 矩陣；lid_r: cosh 陡峭度。"""
    B: jnp.ndarray
    lid_r: float


def _glorot(key, shape):
    fan_in, fan_out = shape[0], shape[1]
    scale = jnp.sqrt(2.0 / (fan_in + fan_out))
    return jax.random.normal(key, shape) * scale


def init_params(key, net_cfg):
    """初始化 Modified MLP 可訓練參數（不含 B）。輸入維度 = 2*n_fourier。"""
    width, depth, nf = net_cfg.width, net_cfg.depth, net_cfg.n_fourier
    keys = jax.random.split(key, 5 + depth)
    in_dim = 2 * nf
    params = {}
    params["Wu"] = _glorot(keys[0], (in_dim, width)); params["bu"] = jnp.zeros(width)
    params["Wv"] = _glorot(keys[1], (in_dim, width)); params["bv"] = jnp.zeros(width)
    params["W0"] = _glorot(keys[2], (in_dim, width)); params["b0"] = jnp.zeros(width)
    params["Wh"] = []; params["bh"] = []
    for i in range(depth - 1):
        params["Wh"].append(_glorot(keys[3 + i], (width, width)))
        params["bh"].append(jnp.zeros(width))
    params["Wout"] = _glorot(keys[-1], (width, 3)); params["bout"] = jnp.zeros(3)
    return params


def build_model(key, net_cfg, lid_r: float):
    """建立 (params, static)。params 可訓練；static 固定（B, lid_r）。"""
    k_b, k_p = jax.random.split(key)
    B = jax.random.normal(k_b, (2, net_cfg.n_fourier)) * net_cfg.fourier_sigma
    params = init_params(k_p, net_cfg)
    return params, NetStatic(B=B, lid_r=lid_r)


def _fourier(B, xy):
    """xy: (...,2) -> (...,2*nf) cos/sin 特徵。"""
    proj = 2.0 * jnp.pi * (xy @ B)
    return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)


def forward(params, static, xy):
    """Modified MLP 原始輸出 (u_hat,v_hat,p_hat)，xy: (...,2) -> (...,3)。"""
    h_in = _fourier(static.B, xy)
    U = jnp.tanh(h_in @ params["Wu"] + params["bu"])
    V = jnp.tanh(h_in @ params["Wv"] + params["bv"])
    H = jnp.tanh(h_in @ params["W0"] + params["b0"])
    for Wh, bh in zip(params["Wh"], params["bh"]):
        Z = jnp.tanh(H @ Wh + bh)
        H = (1.0 - Z) * U + Z * V
    return H @ params["Wout"] + params["bout"]


def predict(params, static, xy):
    """套 hard-BC 後的物理輸出 (u,v,p)，xy: (...,2) -> (...,3)。"""
    raw = forward(params, static, xy)
    x = xy[..., 0]; y = xy[..., 1]
    g = lid_profile(x, r=static.lid_r)
    D = distance_fn(x, y)
    u = y * g + D * raw[..., 0]
    v = D * raw[..., 1]
    p = raw[..., 2]
    return jnp.stack([u, v, p], axis=-1)
