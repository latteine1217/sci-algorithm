"""幾何：距離函數、lid 正則化剖面、collocation 取樣策略。

距離函數 D 在四壁為 0，用於 hard-BC 包裝。
lid 剖面採 cosh 型，角點為 0、中心≈1，避免角點速度不連續。
取樣策略以 SAMPLERS registry 設定驅動，便於比較研究；Re=1000 邊界層
厚 ~1/sqrt(Re)≈0.03，boundary_refined 對壁面與上角點加密以免欠解析。
"""
import jax
import jax.numpy as jnp


# ---- 幾何基元 ----

def distance_fn(x, y):
    """D(x,y)=x(1-x)y(1-y)，四壁為 0、內部為正。"""
    return x * (1.0 - x) * y * (1.0 - y)


def lid_profile(x, r: float = 10.0):
    """g(x)=1 - cosh(r(x-0.5))/cosh(0.5 r)；g(0)=g(1)=0, g(0.5)=1-sech(0.5r)≈1。"""
    return 1.0 - jnp.cosh(r * (x - 0.5)) / jnp.cosh(0.5 * r)


# ---- 取樣策略 ----

def _uniform(key, n):
    return jax.random.uniform(key, shape=(n, 2), minval=0.0, maxval=1.0)


def _wall_band(key, n, width):
    """於隨機選定之壁面 width 帶內取 n 點，法向距離以 U^2 集中近壁。"""
    kw, kt, kn = jax.random.split(key, 3)
    wall = jax.random.randint(kw, (n,), 0, 4)
    t = jax.random.uniform(kt, (n,))                      # 切向 0..1
    d = width * jax.random.uniform(kn, (n,)) ** 2          # 法向，集中近壁
    x = jnp.where(wall == 0, d, jnp.where(wall == 1, 1.0 - d, t))   # 0:左 1:右
    y = jnp.where(wall == 2, d, jnp.where(wall == 3, 1.0 - d, t))   # 2:底 3:頂
    return jnp.stack([x, y], axis=-1)


def _top_corners(key, n, width):
    """於兩上角點 (0,1)、(1,1) 的 width 方塊內取 n 點（最劇梯度區）。"""
    ks, kx, ky = jax.random.split(key, 3)
    side = jax.random.randint(ks, (n,), 0, 2)            # 0:左上 1:右上
    dx = width * jax.random.uniform(kx, (n,))
    dy = width * jax.random.uniform(ky, (n,))
    x = jnp.where(side == 0, dx, 1.0 - dx)
    y = 1.0 - dy
    return jnp.stack([x, y], axis=-1)


def _boundary_refined(key, n, boundary_fraction, boundary_width, corner_fraction):
    """混合取樣：均勻 + 壁面帶 + 上角點。"""
    n_b = int(n * boundary_fraction)
    n_u = n - n_b
    n_c = int(n_b * corner_fraction)
    n_w = n_b - n_c
    k0, k1, k2 = jax.random.split(key, 3)
    parts = [_uniform(k0, n_u)]
    if n_w > 0:
        parts.append(_wall_band(k1, n_w, boundary_width))
    if n_c > 0:
        parts.append(_top_corners(k2, n_c, 2.0 * boundary_width))
    return jnp.concatenate(parts, axis=0)


SAMPLERS = ("uniform", "boundary_refined")


def make_sampler(sampler_cfg):
    """依 SamplerConfig 回傳取樣函式 fn(key, n) -> (n,2)。"""
    name = sampler_cfg.name
    if name == "uniform":
        return lambda key, n: _uniform(key, n)
    if name == "boundary_refined":
        return lambda key, n: _boundary_refined(
            key, n,
            sampler_cfg.boundary_fraction,
            sampler_cfg.boundary_width,
            sampler_cfg.corner_fraction,
        )
    raise ValueError(f"unknown sampler: {name} (available: {SAMPLERS})")


def sample_collocation(key, n: int):
    """便捷：均勻取樣 n 個內部點，形狀 (n,2)。"""
    return _uniform(key, n)
