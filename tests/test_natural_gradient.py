import jax
import jax.numpy as jnp
from pinn_cavity.config import NetworkConfig
from pinn_cavity.networks import build_model
from pinn_cavity.natural_gradient import gn_step, residual_vector


def _setup():
    params, static = build_model(jax.random.PRNGKey(0),
                                 NetworkConfig(width=16, depth=3, n_fourier=8), lid_r=10.0)
    xy = jax.random.uniform(jax.random.PRNGKey(1), (256, 2))
    return params, static, xy


def _loss(p, static, xy, mode):
    return float(jnp.mean(residual_vector(p, static, xy, 1000.0, mode) ** 2))


def test_gn_step_reduces_loss():
    params, static, xy = _setup()
    l0 = _loss(params, static, xy, "taylor")
    p = params
    for _ in range(5):
        p, _ = gn_step(p, static, xy, 1000.0, "taylor", 1.0, 15, True, 1e-3)
    assert _loss(p, static, xy, "taylor") < 0.1 * l0  # 二階應大幅降 loss
    assert all(jnp.isfinite(x).all() for x in jax.tree_util.tree_leaves(p))


def test_linearize_matches_recompute():
    # 快速路徑（linearize）與 robust 路徑（vjp 重算）須給相同更新
    params, static, xy = _setup()
    p1, L1 = gn_step(params, static, xy, 1000.0, "taylor", 1.0, 15, True, 1e-3)
    p2, L2 = gn_step(params, static, xy, 1000.0, "taylor", 1.0, 15, False, 1e-3)
    assert jnp.allclose(L1, L2, atol=1e-6)
    for a, b in zip(jax.tree_util.tree_leaves(p1), jax.tree_util.tree_leaves(p2)):
        assert jnp.allclose(a, b, atol=1e-5)


def test_taylor_matches_hessian_in_gn():
    # GN 內用 taylor 或 hessian 算殘差，更新須一致
    params, static, xy = _setup()
    p1, _ = gn_step(params, static, xy, 1000.0, "taylor", 1.0, 15, True, 1e-3)
    p2, _ = gn_step(params, static, xy, 1000.0, "hessian", 1.0, 15, True, 1e-3)
    for a, b in zip(jax.tree_util.tree_leaves(p1), jax.tree_util.tree_leaves(p2)):
        assert jnp.allclose(a, b, atol=1e-5)
