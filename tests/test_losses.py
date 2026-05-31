import jax
import jax.numpy as jnp
from pinn_cavity.losses import loss_terms, total_loss, update_weights, init_weights, ema_blend
from pinn_cavity.networks import build_model
from pinn_cavity.config import NetworkConfig


def _setup():
    params, static = build_model(jax.random.PRNGKey(0), NetworkConfig(width=16, depth=3, n_fourier=8), lid_r=10.0)
    xy = jax.random.uniform(jax.random.PRNGKey(1), (64, 2))
    return params, static, xy


def test_loss_terms_nonnegative():
    params, static, xy = _setup()
    lx, ly, lc = loss_terms(params, static, xy, re=1000.0)
    assert lx >= 0 and ly >= 0 and lc >= 0


def test_total_loss_is_scalar():
    params, static, xy = _setup()
    L = total_loss(params, static, xy, init_weights(), re=1000.0)
    assert L.shape == ()


def test_total_loss_differentiable():
    params, static, xy = _setup()
    w = init_weights()
    g = jax.grad(lambda p: total_loss(p, static, xy, w, re=1000.0))(params)
    assert jnp.isfinite(jax.tree_util.tree_leaves(g)[0]).all()


def test_gradnorm_weights_positive_and_finite():
    params, static, xy = _setup()
    w = update_weights(params, static, xy, re=1000.0, method="gradnorm")
    assert all(float(v) > 0 and jnp.isfinite(v) for v in (w["x"], w["y"], w["c"]))


def test_fixed_weights_are_unit():
    params, static, xy = _setup()
    w = update_weights(params, static, xy, re=1000.0, method="fixed")
    assert all(float(v) == 1.0 for v in (w["x"], w["y"], w["c"]))


def test_ema_blend_slows_change():
    import jax.numpy as jnp
    old = {"x": jnp.asarray(1.0), "y": jnp.asarray(1.0), "c": jnp.asarray(1.0)}
    new = {"x": jnp.asarray(11.0), "y": jnp.asarray(1.0), "c": jnp.asarray(1.0)}
    w = ema_blend(old, new, alpha=0.9)
    assert abs(float(w["x"]) - 2.0) < 1e-6  # 0.9*1 + 0.1*11 = 2.0
    assert float(w["y"]) == 1.0
