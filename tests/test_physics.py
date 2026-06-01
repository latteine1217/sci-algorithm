import jax
import jax.numpy as jnp
from pinn_cavity.physics import ns_residuals
from pinn_cavity.networks import build_model, predict
from pinn_cavity.config import NetworkConfig


def _model(seed=0):
    cfg = NetworkConfig(width=16, depth=3, n_fourier=8)
    return build_model(jax.random.PRNGKey(seed), cfg, lid_r=10.0)


def test_residual_shapes():
    params, static = _model()
    xy = jax.random.uniform(jax.random.PRNGKey(3), (10, 2))
    rx, ry, rc = ns_residuals(params, static, xy, re=1000.0)
    assert rx.shape == (10,) and ry.shape == (10,) and rc.shape == (10,)


def test_continuity_matches_finite_difference():
    params, static = _model(1)
    p0 = jnp.array([[0.4, 0.6]])
    eps = 1e-4

    def vel(pt):
        return predict(params, static, pt)[0]

    ux = (vel(p0 + jnp.array([[eps, 0]]))[0] - vel(p0 - jnp.array([[eps, 0]]))[0]) / (2 * eps)
    vy = (vel(p0 + jnp.array([[0, eps]]))[1] - vel(p0 - jnp.array([[0, eps]]))[1]) / (2 * eps)
    _, _, rc = ns_residuals(params, static, p0, re=1000.0)
    assert jnp.allclose(rc[0], ux + vy, atol=1e-3)


def test_autodiff_modes_agree():
    # forward-over-reverse 與完整 Hessian 必須給相同殘差
    params, static = _model(2)
    xy = jax.random.uniform(jax.random.PRNGKey(7), (16, 2))
    a = ns_residuals(params, static, xy, re=1000.0, mode="fwd_over_rev")
    b = ns_residuals(params, static, xy, re=1000.0, mode="hessian")
    for ra, rb in zip(a, b):
        assert jnp.allclose(ra, rb, atol=1e-5)
