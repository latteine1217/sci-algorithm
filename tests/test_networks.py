import jax
import jax.numpy as jnp
from pinn_cavity.networks import build_model, predict
from pinn_cavity.geometry import lid_profile


def _model(seed=0):
    from pinn_cavity.config import NetworkConfig
    cfg = NetworkConfig(width=16, depth=3, n_fourier=8, fourier_sigma=2.0)
    return build_model(jax.random.PRNGKey(seed), cfg, lid_r=10.0)


def test_predict_output_shape():
    params, static = _model()
    xy = jnp.array([[0.3, 0.7], [0.5, 0.5]])
    out = predict(params, static, xy)
    assert out.shape == (2, 3)


def test_fourier_B_not_in_trainable_params():
    # B 應在 static、不在可訓練 params（避免最佳化器預條件凍結參數）
    params, static = _model()
    assert "B" not in params
    assert static.B.shape == (2, 8)


def test_hard_bc_on_walls():
    params, static = _model(1)
    walls = jnp.array([[0.0, 0.4], [1.0, 0.6], [0.3, 0.0]])
    out = predict(params, static, walls)
    assert jnp.allclose(out[:, 0], 0.0, atol=1e-6)
    assert jnp.allclose(out[:, 1], 0.0, atol=1e-6)


def test_hard_bc_on_lid():
    params, static = _model(2)
    xs = jnp.array([0.25, 0.5, 0.75])
    lid = jnp.stack([xs, jnp.ones_like(xs)], axis=-1)
    out = predict(params, static, lid)
    assert jnp.allclose(out[:, 0], lid_profile(xs, r=10.0), atol=1e-6)
    assert jnp.allclose(out[:, 1], 0.0, atol=1e-6)


def test_params_are_pytree_of_arrays():
    params, _ = _model()
    leaves = jax.tree_util.tree_leaves(params)
    assert all(isinstance(l, jnp.ndarray) for l in leaves)
