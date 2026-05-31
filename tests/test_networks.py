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


def _rwf_cfg():
    from pinn_cavity.config import NetworkConfig
    return NetworkConfig(width=16, depth=3, n_fourier=8, rwf=True, rwf_mu=1.0, rwf_sigma=0.1)


def test_rwf_off_has_no_scale_params():
    params, _ = _model()  # 預設 rwf=False
    assert "Wu_g" not in params and "Wh_g" not in params


def test_rwf_on_has_scale_params_and_finite_output():
    params, static = build_model(jax.random.PRNGKey(0), _rwf_cfg(), lid_r=10.0)
    assert "Wu_g" in params and "Wout_g" in params and "Wh_g" in params
    assert len(params["Wh_g"]) == len(params["Wh"])
    out = predict(params, static, jnp.array([[0.3, 0.7], [0.5, 0.5]]))
    assert out.shape == (2, 3) and jnp.isfinite(out).all()


def test_rwf_hard_bc_still_satisfied():
    # RWF 下 hard-BC 仍須解析滿足
    params, static = build_model(jax.random.PRNGKey(1), _rwf_cfg(), lid_r=10.0)
    walls = jnp.array([[0.0, 0.4], [1.0, 0.6], [0.3, 0.0]])
    out = predict(params, static, walls)
    assert jnp.allclose(out[:, 0], 0.0, atol=1e-6)
    assert jnp.allclose(out[:, 1], 0.0, atol=1e-6)
