import jax.numpy as jnp
import jax
from pinn_cavity.geometry import distance_fn, lid_profile, sample_collocation


def test_distance_zero_on_walls():
    # 四壁上 D=0
    pts = jnp.array([[0.0, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0]])
    d = distance_fn(pts[:, 0], pts[:, 1])
    assert jnp.allclose(d, 0.0, atol=1e-12)


def test_distance_positive_interior():
    d = distance_fn(jnp.array(0.5), jnp.array(0.5))
    assert float(d) > 0.0


def test_distance_approximates_true_sdf_near_wall():
    # 近左壁 φ≈x（單位梯度，像真 SDF 線性衰減）
    d = distance_fn(jnp.array(0.02), jnp.array(0.5))
    assert abs(float(d) - 0.02) < 5e-3
    # 近底壁 φ≈y
    d2 = distance_fn(jnp.array(0.5), jnp.array(0.03))
    assert abs(float(d2) - 0.03) < 5e-3


def test_lid_profile_corners_and_center():
    g0 = lid_profile(jnp.array(0.0), r=10.0)
    g1 = lid_profile(jnp.array(1.0), r=10.0)
    gc = lid_profile(jnp.array(0.5), r=10.0)
    assert jnp.allclose(g0, 0.0, atol=1e-12)
    assert jnp.allclose(g1, 0.0, atol=1e-12)
    assert 0.95 < float(gc) <= 1.0  # 中心接近 1


def test_sample_shape_and_bounds():
    key = jax.random.PRNGKey(0)
    pts = sample_collocation(key, n=256)
    assert pts.shape == (256, 2)
    assert float(pts.min()) >= 0.0 and float(pts.max()) <= 1.0
