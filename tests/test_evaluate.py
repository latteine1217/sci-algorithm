import jax
import numpy as np
from pinn_cavity.evaluate import centerline_profiles, relative_l2, primary_vortex_center, _field
from pinn_cavity.networks import build_model
from pinn_cavity.config import NetworkConfig


def _model():
    return build_model(jax.random.PRNGKey(0), NetworkConfig(width=16, depth=3, n_fourier=8), lid_r=10.0)


def test_relative_l2_zero_for_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert relative_l2(a, a) < 1e-12


def test_centerline_profiles_shapes():
    params, static = _model()
    res = centerline_profiles(params, static)
    assert res["u_pred"].shape == res["u_ghia"].shape
    assert res["v_pred"].shape == res["v_ghia"].shape


def test_vortex_center_in_interior():
    params, static = _model()
    XX, YY, U, V, P = _field(params, static, n=40)
    cx, cy = primary_vortex_center(XX, YY, U, V)
    assert 0.1 <= cx <= 0.9 and 0.1 <= cy <= 0.9


def test_pressure_is_mean_anchored():
    params, static = _model()
    _, _, _, _, P = _field(params, static, n=40)
    assert abs(float(P.mean())) < 1e-9
