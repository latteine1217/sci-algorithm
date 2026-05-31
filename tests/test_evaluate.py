import jax
import numpy as np
from pinn_cavity.evaluate import centerline_profiles, relative_l2
from pinn_cavity import diagnostics as dg
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
    assert res["u_pred"].shape == res["u_ref"].shape  # PINN vs DNS 同點位
    assert res["v_pred"].shape == res["v_ref"].shape


def test_fields_keys_and_pressure_anchored():
    params, static = _model()
    F = dg.compute_fields(params, static, re=1000.0, n=24)
    for k in ("U", "V", "P", "div", "vort", "rx", "ry", "rc", "psi"):
        assert F[k].shape == (24, 24)
    assert abs(float(F["P"].mean())) < 1e-9  # 壓力錨定


def test_streamfunction_zero_on_bottom():
    params, static = _model()
    F = dg.compute_fields(params, static, re=1000.0, n=24)
    assert np.allclose(F["psi"][0, :], 0.0, atol=1e-12)  # ψ=0 於底壁


def test_aggregate_metrics_has_expected_keys():
    params, static = _model()
    F = dg.compute_fields(params, static, re=1000.0, n=24)
    m = dg.aggregate_metrics(params, static, F)
    for k in ("rel_l2_u", "rel_l2_v", "field_rel_l2_u", "field_rel_l2_v", "field_rel_l2_speed",
              "divergence_max", "divergence_mean", "residual_rms_cont",
              "primary_vortex", "dns_primary_vortex", "secondary_BL_present"):
        assert k in m
    assert m["divergence_max"] >= 0


def test_dns_reference_loads():
    from pinn_cavity import reference as ref
    dns = ref.dns_centerline(n=51)
    assert dns["u"].shape == (51,) and dns["v"].shape == (51,)
    # DNS lid 中心 u≈0.987（cosh r=10）
    assert 0.95 < float(dns["u"][-1]) <= 1.0
    vx, vy = ref.dns_primary_vortex()
    assert 0.45 < vx < 0.65 and 0.5 < vy < 0.7  # Re=1000 主渦右上
