from pinn_cavity.benchmark_ghia import GHIA_RE1000


def test_ghia_structure():
    d = GHIA_RE1000
    assert len(d["y"]) == len(d["u"]) == 17
    assert len(d["x"]) == len(d["v"]) == 17
    # 端點：壁面 u/v = 0、lid u(y=1)=1
    assert abs(d["u"][0]) < 1e-9 and abs(d["u"][-1] - 1.0) < 1e-9
    assert abs(d["v"][0]) < 1e-9 and abs(d["v"][-1]) < 1e-9
    # 垂直中線 u 在中段為負（主渦）
    assert min(d["u"]) < -0.3


def test_vortex_center_reference():
    cx, cy = GHIA_RE1000["primary_vortex"]
    assert 0.50 < cx < 0.56 and 0.55 < cy < 0.58
