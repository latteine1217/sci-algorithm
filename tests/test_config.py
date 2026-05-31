from pinn_cavity.config import Config, load_config


def test_load_smoke_config():
    cfg = load_config("configs/smoke.yaml")
    assert isinstance(cfg, Config)
    assert cfg.re == 1000.0
    assert cfg.network.width > 0
    assert cfg.network.n_fourier > 0
    assert cfg.optimizer.learning_rate > 0
    assert cfg.lid_r > 0


def test_defaults_present():
    cfg = load_config("configs/smoke.yaml")
    # curriculum 預設為空 list 或 None → 直攻目標 Re
    assert cfg.curriculum == [] or cfg.curriculum is None
    assert cfg.weighting in ("gradnorm", "ntk", "fixed")
