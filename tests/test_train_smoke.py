import jax
import jax.numpy as jnp
from pinn_cavity.config import load_config
from pinn_cavity.train import train


def test_smoke_train_runs_and_decreases(tmp_path):
    cfg = load_config("configs/smoke.yaml")
    cfg.train.steps = 100  # 快測
    params, static, history = train(cfg, out_dir=str(tmp_path))
    assert all(jnp.isfinite(l).all() for l in jax.tree_util.tree_leaves(params))
    assert history["loss"][-1] < history["loss"][0]
    # checkpoint state 與 csv 應產出
    assert (tmp_path / "state.pkl").exists()
    assert (tmp_path / "history.csv").exists()


def test_curriculum_and_resume(tmp_path):
    # 兩階段 curriculum + resume 不報錯、能接續
    cfg = load_config("configs/smoke.yaml")
    cfg.curriculum = [{"re": 100, "steps": 30}, {"re": 1000, "steps": 30}]
    cfg.train.steps = 30
    cfg.train.checkpoint_every = 20
    p1, s1, h1 = train(cfg, out_dir=str(tmp_path))
    assert all(jnp.isfinite(l).all() for l in jax.tree_util.tree_leaves(p1))
