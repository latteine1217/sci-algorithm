import jax
import jax.numpy as jnp
import optax
from pinn_cavity.optimizers import build_optimizer
from pinn_cavity.config import OptimizerConfig


def test_soap_reduces_quadratic():
    # 對 f(w)=sum(w^2)，數十步 SOAP 應顯著降 loss
    opt = build_optimizer(OptimizerConfig(learning_rate=1e-1, decay_steps=100))
    w = {"a": jnp.array([3.0, -2.0, 1.0])}
    state = opt.init(w)

    def loss(w):
        return jnp.sum(w["a"] ** 2)

    l0 = float(loss(w))
    for _ in range(50):
        g = jax.grad(loss)(w)
        updates, state = opt.update(g, state, w)
        w = optax.apply_updates(w, updates)
    assert float(loss(w)) < 0.1 * l0
