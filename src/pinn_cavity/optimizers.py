"""最佳化器組裝：以 OPTIMIZERS registry 設定驅動，便於比較研究。

預設 SOAP（haydn-jones/SOAP_JAX, optax 相容）+ cosine 學習率排程。
"""
import jax
import jax.numpy as jnp
import optax
from soap_jax import soap


def _build_soap(opt_cfg):
    schedule = optax.cosine_decay_schedule(
        init_value=opt_cfg.learning_rate,
        decay_steps=max(1, opt_cfg.decay_steps),
        alpha=0.01,
    )
    # SOAP 內部 QR 預設 float32；x64 下需對齊 float64，否則 lax.cond
    # 兩分支（refresh vs keep preconditioner）dtype 不一致而報錯。
    qr_dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    return soap(
        learning_rate=schedule,
        b1=opt_cfg.b1,
        b2=opt_cfg.b2,
        weight_decay=opt_cfg.weight_decay,
        precondition_frequency=opt_cfg.precondition_frequency,
        qr_dtype=qr_dtype,
    )


OPTIMIZERS = {"soap": _build_soap}


def build_optimizer(opt_cfg):
    """依 OptimizerConfig.name 回傳 optax GradientTransformation。"""
    if opt_cfg.name not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer: {opt_cfg.name} "
                         f"(available: {tuple(OPTIMIZERS)})")
    return OPTIMIZERS[opt_cfg.name](opt_cfg)
