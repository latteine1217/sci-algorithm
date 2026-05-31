"""JAX PINN framework for 2D lid-driven cavity Re=1000.

高 Re PINN 在 ν=1e-3 + 二階導下對捨入誤差敏感，預設啟用 float64
（高 Re PINN 標準作法）。如需 float32 加速，設環境變數 PINN_DISABLE_X64=1。
"""
import os as _os
import jax as _jax

if _os.environ.get("PINN_DISABLE_X64", "0") != "1":
    _jax.config.update("jax_enable_x64", True)

__version__ = "0.2.0"
