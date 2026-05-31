"""Checkpoint：存/載完整訓練狀態以支援 resume 長訓練（GPU 正式 run 必要）。

狀態含 params、optimizer state、RNG key、curriculum 進度（stage/step）、
Fourier B（評估端重建 NetStatic 用）、history。pickle 對純 pytree 可接受。
"""
import os
import pickle
import jax


def save_state(path, state: dict):
    """原子寫入：先寫暫存再 rename，避免中斷產生半截檔。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(jax.device_get(state), f)
    os.replace(tmp, path)


def load_state(path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
