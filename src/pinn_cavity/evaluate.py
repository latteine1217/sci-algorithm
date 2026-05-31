"""驗證：Ghia 中線比對、相對 L2、渦心、流線/壓力場圖。

壓力錨定：velocity-pressure formulation 的 loss 僅含 p 的梯度，故 p 有
未固定的常數規範自由度；輸出 p 場前減去域內均值以固定參考壓力。
（訓練端減均值對 loss 為 no-op，故只在評估端做。）
matplotlib 圖標題與 label 用英文（LANGUAGE_POLICY）。
"""
import numpy as np
import jax.numpy as jnp
from .networks import predict, NetStatic
from .benchmark_ghia import GHIA_RE1000


def static_from_state(state, lid_r):
    """由 checkpoint 重建 NetStatic。"""
    return NetStatic(B=state["fourier_B"], lid_r=lid_r)


def relative_l2(pred, ref):
    pred = np.asarray(pred); ref = np.asarray(ref)
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12))


def centerline_profiles(params, static):
    """於 Ghia 取樣點預測中線 u/v，回傳 pred 與 ghia 對照。"""
    g = GHIA_RE1000
    ys = np.array(g["y"]); xs_v = np.array(g["x"])
    pts_u = jnp.stack([jnp.full_like(jnp.array(ys), 0.5), jnp.array(ys)], axis=-1)
    u_pred = np.array(predict(params, static, pts_u)[:, 0])
    pts_v = jnp.stack([jnp.array(xs_v), jnp.full_like(jnp.array(xs_v), 0.5)], axis=-1)
    v_pred = np.array(predict(params, static, pts_v)[:, 1])
    return {
        "y": ys, "u_pred": u_pred, "u_ghia": np.array(g["u"]),
        "x": xs_v, "v_pred": v_pred, "v_ghia": np.array(g["v"]),
    }


def _field(params, static, n=120):
    xs = np.linspace(0, 1, n); ys = np.linspace(0, 1, n)
    XX, YY = np.meshgrid(xs, ys)
    grid = jnp.stack([jnp.array(XX.ravel()), jnp.array(YY.ravel())], axis=-1)
    out = np.array(predict(params, static, grid))
    U = out[:, 0].reshape(n, n); V = out[:, 1].reshape(n, n)
    P = out[:, 2].reshape(n, n)
    P = P - P.mean()  # 壓力錨定：減域內均值
    return XX, YY, U, V, P


def primary_vortex_center(XX, YY, U, V):
    """主渦心估計：內部區（離壁 >0.1）速度大小最小處。"""
    speed = np.sqrt(U ** 2 + V ** 2)
    interior = (XX > 0.1) & (XX < 0.9) & (YY > 0.1) & (YY < 0.9)
    masked = np.where(interior, speed, np.inf)
    j, i = np.unravel_index(np.argmin(masked), masked.shape)
    return float(XX[j, i]), float(YY[j, i])


def evaluate(params, static, out_dir="results"):
    """完整驗證：印指標、存中線/場/壓力圖。回傳指標 dict。"""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    prof = centerline_profiles(params, static)
    l2_u = relative_l2(prof["u_pred"], prof["u_ghia"])
    l2_v = relative_l2(prof["v_pred"], prof["v_ghia"])
    XX, YY, U, V, P = _field(params, static)
    cx, cy = primary_vortex_center(XX, YY, U, V)
    ref = GHIA_RE1000["primary_vortex"]
    print(f"=== Validation === rel-L2 u={l2_u:.4f} v={l2_v:.4f} | "
          f"vortex=({cx:.3f},{cy:.3f}) ref={ref}")

    # 中線剖面對比
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(prof["u_pred"], prof["y"], "b-", label="PINN")
    ax[0].plot(prof["u_ghia"], prof["y"], "ro", label="Ghia 1982")
    ax[0].set_xlabel("u"); ax[0].set_ylabel("y"); ax[0].set_title("u at x=0.5"); ax[0].legend()
    ax[1].plot(prof["x"], prof["v_pred"], "b-", label="PINN")
    ax[1].plot(prof["x"], prof["v_ghia"], "ro", label="Ghia 1982")
    ax[1].set_xlabel("x"); ax[1].set_ylabel("v"); ax[1].set_title("v at y=0.5"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{out_dir}/centerlines.png", dpi=150); plt.close(fig)

    # 速度大小場 + 流線
    speed = np.sqrt(U ** 2 + V ** 2)
    fig, ax = plt.subplots(figsize=(5, 5))
    cf = ax.contourf(XX, YY, speed, levels=30, cmap="viridis")
    ax.streamplot(XX, YY, U, V, color="white", density=1.3, linewidth=0.6)
    ax.plot([cx], [cy], "r+", markersize=12)
    ax.set_title("Velocity magnitude & streamlines (Re=1000)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); fig.colorbar(cf, ax=ax)
    fig.tight_layout(); fig.savefig(f"{out_dir}/field.png", dpi=150); plt.close(fig)

    # 壓力場（已錨定）
    fig, ax = plt.subplots(figsize=(5, 5))
    cf = ax.contourf(XX, YY, P, levels=30, cmap="RdBu_r")
    ax.set_title("Pressure (mean-anchored, Re=1000)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); fig.colorbar(cf, ax=ax)
    fig.tight_layout(); fig.savefig(f"{out_dir}/pressure.png", dpi=150); plt.close(fig)

    return {"rel_l2_u": l2_u, "rel_l2_v": l2_v, "vortex": (cx, cy)}
