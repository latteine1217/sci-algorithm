"""評估：Ghia 對照、物理場診斷、收斂曲線、結構化 layered 報告。

對齊 CFD 評估 rubric（cfd-evaluate）：收斂（loss/殘差/質量守恆）、解析度
（密網格 PDE 殘差）、物理有效性（須看場：渦量/流函數/次渦/有限性）、
對照 Ghia（apples-to-apples）。輸出 summary.json（富指標）+ evaluation.json
（layered verdict）+ 多張診斷圖。matplotlib 標題/label 用英文。
"""
import os
import csv
import json
import numpy as np
import jax.numpy as jnp
from .networks import predict, NetStatic
from .benchmark_ghia import GHIA_RE1000
from . import diagnostics as dg
from .metrics import update_summary


def static_from_state(state, lid_r):
    return NetStatic(B=state["fourier_B"], lid_r=lid_r)


def relative_l2(pred, ref):
    return dg._rel_l2(pred, ref)


def centerline_profiles(params, static):
    return dg.centerline(params, static)


def _read_history(path):
    if not path or not os.path.exists(path):
        return None
    rows = {"step": [], "loss": [], "lx": [], "ly": [], "lc": []}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows["step"].append(float(r["global_step"])); rows["loss"].append(float(r["loss"]))
            rows["lx"].append(float(r["lx"])); rows["ly"].append(float(r["ly"])); rows["lc"].append(float(r["lc"]))
    return {k: np.array(v) for k, v in rows.items()} if rows["step"] else None


# ---------- 繪圖 ----------

def _plot_centerlines(prof, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(prof["u_pred"], prof["y"], "b-", label="PINN")
    ax[0].plot(prof["u_ghia"], prof["y"], "ro", label="Ghia 1982")
    ax[0].set_xlabel("u"); ax[0].set_ylabel("y"); ax[0].set_title("u at x=0.5"); ax[0].legend()
    ax[1].plot(prof["x"], prof["v_pred"], "b-", label="PINN")
    ax[1].plot(prof["x"], prof["v_ghia"], "ro", label="Ghia 1982")
    ax[1].set_xlabel("x"); ax[1].set_ylabel("v"); ax[1].set_title("v at y=0.5"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_field(F, vor, path):
    import matplotlib.pyplot as plt
    XX, YY, U, V = F["XX"], F["YY"], F["U"], F["V"]
    speed = np.sqrt(U ** 2 + V ** 2)
    fig, ax = plt.subplots(figsize=(5, 5))
    cf = ax.contourf(XX, YY, speed, levels=30, cmap="viridis")
    ax.streamplot(XX, YY, U, V, color="white", density=1.3, linewidth=0.6)
    p = vor["primary"]; ax.plot([p["x"]], [p["y"]], "r+", ms=12)
    for k in ("BL1", "BR1"):
        if vor[k]["present"]:
            ax.plot([vor[k]["x"]], [vor[k]["y"]], "y x", ms=8)
    ax.set_title("Velocity magnitude & streamlines (Re=1000)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")  # 正方形 domain
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_scalar(XX, YY, Z, title, cmap, path, marks=None, log=False):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(figsize=(5, 5))
    if log:
        Zp = np.abs(Z) + 1e-12
        cf = ax.contourf(XX, YY, Zp, levels=np.logspace(np.log10(Zp.min() + 1e-12),
                         np.log10(Zp.max()), 30), norm=LogNorm(), cmap=cmap)
    else:
        cf = ax.contourf(XX, YY, Z, levels=30, cmap=cmap)
    if marks:
        for (mx, my) in marks:
            ax.plot([mx], [my], "k+", ms=10)
    ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_streamfunction(F, vor, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    cs = ax.contour(F["XX"], F["YY"], F["psi"], levels=25, colors="k", linewidths=0.5)
    p = vor["primary"]; ax.plot([p["x"]], [p["y"]], "r+", ms=12)
    gx, gy = GHIA_RE1000["primary_vortex"]; ax.plot([gx], [gy], "bo", ms=6, label="Ghia primary")
    ax.set_title("Stream function (Re=1000)"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_residuals(F, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, key, t in zip(ax, ("rx", "ry", "rc"),
                         ("|mom-x residual|", "|mom-y residual|", "|continuity residual|")):
        Z = np.abs(F[key]) + 1e-12
        cf = a.contourf(F["XX"], F["YY"], Z, levels=np.logspace(np.log10(Z.min()),
                        np.log10(Z.max()), 25), norm=LogNorm(), cmap="magma")
        a.set_title(t); a.set_xlabel("x"); a.set_ylabel("y"); a.set_aspect("equal")
        fig.colorbar(cf, ax=a, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_convergence(hist, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(hist["step"], hist["loss"], "k-", label="total")
    ax.semilogy(hist["step"], hist["lx"], label="mom-x")
    ax.semilogy(hist["step"], hist["ly"], label="mom-y")
    ax.semilogy(hist["step"], hist["lc"], label="continuity")
    ax.set_xlabel("step"); ax.set_ylabel("loss / residual MSE"); ax.set_title("Convergence")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------- layered 評估 ----------

def _assess(metrics, hist):
    """以透明啟發式門檻產生 cfd-evaluate 風格 layered verdict。"""
    unverified = []
    # 收斂：loss 末段相對變化（plateau?）+ 質量守恆（散度）
    conv_note = "no history"
    plateaued = None
    if hist is not None and len(hist["loss"]) >= 4:
        tail = hist["loss"][-max(3, len(hist["loss"]) // 5):]
        rel = float(np.std(tail) / (np.mean(tail) + 1e-12))
        plateaued = rel < 0.5
        conv_note = f"loss tail rel-std={rel:.2f}"
    else:
        unverified.append("convergence history unavailable")
    mass_ok = metrics["divergence_mean"] < 1e-2
    conv_status = "PASS" if (plateaued and mass_ok) else ("SUSPECT" if mass_ok or plateaued else "FAIL")

    # 解析度：密網格 PDE 殘差（單一表示，無 mesh independence）
    res_ok = metrics["residual_rms_cont"] < 1e-1
    disc_status = "PASS" if res_ok else "SUSPECT"
    unverified.append("mesh/representation independence not assessed (single network)")

    # 物理有效性：有限 + 主渦位置 + 次渦
    feats = []
    if metrics["primary_vortex_pos_err"] < 0.1:
        feats.append("primary vortex located")
    sec = metrics["secondary_BL_present"] and metrics["secondary_BR_present"]
    if sec:
        feats.append("bottom corner vortices present")
    phys_status = "PASS" if (metrics["primary_vortex_pos_err"] < 0.1 and sec) else "SUSPECT"

    # 對照 Ghia：apples-to-apples（同 Re=1000、同定義），cosh lid 為 modeling 差異
    l2 = max(metrics["rel_l2_u"], metrics["rel_l2_v"])
    val_status = "PASS" if l2 < 0.10 else ("SUSPECT" if l2 < 0.25 else "FAIL")

    statuses = [conv_status, disc_status, phys_status, val_status]
    if "FAIL" in statuses:
        verdict = "NOT_TRUSTWORTHY"; action = "RERUN"
    elif "SUSPECT" in statuses:
        verdict = "CONDITIONAL"; action = "INVESTIGATE"
    else:
        verdict = "TRUSTWORTHY"; action = "ACCEPT"

    return {
        "verdict": verdict, "recommended_action": action,
        "layers": {
            "convergence": {"status": conv_status, "mass_imbalance_mean_div": metrics["divergence_mean"],
                            "divergence_max": metrics["divergence_max"], "notes": conv_note},
            "discretization": {"status": disc_status,
                               "residual_rms": {"momx": metrics["residual_rms_momx"],
                                                "momy": metrics["residual_rms_momy"],
                                                "cont": metrics["residual_rms_cont"]}},
            "physical_validity": {"status": phys_status, "features": feats,
                                  "primary_vortex_pos_err": metrics["primary_vortex_pos_err"],
                                  "secondary_present": sec},
            "validation": {"status": val_status, "reference": "Ghia et al. 1982 (Re=1000)",
                           "rel_l2_u": metrics["rel_l2_u"], "rel_l2_v": metrics["rel_l2_v"],
                           "error_attribution": "centerline gap partly modeling (cosh-regularized lid vs sharp lid)"},
        },
        "unverified": unverified,
    }


def evaluate(params, static, re=1000.0, out_dir="results", history_path=None, grid_n=120):
    """完整評估：場診斷 + 全套圖 + 富指標 + layered 報告。回傳 metrics dict。"""
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(out_dir, exist_ok=True)

    F = dg.compute_fields(params, static, re, n=grid_n)
    vor = dg.detect_vortices(F)
    metrics = dg.aggregate_metrics(params, static, F)
    prof = dg.centerline(params, static)
    hist = _read_history(history_path or os.path.join(out_dir, "history.csv"))

    # 圖
    _plot_centerlines(prof, f"{out_dir}/centerlines.png")
    _plot_field(F, vor, f"{out_dir}/field.png")
    _plot_scalar(F["XX"], F["YY"], F["P"], "Pressure (mean-anchored)", "RdBu_r", f"{out_dir}/pressure.png")
    _plot_scalar(F["XX"], F["YY"], F["vort"], "Vorticity", "RdBu_r", f"{out_dir}/vorticity.png")
    _plot_scalar(F["XX"], F["YY"], F["div"], "|divergence| (continuity error)", "magma",
                 f"{out_dir}/divergence.png", log=True)
    _plot_residuals(F, f"{out_dir}/residuals.png")
    _plot_streamfunction(F, vor, f"{out_dir}/streamfunction.png")
    if hist is not None:
        _plot_convergence(hist, f"{out_dir}/convergence.png")

    assessment = _assess(metrics, hist)

    # 落盤
    update_summary(os.path.join(out_dir, "summary.json"), {"accuracy": metrics, "assessment": assessment})
    with open(os.path.join(out_dir, "evaluation.json"), "w") as f:
        json.dump({"metrics": metrics, "assessment": assessment}, f, indent=2, ensure_ascii=False)

    print(f"=== Validation === verdict={assessment['verdict']} "
          f"| rel-L2 u={metrics['rel_l2_u']:.4f} v={metrics['rel_l2_v']:.4f} "
          f"| div_max={metrics['divergence_max']:.2e} div_mean={metrics['divergence_mean']:.2e}")
    print(f"    primary vortex ({metrics['primary_vortex'][0]:.3f},{metrics['primary_vortex'][1]:.3f}) "
          f"err={metrics['primary_vortex_pos_err']:.3f} | "
          f"secondary BL={metrics['secondary_BL_present']} BR={metrics['secondary_BR_present']} "
          f"| cont-residual rms={metrics['residual_rms_cont']:.2e}")
    return metrics
