"""訓練編排：SOAP 訓練迴圈 + Re curriculum（per-stage 步數）+ checkpoint/resume + csv 日誌。

Observability：每 log_every 記 loss 與各殘差項到 stdout 與 history.csv，
便於回答「為何收斂/不收斂」。Reproducibility：checkpoint 存完整狀態可 resume。
"""
import os
import csv
import time
import jax
import optax

from .networks import build_model, NetStatic
from .losses import total_loss, loss_terms, update_weights, init_weights
from .optimizers import build_optimizer
from .geometry import make_sampler
from .config import device_info, curriculum_stages, apply_runtime
from .checkpoint import save_state, load_state
from .metrics import device_memory_mb, config_snapshot, write_summary


def _stage_offsets(stages):
    """各階段的 global step 起點。"""
    offs, acc = [], 0
    for _, steps in stages:
        offs.append(acc); acc += steps
    return offs


def _open_csv(path, resume):
    new = (not resume) or (not os.path.exists(path))
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["global_step", "re", "loss", "lx", "ly", "lc"])
        f.flush()
    return f, w


def _save(out_dir, params, static, opt_state, key, stage_idx, step, history):
    save_state(os.path.join(out_dir, "state.pkl"), {
        "params": params, "fourier_B": static.B, "opt_state": opt_state,
        "key": key, "stage_idx": stage_idx, "step": step, "history": history,
    })


def _run_stage(params, static, cfg, re, steps, key, sampler, opt, opt_state,
               start_step, offset, stage_idx, out_dir, history, csv_w):
    """跑單一 curriculum 階段，回傳 (params, opt_state, key)。"""

    @jax.jit
    def step(params, opt_state, xy, weights):
        L, grads = jax.value_and_grad(
            lambda p: total_loss(p, static, xy, weights, re)
        )(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, L

    key, sk = jax.random.split(key)
    xy = sampler(sk, cfg.train.n_collocation)
    weights = init_weights()

    for it in range(start_step, steps):
        if it % cfg.train.resample_every == 0 and it > start_step:
            key, sk = jax.random.split(key)
            xy = sampler(sk, cfg.train.n_collocation)
        if cfg.weighting != "fixed" and it % cfg.train.weight_update_every == 0 and it > 0:
            weights = update_weights(params, static, xy, re, method=cfg.weighting)

        params, opt_state, L = step(params, opt_state, xy, weights)

        if it % cfg.train.log_every == 0 or it == steps - 1:
            lx, ly, lc = loss_terms(params, static, xy, re)
            gstep = offset + it
            history["step"].append(gstep); history["loss"].append(float(L))
            history["lx"].append(float(lx)); history["ly"].append(float(ly)); history["lc"].append(float(lc))
            csv_w.writerow([gstep, float(re), float(L), float(lx), float(ly), float(lc)])
            print(f"[Re={re:.0f}] it={it} loss={float(L):.3e} "
                  f"lx={float(lx):.2e} ly={float(ly):.2e} lc={float(lc):.2e}")

        if cfg.train.checkpoint_every and (it + 1) % cfg.train.checkpoint_every == 0:
            _save(out_dir, params, static, opt_state, key, stage_idx, it + 1, history)

    return params, opt_state, key


def train(cfg, out_dir="results", resume_path=None):
    """主入口。回傳 (params, static, history)。"""
    os.makedirs(out_dir, exist_ok=True)
    apply_runtime(cfg)
    print(f"=== Train start | {device_info()} ===")

    stages = curriculum_stages(cfg)
    offsets = _stage_offsets(stages)
    sampler = make_sampler(cfg.sampler)
    history = {"step": [], "loss": [], "lx": [], "ly": [], "lc": []}

    resuming = resume_path is not None and os.path.exists(resume_path)
    if resuming:
        st = load_state(resume_path)
        params = st["params"]
        static = NetStatic(B=st["fourier_B"], lid_r=cfg.lid_r)
        key = st["key"]
        start_stage, start_step = st["stage_idx"], st["step"]
        history = st.get("history", history)
        resumed_opt_state = st["opt_state"]
        print(f"=== Resume from stage {start_stage} step {start_step} ===")
    else:
        key = jax.random.PRNGKey(cfg.seed)
        key, ik = jax.random.split(key)
        params, static = build_model(ik, cfg.network, cfg.lid_r)
        start_stage, start_step, resumed_opt_state = 0, 0, None

    csv_f, csv_w = _open_csv(os.path.join(out_dir, "history.csv"), resuming)
    stage_records = []
    steps_run = 0
    t_total0 = time.time()
    try:
        for si in range(start_stage, len(stages)):
            re, steps = stages[si]
            opt = build_optimizer(cfg.optimizer)
            if si == start_stage and resumed_opt_state is not None:
                opt_state, s_begin = resumed_opt_state, start_step
            else:
                opt_state, s_begin = opt.init(params), 0
            t_stage0 = time.time()
            params, opt_state, key = _run_stage(
                params, static, cfg, re, steps, key, sampler, opt, opt_state,
                s_begin, offsets[si], si, out_dir, history, csv_w)
            dt = time.time() - t_stage0
            n = steps - s_begin
            steps_run += n
            stage_records.append({
                "re": re, "steps": n, "wall_seconds": round(dt, 1),
                "steps_per_sec": round(n / dt, 2) if dt > 0 else None,
            })
            csv_f.flush()
            _save(out_dir, params, static, opt_state, key, si, steps, history)
    finally:
        csv_f.close()

    wall_total = time.time() - t_total0
    summary = {
        "device": device_info(),
        "config": config_snapshot(cfg),
        "wall_seconds_total": round(wall_total, 1),
        "steps_total": steps_run,
        "steps_per_sec": round(steps_run / wall_total, 2) if wall_total > 0 else None,
        "stages": stage_records,
        "peak_memory_mb": device_memory_mb(),
        "final_loss": history["loss"][-1] if history["loss"] else None,
    }
    write_summary(os.path.join(out_dir, "summary.json"), summary)
    print(f"=== wall={wall_total:.1f}s steps={steps_run} "
          f"({summary['steps_per_sec']}/s) mem={summary['peak_memory_mb']} ===")

    return params, static, history
