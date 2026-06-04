"""SOAP 批次化版本：對同形狀的 2D 層做 vmap 減少 kernel launch 開銷。

設計決策（基於 bench_batching.py 量測結果）：
  - matmul (project / outer product)：vmap 批次化，1.80× speedup
  - QR decomposition：保持 sequential，批次化反而 12× 更慢（cuSolver 小矩陣特性）

只批次化 project / update_preconditioner（每步執行），
refresh_preconditioner（每 precondition_frequency 步）保持原始 sequential。

用法：在 optimizers.py 的 OPTIMIZERS registry 中以 "soap_batched" 呼叫。
"""
from itertools import chain
from typing import Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import optax.tree_utils as otu
from jaxtyping import Array
from optax import GradientTransformation, Updates

# 從 soap_jax 匯入不需修改的元件（避免重複維護）
from soap_jax.soap import (
    Preconditioner,
    SOAPState,
    PostDecayState,
    lerp,
    init_conditioner,
    get_orthogonal_matrix,
    get_orthogonal_matrix_QR,
    project,
    project_back,
    update_preconditioner,
    _is_preconditioner,
    _resolve_learning_rate,
    add_decayed_weights_post,
)


# ── 形狀分組工具 ─────────────────────────────────────────────────────────

def _leaf_shape_key(leaf):
    """回傳可 hash 的 (ndim, shape, dtype) key，供分組用。"""
    return (leaf.ndim, leaf.shape, leaf.dtype)


def _compute_groups(leaves: list) -> dict:
    """將 leaf 清單按 (ndim, shape, dtype) 分組。

    回傳 {shape_key: [index, ...], ...}，只包含 ndim==2 且組內 N>1 的組。
    單例組（N=1）或 1D 組不批次化。
    """
    shape_to_indices = {}
    for i, leaf in enumerate(leaves):
        key = _leaf_shape_key(leaf)
        if key not in shape_to_indices:
            shape_to_indices[key] = []
        shape_to_indices[key].append(i)
    # 只保留 2D、N>1 的組
    return {k: v for k, v in shape_to_indices.items() if k[0] == 2 and len(v) > 1}


def _stack_preconditioners(preconds: list) -> "Preconditioner":
    """將 N 個 Preconditioner 沿 axis=0 堆疊成一個 batched Preconditioner。

    例：N 個 Preconditioner([(m,m), (n,n)]) → Preconditioner([(N,m,m), (N,n,n)])
    JAX vmap 可以正確地在 Preconditioner pytree 的 batch dim 上映射。
    """
    n_slots = len(preconds[0].matrices)
    batched = []
    for k in range(n_slots):
        slot_mats = [p.matrices[k] for p in preconds]
        if slot_mats[0] is None:
            batched.append(None)
        else:
            batched.append(jnp.stack(slot_mats, axis=0))
    return Preconditioner(batched)


def _unstack_preconditioners(batched: "Preconditioner", n: int) -> list:
    """將 batched Preconditioner 解堆疊成 N 個個別 Preconditioner。"""
    result = []
    for i in range(n):
        mats = []
        for slot_mat in batched.matrices:
            if slot_mat is None:
                mats.append(None)
            else:
                mats.append(slot_mat[i])
        result.append(Preconditioner(mats))
    return result


# ── 批次化 scale_by_soap ─────────────────────────────────────────────────

def scale_by_soap_batched(
    b1: float = 0.95,
    b2: float = 0.95,
    shampoo_beta: float = -1,
    eps: float = 1e-8,
    correct_bias: bool = True,
    precondition_frequency: int = 10,
    max_precond_dim: int = 10000,
    precondition_1d: bool = False,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    mu_dtype=None,
    qr_dtype=jnp.float32,
) -> GradientTransformation:
    """SOAP 批次化版本。

    與 soap_jax.scale_by_soap 行為完全相同，差異：
      - project / update_preconditioner：同形狀 2D 層透過 jax.vmap 批次化
      - refresh_preconditioner（QR）：保持 sequential（批次 QR 反而慢 12×）
    """
    if shampoo_beta < 0:
        shampoo_beta = b2

    # ---- 靜態分組資訊（在 init_fn 時確定，不存入 JAX state）----
    # 用 Python dict 存在 closure，trace 期間為 concrete Python 值
    _groups = {}          # {shape_key: [indices]}
    _treedef = [None]     # 存 params 的 treedef

    def init_fn(params: Updates) -> SOAPState:
        leaves, treedef = jtu.tree_flatten(params)
        _treedef[0] = treedef
        groups = _compute_groups(leaves)
        _groups.update(groups)

        exp_avg = otu.tree_zeros_like(params, dtype=mu_dtype)
        exp_avg_sq = otu.tree_zeros_like(params, dtype=mu_dtype)
        GG = jtu.tree_map(
            lambda p: init_conditioner(p, max_precond_dim, precondition_1d, qr_dtype),
            params,
        )
        Q = jtu.tree_map(
            lambda p: init_conditioner(p, max_precond_dim, precondition_1d, qr_dtype),
            params,
        )
        return SOAPState(
            count=jnp.zeros([], jnp.int32),
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            GG=GG,
            Q=Q,
        )

    def init_step(updates, state):
        """第一步：用原始 tree_map（init_step 只跑一次，不需優化）。"""
        new_GG = jtu.tree_map(
            lambda grad, gg: update_preconditioner(grad, gg, shampoo_beta, precision),
            updates, state.GG, is_leaf=_is_preconditioner,
        )
        new_Q = jtu.tree_map(
            lambda gg: gg.map(lambda m: get_orthogonal_matrix(m, qr_dtype)),
            new_GG, is_leaf=_is_preconditioner,
        )
        new_updates = otu.tree_zeros_like(updates)
        return new_updates, state._replace(GG=new_GG, Q=new_Q)

    def _batched_project(updates_leaves, Q_leaves):
        """對 updates / Q leaves 執行批次化 project，回傳 projected leaves list。

        同形狀 N>1 的 2D 層：vmap 批次化。
        singleton 2D 層（input/output）和 1D 層：sequential fallback，保證每層都被處理。
        """
        projected = list(updates_leaves)
        batched_indices = set()
        for indices in _groups.values():
            batched_indices.update(indices)
            grad_batch = jnp.stack([updates_leaves[i] for i in indices])
            Q_batch = _stack_preconditioners([Q_leaves[i] for i in indices])
            proj_batch = jax.vmap(lambda g, q: project(g, q, precision))(
                grad_batch, Q_batch
            )
            for j, orig_i in enumerate(indices):
                projected[orig_i] = proj_batch[j]
        # singleton 2D 層：sequential fallback（保證 input/output 也套用 preconditioner）
        for i in range(len(updates_leaves)):
            if i not in batched_indices and updates_leaves[i].ndim == 2:
                projected[i] = project(updates_leaves[i], Q_leaves[i], precision)
        return projected

    def _batched_update_GG(updates_leaves, GG_leaves):
        """對 updates / GG leaves 執行批次化 update_preconditioner，回傳新 GG leaves list。

        同形狀 N>1 的 2D 層：vmap 批次化。
        singleton 2D 層和 1D 層：sequential fallback。
        """
        new_GG_leaves = list(GG_leaves)
        batched_indices = set()
        for indices in _groups.values():
            batched_indices.update(indices)
            n = len(indices)
            grad_batch = jnp.stack([updates_leaves[i] for i in indices])
            GG_batch = _stack_preconditioners([GG_leaves[i] for i in indices])
            new_GG_batch = jax.vmap(
                lambda g, gg: update_preconditioner(g, gg, shampoo_beta, precision)
            )(grad_batch, GG_batch)
            unstacked = _unstack_preconditioners(new_GG_batch, n)
            for j, orig_i in enumerate(indices):
                new_GG_leaves[orig_i] = unstacked[j]
        # singleton 2D 層：sequential fallback
        for i in range(len(updates_leaves)):
            if i not in batched_indices:
                new_GG_leaves[i] = update_preconditioner(
                    updates_leaves[i], GG_leaves[i], shampoo_beta, precision
                )
        return new_GG_leaves

    def _batched_project_back(norm_updates_leaves, Q_leaves):
        """對 norm_updates / Q leaves 執行批次化 project_back。

        同形狀 N>1 的 2D 層：vmap 批次化。
        singleton 2D 層和 1D 層：sequential fallback。
        """
        projback = list(norm_updates_leaves)
        batched_indices = set()
        for indices in _groups.values():
            batched_indices.update(indices)
            u_batch = jnp.stack([norm_updates_leaves[i] for i in indices])
            Q_batch = _stack_preconditioners([Q_leaves[i] for i in indices])
            pb_batch = jax.vmap(lambda u, q: project_back(u, q, precision))(
                u_batch, Q_batch
            )
            for j, orig_i in enumerate(indices):
                projback[orig_i] = pb_batch[j]
        # singleton 2D 層：sequential fallback
        for i in range(len(norm_updates_leaves)):
            if i not in batched_indices and norm_updates_leaves[i].ndim == 2:
                projback[i] = project_back(norm_updates_leaves[i], Q_leaves[i], precision)
        return projback

    def update_step(updates, state):
        """主要更新步：批次化 project / update_GG / project_back；QR 保持 sequential。"""
        treedef = _treedef[0]
        if treedef is None:
            # fallback: 沒有呼叫過 init_fn（通常不會發生）
            return _update_step_fallback(updates, state)

        # 展平所有 pytree leaves（保留 Preconditioner 為 leaf）
        upd_leaves, _ = jtu.tree_flatten(updates)
        GG_leaves, _ = jtu.tree_flatten(state.GG, is_leaf=_is_preconditioner)
        Q_leaves, _ = jtu.tree_flatten(state.Q, is_leaf=_is_preconditioner)

        # ── 1. Batched project（前向投影）──────────────────────────────
        proj_leaves = _batched_project(upd_leaves, Q_leaves)
        grad_projected = treedef.unflatten(proj_leaves)

        # ── 2. Moment 更新（EMA，純 element-wise，不需批次化）──────────
        exp_avg = otu.tree_update_moment(grad_projected, state.exp_avg, b1, 1)
        exp_avg_sq = otu.tree_update_moment_per_elem_norm(grad_projected, state.exp_avg_sq, b2, 2)
        if mu_dtype is not None:
            exp_avg = otu.tree_cast(exp_avg, mu_dtype)
            exp_avg_sq = otu.tree_cast(exp_avg_sq, mu_dtype)

        # ── 3. Batched project_back（後向投影）─────────────────────────
        ea_leaves, _ = jtu.tree_flatten(exp_avg)
        eas_leaves, _ = jtu.tree_flatten(exp_avg_sq)
        norm_upd_raw = [
            e / (jnp.sqrt(es) + eps)
            for e, es in zip(ea_leaves, eas_leaves)
        ]
        Q_leaves_fresh, _ = jtu.tree_flatten(state.Q, is_leaf=_is_preconditioner)
        pb_leaves = _batched_project_back(norm_upd_raw, Q_leaves_fresh)
        norm_updates = treedef.unflatten(pb_leaves)

        # ── 4. Bias correction ─────────────────────────────────────────
        if correct_bias:
            effective_step = state.count - 1
            bc1 = 1 - b1 ** effective_step
            bc2 = 1 - b2 ** effective_step
            corr = jnp.sqrt(bc2) / bc1
            norm_updates = jtu.tree_map(lambda p: p * corr, norm_updates)

        # ── 5. Batched update_GG（preconditioner EMA）─────────────────
        new_GG_leaves = _batched_update_GG(upd_leaves, GG_leaves)
        new_GG = jtu.tree_unflatten(
            jtu.tree_structure(state.GG, is_leaf=_is_preconditioner),
            new_GG_leaves,
        )

        # ── 6. QR refresh（保持 sequential！批次化反而 12× 更慢）───────
        def refresh_preconditioner():
            new_Q_and_eas = jtu.tree_map(
                lambda e, gg, q: get_orthogonal_matrix_QR(gg, q, e, precision, qr_dtype),
                exp_avg_sq, new_GG, state.Q, is_leaf=_is_preconditioner,
            )
            new_Q = jtu.tree_map(lambda _, x: x[0], updates, new_Q_and_eas)
            new_eas = jtu.tree_map(lambda _, x: x[1], updates, new_Q_and_eas)
            new_ea = jtu.tree_map(
                lambda e, old_q, new_q: project(project_back(e, old_q, precision), new_q, precision),
                exp_avg, state.Q, new_Q, is_leaf=_is_preconditioner,
            )
            return new_Q, new_eas, new_ea

        def keep_preconditioner():
            return state.Q, exp_avg_sq, exp_avg

        new_Q, exp_avg_sq, exp_avg = jax.lax.cond(
            (state.count - 1) % precondition_frequency == 0,
            refresh_preconditioner,
            keep_preconditioner,
        )

        new_state = SOAPState(
            count=state.count,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            GG=new_GG,
            Q=new_Q,
        )
        return norm_updates, new_state

    def _update_step_fallback(updates, state):
        """純 tree_map 版本（fallback，行為與 soap_jax 相同）。"""
        grad_projected = jtu.tree_map(
            lambda grad, q: project(grad, q, precision),
            updates, state.Q, is_leaf=_is_preconditioner,
        )
        exp_avg = otu.tree_update_moment(grad_projected, state.exp_avg, b1, 1)
        exp_avg_sq = otu.tree_update_moment_per_elem_norm(grad_projected, state.exp_avg_sq, b2, 2)
        norm_updates = jtu.tree_map(
            lambda e, es, q: project_back(e / (jnp.sqrt(es) + eps), q, precision),
            exp_avg, exp_avg_sq, state.Q, is_leaf=_is_preconditioner,
        )
        if correct_bias:
            effective_step = state.count - 1
            bc1 = 1 - b1 ** effective_step
            bc2 = 1 - b2 ** effective_step
            corr = jnp.sqrt(bc2) / bc1
            norm_updates = jtu.tree_map(lambda p: p * corr, norm_updates)
        new_GG = jtu.tree_map(
            lambda grad, gg: update_preconditioner(grad, gg, shampoo_beta, precision),
            updates, state.GG, is_leaf=_is_preconditioner,
        )

        def refresh_preconditioner():
            new_Q_and_eas = jtu.tree_map(
                lambda e, gg, q: get_orthogonal_matrix_QR(gg, q, e, precision, qr_dtype),
                exp_avg_sq, new_GG, state.Q, is_leaf=_is_preconditioner,
            )
            new_Q = jtu.tree_map(lambda _, x: x[0], updates, new_Q_and_eas)
            new_eas = jtu.tree_map(lambda _, x: x[1], updates, new_Q_and_eas)
            new_ea = jtu.tree_map(
                lambda e, old_q, new_q: project(project_back(e, old_q, precision), new_q, precision),
                exp_avg, state.Q, new_Q, is_leaf=_is_preconditioner,
            )
            return new_Q, new_eas, new_ea

        def keep_preconditioner():
            return state.Q, exp_avg_sq, exp_avg

        new_Q, exp_avg_sq, exp_avg = jax.lax.cond(
            (state.count - 1) % precondition_frequency == 0,
            refresh_preconditioner,
            keep_preconditioner,
        )
        return norm_updates, SOAPState(
            count=state.count, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq,
            GG=new_GG, Q=new_Q,
        )

    def update_fn(updates, state, params=None):
        del params
        count_inc = jnp.asarray(optax.safe_int32_increment(state.count))
        state = state._replace(count=count_inc)
        updates, new_state = jax.lax.cond(
            count_inc == 1,
            lambda: init_step(updates, state),
            lambda: update_step(updates, state),
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)  # type: ignore


def soap_batched(
    learning_rate: optax.ScalarOrSchedule = 3e-3,
    b1: float = 0.95,
    b2: float = 0.95,
    shampoo_beta: float = -1,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    correct_bias: bool = True,
    precondition_frequency: int = 10,
    max_precond_dim: int = 10000,
    precondition_1d: bool = False,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    mu_dtype=None,
    qr_dtype=jnp.float32,
) -> optax.GradientTransformationExtraArgs:
    """SOAP 批次化版本（drop-in replacement for soap_jax.soap）。

    per-step matmul 批次化 → ~1.8× speedup on same-shape layers
    QR refresh 保持 sequential → 避免 12× regression
    """
    return optax.chain(
        scale_by_soap_batched(
            b1=b1, b2=b2, shampoo_beta=shampoo_beta, eps=eps,
            correct_bias=correct_bias, precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim, precondition_1d=precondition_1d,
            precision=precision, mu_dtype=mu_dtype, qr_dtype=qr_dtype,
        ),
        optax.scale_by_learning_rate(learning_rate),
        add_decayed_weights_post(weight_decay, learning_rate),
    )
