# PINN 框架設計：2D Lid-Driven Cavity, Re=1000 (JAX)

- 日期：2026-05-31
- 目標：以 JAX 從頭建立 PINN 框架，對齊 2D lid-driven cavity Re=1000 基準（Ghia et al. 1982），先確定能穩定收斂出好結果，再進入演算法優化研究。
- 範圍：單一 case（Re=1000 方腔），模組化、可延伸，為後續演算法研究預留介面。

## 1. 問題設定與控制方程

- 域：單位正方形 $\Omega=[0,1]^2$。
- 無因次化：$U_{lid}=L=1 \Rightarrow \nu = 1/Re = 10^{-3}$（Re=1000）。
- 穩態不可壓 Navier–Stokes（velocity-pressure formulation）：
  - x-動量：$u u_x + v u_y + p_x - \frac{1}{Re}(u_{xx}+u_{yy}) = 0$
  - y-動量：$u v_x + v v_y + p_y - \frac{1}{Re}(v_{xx}+v_{yy}) = 0$
  - 連續：$u_x + v_y = 0$（soft，列入殘差 loss）

## 2. 邊界條件（hard BC + cosh 角點平滑化）

四壁 $u=v=0$；頂部 lid $u=g(x),\ v=0$。以 hard constraint 透過解析包裝施加，BC 不進 loss。

- 距離函數：$D(x,y) = x(1-x)\,y(1-y)$（四壁為 0）。
- lid 正則化剖面（cosh 型）：
  $$g(x) = 1 - \frac{\cosh\!\big(r\,(x-0.5)\big)}{\cosh(0.5\,r)}$$
  - 角點 $x\in\{0,1\}$：$g=0$；中心 $x=0.5$：$g=1-\mathrm{sech}(0.5r)\approx 1$。
  - 陡峭度 $r$ 由 config 控制，預設 $r=10$（內部接近平台、僅角點附近平滑下降）。
- hard-BC 輸出包裝（網路原始輸出 $\hat u,\hat v,\hat p$）：
  - $u(x,y) = y\,g(x) + D(x,y)\,\hat u$
  - $v(x,y) = D(x,y)\,\hat v$
  - $p(x,y) = \hat p$
  - 驗證：三面壁 $D=0$ 且 $y g(x)$ 於壁面為 0 ⇒ $u=v=0$；頂部 $y=1,D=0 \Rightarrow u=g(x),v=0$。BC 解析滿足。
- 壓力錨定：$p$ 無 Dirichlet，有常數規範自由度。**修正（v0.2，依專家審查）**：velocity-pressure 的 loss 僅含 $p$ 的梯度，故「訓練端減均值」對 loss 為 no-op；錨定只需在**評估/輸出端**對 $p$ 場減域內均值即可。

## 3. 網路架構

輸入 $(x,y)$ →（A）Random Fourier Features →（B）Modified MLP →（C）hard-BC 包裝 → $(u,v,p)$。

- (A) Random Fourier Features：$\gamma(\mathbf{x}) = [\cos(2\pi \mathbf{B}\mathbf{x}), \sin(2\pi \mathbf{B}\mathbf{x})]$，$\mathbf{B}\sim\mathcal{N}(0,\sigma^2)$，$\sigma$、特徵數可配置（預設 $\sigma=2.0$，特徵數 128）。
- (B) Modified MLP（Wang et al. 2021, "gradient pathologies"）：兩個編碼器 $U,V$；每層閘控 $H^{(l+1)} = (1-Z^{(l)})\odot U + Z^{(l)}\odot V$，$Z^{(l)}=\phi(W^{(l)}H^{(l)}+b^{(l)})$。激活 $\tanh$。預設寬度 128、深度 4。
- 參數管理：以純函式 + pytree 參數字典手寫（不引入 Flax/Equinox），保持 JAX 變換（grad/jit/vmap）透明、利於後續演算法研究。

## 4. Loss 與自適應權重

- 僅 PDE 三項殘差（hard BC 免去 BC penalty）：$\mathcal{L} = \lambda_x \mathcal{L}_{mom_x} + \lambda_y \mathcal{L}_{mom_y} + \lambda_c \mathcal{L}_{cont}$，各項為對應殘差的 MSE。
- 殘差以 `jax.grad`/`jax.jacfwd` 自動微分逐點計算，`vmap` 批次化。
- 自適應權重基建（可切換）：
  - `gradnorm`：learning-rate annealing（Wang et al. 2021），依各 loss 對參數梯度範數調 $\lambda$。
  - `ntk`：NTK 對角近似（Wang et al. 2022）。
  - `fixed`：等權。
  - 預設 `gradnorm`，每 K 步更新一次權重。

## 5. 最佳化器：SOAP

- 採 **SOAP**（Vyas et al. 2024, arXiv:2409.11321）單一最佳化器，取代傳統 Adam→L-BFGS 兩段式。
- 來源：外部套件 [haydn-jones/SOAP_JAX](https://github.com/haydn-jones/SOAP_JAX)，optax 相容。
  - 安裝：`uv add "soap-jax @ git+https://github.com/haydn-jones/SOAP_JAX"`。
  - 介面：`from soap_jax import soap`，回傳 optax `GradientTransformation`；訓練迴圈維持標準 optax `update`/`apply_updates`。
- 超參納入 config：`learning_rate`、`b1`、`b2`、`weight_decay`、`precondition_frequency`、學習率排程（cosine decay）。
- `optimizers.py` 保留可插拔介面（預設且僅實作 SOAP），便於日後比較其他最佳化器。

## 6. 取樣

- 內部 collocation 點：均勻隨機（jax PRNG），每 K 步重抽；$N_f \sim 1\text{–}2\times10^4$。
- 邊界點不需取樣（hard BC）。壓力錨定使用當前 collocation batch 均值。
- 後續可擴充 RAD/RAR 自適應取樣：先在 `geometry.py` 預留取樣策略介面，本階段不實作。

## 7. Re Curriculum（可選，預設關閉）

- 階段式 warm-start：Re $=100 \to 400 \to 1000$，每階段載入前階段參數續訓。
- 由 config `curriculum: [100, 400, 1000]` 啟用；空或單元素則直攻目標 Re。
- 由 `train.py` 編排，各階段獨立 SOAP 訓練迴圈。

## 8. 驗證與成功標準

- 內嵌 Ghia et al. 1982 Re=1000 表列基準（`benchmark_ghia.py`）：
  - $x=0.5$ 垂直中線的 $u(y)$；$y=0.5$ 水平中線的 $v(x)$。
- 指標：中線剖面相對 L2 誤差；主渦心位置（參考 ≈ $(0.531, 0.565)$）。
- 輸出：中線剖面對比圖、速度大小場、流線圖（含角落次渦）、loss 曲線。
- **收斂成功標準**：中線剖面相對 L2 誤差 < ~5–10%，且流線圖可見主渦 + 至少底部兩角落次渦。

## 9. 程式結構

```
sci-algorithm/
  pyproject.toml            # uv 管理；deps: jax, jaxlib, optax, soap-jax(git), numpy, matplotlib, pyyaml
  README.md
  STATUS.md                 # 狀態/進度檔（與 protocol 分離）
  configs/
    re1000.yaml             # lab-server GPU 正式配置
    smoke.yaml              # Mac CPU 快測：小網路、少步數、小 N_f
  src/pinn_cavity/
    __init__.py
    config.py               # dataclass 設定 + yaml 載入；裝置(CPU/GPU)偵測切換
    geometry.py             # 取樣、距離函數 D、lid 剖面 g(x)、(預留)取樣策略介面
    networks.py             # Fourier features、Modified MLP、hard-BC 包裝、參數初始化
    physics.py              # autodiff NS 殘差（單點函式 + vmap）
    losses.py               # 殘差 loss 組裝 + 自適應權重（gradnorm/ntk/fixed）
    optimizers.py           # SOAP 組裝（optax 相容）+ lr 排程
    train.py                # 單階段訓練迴圈 + curriculum 編排 + checkpoint
    evaluate.py             # Ghia 比對、L2 指標、繪圖
    benchmark_ghia.py       # Ghia 1982 Re=1000 表列資料
  scripts/
    train.py                # CLI 入口：載 config → 訓練 → 存 checkpoint
    evaluate.py             # CLI 入口：載 checkpoint → 驗證 + 繪圖
  results/                  # checkpoint、圖、指標輸出（gitignore）
```

模組單一職責、可獨立推理；JAX 變換在 `networks/physics/losses` 維持純函式以利 jit/grad。

## 10. 跨平台

- `config.py` 偵測裝置：預設依 JAX 後端；`smoke.yaml` 強制小規模供 Mac CPU 驗證流程正確性（非追求收斂）。
- `re1000.yaml` 為 lab-server GPU 正式訓練配置（大 $N_f$、長步數）。
- 正式訓練經 SLURM 於 compute node 執行（依 lab-server 既有 helper），head node 僅 `uv sync`。

## 11. 依賴

`jax`, `jaxlib`, `optax`, `soap-jax`(git), `numpy`, `matplotlib`, `pyyaml`。以 `uv` 管理（`pyproject.toml`）。

## 12. 非目標（YAGNI）

- 不做 3D、不做非穩態、不做多 case 通用求解器。
- 不實作 RAD/RAR、其他最佳化器、其他 formulation（僅留介面）。
- 不引入 Flax/Equinox/PyTorch。

## 13. 待驗證假設（實作期確認）

- SOAP_JAX 套件版本/API 穩定性與 JAX 版本相容性（Python 3.10–3.13）。
- Modified MLP + Fourier features + SOAP 在 hard-BC 下能否於合理步數達標；若 Re=1000 直攻不穩，啟用 Re curriculum。
- 壓力均值錨定是否足以固定壓力規範（必要時改為單點 pin）。
