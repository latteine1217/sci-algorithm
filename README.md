# PINN Lid-Driven Cavity (Re=1000, JAX)

velocity-pressure PINN，hard-BC + cosh 角點平滑 lid，Modified MLP + Fourier features，SOAP 最佳化，Ghia 1982 驗證。架構以設定驅動、元件可插拔，為演算法優化研究預留擴充點。

## 安裝

```bash
uv sync --extra dev            # 本機 CPU（開發/測試）
uv sync --extra cuda           # lab-server GPU（head node 線上執行）
```

預設啟用 float64（高 Re PINN 標準）。如需 float32 加速：`PINN_DISABLE_X64=1`。

## 快測（Mac CPU）

```bash
uv run pytest -q
uv run python scripts/train.py    --config configs/smoke.yaml --out results/smoke
uv run python scripts/evaluate.py --config configs/smoke.yaml --state results/smoke/state.pkl --out results/smoke
```

## 正式訓練（lab-server r740 GPU, SLURM）

```bash
# head node（線上）：同步 GPU 依賴
uv sync --extra cuda
# 提交到 compute node r740
scripts/slurm/submit_exp.sh re1000-exp01 configs/re1000.yaml
# 中斷後重跑同指令會自動 resume（state.pkl 存在時）
```

輸出落在 `results/<EXP_ID>/`：`state.pkl`（可 resume）、`history.csv`、`centerlines.png`、`field.png`、`pressure.png`。

## 方法摘要

- **Formulation**：穩態不可壓 velocity-pressure NS，連續方程 soft。
- **BC**：hard constraint，`u = y·g(x) + D·û`，`v = D·v̂`，`D = x(1-x)y(1-y)`，cosh lid 剖面 `g(x) = 1 - cosh(r(x-0.5))/cosh(0.5r)`。
- **網路**：Random Fourier Features（σ 可調）→ Modified MLP（Wang et al. 2021）。Fourier B 為固定量，與可訓練參數分離。
- **最佳化**：SOAP（Vyas et al. 2024；x64 下對齊 qr_dtype）。
- **取樣**：`uniform` / `boundary_refined`（邊界層+上角點加密）。
- **權重**：`fixed` / `gradnorm` / `ntk`（設定驅動）。
- **Curriculum**：可選 Re warm-start，per-stage 步數。
- **壓力錨定**：評估端減域內均值固定參考壓力（訓練端對 loss 為 no-op）。

## 結構

```
src/pinn_cavity/
  config.py        設定 + curriculum 正規化 + x64
  geometry.py      距離函數、lid 剖面、取樣策略 (SAMPLERS)
  networks.py      Fourier features、Modified MLP、hard-BC、NetStatic
  physics.py       autodiff NS 殘差（formulation 擴充點）
  losses.py        殘差 loss + 權重策略 (WEIGHTERS)
  optimizers.py    SOAP 組裝 (OPTIMIZERS)
  checkpoint.py    完整狀態存載（resume）
  train.py         訓練編排 + curriculum + checkpoint + csv
  evaluate.py      Ghia 比對、渦心、流線/壓力圖
  benchmark_ghia.py  Ghia 1982 Re=1000 表列
configs/           smoke（快測）、re1000（r740 正式）
scripts/           train.py、evaluate.py、slurm/{submit_exp.sh,train.sbatch}
tests/             pytest 單元測試
docs/superpowers/  設計 spec 與實作計畫
```

## 擴充點（未來研究）

- **最佳化器比較**：`optimizers.OPTIMIZERS` 註冊新項，config `optimizer.name` 切換。
- **取樣策略**：`geometry.SAMPLERS`（RAD/RAR 等），config `sampler.name`。
- **權重策略**：`losses.WEIGHTERS`，config `weighting`。
- **Formulation**：`physics.py` 標註擴充點（stream-function、湍流模型）；目前單一 formulation 依 YAGNI 不預抽象。

## 收斂標準

中線剖面相對 L2 < ~5–10%，流線圖可見主渦（中心≈0.53,0.56）+ 角落次渦。

## 參考

- Ghia, Ghia & Shin (1982), JCP 48, 387.
- Wang, Teng & Perdikaris (2021) Modified MLP / gradient pathologies.
- Vyas et al. (2024) SOAP, arXiv:2409.11321.
