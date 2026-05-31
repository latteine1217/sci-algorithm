# STATUS — PINN Cavity Re=1000

## 目標
2D lid-driven cavity Re=1000，中線剖面相對 L2 < ~5–10%，可見主渦+次渦。

## 進度
- [x] 框架 v0.1：config/geometry/networks/physics/losses/optimizers/train/evaluate（22 tests）
- [x] PINNs 專家審查（subagent）
- [x] 框架 v0.2：套用審查修正（27 tests pass）
- [x] git + 推上 GitHub (private: latteine1217/sci-algorithm)
- [x] SLURM 設施（scripts/slurm/submit_exp.sh + train.sbatch，target r740）
- [x] r740 正式 run #1（fp32 SOAP baseline, job 3765）— 管線通，**未達收斂**
- [x] 結構化指標 summary.json（wall/mem/accuracy）+ EXPERIMENTS.md 對照表
- [ ] 收斂改善：壓低連續殘差（見下）

## v0.2 審查修正（2026-05-31）
- [x] **float64** 預設啟用（__init__；高 Re 標準）；連帶修 SOAP qr_dtype 對齊
- [x] **壓力錨定** 評估端減均值（修正 spec 概念：訓練端對 loss 為 no-op）
- [x] **邊界層/角點加密取樣** boundary_refined + SAMPLERS registry
- [x] **fourier_sigma** re1000 提到 6.0
- [x] **weighting=fixed** 作 re1000 baseline（gradnorm/ntk 仍可切）
- [x] **checkpoint resume** 完整狀態（params/opt_state/key/stage/step/history）
- [x] **Fourier B 移出 trainable params**（NetStatic）
- [x] **registry 化** optimizer/sampler/weighting，設定驅動
- [x] **SOAP commit pin** + chex 依賴補宣告
- [x] csv 日誌、渦心偵測、壓力場圖

## 紀錄
- 2026-05-31 v0.1→v0.2 重構完成。27 tests pass。
  smoke(200步) 管線 OK（壓力錨定/渦心/三圖/csv/state 全產出），L2≈1.0 屬欠訓練。
  下一步：r740 跑 re1000.yaml（curriculum 100→400→1000，主階段 40k 步）。
- 待 r740 正式 run 後記錄：rel-L2 u/v、渦心位置、是否見次渦。
