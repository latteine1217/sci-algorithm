"""設定資料結構與 YAML 載入；不在此寫任務流水帳。

設定驅動元件選擇（sampler / optimizer / weighting），便於後續比較研究。
"""
from __future__ import annotations
from dataclasses import dataclass, field
import yaml
import jax


@dataclass
class NetworkConfig:
    width: int = 64
    depth: int = 4
    n_fourier: int = 64
    fourier_sigma: float = 5.0  # Re=1000 薄邊界層需較高頻；過低會欠解析
    rwf: bool = False           # Random Weight Factorization（Wang et al. 2023）
    rwf_mu: float = 1.0         # 尺度 g~N(μ,σ)；W = V·exp(g)
    rwf_sigma: float = 0.1


@dataclass
class OptimizerConfig:
    name: str = "soap"           # registry: optimizers.OPTIMIZERS
    learning_rate: float = 1.0e-3
    b1: float = 0.95
    b2: float = 0.95
    weight_decay: float = 0.0
    precondition_frequency: int = 10  # SOAP 專用：preconditioner 更新頻率
    decay_steps: int = 20000
    muon_momentum: float = 0.95       # Muon 專用：Nesterov momentum 係數
    muon_ns_steps: int = 5            # Muon 專用：Newton-Schulz 迭代步數


@dataclass
class SamplerConfig:
    name: str = "uniform"        # registry: geometry.SAMPLERS
    boundary_fraction: float = 0.3   # 邊界加密點佔比（boundary_refined）
    boundary_width: float = 0.05     # 邊界層帶寬
    corner_fraction: float = 0.15    # 邊界點中分配給上角點之比例


@dataclass
class TrainConfig:
    steps: int = 20000               # curriculum 階段未指定步數時的預設
    n_collocation: int = 16384
    resample_every: int = 100
    weight_update_every: int = 100
    log_every: int = 500
    checkpoint_every: int = 5000


@dataclass
class Config:
    re: float = 1000.0
    lid_r: float = 10.0
    seed: int = 0
    x64: bool = True
    weighting: str = "fixed"     # registry: losses.WEIGHTERS（fixed/gradnorm/ntk）
    weight_ema: float = 0.9      # 權重 EMA 平滑係數：w←ema·w + (1-ema)·new
    autodiff: str = "taylor"  # physics 二階導模式：taylor（jet/Forward-Laplacian，3×快2.9×省記憶體）/ fwd_over_rev / hessian
    optimizer_mode: str = "soap"  # soap（一階，SOAP）/ gn（matrix-free Gauss-Newton）
    gn_cg_iters: int = 10        # GN：CG 迭代數（tradeoff per-step cost vs 解精度）
    gn_lr: float = 1.0           # GN：步長（Levenberg-Marquardt scaling）
    gn_damping: float = 1e-3     # GN：LM damping λ（穩定化）
    curriculum: list = field(default_factory=list)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    net = NetworkConfig(**raw.pop("network", {}))
    opt = OptimizerConfig(**raw.pop("optimizer", {}))
    smp = SamplerConfig(**raw.pop("sampler", {}))
    tr = TrainConfig(**raw.pop("train", {}))
    return Config(network=net, optimizer=opt, sampler=smp, train=tr, **raw)


def curriculum_stages(cfg: Config):
    """正規化 curriculum 為 [(re, steps), ...]。

    支援三種寫法：
      - []                              → 直攻 cfg.re，步數 cfg.train.steps
      - [100, 400, 1000]                → 各階段步數皆 cfg.train.steps
      - [{re:100,steps:N}, ...]         → 各階段自訂步數
    最終階段強制為 cfg.re。
    """
    items = list(cfg.curriculum) if cfg.curriculum else []
    if not items:
        return [(float(cfg.re), int(cfg.train.steps))]
    stages = []
    for it in items:
        if isinstance(it, dict):
            stages.append((float(it["re"]), int(it.get("steps", cfg.train.steps))))
        else:
            stages.append((float(it), int(cfg.train.steps)))
    if stages[-1][0] != float(cfg.re):
        stages.append((float(cfg.re), int(cfg.train.steps)))
    return stages


def apply_runtime(cfg: Config):
    """依 config 套用執行期設定（x64）。應在建立任何陣列前呼叫。"""
    jax.config.update("jax_enable_x64", bool(cfg.x64))


def device_info() -> str:
    """回報 JAX 後端與裝置 + x64 狀態，供日誌（Observability）。"""
    return (f"backend={jax.default_backend()} devices={jax.devices()} "
            f"x64={jax.config.read('jax_enable_x64')}")
