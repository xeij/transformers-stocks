"""Plain dataclasses mirroring configs/*.yaml."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    tickers: list[str]
    start: str = "2010-01-01"
    end: str = "2026-10-01"
    cache_dir: str = "data/raw"


@dataclass
class SplitConfig:
    train_months: int = 60
    val_months: int = 6
    test_months: int = 12
    step_months: int = 12
    min_folds: int = 5


@dataclass
class NetConfig:
    """Sequence models (phase 2). Architecture numbers follow the brief."""

    lookback: int = 64
    batch_size: int = 512
    max_epochs: int = 20
    patience: int = 3
    lr: float = 1e-3
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    device: str = "auto"  # "auto" = mps if available else cpu
    clip: float = 5.0  # standardised features are clipped to +-clip
    lstm_hidden: int = 64
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    patch_len: int = 8
    patch_stride: int = 4
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.3


@dataclass
class Config:
    name: str
    seed: int
    tasks: list[str]
    models: list[str]
    data: DataConfig
    splits: SplitConfig = field(default_factory=SplitConfig)
    nets: NetConfig = field(default_factory=NetConfig)
    out_dir: str = "reports"


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    data = dict(raw["data"])
    data["start"], data["end"] = str(data["start"]), str(data["end"])  # yaml may parse dates
    raw["data"] = DataConfig(**data)
    raw["splits"] = SplitConfig(**raw.get("splits", {}))
    raw["nets"] = NetConfig(**raw.get("nets", {}))
    return Config(**raw)


def config_hash(cfg: Config) -> str:
    return hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()[:12]


def fit_hash(cfg: Config) -> str:
    """Hash of everything that changes a fit's predictions (not the model list or run name)."""
    d = asdict(cfg)
    key = {k: d[k] for k in ("seed", "data", "splits", "nets")}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
