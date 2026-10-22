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
class Config:
    name: str
    seed: int
    tasks: list[str]
    models: list[str]
    data: DataConfig
    splits: SplitConfig = field(default_factory=SplitConfig)
    out_dir: str = "reports"


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    data = dict(raw["data"])
    data["start"], data["end"] = str(data["start"]), str(data["end"])  # yaml may parse dates
    raw["data"] = DataConfig(**data)
    raw["splits"] = SplitConfig(**raw.get("splits", {}))
    return Config(**raw)


def config_hash(cfg: Config) -> str:
    return hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()[:12]
