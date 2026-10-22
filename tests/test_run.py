import json

import pandas as pd

from stockbench.config import Config, DataConfig, NetConfig, SplitConfig
from stockbench.run import run
from tests.conftest import TICKERS


def test_run_end_to_end_on_synthetic_data(tmp_path, prices):
    def source(ticker, start, end):
        return prices[prices["ticker"] == ticker].reset_index(drop=True)

    cfg = Config(
        name="e2e",
        seed=0,
        tasks=["vol_5d", "dir_5d", "rank_5d"],
        models=["naive", "linear", "lgbm", "lstm"],
        data=DataConfig(
            tickers=TICKERS, start="2010-01-01", end="2013-02-01", cache_dir=str(tmp_path / "raw")
        ),
        splits=SplitConfig(
            train_months=12, val_months=3, test_months=6, step_months=6, min_folds=2
        ),
        out_dir=str(tmp_path / "reports"),
        nets=NetConfig(
            lookback=16, patch_len=4, patch_stride=2, max_epochs=1, seeds=[0, 1], device="cpu"
        ),
    )
    out = run(cfg, source=source)

    preds = pd.read_parquet(out / "predictions.parquet")
    assert set(preds["model"]) == {"naive", "linear", "lgbm", "lstm"}
    assert preds.groupby("model")["seed"].nunique().to_dict() == {
        "naive": 1,
        "linear": 1,
        "lgbm": 1,
        "lstm": 2,
    }
    per_run = preds.groupby(["task", "model", "seed"]).size()
    assert (per_run.groupby("task").nunique() == 1).all()  # identical test rows per task
    assert set(preds["task"]) == {"vol_5d", "dir_5d", "rank_5d"}
    assert preds["fold"].nunique() >= 2
    assert preds["y_pred"].notna().all()

    metrics = pd.read_parquet(out / "metrics.parquet")
    assert {"task", "model", "fold", "metric", "value"} <= set(metrics.columns)
    assert (metrics.query("task == 'vol_5d' and metric == 'qlike'")["value"] >= 0).all()

    dm = pd.read_parquet(out / "dm.parquet")
    assert set(dm["baseline"]) == {"naive", "lgbm"}
    assert set(dm.query("model == 'lstm'")["seed"]) == {0, 1}
    assert {"stat", "pvalue"} <= set(dm.columns)

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config_hash"] and manifest["data_start"] and manifest["n_folds"] >= 2
    summary = (out / "summary.md").read_text()
    assert "vol_5d" in summary and "lgbm" in summary and "lstm" in summary
