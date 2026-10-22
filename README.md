# stockbench

A walk-forward benchmark that asks whether transformer models add anything over strong baselines
for short-horizon US equity prediction. Every model sees identical features and folds, is scored
with Diebold-Mariano tests against a naive forecast and LightGBM, and a defensible null result
counts as a valid outcome.

Stack: Python 3.11, pandas, scikit-learn, LightGBM, PyTorch, statsmodels, yfinance; `uv` for the
environment, `ruff` and `pytest` for checks.

## Quickstart

Prerequisites: [uv](https://docs.astral.sh/uv/) 0.4+, and on macOS the OpenMP runtime LightGBM
links against (`brew install libomp`).

```bash
uv sync
sh scripts/fix_macos_venv.sh   # macOS only, see below
uv run pytest
uv run python -m stockbench run --config configs/exp01.yaml
```

The first run downloads daily bars for the 46-ticker universe from yfinance into `data/raw/`
(gitignored, cached as one parquet per ticker). Results land in `reports/exp01/`:
`predictions.parquet`, `metrics.parquet`, `dm.parquet`, `summary.md`, and `manifest.json`
(config hash, git sha, data range, library versions).

## How it works

Three targets over the next five trading days: log realised volatility (regression), return
direction with a 0.25% dead zone (classification), and the cross-sectional percentile rank of the
return (regression). Features are trailing returns, realised vol, volume z-score, RSI, MACD,
rolling max drawdown, a cross-asset block (SPY, VIX, DXY, 10y yield) and calendar one-hots, each
registered as one function in `features.py`.

Evaluation is rolling walk-forward: 5y train, 6m validation, 1y test, stepping 1y, which gives 11
test years from mid-2015 to mid-2026. Training and validation samples whose target window would
reach into the next block are purged, scalers live inside each model and are fit on train rows
only, and the Diebold-Mariano test runs on per-date mean loss differentials with a Newey-West
variance (lag 4) to absorb the overlapping targets. `PLAN.md` has the exact date math and the
argument for why sequence lookbacks cannot leak.

`tests/test_leakage.py` is the contract: it perturbs every bar after a cutoff and asserts all
features before the cutoff are unchanged, and it asserts no train or validation target overlaps a
test block.

## Status

Phase 1 is done: data cache, features, targets, splitter, leakage tests, naive / ridge-logistic /
LightGBM baselines, metrics, DM tests, CLI, first report. Phase 2 (LSTM and PatchTST, multi-seed)
and Phase 3 (long/short backtest with costs) are not started.

Known limitations: the universe is today's large caps with history since 2010, so the panel has
survivorship bias; yfinance adjusted prices are the only source wired up.

`scripts/fix_macos_venv.sh` works around two macOS problems. The torch wheel bundles its own
`libomp.dylib` while LightGBM links Homebrew's, and two OpenMP runtimes in one process hang or
segfault, so the script points torch at the Homebrew copy. It also clears the macOS hidden flag
that some process keeps setting on `.venv`, which makes Python skip the editable-install `.pth`
file ("No module named stockbench"). Rerun it after any `uv sync` that reinstalls torch.
