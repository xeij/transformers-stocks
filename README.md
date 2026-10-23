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
`predictions.parquet`, `metrics.parquet`, `dm.parquet`, `backtest.parquet`, `plots/`,
`summary.md`, and `manifest.json` (config hash, git sha, data range, library versions).
`configs/exp02.yaml` adds the LSTM and PatchTST; its fits are cached one parquet each under
`reports/exp02/cache/`, so an interrupted run resumes.

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

The backtest turns the direction and rank predictions into a dollar-neutral book: on each test
date it goes equal-weight long the top decile of predictions and short the bottom decile, holds
each book for five days with capital spread across the five live books, and charges 10 bps per
unit of notional traded. A constant prediction takes no positions, so the naive strategies sit
flat. SPY buy-and-hold over the same days is the benchmark.

## Status

All three phases are done: data cache, features, targets, splitter, leakage tests, naive /
ridge-logistic / LightGBM baselines, metrics, DM tests, CLI; the LSTM and PatchTST sequence
models trained on 64-bar windows pooled across tickers with three seeds each; and the long/short
backtest with costs and the figures. `reports/exp02/summary.md` is the full report.

Results (11 test years, 2015-07 to 2026-07): ridge is the best volatility model (RMSE 0.495
against 0.634 for the random-walk naive and 0.508 for LightGBM); PatchTST ties LightGBM (0.510,
DM t = +0.7) and the LSTM trails both. Direction and cross-sectional rank are null results for
every model: nothing beats the base rate on direction, and the only rank IC with a t-stat above 2
(LSTM, 0.019) comes with a squared error significantly worse than the constant predictor. In the
backtest no strategy survives costs: every decile book loses money net of 10 bps per side
(annualised -0.1% to -18%, Sharpe -0.8 to +0.1) while SPY buy-and-hold returns 14.2% a year at
Sharpe 0.83. Turnover of 100 to 150 units of capital a year costs 10 to 15 points of return, and
the best gross Sharpe (LSTM on rank, 0.51 +- 0.22 across seeds) is below SPY's before costs. A
full exp02 run takes about ten hours on an Apple M5 laptop, nearly all of it PatchTST;
interrupted runs resume from the per-fit cache.

Known limitations: the universe is today's large caps with history since 2010, so the panel has
survivorship bias; yfinance adjusted prices are the only source wired up.

`scripts/fix_macos_venv.sh` works around two macOS problems. The torch wheel bundles its own
`libomp.dylib` while LightGBM links Homebrew's, and two OpenMP runtimes in one process hang or
segfault, so the script points torch at the Homebrew copy. It also clears the macOS hidden flag
that some process keeps setting on `.venv`, which makes Python skip the editable-install `.pth`
file ("No module named stockbench"). Rerun it after any `uv sync` that reinstalls torch.
