# PLAN

Benchmark: do transformers beat honest baselines for 5-day equity volatility, direction and
cross-sectional rank prediction under leak-free rolling walk-forward evaluation? A defensible null
result is a success. Package name: `stockbench`.

## Module responsibilities (`src/stockbench/`)

| module | responsibility |
|---|---|
| `config.py` | dataclasses mirroring `configs/*.yaml`; `load_config`, `config_hash` |
| `data.py` | `load_prices(tickers, start, end, source, cache_dir)`: source is any callable `(ticker, start, end) -> OHLCV frame` (default `fetch_yfinance`); results cached per ticker as parquet in `data/raw/`. `market_frame` aligns SPY, ^VIX, DX-Y.NYB, ^TNX to the SPY calendar (forward-fill = past only) |
| `features.py` | feature registry: `@feature(name, scope="ticker"|"market")`, one function per feature; `build_features(panel, market)` -> frame indexed by (date, ticker) |
| `targets.py` | `fwd_ret_5d`, `vol_5d`, `dir_5d`, `rank_5d`, and `target_end` (date of bar t+H) |
| `splits.py` | `walk_forward(first, last, cfg) -> list[Fold]`; `Fold.assign(dates, target_end)` -> train/val/test boolean masks with purge |
| `models.py` | `Model` protocol `fit(X, y, X_val, y_val)` / `predict(X)`; `naive`, `linear` (ridge or logistic by task), `lgbm`; `make_model(name, task, seed)` |
| `eval.py` | per-task metrics, `dm_test` (HAC), fold/pooled summaries |
| `run.py` | the experiment loop; writes `reports/<exp>/{predictions,metrics,dm,backtest,backtest_daily}.parquet`, `plots/*.png`, `summary.md`, `manifest.json` |
| `__main__.py` | `python -m stockbench run --config configs/exp01.yaml` |
| `nets.py` (phase 2) | window dataset, LSTM, PatchTST, torch training loop with seeds |
| `backtest.py` (phase 3) | decile long/short books from the direction and rank predictions, overlapping 5-day holding, costs, SPY benchmark |
| `plots.py` (phase 3) | PNG figures: per-fold metric per task, net wealth curves |

`tests/` builds synthetic random-walk panels (no network) and holds the leakage tests, which were
written before any model code.

## Walk-forward date math

Config: `start=2010-01-01`, `train=5y`, `val=6mo`, `test=1y`, `step=1y`, horizon `H=5` bars.
All blocks are half-open calendar intervals `[start, end)`.

```
fold k (k = 0, 1, ...):
  train = [2010-01-01 + k·1y, 2015-01-01 + k·1y)
  val   = [2015-01-01 + k·1y, 2015-07-01 + k·1y)
  test  = [2015-07-01 + k·1y, 2016-07-01 + k·1y)
keep fold k iff test.end <= last available bar  ->  data through 2026-10 gives k = 0..10, 11 folds
(minimum of 5 folds is asserted at run time)
```

A sample is `(ticker, t)`. Its target uses bars `t+1 .. t+H`; `target_end(t)` is the date of bar
`t+H`, taken as the max across tickers on date `t` because the rank target mixes tickers. Block
membership (the purge is the `target_end` condition):

```
train: train.start <= t  and  target_end(t) <  train.end     # target realised before val starts
val:   val.start   <= t  and  target_end(t) <  val.end       # realised before test starts
test:  test.start  <= t  and  t < test.end                   # target may run H bars past test.end:
                                                             # that is the future for every model
```

Consequences: no sample is in two blocks of the same fold; no train or val target overlaps the
test block; test blocks of different folds are disjoint and contiguous (2015-07 .. 2026-07).
Later folds train on earlier folds' test years, which is what rolling walk-forward means.

The train window is used only for fitting; val only for early stopping / regularisation choice.
No refit on train+val, so every model sees exactly the same rows.

Scalers live inside each model (sklearn `Pipeline(StandardScaler, ...)`, or the torch wrapper)
and are fit on the fold's train rows only.

## Sequence tensors (phase 2) and why the lookback cannot leak

Per ticker `i`, the scaled feature matrix `F_i` has shape `(T_i, d)`, one row per bar, sorted by
date. The window for sample `(i, t)` is

```
W(i, t) = F_i[pos(t) - L + 1 : pos(t) + 1]        # L = 64 bars, ends at t inclusive
```

built with `np.lib.stride_tricks.sliding_window_view(F_i, L, axis=0)` (zero copy) and indexed by
the sample's row position. Every row in `W(i,t)` has date `<= t`, and every feature row is itself
computed from bars `<= its date`, so the window contains no information after `t`.

* Train windows: `t <= train.end - H bars`, so no train window or train target touches val/test.
  Same for val windows and `val.end`.
* Test windows: rows with date `<= t`. The first `L-1` test dates of a fold have windows reaching
  back into val/train rows. That is the past at prediction time, and it keeps the test set
  identical across tabular and sequence models. **Interpretation of "lookback must not cross the
  train/test boundary": no training or validation window or target may touch a later block.**
  If the stricter rule is wanted (test windows entirely inside the test block), the first `L-1`
  test dates per fold are dropped for all models; one flag, say so.
* Samples with fewer than `L` prior bars (the first 63 bars of 2010 only) are dropped for the
  sequence models; this touches fold 0's train block only. Val/test sets are identical for all
  models.
* The phase-2 leakage test asserts: every window row date `<= t`; every train/val window row date
  `< block end`; `W(i,t)` equals the per-row features at dates `t-L+1 .. t` bit for bit.

## Choices made (not in the brief)

1. Config = plain dataclasses + YAML. No hydra.
2. Universe = 45 large-cap US names with continuous history since 2010-01 + SPY (listed in
   `configs/exp01.yaml`). Chosen as of 2026, so the panel has survivorship bias; the rank task is
   the most exposed. Point-in-time constituents would need a paid source.
3. "Log return at lag k" = cumulative k-bar log return ending at t (`log P_t - log P_{t-k}`).
4. Realised vol = `log sqrt(mean r^2)` over the window, same functional form for the features
   (`rv_5/21/63`, bars `t-w+1..t`) and the target `vol_5d` (bars `t+1..t+5`), so the random-walk
   naive is exactly the feature `rv_5`.
5. Volume z-score = log volume vs trailing 21-bar mean/std. RSI(14) Wilder EWM. MACD =
   `(EMA12 - EMA26) / close` plus histogram vs 9-bar signal. Max drawdown 21d = min over the 21-bar
   window of `price / running max - 1`.
6. Cross-asset block: SPY 1/5/21-bar log return and `rv_21`; log VIX level, 1/5-bar change; DXY
   1/5-bar log return; TNX level, 1/5-bar change. All on the SPY calendar, forward-filled.
7. Targets: `fwd_ret_5d = log(P_{t+5}/P_t)`; `dir_5d` dead zone on that log return; `rank_5d` =
   percentile rank in (0, 1] across the whole universe (SPY included) per date.
8. Naive per task: vol -> random walk (`rv_5`); dir -> train majority class with probability =
   class frequency; rank -> train mean (constant; IC reported as 0).
9. Linear = ridge (regression) / logistic (dir) on standardised features. The penalty is on the
   per-sample scale, lambda in `{1e-4, 1e-3, 1e-2, 1e-1, 1}` (ridge alpha = lambda*n, logistic
   C = 1/(lambda*n)), picked on val by squared error (Brier for dir). LightGBM: 2000 trees max,
   lr 0.02, 31 leaves, min 100 samples per leaf, 0.8 row/column subsampling, early stop 100 on
   val. No hyperparameter search beyond that; the baselines are deliberately plain.
10. DM test: per-date mean loss differential across tickers (removes cross-sectional correlation
    from the series), then a HAC (Newey-West, lag H-1 = 4) t-test on its mean. Reported per fold
    and pooled over all test dates. Losses: squared error (vol, rank), Brier (dir). No small-sample
    correction (about 250 dates per fold).
11. "Spearman IC" and "rank IC" are the same statistic. The report gives, per date, the Pearson
    IC and the Spearman IC of the prediction against the rank target (Spearman against the raw
    return is identical), then the mean across dates and a HAC t-stat for each.
12. Rows with any NaN feature are dropped once; rows with a NaN target are dropped per task only
    (the dir dead zone removes rows from the dir task, nothing else).
13. Seeds: `seed` in config feeds numpy, LightGBM and (phase 2) torch. Manifest records config
    hash, git sha, data range, library versions.

14. LSTM: 2 layers x 64 hidden, dropout 0.2 between layers and before the head; the last hidden
    state feeds a linear output.
15. PatchTST: channel-independent. Each of the 43 feature channels is cut into 15 patches
    (length 8, stride 4), linearly embedded to d_model 64 with a learnable positional embedding,
    and run through 2 pre-norm encoder layers (4 heads, FFN 128, GELU, dropout 0.3) shared across
    channels. A shared flatten head (15 x 64 -> 1) reads one number per channel and a linear mixer
    over the 43 channels gives the output. No RevIN: the level of inputs such as `rv_5` is signal
    for these targets, not nuisance.
16. Net training: standardised features clipped to +-5; regression targets standardised with train
    stats; MSE or BCE-with-logits; Adam lr 1e-3, batch 512, gradient clipping 1.0; up to 20 epochs,
    early stopping on val loss with patience 3 and best-epoch restore. Seeds `[0, 1, 2]` are three
    separate runs each; the report shows the across-seed mean and the seed std, and the DM test is
    run per seed.
17. Device: Apple GPU via MPS. Seeds fix initialisation, shuffling and dropout masks; CPU runs are
    bitwise reproducible (tested). PyTorch does not guarantee MPS determinism, but repeated fits of
    both nets on MPS produced bitwise identical predictions on the development machine.
18. Rows without a full 64-bar window (the first 63 bars of each ticker) are dropped from the
    sequence models' training set only; a val or test row without a window raises.
19. macOS: the torch wheel bundles its own `libomp.dylib` and LightGBM links Homebrew's; two
    OpenMP runtimes in one process hang or segfault. `scripts/fix_macos_venv.sh` points torch at
    the Homebrew copy (and clears the hidden flag some process keeps putting on `.venv`).
20. Backtest book: on each test date the names are mid-ranked by prediction, `(rank - 0.5) / n`;
    the top decile (`> 0.9`) is held long and the bottom decile (`< 0.1`) short, equal weight,
    1 long / 1 short per unit of capital (46 names give 5 per side). Ties share the average rank,
    so a constant prediction (the naive, or LightGBM collapsed to the base rate) takes no
    positions rather than arbitrary ones. Direction trades on the predicted probability, rank on
    the predicted percentile; volatility has no backtest.
21. Holding: overlapping books. The book formed at the close of t earns the daily returns of
    t+1..t+5 and capital is split 1/5 across the five live books, so the weights held on day d
    are the average of the books formed on the five previous dates (the calendar-time portfolio
    of Jegadeesh and Titman). Weights are reset to target every day; drift inside a holding
    period is ignored. Daily simple returns from adjusted closes.
22. Costs and statistics: 10 bps per unit of notional traded, charged on `sum |W_d - W_{d-1}|`
    (a complete change of book costs 4 units x 10 bps per 5 days). Returns are raw (no risk-free
    rate, no short rebate) and annualised geometrically; Sharpe is mean / std of daily returns
    x sqrt(252); turnover is annualised notional traded per unit of capital; the HAC t-stat is
    the Newey-West (lag 4) test of mean daily net return = 0. The benchmark is SPY buy-and-hold
    over the same days with no costs.
23. The direction model predicts on every test row, including the dead-zone rows that carry no
    label; metrics and DM tests use labelled rows only. Otherwise the tradable universe on a
    date would be the names that in hindsight moved more than 0.25%.
24. Figures: the per-fold primary metric per task, and the net wealth curve per backtest task
    where a multi-seed model's seeds are averaged into one daily return series (an equal-weight
    portfolio of the seed strategies).

## Phases

1. Scaffold, loader + cache, features, targets, splitter, leakage tests, naive/linear/lgbm,
   metrics + DM, CLI, first report. Stop for review.
2. `nets.py`: LSTM and PatchTST (pooled across tickers), >= 3 seeds, same harness, mean +- std.
3. `backtest.py`: 5-day holding long/short from direction and rank, 10 bps per side, versus SPY;
   final report with plots.
