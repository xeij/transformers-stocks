"""Long/short backtest of the test-fold predictions.

Every test date t forms a book at the close: equal-weight long the top decile of the day's
predictions and equal-weight short the bottom decile, 1 long / 1 short per unit of capital. A book
is held for HORIZON daily returns and capital is split 1/HORIZON across the live books, so the
weights held through day d average the books formed on the HORIZON dates before d. Weights are
rebalanced daily to target (drift inside a holding period is ignored) and COST_BPS is charged per
unit of notional traded on the day-to-day change in held weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockbench.eval import hac_tstat
from stockbench.targets import HORIZON

COST_BPS = 10.0  # per side, i.e. per unit of notional traded
DECILE = 0.1
DAYS = 252
BT_TASKS = ("dir_5d", "rank_5d")
BENCH = "SPY"


def book(pred: pd.DataFrame, decile: float = DECILE) -> pd.DataFrame:
    """Book weights, formation date x ticker: +1/n on the top decile, -1/n on the bottom.

    Mid-ranks (rank - 0.5) / n put a constant prediction at 0.5 for every name, so the naive (and
    a LightGBM collapsed to the base rate) takes no positions instead of arbitrary ones.
    """
    S = pred.pivot(index="date", columns="ticker", values="y_pred")
    r = (S.rank(axis=1) - 0.5).div(S.count(axis=1), axis=0)
    long = (r > 1 - decile).astype(float)
    short = (r < decile).astype(float)
    long = long.div(long.sum(axis=1), axis=0).fillna(0.0)
    short = short.div(short.sum(axis=1), axis=0).fillna(0.0)
    return long - short


def daily_returns(
    pred: pd.DataFrame,
    prices: pd.DataFrame,
    cost_bps: float = COST_BPS,
    horizon: int = HORIZON,
    decile: float = DECILE,
) -> pd.DataFrame:
    """Daily gross / net simple returns and turnover, first live day to last, for one strategy."""
    C = prices[prices["ticker"].isin(pred["ticker"].unique())]
    R = C.pivot(index="date", columns="ticker", values="adj_close").sort_index().pct_change()
    w = book(pred, decile).reindex(index=R.index, columns=R.columns).fillna(0.0)
    W = sum(w.shift(j, fill_value=0.0) for j in range(1, horizon + 1)) / horizon  # held on day d
    gross = (W * R).sum(axis=1)
    turnover = (W - W.shift(1, fill_value=0.0)).abs().sum(axis=1)
    out = pd.DataFrame({"gross": gross, "net": gross - cost_bps / 1e4 * turnover,
                        "turnover": turnover})  # fmt: skip
    i0 = R.index.get_loc(pred["date"].min()) + 1
    i1 = R.index.get_loc(pred["date"].max()) + horizon
    return out.iloc[i0 : i1 + 1]


def benchmark(prices: pd.DataFrame, index: pd.Index, ticker: str = BENCH) -> pd.DataFrame:
    """Buy and hold `ticker` over the same days, no costs."""
    c = prices[prices["ticker"] == ticker].set_index("date")["adj_close"].sort_index()
    r = c.pct_change().reindex(index).fillna(0.0)
    return pd.DataFrame({"gross": r, "net": r, "turnover": 0.0})


def summarize(d: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in ("gross", "net"):
        r = d[col].to_numpy(float)
        wealth = np.cumprod(1 + r)
        sd = r.std(ddof=1)
        out[f"ann_return_{col}"] = float(wealth[-1] ** (DAYS / len(r)) - 1)
        out[f"sharpe_{col}"] = float(r.mean() / sd * np.sqrt(DAYS)) if sd > 0 else float("nan")
        out[f"max_dd_{col}"] = float((wealth / np.maximum.accumulate(wealth) - 1).min())
    out["tstat_net"] = hac_tstat(d["net"].to_numpy(float))[0]
    out["turnover"] = float(d["turnover"].mean() * DAYS)  # notional traded per year
    out["n_days"] = float(len(d))
    return out


def backtest_table(P: pd.DataFrame, prices: pd.DataFrame, **kw) -> tuple[pd.DataFrame, ...]:
    """Metrics per (task, model, seed) plus a `spy` row per task, and the daily return curves."""
    rows, curves, bench = [], [], {}
    sel = P[P["task"].isin(BT_TASKS)]
    for (task, model, seed), g in sel.groupby(["task", "model", "seed"], sort=False):
        d = daily_returns(g, prices, **kw)
        bench.setdefault(task, benchmark(prices, d.index))
        rows.append({"task": task, "model": model, "seed": seed, **summarize(d)})
        curves.append(d.reset_index().assign(task=task, model=model, seed=seed))
    for task, d in bench.items():
        rows.append({"task": task, "model": "spy", "seed": 0, **summarize(d)})
        curves.append(d.reset_index().assign(task=task, model="spy", seed=0))
    if not rows:
        return pd.DataFrame(columns=["task", "model", "seed"]), pd.DataFrame(columns=["date"])
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)
