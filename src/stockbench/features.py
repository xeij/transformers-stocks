"""Feature registry. One function per feature; every value at date t uses bars <= t only."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

FeatureFn = Callable[[pd.DataFrame], "pd.Series | pd.DataFrame"]


@dataclass(frozen=True)
class Feature:
    fn: FeatureFn
    scope: str  # "ticker": gets one ticker's bars (date index); "market": gets market_frame()


REGISTRY: dict[str, Feature] = {}


def feature(name: str, scope: str = "ticker"):
    def deco(fn: FeatureFn) -> FeatureFn:
        REGISTRY[name] = Feature(fn, scope)
        return fn

    return deco


def _log_ret(p: pd.Series) -> pd.Series:
    return np.log(p).diff()


def _log_rv(r: pd.Series, w: int) -> pd.Series:
    """log sqrt(mean r^2) over the trailing w bars, inclusive of t."""
    return np.log(np.sqrt((r**2).rolling(w).mean()))


# --- per-ticker -------------------------------------------------------------------------------
for _k in (1, 2, 3, 5, 10, 21):
    feature(f"ret_{_k}")(lambda df, k=_k: np.log(df["adj_close"]).diff(k))
for _w in (5, 21, 63):
    feature(f"rv_{_w}")(lambda df, w=_w: _log_rv(_log_ret(df["adj_close"]), w))


@feature("vol_z")
def vol_z(df: pd.DataFrame) -> pd.Series:
    lv = np.log(df["volume"].where(df["volume"] > 0))
    return (lv - lv.rolling(21).mean()) / lv.rolling(21).std()


@feature("rsi_14")
def rsi_14(df: pd.DataFrame) -> pd.Series:
    d = df["adj_close"].diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def _macd_line(p: pd.Series) -> pd.Series:
    return p.ewm(span=12, adjust=False).mean() - p.ewm(span=26, adjust=False).mean()


@feature("macd")
def macd(df: pd.DataFrame) -> pd.Series:
    return _macd_line(df["adj_close"]) / df["adj_close"]


@feature("macd_hist")
def macd_hist(df: pd.DataFrame) -> pd.Series:
    line = _macd_line(df["adj_close"])
    return (line - line.ewm(span=9, adjust=False).mean()) / df["adj_close"]


@feature("mdd_21")
def mdd_21(df: pd.DataFrame) -> pd.Series:
    """Worst peak-to-trough drawdown inside the trailing 21-bar window."""
    p = df["adj_close"].to_numpy(dtype=float)
    out = np.full(len(p), np.nan)
    if len(p) >= 21:
        w = np.lib.stride_tricks.sliding_window_view(p, 21)
        out[20:] = (w / np.maximum.accumulate(w, axis=1) - 1).min(axis=1)
    return pd.Series(out, index=df.index)


# --- market-wide (same value for every ticker on a date) -----------------------------------------
for _k in (1, 5, 21):
    feature(f"spy_ret_{_k}", "market")(lambda m, k=_k: np.log(m["spy"]).diff(k))
feature("spy_rv_21", "market")(lambda m: _log_rv(_log_ret(m["spy"]), 21))
feature("vix", "market")(lambda m: np.log(m["vix"]))
feature("tnx", "market")(lambda m: m["tnx"])
for _k in (1, 5):
    feature(f"vix_chg_{_k}", "market")(lambda m, k=_k: np.log(m["vix"]).diff(k))
    feature(f"dxy_ret_{_k}", "market")(lambda m, k=_k: np.log(m["dxy"]).diff(k))
    feature(f"tnx_chg_{_k}", "market")(lambda m, k=_k: m["tnx"].diff(k))


def _onehot(values, categories, prefix: str, index) -> pd.DataFrame:
    cat = pd.Categorical(values, categories=list(categories))
    return pd.get_dummies(cat, prefix=prefix, dtype=float).set_axis(index)


@feature("dow", "market")
def dow(m: pd.DataFrame) -> pd.DataFrame:
    return _onehot(m.index.day_of_week, range(5), "dow", m.index)


@feature("month", "market")
def month(m: pd.DataFrame) -> pd.DataFrame:
    return _onehot(m.index.month, range(1, 13), "month", m.index)


# --- assembly ---------------------------------------------------------------------------------
def _named(out: pd.Series | pd.DataFrame, name: str) -> pd.Series | pd.DataFrame:
    return out.rename(name) if isinstance(out, pd.Series) else out


def build_features(prices: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Every registered feature for the panel, indexed by (date, ticker); NaN rows dropped."""
    ticker_fns = {n: f.fn for n, f in REGISTRY.items() if f.scope == "ticker"}
    market_fns = {n: f.fn for n, f in REGISTRY.items() if f.scope == "market"}

    per_ticker = []
    for t, g in prices.groupby("ticker", sort=True):
        g = g.sort_values("date").set_index("date")
        cols = pd.concat([_named(fn(g), n) for n, fn in ticker_fns.items()], axis=1)
        per_ticker.append(cols.assign(ticker=t).set_index("ticker", append=True))
    X = pd.concat(per_ticker)
    M = pd.concat([_named(fn(market), n) for n, fn in market_fns.items()], axis=1)
    M.index.name = "date"
    X = X.join(M, how="left")
    return X.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
