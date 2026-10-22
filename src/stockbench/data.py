"""Price loading with a swappable source and a per-ticker parquet cache."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
MARKET = {"spy": "SPY", "vix": "^VIX", "dxy": "DX-Y.NYB", "tnx": "^TNX"}

# A source maps (ticker, start, end) -> frame with COLUMNS. Swap in another vendor here.
Source = Callable[[str, str, str], pd.DataFrame]


def fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    h = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False, actions=False)
    if h.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")
    h = h.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                 "Adj Close": "adj_close", "Volume": "volume"}
    )  # fmt: skip
    h.index = h.index.tz_localize(None).normalize()
    h.index.name = "date"
    out = h[COLUMNS[2:]].reset_index()
    out.insert(1, "ticker", ticker)
    return out


def load_prices(
    tickers: list[str],
    start: str,
    end: str,
    source: Source = fetch_yfinance,
    cache_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Long panel sorted by (ticker, date). Each ticker/range is cached once as parquet."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in tickers:
        f = cache_dir / f"{t}_{start}_{end}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
        else:
            log.info("fetching %s %s..%s", t, start, end)
            df = source(t, start, end)
            df.to_parquet(f, index=False)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)[COLUMNS]
    out["date"] = pd.to_datetime(out["date"])
    out = out.dropna(subset=["adj_close"])
    return out.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)


def market_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """SPY/VIX/DXY/TNX levels on the SPY calendar; gaps forward-filled (past data only)."""
    wide = prices.pivot(index="date", columns="ticker", values="adj_close")
    out = wide[list(MARKET.values())].rename(columns={v: k for k, v in MARKET.items()})
    out = out[out["spy"].notna()].ffill()
    out.columns.name = None
    return out[list(MARKET)]
