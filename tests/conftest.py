import numpy as np
import pandas as pd
import pytest

TICKERS = ["AAA", "BBB", "CCC", "DDD"]
MARKET_TICKERS = ["SPY", "^VIX", "DX-Y.NYB", "^TNX"]


def random_walk_prices(tickers, dates, rng):
    frames = []
    for t in tickers:
        n = len(dates)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": t,
                    "open": close * (1 + rng.normal(0, 0.002, n)),
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "adj_close": close,
                    "volume": rng.lognormal(15, 0.5, n),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def prices():
    """Synthetic long panel: 4 universe tickers + the 4 market series, 800 business days."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2010-01-01", periods=800)
    return random_walk_prices(TICKERS + MARKET_TICKERS, dates, rng)


@pytest.fixture
def universe(prices):
    return prices[prices["ticker"].isin(TICKERS)].reset_index(drop=True)
