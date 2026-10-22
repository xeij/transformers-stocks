import pandas as pd

from stockbench.data import COLUMNS, load_prices, market_frame


def test_load_prices_caches_each_ticker_as_parquet(tmp_path):
    calls = []

    def source(ticker, start, end):
        calls.append(ticker)
        dates = pd.bdate_range(start, periods=10)
        return pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "adj_close": 1.0,
                "volume": 1.0,
            }
        )

    a = load_prices(["X", "Y"], "2020-01-01", "2020-02-01", source=source, cache_dir=tmp_path)
    b = load_prices(["X", "Y"], "2020-01-01", "2020-02-01", source=source, cache_dir=tmp_path)
    assert calls == ["X", "Y"]  # second call served from cache
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == COLUMNS and len(a) == 20
    assert len(list(tmp_path.glob("*.parquet"))) == 2


def test_market_frame_aligns_to_spy_calendar_and_forward_fills():
    spy_dates = pd.bdate_range("2020-01-06", periods=5)
    rows = []
    for i, d in enumerate(spy_dates):
        rows.append({"date": d, "ticker": "SPY", "adj_close": 100 + i})
        if i != 2:  # vix missing on the third day
            rows.append({"date": d, "ticker": "^VIX", "adj_close": 20 + i})
        rows.append({"date": d, "ticker": "^TNX", "adj_close": 1.5})
        rows.append({"date": d, "ticker": "DX-Y.NYB", "adj_close": 95.0})
    rows.append(
        {"date": pd.Timestamp("2020-01-11"), "ticker": "DX-Y.NYB", "adj_close": 99.0}
    )  # Sat
    panel = pd.DataFrame(rows)
    for c in ["open", "high", "low", "close", "volume"]:
        panel[c] = panel["adj_close"]

    m = market_frame(panel)
    assert list(m.columns) == ["spy", "vix", "dxy", "tnx"]
    assert m.index.equals(spy_dates)
    assert m.loc[spy_dates[2], "vix"] == 21  # forward-filled from the previous day
    assert m["spy"].tolist() == [100, 101, 102, 103, 104]
