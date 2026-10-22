import numpy as np
import pytest

from stockbench.data import market_frame
from stockbench.features import REGISTRY, build_features

EXPECTED = {
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_21",
    "rv_5", "rv_21", "rv_63", "vol_z", "rsi_14", "macd", "macd_hist", "mdd_21",
    "spy_ret_1", "spy_ret_5", "spy_ret_21", "spy_rv_21",
    "vix", "vix_chg_1", "vix_chg_5", "dxy_ret_1", "dxy_ret_5", "tnx", "tnx_chg_1", "tnx_chg_5",
    "dow_0", "dow_4", "month_1", "month_12",
}  # fmt: skip


def test_feature_definitions(prices, universe):
    X = build_features(universe, market_frame(prices))
    assert EXPECTED <= set(X.columns)
    assert not X.isna().any().any()
    assert {"ret_1", "rv_5", "spy_ret_1", "dow", "month"} <= set(REGISTRY)

    one = X.xs("AAA", level="ticker")
    p = universe[universe["ticker"] == "AAA"].set_index("date")
    lp = np.log(p["adj_close"])
    r = lp.diff()
    t = one.index[100]
    i = p.index.get_loc(t)

    assert one.loc[t, "ret_5"] == pytest.approx(lp.iloc[i] - lp.iloc[i - 5])
    assert one.loc[t, "rv_21"] == pytest.approx(
        np.log(np.sqrt(np.mean(r.iloc[i - 20 : i + 1] ** 2)))
    )
    window = p["adj_close"].iloc[i - 20 : i + 1].to_numpy()
    assert one.loc[t, "mdd_21"] == pytest.approx((window / np.maximum.accumulate(window) - 1).min())
    assert one["mdd_21"].max() <= 0
    assert 0 <= one["rsi_14"].min() and one["rsi_14"].max() <= 100
    assert one.loc[t, f"dow_{t.day_of_week}"] == 1 and one.filter(like="dow_").loc[t].sum() == 1
    assert one.loc[t, f"month_{t.month}"] == 1 and one.filter(like="month_").loc[t].sum() == 1

    m = market_frame(prices)
    ls = np.log(m["spy"])
    assert one.loc[t, "spy_ret_5"] == pytest.approx(ls.loc[t] - ls.iloc[m.index.get_loc(t) - 5])
    assert one.loc[t, "vix"] == pytest.approx(np.log(m.loc[t, "vix"]))
