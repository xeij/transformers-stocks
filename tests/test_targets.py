import numpy as np
import pytest

from stockbench.targets import DEAD_ZONE, build_targets
from tests.conftest import TICKERS


def test_vol_and_return_use_the_next_five_bars(universe):
    Y = build_targets(universe)
    one = Y.xs("AAA", level="ticker")
    p = universe[universe["ticker"] == "AAA"].set_index("date")
    lp = np.log(p["adj_close"])
    r = lp.diff()
    t = one.index[50]
    i = p.index.get_loc(t)
    assert one.loc[t, "fwd_ret_5d"] == pytest.approx(lp.iloc[i + 5] - lp.iloc[i])
    assert one.loc[t, "vol_5d"] == pytest.approx(
        np.log(np.sqrt(np.mean(r.iloc[i + 1 : i + 6] ** 2)))
    )
    assert one.loc[t, "target_end"] == p.index[i + 5]
    assert one["target_end"].iloc[-5:].isna().all()
    assert one["vol_5d"].iloc[-5:].isna().all()


def test_direction_has_dead_zone(universe):
    Y = build_targets(universe)
    small = Y["fwd_ret_5d"].abs() < DEAD_ZONE
    assert small.any()
    assert Y.loc[small, "dir_5d"].isna().all()
    big = ~small & Y["fwd_ret_5d"].notna()
    assert (Y.loc[big, "dir_5d"] == (Y.loc[big, "fwd_ret_5d"] > 0).astype(float)).all()


def test_rank_is_percentile_within_each_date(universe):
    Y = build_targets(universe).dropna(subset=["fwd_ret_5d"])
    per_date = Y.groupby(level="date")["rank_5d"]
    assert (per_date.max() == 1).all()
    assert np.allclose(per_date.min(), 1 / len(TICKERS))
    d0 = Y.index.get_level_values("date")[0]
    day = Y.xs(d0, level="date")
    assert day["rank_5d"].idxmax() == day["fwd_ret_5d"].idxmax()
