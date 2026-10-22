"""Mandatory leakage tests. Written before any model code."""

import numpy as np
import pandas as pd

from stockbench.config import SplitConfig
from stockbench.data import market_frame
from stockbench.features import build_features
from stockbench.models import make_model
from stockbench.splits import walk_forward
from stockbench.targets import HORIZON, build_targets
from tests.conftest import TICKERS

PRICE_COLS = ["open", "high", "low", "close", "adj_close"]


def shock_after(prices, cutoff, rng):
    """Randomly perturb every bar strictly after `cutoff`."""
    out = prices.copy()
    after = out["date"] > cutoff
    n = int(after.sum())
    for c in PRICE_COLS:
        out.loc[after, c] = out.loc[after, c] * np.exp(rng.normal(0, 0.05, n))
    out.loc[after, "volume"] = out.loc[after, "volume"] * rng.lognormal(0, 0.5, n)
    return out


def test_every_feature_at_t_uses_only_bars_up_to_t(prices):
    rng = np.random.default_rng(1)
    cutoff = pd.Timestamp(np.sort(prices["date"].unique())[500])
    universe = prices[prices["ticker"].isin(TICKERS)]
    base = build_features(universe, market_frame(prices))

    shocked_all = shock_after(prices, cutoff, rng)
    shocked_universe = shocked_all[shocked_all["ticker"].isin(TICKERS)]
    shocked = build_features(shocked_universe, market_frame(shocked_all))

    past = base.index.get_level_values("date") <= cutoff
    pd.testing.assert_frame_equal(base[past], shocked[past])

    # teeth: every price/volume-derived feature does change after the cutoff
    changed = (base[~past] != shocked[~past]).any()
    calendar = [c for c in base.columns if c.startswith(("dow_", "month_"))]
    assert changed.drop(calendar).all(), changed[~changed].index.tolist()


def test_every_target_at_t_uses_only_bars_t_plus_1_to_t_plus_h(universe):
    rng = np.random.default_rng(2)
    cutoff = pd.Timestamp(np.sort(universe["date"].unique())[500])
    base = build_targets(universe)
    shocked = build_targets(shock_after(universe, cutoff, rng))

    realised = base["target_end"] <= cutoff
    pd.testing.assert_frame_equal(base[realised], shocked[realised])

    # teeth: samples whose horizon straddles the cutoff change, and there are exactly H per ticker
    straddle = (base.index.get_level_values("date") <= cutoff) & ~realised
    assert straddle.sum() == HORIZON * len(TICKERS)
    assert (base.loc[straddle, "vol_5d"] != shocked.loc[straddle, "vol_5d"]).all()


def test_no_test_sample_in_its_folds_train_or_val_and_targets_are_purged():
    dates = pd.bdate_range("2010-01-01", "2026-10-01")
    target_end = pd.Series(dates).shift(-HORIZON)  # date of bar t+H, NaT at the tail
    folds = walk_forward(dates[0], dates[-1], SplitConfig())
    assert len(folds) >= 5

    test_cover = np.zeros(len(dates), dtype=int)
    for f in folds:
        train, val, test = f.assign(dates, target_end)
        assert not (train & val).any()
        assert not (train & test).any()
        assert not (val & test).any()
        assert dates[train].min() >= f.train_start
        assert target_end[train].max() < f.val_start  # train targets realised before val
        assert target_end[val].max() < f.test_start  # val targets realised before test
        assert dates[test].min() >= f.test_start
        assert dates[test].max() < f.test_end
        test_cover += test
    assert test_cover.max() == 1  # every date is tested at most once


def test_fold_dates_match_plan():
    dates = pd.bdate_range("2010-01-01", "2026-10-01")
    folds = walk_forward(dates[0], dates[-1], SplitConfig())
    f0 = folds[0]
    assert (f0.train_start, f0.val_start, f0.test_start, f0.test_end) == (
        pd.Timestamp("2010-01-01"),
        pd.Timestamp("2015-01-01"),
        pd.Timestamp("2015-07-01"),
        pd.Timestamp("2016-07-01"),
    )
    assert len(folds) == 11
    assert folds[-1].test_end == pd.Timestamp("2026-07-01")


def test_model_scaler_is_fit_on_train_rows_only():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    y = pd.Series(rng.normal(size=200))
    X_val = X * 3 + 10  # a different distribution; must not touch the scaler
    m = make_model("linear", "vol_5d", seed=0).fit(X, y, X_val, y)
    np.testing.assert_allclose(m.scaler.mean_, X.mean().values)
    np.testing.assert_allclose(m.scaler.scale_, X.std(ddof=0).values)
