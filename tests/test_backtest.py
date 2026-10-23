import numpy as np
import pandas as pd
import pytest

from stockbench.backtest import backtest_table, book, daily_returns
from stockbench.targets import HORIZON, build_targets

D = 0.25  # decile on the 4-name fixture gives one long and one short; 0.1 would give none


def _pred(universe, fn):
    Y = build_targets(universe).dropna(subset=["fwd_ret_5d"]).reset_index()
    Y = Y[Y["date"] >= "2011-01-01"].reset_index(drop=True)
    return pd.DataFrame(
        {"task": "rank_5d", "model": "m", "seed": 0, "fold": 0, "date": Y["date"],
         "ticker": Y["ticker"], "y_true": Y["rank_5d"], "y_pred": fn(Y),
         "fwd_ret_5d": Y["fwd_ret_5d"]}
    )  # fmt: skip


def test_constant_prediction_takes_no_positions(universe):
    pred = _pred(universe, lambda Y: np.full(len(Y), 0.55))
    assert (book(pred, decile=D) == 0).all().all()
    d = daily_returns(pred, universe, decile=D)
    assert (d["net"] == 0).all() and (d["turnover"] == 0).all()


def test_book_is_dollar_neutral_equal_weight(universe):
    rng = np.random.default_rng(1)
    pred = _pred(universe, lambda Y: rng.normal(size=len(Y)))
    w = book(pred, decile=D)
    assert np.allclose(w.sum(axis=1), 0) and np.allclose(w.abs().sum(axis=1), 2)
    assert set(np.unique(w.to_numpy())) == {-1.0, 0.0, 1.0}


def test_one_book_pnl_and_cost_by_hand(universe):
    """A single formation date t: day t+1 earns 1/H of (long return - short return), minus cost."""
    rng = np.random.default_rng(2)
    pred = _pred(universe, lambda Y: rng.normal(size=len(Y)))
    t = pred["date"].iloc[100]
    one = pred[pred["date"] == t]
    d = daily_returns(one, universe, decile=D, cost_bps=10)
    C = universe.pivot(index="date", columns="ticker", values="adj_close")
    nxt = C.index[C.index.get_loc(t) + 1]
    expected = (book(one, decile=D).iloc[0] * C.pct_change().loc[nxt]).sum() / HORIZON
    assert len(d) == HORIZON and d.index[0] == nxt
    assert d.loc[nxt, "gross"] == pytest.approx(expected)
    assert d.loc[nxt, "turnover"] == pytest.approx(2 / HORIZON)
    assert d.loc[nxt, "net"] == pytest.approx(expected - 0.001 * 2 / HORIZON)
    assert (d["turnover"].iloc[1:] == 0).all()


def test_perfect_foresight_profits_and_costs_bite(prices, universe):
    pred = _pred(universe, lambda Y: Y["fwd_ret_5d"].to_numpy())
    d = daily_returns(pred, universe, decile=D)
    assert (1 + d["gross"]).prod() > 2
    assert d["net"].sum() < d["gross"].sum()
    B, curves = backtest_table(pred, prices, decile=D)
    assert set(B["model"]) == {"m", "spy"} and set(curves["model"]) == {"m", "spy"}
    assert B.set_index("model").loc["m", "sharpe_gross"] > 3
    assert B.set_index("model").loc["spy", "turnover"] == 0


def test_returns_use_only_earlier_predictions(universe):
    rng = np.random.default_rng(3)
    pred = _pred(universe, lambda Y: rng.normal(size=len(Y)))
    d0 = daily_returns(pred, universe, decile=D)
    T = pred["date"].iloc[len(pred) // 2]
    pert = pred.copy()
    after = (pert["date"] >= T).to_numpy()
    pert.loc[after, "y_pred"] = rng.normal(size=after.sum())
    d1 = daily_returns(pert, universe, decile=D)
    assert d0.loc[:T].equals(d1.loc[:T])  # day T is held on books formed before T
    assert not d0.loc[T:].equals(d1.loc[T:])
