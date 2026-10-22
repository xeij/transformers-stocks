import numpy as np
import pandas as pd
import pytest

from stockbench.eval import dm_test, metrics, sample_loss


def test_vol_metrics_are_zero_for_a_perfect_forecast():
    y = pd.Series(np.log([0.01, 0.02, 0.03]))
    df = pd.DataFrame(
        {"y_true": y, "y_pred": y, "date": pd.Timestamp("2020-01-02"), "fwd_ret_5d": 0.0}
    )
    m = metrics("vol_5d", df)
    assert m["rmse"] == 0 and m["mae"] == 0 and m["qlike"] == pytest.approx(0)


def test_qlike_uses_variance_scale():
    # true vol 0.02, predicted 0.01: sigma2/h = 4 -> 4 - log 4 - 1
    df = pd.DataFrame(
        {
            "y_true": [np.log(0.02)],
            "y_pred": [np.log(0.01)],
            "date": pd.Timestamp("2020-01-02"),
            "fwd_ret_5d": 0.0,
        }
    )
    assert metrics("vol_5d", df)["qlike"] == pytest.approx(4 - np.log(4) - 1)


def test_direction_metrics():
    df = pd.DataFrame(
        {
            "y_true": [1, 1, 0, 0],
            "y_pred": [0.9, 0.6, 0.4, 0.2],
            "date": pd.Timestamp("2020-01-02"),
            "fwd_ret_5d": 0.0,
        }
    )
    m = metrics("dir_5d", df)
    assert m["accuracy"] == 1 and m["balanced_accuracy"] == 1 and m["auc"] == 1


def test_rank_ic_is_one_for_a_perfect_ordering():
    rng = np.random.default_rng(0)
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=30), 10)
    fwd = rng.normal(size=300)
    df = pd.DataFrame({"date": dates, "fwd_ret_5d": fwd})
    df["y_true"] = df.groupby("date")["fwd_ret_5d"].rank(pct=True)
    df["y_pred"] = df["y_true"] * 3  # monotone transform keeps the ordering
    m = metrics("rank_5d", df)
    assert m["rank_ic"] == pytest.approx(1)
    assert m["ic"] == pytest.approx(1)  # pearson of pred vs the rank target
    assert m["rank_ic_tstat"] > 10


def test_constant_prediction_has_zero_ic():
    df = pd.DataFrame(
        {
            "date": np.repeat(pd.bdate_range("2020-01-01", periods=5), 4),
            "fwd_ret_5d": np.arange(20.0),
            "y_true": np.arange(20.0),
            "y_pred": 0.5,
        }
    )
    m = metrics("rank_5d", df)
    assert m["rank_ic"] == 0 and m["ic"] == 0


def test_sample_losses():
    np.testing.assert_allclose(
        sample_loss("dir_5d", np.array([1, 0]), np.array([0.8, 0.3])), [0.04, 0.09]
    )
    np.testing.assert_allclose(
        sample_loss("vol_5d", np.array([1.0, 2.0]), np.array([0.5, 2.0])), [0.25, 0.0]
    )
    np.testing.assert_allclose(sample_loss("rank_5d", np.array([1.0]), np.array([0.0])), [1.0])


def test_dm_test_sign_and_significance():
    rng = np.random.default_rng(0)
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=300), 3)  # panel: 3 tickers per date
    loss_b = rng.normal(1.0, 0.1, size=900)
    loss_a = loss_b - 0.5 + rng.normal(0, 0.1, size=900)
    stat, p = dm_test(loss_a, loss_b, dates, lag=4)
    assert stat < -5 and p < 1e-3  # negative = model a is better
    stat_rev, p_rev = dm_test(loss_b, loss_a, dates, lag=4)
    assert stat_rev == pytest.approx(-stat) and p_rev == pytest.approx(p)


def test_dm_test_is_insignificant_for_equal_models():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=300)
    loss_a = rng.normal(1.0, 0.1, size=300)
    loss_b = rng.normal(1.0, 0.1, size=300)
    stat, p = dm_test(loss_a, loss_b, dates, lag=4)
    assert abs(stat) < 2.5 and p > 0.01
