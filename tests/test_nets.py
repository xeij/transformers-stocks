import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from stockbench.config import NetConfig
from stockbench.models import make_model
from stockbench.nets import make_panel

FEATS = ["f0", "f1", "f2", "f3"]


def lookback_panel(n_tickers=3, n_dates=1200, seed=0):
    """Target = rolling 5-bar sum of f0: a row's own features carry only 1/sqrt(5) of it."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_dates)
    frames = []
    for i in range(n_tickers):
        F = rng.normal(size=(n_dates, len(FEATS)))
        y = pd.Series(F[:, 0]).rolling(5, min_periods=1).sum().to_numpy() + rng.normal(
            0, 0.1, n_dates
        )
        df = pd.DataFrame(F, columns=FEATS)
        df["y"], df["date"], df["ticker"] = y, dates, f"T{i}"
        frames.append(df)
    return pd.concat(frames).set_index(["date", "ticker"]).sort_index()


SMALL = dict(lookback=16, patch_len=4, patch_stride=2, batch_size=256, device="cpu")


def date_split(df, a=0.6, b=0.75):
    d = df.index.get_level_values("date")
    u = np.sort(d.unique())
    cut1, cut2 = u[int(len(u) * a)], u[int(len(u) * b)]
    return np.asarray(d < cut1), np.asarray((d >= cut1) & (d < cut2)), np.asarray(d >= cut2)


@pytest.mark.parametrize("name", ["lstm", "patchtst"])
def test_sequence_models_learn_a_lookback_signal(name):
    df, panel = make_panel(lookback_panel(), FEATS, lookback=16)
    tr, va, te = date_split(df)
    y = df["y"].to_numpy()
    cfg = NetConfig(max_epochs=25, patience=6, lr=3e-3, **SMALL)
    m = make_model(name, "vol_5d", seed=0, nets=cfg).fit(panel, y, tr, va)
    p = m.predict(panel, te)
    assert p.shape == (te.sum(),)
    assert np.corrcoef(p, y[te])[0, 1] > 0.8, name


@pytest.mark.parametrize("name", ["lstm", "patchtst"])
def test_sequence_models_classify(name):
    df, panel = make_panel(lookback_panel(), FEATS, lookback=16)
    tr, va, te = date_split(df)
    y = (df["y"].to_numpy() > 0).astype(float)
    cfg = NetConfig(max_epochs=25, patience=6, lr=3e-3, **SMALL)
    m = make_model(name, "dir_5d", seed=0, nets=cfg).fit(panel, y, tr, va)
    p = m.predict(panel, te)
    assert p.min() >= 0 and p.max() <= 1
    assert roc_auc_score(y[te], p) > 0.8, name


@pytest.mark.parametrize("name", ["lstm", "patchtst"])
def test_sequence_models_are_deterministic_for_a_seed(name):
    df, panel = make_panel(lookback_panel(n_dates=400), FEATS, lookback=16)
    tr, va, te = date_split(df)
    y = df["y"].to_numpy()
    cfg = NetConfig(max_epochs=2, **SMALL)
    a = make_model(name, "vol_5d", seed=1, nets=cfg).fit(panel, y, tr, va).predict(panel, te)
    b = make_model(name, "vol_5d", seed=1, nets=cfg).fit(panel, y, tr, va).predict(panel, te)
    np.testing.assert_array_equal(a, b)


def test_sequence_model_scaler_and_target_stats_use_train_rows_only():
    df, panel = make_panel(lookback_panel(n_dates=400), FEATS, lookback=16)
    tr, va, te = date_split(df)
    y = df["y"].to_numpy()
    m = make_model("lstm", "vol_5d", seed=0, nets=NetConfig(max_epochs=1, **SMALL))
    m.fit(panel, y, tr, va)
    np.testing.assert_allclose(m.x_mean, panel.F[tr].mean(0), rtol=1e-5)
    np.testing.assert_allclose(m.x_std, panel.F[tr].std(0), rtol=1e-5)
    assert m.y_mean == pytest.approx(y[tr].mean()) and m.y_std == pytest.approx(y[tr].std())
