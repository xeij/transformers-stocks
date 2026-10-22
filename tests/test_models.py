import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from stockbench.models import make_model


def test_naive_vol_is_the_random_walk_forecast():
    X = pd.DataFrame({"rv_5": [1.0, 2.0, 3.0], "other": [9.0, 9.0, 9.0]})
    y = pd.Series([0.0, 0.0, 0.0])
    m = make_model("naive", "vol_5d", seed=0).fit(X, y, X, y)
    np.testing.assert_array_equal(m.predict(X), X["rv_5"].to_numpy())


def test_naive_dir_predicts_the_train_majority_frequency():
    X = pd.DataFrame({"rv_5": np.zeros(4)})
    y = pd.Series([1.0, 1.0, 1.0, 0.0])
    m = make_model("naive", "dir_5d", seed=0).fit(X, y, X, y)
    np.testing.assert_allclose(m.predict(X.iloc[:2]), [0.75, 0.75])


def test_naive_rank_predicts_the_train_mean():
    X = pd.DataFrame({"rv_5": np.zeros(4)})
    y = pd.Series([0.2, 0.4, 0.6, 0.8])
    m = make_model("naive", "rank_5d", seed=0).fit(X, y, X, y)
    np.testing.assert_allclose(m.predict(X), 0.5)


def _linear_problem(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y = 2 * X["f0"] - X["f1"] + rng.normal(0, 0.1, n)
    tr, va, te = slice(0, 2000), slice(2000, 2500), slice(2500, n)
    return X, y, tr, va, te


def test_linear_and_lgbm_learn_a_regression_signal():
    X, y, tr, va, te = _linear_problem()
    for name in ["linear", "lgbm"]:
        m = make_model(name, "vol_5d", seed=0).fit(X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va])
        pred = m.predict(X.iloc[te])
        assert pred.shape == (500,)
        assert np.corrcoef(pred, y.iloc[te])[0, 1] > 0.9, name


def test_linear_and_lgbm_learn_a_classification_signal():
    X, y, tr, va, te = _linear_problem()
    yb = (y > 0).astype(float)
    for name in ["linear", "lgbm"]:
        m = make_model(name, "dir_5d", seed=0).fit(X.iloc[tr], yb.iloc[tr], X.iloc[va], yb.iloc[va])
        p = m.predict(X.iloc[te])
        assert p.min() >= 0 and p.max() <= 1
        assert roc_auc_score(yb.iloc[te], p) > 0.9, name


def test_lgbm_is_deterministic_for_a_seed():
    X, y, tr, va, te = _linear_problem()
    a = make_model("lgbm", "vol_5d", seed=1).fit(X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va])
    b = make_model("lgbm", "vol_5d", seed=1).fit(X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va])
    np.testing.assert_array_equal(a.predict(X.iloc[te]), b.predict(X.iloc[te]))
