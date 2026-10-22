"""Metrics per task and the Diebold-Mariano test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from stockbench.targets import HORIZON

DM_LAG = HORIZON - 1  # overlapping 5-day targets -> MA(4) loss differentials


def sample_loss(task: str, y_true, y_pred) -> np.ndarray:
    """Per-sample loss used by the DM test: squared error (Brier for the direction task)."""
    return (np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2


def hac_tstat(x, lag: int = DM_LAG) -> tuple[float, float]:
    """t-statistic and (normal) p-value for mean(x) = 0 with Newey-West HAC variance."""
    x = np.asarray(x, float)
    if np.ptp(x) == 0:  # constant series: zero variance, t-stat is 0 or infinite by sign
        m = float(x.mean())
        return (0.0, 1.0) if m == 0 else (float(np.sign(m) * np.inf), 0.0)
    res = sm.OLS(x, np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    return float(res.tvalues[0]), float(res.pvalues[0])


def dm_test(loss_a, loss_b, dates, lag: int = DM_LAG) -> tuple[float, float]:
    """DM statistic for model a vs model b. Negative = a has lower loss.

    Losses are averaged across tickers on each date first, so the series the HAC variance is
    computed on is one observation per date.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    per_date = pd.Series(d).groupby(np.asarray(dates)).mean().sort_index()
    return hac_tstat(per_date.to_numpy(), lag)


def _corr(a, b, method: str) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:  # constant input: correlation undefined, report 0
        return 0.0
    c = pd.Series(a).corr(pd.Series(b), method=method)
    return 0.0 if np.isnan(c) else float(c)


def metrics(task: str, df: pd.DataFrame) -> dict[str, float]:
    """df has columns y_true, y_pred, date, fwd_ret_5d (rows = test samples)."""
    y = df["y_true"].to_numpy(float)
    p = df["y_pred"].to_numpy(float)
    if task == "vol_5d":
        ratio = np.exp(2 * (y - p))  # realised variance / forecast variance
        return {
            "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            "mae": float(np.mean(np.abs(y - p))),
            "qlike": float(np.mean(ratio - np.log(ratio) - 1)),
        }
    if task == "dir_5d":
        yhat = (p > 0.5).astype(float)
        return {
            "accuracy": float(accuracy_score(y, yhat)),
            "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
            "auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan"),
        }
    if task == "rank_5d":
        by_date = df.groupby("date", sort=True)
        ic = by_date.apply(lambda g: _corr(g["y_pred"], g["y_true"], "pearson"))
        ric = by_date.apply(lambda g: _corr(g["y_pred"], g["y_true"], "spearman"))
        return {
            "ic": float(ic.mean()),
            "rank_ic": float(ric.mean()),
            "ic_tstat": hac_tstat(ic)[0],
            "rank_ic_tstat": hac_tstat(ric)[0],
        }
    raise KeyError(task)
