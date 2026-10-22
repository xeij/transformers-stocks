"""Phase-1 models. Each one owns its scaler, fit on the train rows it is given."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stockbench.targets import TASKS


class Naive:
    """vol: random walk (last 5-bar log RV); dir: train class-1 frequency; rank: train mean."""

    def __init__(self, task: str):
        self.task = task

    def fit(self, X, y, X_val, y_val):
        self.const = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "vol_5d":
            return X["rv_5"].to_numpy(dtype=float)
        return np.full(len(X), self.const)


class Linear:
    """Ridge (regression) or logistic (classification) on standardised features.

    The penalty lambda is on the per-sample scale (alpha = lambda * n, C = 1 / (lambda * n))
    and is picked on the validation block by squared error (Brier for classification).
    """

    GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)

    def __init__(self, task: str, seed: int):
        self.classify = TASKS[task] == "classification"

    def _make(self, lam: float, n: int):
        if self.classify:
            est = LogisticRegression(C=1 / (lam * n), max_iter=5000)
        else:
            est = Ridge(alpha=lam * n)
        return make_pipeline(StandardScaler(), est)

    @staticmethod
    def _predict(pipe, X) -> np.ndarray:
        if hasattr(pipe[-1], "predict_proba"):
            return pipe.predict_proba(X)[:, 1]
        return pipe.predict(X)

    def fit(self, X, y, X_val, y_val):
        best = (np.inf, None)
        for lam in self.GRID:
            pipe = self._make(lam, len(X)).fit(X, y)
            loss = float(np.mean((np.asarray(y_val, float) - self._predict(pipe, X_val)) ** 2))
            if loss < best[0]:
                best = (loss, pipe)
        self.pipe = best[1]
        return self

    def predict(self, X) -> np.ndarray:
        return self._predict(self.pipe, X)

    @property
    def scaler(self) -> StandardScaler:
        return self.pipe[0]


class LGBM:
    def __init__(self, task: str, seed: int):
        self.classify = TASKS[task] == "classification"
        params = dict(
            n_estimators=2000, learning_rate=0.02, num_leaves=31, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, deterministic=True, force_row_wise=True, verbose=-1,
        )  # fmt: skip
        self.model = lgb.LGBMClassifier(**params) if self.classify else lgb.LGBMRegressor(**params)

    def fit(self, X, y, X_val, y_val):
        self.model.fit(
            X,
            y,
            eval_X=X_val,
            eval_y=y_val,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        return self

    def predict(self, X) -> np.ndarray:
        if self.classify:
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)


def make_model(name: str, task: str, seed: int):
    if name == "naive":
        return Naive(task)
    if name == "linear":
        return Linear(task, seed)
    if name == "lgbm":
        return LGBM(task, seed)
    raise KeyError(f"unknown model {name!r}")
