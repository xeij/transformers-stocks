"""Rolling walk-forward folds on calendar dates with a horizon purge. See PLAN.md for the math."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockbench.config import SplitConfig


@dataclass(frozen=True)
class Fold:
    k: int
    train_start: pd.Timestamp
    val_start: pd.Timestamp  # == train end
    test_start: pd.Timestamp  # == val end
    test_end: pd.Timestamp

    def assign(self, dates, target_end) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Boolean masks (train, val, test) for samples at `dates` with targets ending `target_end`.

        Train/val samples must have their target fully realised before the next block starts.
        NaT target_end (unrealised) never enters train or val.
        """
        d = pd.DatetimeIndex(np.asarray(dates))
        te = pd.DatetimeIndex(np.asarray(target_end))
        train = (d >= self.train_start) & (te < self.val_start)
        val = (d >= self.val_start) & (te < self.test_start)
        test = (d >= self.test_start) & (d < self.test_end)
        return np.asarray(train), np.asarray(val), np.asarray(test)


def walk_forward(origin, last, cfg: SplitConfig) -> list[Fold]:
    """Folds anchored at `origin`; a fold is kept only if its whole test block is <= `last`."""
    origin, last = pd.Timestamp(origin), pd.Timestamp(last)
    folds: list[Fold] = []
    k = 0
    while True:
        train_start = origin + pd.DateOffset(months=k * cfg.step_months)
        val_start = train_start + pd.DateOffset(months=cfg.train_months)
        test_start = val_start + pd.DateOffset(months=cfg.val_months)
        test_end = test_start + pd.DateOffset(months=cfg.test_months)
        if test_end - pd.Timedelta(days=1) > last:
            return folds
        folds.append(Fold(k, train_start, val_start, test_start, test_end))
        k += 1
