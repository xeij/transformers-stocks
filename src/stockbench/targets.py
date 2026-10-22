"""Targets over the next HORIZON bars. `target_end` is the date of bar t+H (the purge key)."""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 5
DEAD_ZONE = 0.0025
TASKS = {"vol_5d": "regression", "dir_5d": "classification", "rank_5d": "regression"}
TARGET_COLS = ["fwd_ret_5d", "vol_5d", "dir_5d", "rank_5d", "target_end"]


def build_targets(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.sort_values(["ticker", "date"], kind="stable")[
        ["date", "ticker", "adj_close"]
    ].copy()
    df["lp"] = np.log(df["adj_close"])
    g = df.groupby("ticker", sort=False)
    H = HORIZON

    df["fwd_ret_5d"] = g["lp"].shift(-H) - df["lp"]
    r2 = g["lp"].diff() ** 2
    trailing = r2.groupby(df["ticker"]).transform(lambda s: s.rolling(H).mean())  # bars j-H+1..j
    df["vol_5d"] = np.log(np.sqrt(trailing.groupby(df["ticker"]).shift(-H)))  # bars t+1..t+H
    fr = df["fwd_ret_5d"]
    df["dir_5d"] = (fr > 0).astype(float).where(fr.abs() >= DEAD_ZONE)
    df["rank_5d"] = df.groupby("date")["fwd_ret_5d"].rank(pct=True)
    # the rank target mixes tickers, so a date's targets are realised when the last of them is
    df["target_end"] = g["date"].shift(-H)
    df["target_end"] = df.groupby("date")["target_end"].transform("max")
    return df.set_index(["date", "ticker"]).sort_index()[TARGET_COLS]
