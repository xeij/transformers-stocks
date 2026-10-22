"""The experiment loop: data -> features/targets -> folds -> models -> metrics/DM -> report."""

from __future__ import annotations

import json
import logging
import platform
import random
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from stockbench.config import Config, config_hash
from stockbench.data import MARKET, Source, fetch_yfinance, load_prices, market_frame
from stockbench.eval import dm_test, metrics, sample_loss
from stockbench.features import build_features
from stockbench.models import make_model
from stockbench.splits import walk_forward
from stockbench.targets import TASKS, build_targets

log = logging.getLogger("stockbench")
POOLED = -1  # fold id for rows computed over all test folds together
BASELINES = ("naive", "lgbm")


def run(cfg: Config, source: Source = fetch_yfinance) -> Path:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)

    tickers = list(dict.fromkeys(cfg.data.tickers + list(MARKET.values())))
    prices = load_prices(tickers, cfg.data.start, cfg.data.end, source, cfg.data.cache_dir)
    universe = prices[prices["ticker"].isin(cfg.data.tickers)]
    X = build_features(universe, market_frame(prices))
    Y = build_targets(universe)
    df = X.join(Y, how="inner")
    feature_cols = list(X.columns)
    dates = df.index.get_level_values("date")
    tickers_idx = df.index.get_level_values("ticker")
    log.info("panel: %d rows, %d features, %s..%s", len(df), len(feature_cols), dates.min().date(),
             dates.max().date())  # fmt: skip

    folds = walk_forward(cfg.data.start, dates.max(), cfg.splits)
    if len(folds) < cfg.splits.min_folds:
        raise RuntimeError(f"only {len(folds)} folds, need {cfg.splits.min_folds}")

    preds = []
    for task in cfg.tasks:
        valid = df[task].notna().to_numpy()
        for f in folds:
            tr, va, te = (m & valid for m in f.assign(dates, df["target_end"]))
            Xtr, Xva, Xte = (df.loc[m, feature_cols] for m in (tr, va, te))
            ytr, yva, yte = (df.loc[m, task] for m in (tr, va, te))
            log.info("%s fold %d: train %d val %d test %d (test %s..%s)", task, f.k, len(Xtr),
                     len(Xva), len(Xte), f.test_start.date(), f.test_end.date())  # fmt: skip
            for name in cfg.models:
                t0 = time.time()
                p = make_model(name, task, cfg.seed).fit(Xtr, ytr, Xva, yva).predict(Xte)
                log.info("  %-6s %.1fs", name, time.time() - t0)
                preds.append(
                    pd.DataFrame(
                        {
                            "task": task,
                            "model": name,
                            "fold": f.k,
                            "date": dates[te],
                            "ticker": tickers_idx[te],
                            "y_true": yte.to_numpy(float),
                            "y_pred": p,
                            "fwd_ret_5d": df.loc[te, "fwd_ret_5d"].to_numpy(float),
                        }
                    )  # fmt: skip
                )
    P = pd.concat(preds, ignore_index=True)
    M = metrics_table(P)
    D = dm_table(P)

    P.to_parquet(out / "predictions.parquet", index=False)
    M.to_parquet(out / "metrics.parquet", index=False)
    D.to_parquet(out / "dm.parquet", index=False)
    manifest = {
        "name": cfg.name,
        "config_hash": config_hash(cfg),
        "config": asdict(cfg),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_start": str(dates.min().date()),
        "data_end": str(dates.max().date()),
        "n_rows": int(len(df)),
        "n_features": len(feature_cols),
        "n_folds": len(folds),
        "folds": [
            {
                "k": f.k,
                "train_start": str(f.train_start.date()),
                "val_start": str(f.val_start.date()),
                "test_start": str(f.test_start.date()),
                "test_end": str(f.test_end.date()),
            }
            for f in folds
        ],  # fmt: skip
        "python": platform.python_version(),
        "versions": {
            p: version(p)
            for p in ("numpy", "pandas", "scikit-learn", "lightgbm", "statsmodels", "yfinance")
        },  # fmt: skip
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "summary.md").write_text(render_summary(cfg, P, M, D, manifest))
    log.info("wrote %s", out)
    return out


def metrics_table(P: pd.DataFrame) -> pd.DataFrame:
    """Long table: task, model, fold (POOLED = all folds), metric, value."""
    rows = []
    for (task, model, fold), g in P.groupby(["task", "model", "fold"], sort=False):
        rows += [{"task": task, "model": model, "fold": fold, "metric": k, "value": v}
                 for k, v in metrics(task, g).items()]  # fmt: skip
    for (task, model), g in P.groupby(["task", "model"], sort=False):
        rows += [{"task": task, "model": model, "fold": POOLED, "metric": k, "value": v}
                 for k, v in metrics(task, g).items()]  # fmt: skip
    return pd.DataFrame(rows)


def dm_table(P: pd.DataFrame) -> pd.DataFrame:
    """DM test of every model against each baseline, per fold and pooled over all test dates."""
    rows = []
    key = ["task", "fold", "date", "ticker"]
    for task, g in P.groupby("task", sort=False):
        wide = g.pivot(index=key, columns="model", values="y_pred")
        y = g.drop_duplicates(key).set_index(key)["y_true"].reindex(wide.index)
        loss = {m: sample_loss(task, y, wide[m]) for m in wide.columns}
        folds = wide.index.get_level_values("fold").to_numpy()
        dts = wide.index.get_level_values("date").to_numpy()
        for base in BASELINES:
            if base not in loss:
                continue
            for m in wide.columns:
                if m == base:
                    continue
                for k in [*np.unique(folds), POOLED]:
                    sel = np.ones(len(folds), bool) if k == POOLED else folds == k
                    stat, p = dm_test(loss[m][sel], loss[base][sel], dts[sel])
                    rows.append({"task": task, "model": m, "baseline": base, "fold": int(k),
                                 "stat": stat, "pvalue": p})  # fmt: skip
    return pd.DataFrame(rows)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _sel(df: pd.DataFrame, **eq) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    for col, val in eq.items():
        mask &= (df[col] == val).to_numpy()
    return df[mask]


def render_summary(cfg, P, M, D, manifest) -> str:
    per_fold = M[M["fold"] != POOLED]
    pooled = M[M["fold"] == POOLED]
    n_folds = manifest["n_folds"]
    sha = manifest["git_sha"][:10] or "n/a"
    dirty = " (dirty)" if manifest["git_dirty"] else ""
    lines = [
        f"# {cfg.name}: walk-forward results",
        "",
        f"Generated {manifest['generated_at']} | config `{manifest['config_hash']}` | "
        f"git `{sha}`{dirty}",
        "",
        f"Samples {manifest['data_start']}..{manifest['data_end']}, {manifest['n_rows']} rows x "
        f"{manifest['n_features']} features, {len(cfg.data.tickers)} tickers. "
        f"{n_folds} folds: train {cfg.splits.train_months}m / val {cfg.splits.val_months}m / "
        f"test {cfg.splits.test_months}m, step {cfg.splits.step_months}m; test blocks "
        f"{manifest['folds'][0]['test_start']}..{manifest['folds'][-1]['test_end']}.",
        "",
        "Metrics are mean +- std across test folds. DM = Diebold-Mariano statistic on the pooled "
        "per-date loss differential (negative = the row's model has lower loss than the baseline), "
        "HAC lag 4; `folds<.05` counts folds where that sign holds with p < 0.05. Losses: squared "
        "error on log vol / rank, Brier for direction. `linear` is ridge for vol/rank and logistic "
        "for direction.",
        "",
    ]
    for task in cfg.tasks:
        names = list(dict.fromkeys(_sel(per_fold, task=task)["metric"]))
        hdr = ["model", *names, "DM vs naive", "folds<.05", "DM vs lgbm", "folds<.05"]
        lines += [f"## {task} ({TASKS[task]})", ""]
        lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for model in cfg.models:
            cells = [model]
            for met in names:
                v = _sel(per_fold, task=task, model=model, metric=met)["value"]
                cells.append(f"{v.mean():.4f} +- {v.std():.4f}")
            for base in BASELINES:
                d = _sel(D, task=task, model=model, baseline=base)
                if d.empty:
                    cells += ["-", "-"]
                    continue
                pl = d[d["fold"] == POOLED].iloc[0]
                same_sign = np.sign(d["stat"]) == np.sign(pl["stat"])
                sig = d[(d["fold"] != POOLED) & (d["pvalue"] < 0.05) & same_sign]
                cells += [f"{pl['stat']:+.2f} (p={pl['pvalue']:.3f})", f"{len(sig)}/{n_folds}"]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        primary = "rank_ic" if task == "rank_5d" else names[0]
        pf = _sel(per_fold, task=task, metric=primary)
        pf = pf.pivot(index="fold", columns="model", values="value")
        pooled_row = _sel(pooled, task=task, metric=primary).set_index("model")["value"]
        head = "| fold | test block | " + " | ".join(cfg.models) + " |"
        lines += [f"Per-fold `{primary}`:", "", head, "|---|---|" + "---|" * len(cfg.models)]
        for f in manifest["folds"]:
            vals = " | ".join(f"{pf.loc[f['k'], m]:.4f}" for m in cfg.models)
            lines.append(f"| {f['k']} | {f['test_start']}..{f['test_end']} | {vals} |")
        vals = " | ".join(f"{pooled_row[m]:.4f}" for m in cfg.models)
        lines += [f"| pooled | all | {vals} |", ""]
    return "\n".join(lines)
