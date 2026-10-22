"""Experiment loop: data -> features/targets -> folds -> models x seeds -> metrics/DM -> report."""

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

from stockbench.config import Config, config_hash, fit_hash
from stockbench.data import MARKET, Source, fetch_yfinance, load_prices, market_frame
from stockbench.eval import dm_test, metrics, sample_loss
from stockbench.features import build_features
from stockbench.models import SEQ_MODELS, make_model
from stockbench.nets import make_panel
from stockbench.splits import walk_forward
from stockbench.targets import TASKS, build_targets

log = logging.getLogger("stockbench")
POOLED = -1  # fold id for rows computed over all test folds together
BASELINES = ("naive", "lgbm")
PRED_COLS = ["task", "model", "seed", "fold", "date", "ticker", "y_true", "y_pred", "fwd_ret_5d"]


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
    feature_cols = list(X.columns)
    df, panel = make_panel(X.join(Y, how="inner"), feature_cols, cfg.nets.lookback)
    dates = df.index.get_level_values("date")
    tickers_idx = df.index.get_level_values("ticker")
    log.info("panel: %d rows, %d features, %s..%s", len(df), len(feature_cols), dates.min().date(),
             dates.max().date())  # fmt: skip

    folds = walk_forward(cfg.data.start, dates.max(), cfg.splits)
    if len(folds) < cfg.splits.min_folds:
        raise RuntimeError(f"only {len(folds)} folds, need {cfg.splits.min_folds}")

    cache = out / "cache"  # one parquet per fit so an interrupted run resumes
    cache.mkdir(exist_ok=True)
    fh = fit_hash(cfg)
    preds = []
    for task in cfg.tasks:
        valid = df[task].notna().to_numpy()
        y_all = df[task].to_numpy(dtype=float)
        for f in folds:
            tr, va, te = (m & valid for m in f.assign(dates, df["target_end"]))
            Xtr, Xva, Xte = (df.loc[m, feature_cols] for m in (tr, va, te))
            ytr, yva, yte = (df.loc[m, task] for m in (tr, va, te))
            log.info("%s fold %d: train %d val %d test %d (test %s..%s)", task, f.k, len(Xtr),
                     len(Xva), len(Xte), f.test_start.date(), f.test_end.date())  # fmt: skip
            for name in cfg.models:
                for seed in cfg.nets.seeds if name in SEQ_MODELS else [cfg.seed]:
                    cached = cache / f"{task}_{f.k}_{name}_{seed}_{fh}.parquet"
                    if cached.exists():
                        preds.append(pd.read_parquet(cached))
                        continue
                    t0 = time.time()
                    m = make_model(name, task, seed, cfg.nets)
                    if getattr(m, "needs_history", False):
                        p = m.fit(panel, y_all, tr, va).predict(panel, te)
                    else:
                        p = m.fit(Xtr, ytr, Xva, yva).predict(Xte)
                    log.info("  %-8s seed %d %6.1fs %s", name, seed, time.time() - t0, m.info)
                    pred = pd.DataFrame(
                        {
                            "task": task,
                            "model": name,
                            "seed": seed,
                            "fold": f.k,
                            "date": dates[te],
                            "ticker": tickers_idx[te],
                            "y_true": yte.to_numpy(float),
                            "y_pred": p,
                            "fwd_ret_5d": df.loc[te, "fwd_ret_5d"].to_numpy(float),
                        }
                    )
                    pred.to_parquet(cached, index=False)
                    preds.append(pred)
    P = pd.concat(preds, ignore_index=True)[PRED_COLS]
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
            for p in (
                "numpy",
                "pandas",
                "scikit-learn",
                "lightgbm",
                "statsmodels",
                "torch",
                "yfinance",
            )
        },  # fmt: skip
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "summary.md").write_text(render_summary(cfg, M, D, manifest))
    log.info("wrote %s", out)
    return out


def metrics_table(P: pd.DataFrame) -> pd.DataFrame:
    """Long table: task, model, seed, fold (POOLED = all folds), metric, value."""
    rows = []
    for (task, model, seed, fold), g in P.groupby(["task", "model", "seed", "fold"], sort=False):
        rows += [{"task": task, "model": model, "seed": seed, "fold": fold, "metric": k,
                  "value": v} for k, v in metrics(task, g).items()]  # fmt: skip
    for (task, model, seed), g in P.groupby(["task", "model", "seed"], sort=False):
        rows += [{"task": task, "model": model, "seed": seed, "fold": POOLED, "metric": k,
                  "value": v} for k, v in metrics(task, g).items()]  # fmt: skip
    return pd.DataFrame(rows)


def dm_table(P: pd.DataFrame) -> pd.DataFrame:
    """DM test of every (model, seed) against each baseline, per fold and pooled over all dates."""
    rows = []
    key = ["fold", "date", "ticker"]
    for task, g in P.groupby("task", sort=False):
        wide = g.pivot(index=key, columns=["model", "seed"], values="y_pred")
        y = g.drop_duplicates(key).set_index(key)["y_true"].reindex(wide.index)
        loss = {col: sample_loss(task, y, wide[col]) for col in wide.columns}
        folds = wide.index.get_level_values("fold").to_numpy()
        dts = wide.index.get_level_values("date").to_numpy()
        for base in BASELINES:
            bcols = [c for c in wide.columns if c[0] == base]
            if not bcols:
                continue
            for col in wide.columns:
                if col[0] == base:
                    continue
                for k in [*np.unique(folds), POOLED]:
                    sel = np.ones(len(folds), bool) if k == POOLED else folds == k
                    stat, p = dm_test(loss[col][sel], loss[bcols[0]][sel], dts[sel])
                    rows.append({"task": task, "model": col[0], "seed": col[1], "baseline": base,
                                 "fold": int(k), "stat": stat, "pvalue": p})  # fmt: skip
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


def _dm_cells(d: pd.DataFrame) -> list[str]:
    """Pooled DM across seeds (with range and seeds significant) and significant (seed, fold)."""
    pl = d[d["fold"] == POOLED]
    stat = pl["stat"].mean()
    if len(pl) == 1:
        txt = f"{stat:+.2f} (p={pl['pvalue'].iloc[0]:.3f})"
    else:
        n_sig = int(((pl["pvalue"] < 0.05) & (np.sign(pl["stat"]) == np.sign(stat))).sum())
        lo, hi = pl["stat"].min(), pl["stat"].max()
        txt = f"{stat:+.2f} [{lo:+.2f}, {hi:+.2f}] {n_sig}/{len(pl)} seeds"
    pf = d[d["fold"] != POOLED]
    sig = pf[(pf["pvalue"] < 0.05) & (np.sign(pf["stat"]) == np.sign(stat))]
    return [txt, f"{len(sig)}/{len(pf)}"]


def render_summary(cfg, M, D, manifest) -> str:
    per_fold = M[M["fold"] != POOLED]
    pooled = M[M["fold"] == POOLED]
    sha = manifest["git_sha"][:10] or "n/a"
    dirty = " (dirty)" if manifest["git_dirty"] else ""
    n = cfg.nets
    lines = [
        f"# {cfg.name}: walk-forward results",
        "",
        f"Generated {manifest['generated_at']} | config `{manifest['config_hash']}` | "
        f"git `{sha}`{dirty}",
        "",
        f"Samples {manifest['data_start']}..{manifest['data_end']}, {manifest['n_rows']} rows x "
        f"{manifest['n_features']} features, {len(cfg.data.tickers)} tickers. "
        f"{manifest['n_folds']} folds: train {cfg.splits.train_months}m / val "
        f"{cfg.splits.val_months}m / test {cfg.splits.test_months}m, step "
        f"{cfg.splits.step_months}m; test blocks {manifest['folds'][0]['test_start']}.."
        f"{manifest['folds'][-1]['test_end']}.",
        "",
        "Metrics are mean +- std across test folds (seed-averaged per fold first); multi-seed "
        "models also show the std across seeds of the all-fold mean. `*_tstat` columns are HAC "
        "t-stats of the mean daily IC over the pooled test dates, averaged over seeds. DM = "
        "Diebold-Mariano statistic on the pooled per-date loss differential (negative = the row's "
        "model has lower loss than the baseline), HAC lag 4; for multi-seed models: mean over "
        "seeds, [min, max], and seeds significant at 5%. `sig` counts (seed, fold) pairs with that "
        "sign and p < 0.05 out of seeds x folds. Losses: squared error on log vol / rank, Brier "
        "for direction. `linear` is ridge for vol/rank and logistic for direction.",
        "",
    ]
    if any(m in SEQ_MODELS for m in cfg.models):
        lines += [
            f"Sequence models: lookback {n.lookback} bars, batch {n.batch_size}, Adam lr {n.lr}, "
            f"early stopping on val loss (patience {n.patience}, max {n.max_epochs} epochs), "
            f"seeds {n.seeds}, device `{n.device}`. LSTM: {n.lstm_layers} layers x "
            f"{n.lstm_hidden} hidden, dropout {n.lstm_dropout}. PatchTST: patch {n.patch_len} / "
            f"stride {n.patch_stride}, {n.n_layers} layers, d_model {n.d_model}, {n.n_heads} "
            f"heads, dropout {n.dropout}, channel-independent.",
            "",
        ]
    for task in cfg.tasks:
        names = list(dict.fromkeys(_sel(per_fold, task=task)["metric"]))
        hdr = ["model", *names, "DM vs naive", "sig", "DM vs lgbm", "sig"]
        lines += [f"## {task} ({TASKS[task]})", ""]
        lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for model in cfg.models:
            seeds = sorted(_sel(per_fold, task=task, model=model)["seed"].unique())
            cells = [f"{model} ({len(seeds)} seeds)" if len(seeds) > 1 else model]
            for met in names:
                if met.endswith("_tstat"):
                    v = _sel(pooled, task=task, model=model, metric=met)["value"]
                    cells.append(f"{v.mean():.2f}")
                    continue
                pf = _sel(per_fold, task=task, model=model, metric=met)
                by_fold = pf.groupby("fold")["value"].mean()
                cell = f"{by_fold.mean():.4f} +- {by_fold.std():.4f}"
                if len(seeds) > 1:
                    cell += f" (seed sd {pf.groupby('seed')['value'].mean().std():.4f})"
                cells.append(cell)
            for base in BASELINES:
                d = _sel(D, task=task, model=model, baseline=base)
                cells += ["-", "-"] if d.empty else _dm_cells(d)
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        primary = "rank_ic" if task == "rank_5d" else names[0]
        pf = _sel(per_fold, task=task, metric=primary)
        pf = pf.groupby(["fold", "model"])["value"].mean().unstack("model")
        pooled_row = _sel(pooled, task=task, metric=primary).groupby("model")["value"].mean()
        head = "| fold | test block | " + " | ".join(cfg.models) + " |"
        lines += [f"Per-fold `{primary}` (seed-averaged):", "", head]
        lines += ["|---|---|" + "---|" * len(cfg.models)]
        for f in manifest["folds"]:
            vals = " | ".join(f"{pf.loc[f['k'], m]:.4f}" for m in cfg.models)
            lines.append(f"| {f['k']} | {f['test_start']}..{f['test_end']} | {vals} |")
        vals = " | ".join(f"{pooled_row[m]:.4f}" for m in cfg.models)
        lines += [f"| pooled | all | {vals} |", ""]
    return "\n".join(lines)
