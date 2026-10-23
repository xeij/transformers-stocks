"""Build paper/main.pdf: tables and figures from reports/<exp>, then pdflatex + bibtex.

    PYTHONPATH=src uv run python scripts/build_paper.py [reports/exp02] [configs/exp02.yaml]
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stockbench import plots
from stockbench.backtest import BT_TASKS
from stockbench.config import load_config
from stockbench.data import MARKET, load_prices
from stockbench.features import build_features
from stockbench.run import BASELINES, POOLED, PRIMARY, _sel, fold_pivot
from stockbench.splits import walk_forward
from stockbench.targets import build_targets

NAME = {"naive": "Naive", "linear": "Ridge / logistic", "lgbm": "LightGBM", "lstm": "LSTM",
        "patchtst": "PatchTST", "spy": "SPY buy-and-hold"}  # fmt: skip
LABEL = {"rmse": "RMSE", "mae": "MAE", "qlike": "QLIKE", "accuracy": "Accuracy",
         "balanced_accuracy": "Bal.\\ acc.", "auc": "AUC", "ic": "IC", "rank_ic": "Rank IC",
         "ic_tstat": "$t$(IC)", "rank_ic_tstat": "$t$(Rank IC)"}  # fmt: skip
TASK = {"vol_5d": "vol", "dir_5d": "dir", "rank_5d": "rank"}


def stars(p: float) -> str:
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.10 else ""


def pm(mean: float, sd: float, d: int = 3) -> str:
    return f"${mean:.{d}f} \\pm {sd:.{d}f}$"


def tabular(header: list[str], rows: list[list[str]], align: str) -> str:
    out = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def dm_cells(d: pd.DataFrame) -> list[str]:
    pl = d[d["fold"] == POOLED]
    stat = pl["stat"].mean()
    if len(pl) == 1:
        txt = f"${stat:+.1f}{stars(pl['pvalue'].iloc[0])}$"
    else:
        txt = f"${stat:+.1f}\\ [{pl['stat'].min():+.1f},\\ {pl['stat'].max():+.1f}]$"
    pf = d[d["fold"] != POOLED]
    sig = pf[(pf["pvalue"] < 0.05) & (np.sign(pf["stat"]) == np.sign(stat))]
    return [txt, f"{len(sig)}/{len(pf)}"]


def metrics_table(M: pd.DataFrame, D: pd.DataFrame, task: str, models: list[str]) -> str:
    per_fold, pooled = M[M["fold"] != POOLED], M[M["fold"] == POOLED]
    names = list(dict.fromkeys(_sel(per_fold, task=task)["metric"]))
    cols = [n for n in names if not n.endswith("_tstat")]
    tstats = [n for n in names if n.endswith("_tstat")]
    header = ["Model", *(LABEL[c] for c in cols), "Seed sd", *(LABEL[t] for t in tstats),
              "DM vs naive", "sig.", "DM vs LightGBM", "sig."]  # fmt: skip
    rows = []
    for m in models:
        seeds = sorted(_sel(per_fold, task=task, model=m)["seed"].unique())
        cells = [NAME[m] + (f" ({len(seeds)} seeds)" if len(seeds) > 1 else "")]
        for c in cols:
            by_fold = _sel(per_fold, task=task, model=m, metric=c).groupby("fold")["value"].mean()
            cells.append(pm(by_fold.mean(), by_fold.std()))
        pf = _sel(per_fold, task=task, model=m, metric=PRIMARY[task])
        cells.append(f"${pf.groupby('seed')['value'].mean().std():.3f}$" if len(seeds) > 1 else "--")
        for t in tstats:
            cells.append(f"${_sel(pooled, task=task, model=m, metric=t)['value'].mean():.2f}$")
        for base in BASELINES:
            d = _sel(D, task=task, model=m, baseline=base)
            cells += ["--", "--"] if d.empty else dm_cells(d)
        rows.append(cells)
    return tabular(header, rows, "l" + "r" * (len(header) - 1))


def per_fold_table(M: pd.DataFrame, manifest: dict, task: str, models: list[str]) -> str:
    pf = fold_pivot(M, task)
    pooled = _sel(M[M["fold"] == POOLED], task=task, metric=PRIMARY[task])
    pooled = pooled.groupby("model")["value"].mean()
    rows = [[str(f["k"]), f["test_start"][:7], *(f"${pf.loc[f['k'], m]:.4f}$" for m in models)]
            for f in manifest["folds"]]  # fmt: skip
    rows.append(["\\midrule Pooled", "all", *(f"${pooled[m]:.4f}$" for m in models)])
    return tabular(["Fold", "Test start", *(NAME[m] for m in models)], rows, "ll" + "r" * len(models))


def backtest_table(B: pd.DataFrame, task: str, models: list[str]) -> str:
    cols = [("ann_return_net", "+.1%"), ("sharpe_net", "+.2f"), ("max_dd_net", ".1%"),
            ("tstat_net", "+.2f"), ("ann_return_gross", "+.1%"), ("sharpe_gross", "+.2f"),
            ("turnover", ".0f")]  # fmt: skip
    header = ["Model", "Net return", "Net Sharpe", "Max DD", "$t$(net)", "Gross return",
              "Gross Sharpe", "Turnover"]  # fmt: skip
    rows = []
    for m in [*models, "spy"]:
        g = _sel(B, task=task, model=m)
        if g.empty:
            continue
        cells = [NAME[m] + (f" ({len(g)} seeds)" if len(g) > 1 else "")]
        for col, spec in cols:
            v = g[col].dropna()
            if v.empty:
                cells.append("--")
                continue
            cell = f"{v.mean():{spec}}"
            if len(v) > 1:
                cell += f" \\pm {v.std():{spec.lstrip('+')}}"
            cells.append("$" + cell.replace("%", "\\%") + "$")
        rows.append(cells)
    return tabular(header, rows, "l" + "r" * (len(header) - 1))


def seeds_table(M: pd.DataFrame, B: pd.DataFrame) -> str:
    pooled = M[M["fold"] == POOLED]
    rows = []
    for m in ("lstm", "patchtst"):
        for s in sorted(_sel(pooled, model=m)["seed"].unique()):
            cells = [NAME[m], str(s)]
            for task in ("vol_5d", "dir_5d", "rank_5d"):
                v = _sel(pooled, task=task, model=m, seed=s, metric=PRIMARY[task])["value"]
                cells.append(f"${v.iloc[0]:.4f}$")
            for task in BT_TASKS:
                v = _sel(B, task=task, model=m, seed=s)["sharpe_net"]
                cells.append(f"${v.iloc[0]:+.2f}$")
            rows.append(cells)
    header = ["Model", "Seed", "Vol RMSE", "Dir accuracy", "Rank IC", "Dir net Sharpe",
              "Rank net Sharpe"]  # fmt: skip
    return tabular(header, rows, "ll" + "r" * 5)


def folds_table(cfg, manifest: dict) -> str:
    tickers = list(dict.fromkeys(cfg.data.tickers + list(MARKET.values())))
    prices = load_prices(tickers, cfg.data.start, cfg.data.end, cache_dir=cfg.data.cache_dir)
    universe = prices[prices["ticker"].isin(cfg.data.tickers)]
    from stockbench.data import market_frame

    df = build_features(universe, market_frame(prices)).join(build_targets(universe), how="inner")
    dates = df.index.get_level_values("date")
    rows = []
    for f in walk_forward(cfg.data.start, dates.max(), cfg.splits):
        tr, va, te = f.assign(dates, df["target_end"])
        labelled = df["dir_5d"].notna().to_numpy()
        rows.append([str(f.k), str(f.train_start.date()), str(f.val_start.date()),
                     str(f.test_start.date()), str(f.test_end.date()), f"{tr.sum():,}",
                     f"{va.sum():,}", f"{te.sum():,}", f"{(te & labelled).sum():,}"])  # fmt: skip
    header = ["$k$", "Train start", "Val.\\ start", "Test start", "Test end", "$n_{\\text{train}}$",
              "$n_{\\text{val}}$", "$n_{\\text{test}}$", "$n_{\\text{test}}^{\\text{dir}}$"]  # fmt: skip
    return tabular(header, rows, "lllllrrrr")


def main(exp: str, config: str) -> None:
    out = Path("paper")
    tables, figs = out / "tables", out / "figures"
    cfg = load_config(config)
    rep = Path(exp)
    manifest = json.loads((rep / "manifest.json").read_text())
    M = pd.read_parquet(rep / "metrics.parquet")
    D = pd.read_parquet(rep / "dm.parquet")
    B = pd.read_parquet(rep / "backtest.parquet")
    curves = pd.read_parquet(rep / "backtest_daily.parquet")
    models = cfg.models
    for task in cfg.tasks:
        (tables / f"metrics_{TASK[task]}.tex").write_text(metrics_table(M, D, task, models))
        (tables / f"folds_{TASK[task]}.tex").write_text(per_fold_table(M, manifest, task, models))
        labels = [f["test_start"][:7] for f in manifest["folds"]]
        plots.fold_metric(fold_pivot(M, task)[models], labels, "",
                          LABEL[PRIMARY[task]].replace("\\", ""), figs / f"fold_{TASK[task]}.pdf")
    for task in BT_TASKS:
        (tables / f"backtest_{TASK[task]}.tex").write_text(backtest_table(B, task, models))
        plots.wealth(curves[curves["task"] == task], "", figs / f"wealth_{TASK[task]}.pdf")
    (tables / "seeds.tex").write_text(seeds_table(M, B))
    (tables / "folds.tex").write_text(folds_table(cfg, manifest))
    (tables / "universe.tex").write_text(", ".join(cfg.data.tickers).replace("&", "\\&") + "\n")
    meta = {
        "generated": manifest["generated_at"][:10],
        "sha": manifest["git_sha"][:10],
        "confighash": manifest["config_hash"],
        "nrows": f"{manifest['n_rows']:,}",
        "nfeatures": str(manifest["n_features"]),
        "datastart": manifest["data_start"],
        "dataend": manifest["data_end"],
        "ntickers": str(len(cfg.data.tickers)),
        **{f"v{k.replace('-', '')}": v for k, v in manifest["versions"].items()},
        "vpython": manifest["python"],
    }
    (tables / "meta.tex").write_text(
        "".join(f"\\newcommand{{\\meta{k}}}{{{v}}}\n" for k, v in meta.items())
    )
    if (out / "main.tex").exists():
        for cmd in (["pdflatex"], ["bibtex"], ["pdflatex"], ["pdflatex"]):
            arg = "main" if cmd[0] == "bibtex" else "main.tex"
            flags = [] if cmd[0] == "bibtex" else ["-interaction=nonstopmode", "-halt-on-error"]
            r = subprocess.run([*cmd, *flags, arg], cwd=out, capture_output=True, text=True)
            if r.returncode and cmd[0] != "bibtex":
                print(r.stdout[-3000:])
                sys.exit(f"{cmd[0]} failed")
        print("wrote", out / "main.pdf")


if __name__ == "__main__":
    main(*(sys.argv[1:3] or ["reports/exp02", "configs/exp02.yaml"]))
