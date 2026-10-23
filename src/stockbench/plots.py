"""PNG figures for the report."""

from __future__ import annotations

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

COLORS = {  # one fixed slot per model, never cycled
    "naive": "#2a78d6",
    "linear": "#eb6834",
    "lgbm": "#1baf7a",
    "lstm": "#eda100",
    "patchtst": "#e87ba4",
    "spy": "#52514e",
}


def _finish(ax, title: str, ylabel: str, path) -> None:
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", color="#e5e5e2", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.figure.legend(frameon=False, fontsize=8, ncol=6, loc="outside lower center")
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)


def fold_metric(pf: pd.DataFrame, labels: list[str], title: str, ylabel: str, path) -> None:
    """One line per model (columns of pf) across the test folds (rows), x labelled by `labels`."""
    _, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    x = list(range(len(pf)))
    for m in pf.columns:
        ax.plot(x, pf[m], marker="o", ms=4, lw=2, color=COLORS.get(m, "#888888"), label=m)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    _finish(ax, title, ylabel, path)


def wealth(curves: pd.DataFrame, title: str, path) -> None:
    """Cumulative net wealth per model; a model's seeds are averaged into one daily series."""
    _, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    for m, g in curves.groupby("model", sort=False):
        r = g.groupby("date")["net"].mean().sort_index()
        spy = m == "spy"
        ax.plot(r.index, (1 + r).cumprod(), lw=1.5 if spy else 2, ls="--" if spy else "-",
                color=COLORS.get(m, "#888888"), label=m)  # fmt: skip
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    _finish(ax, title, "wealth, log scale", path)
