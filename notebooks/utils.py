"""Notebook utility functions for 01_regime_gating_walkforward.ipynb.

Provides:
  - load_results_csv / load_equity_debug  (data loading)
  - plot_equity_vs_spikes                 (2-panel equity + drawdown plot)
  - plot_selected_timeline                (policy-selection step chart)

All plotting functions accept pre-loaded DataFrames so the notebook stays
read-only with respect to artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_NOTEBOOK_DIR = Path(__file__).parent
_RESULTS_DIR = _NOTEBOOK_DIR.parent / "results"


def _default_results_dir() -> Path:
    return _RESULTS_DIR


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results_csv(filename: str, results_dir: str | Path | None = None) -> pd.DataFrame:
    """Load a walk-forward results CSV from the results directory.

    Parameters
    ----------
    filename : str
        CSV filename, e.g. ``"walkforward_results_summary.csv"``.
    results_dir : path-like, optional
        Override the default ``../results/`` directory.
    """
    d = Path(results_dir) if results_dir is not None else _default_results_dir()
    return pd.read_csv(d / filename)


def load_equity_debug(pair: str, fold: int, results_dir: str | Path | None = None) -> pd.DataFrame:
    """Load an equity debug CSV: ``equity_debug_{pair}_fold{fold}.csv``.

    The ``date`` column is parsed to datetime automatically.
    """
    d = Path(results_dir) if results_dir is not None else _default_results_dir()
    path = d / f"equity_debug_{pair}_fold{fold}.csv"
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SELECTED_MAP = {
    "PhaseAware": 0,
    "TrendFollowing": 1,
    "MeanReversion": 2,
}
_SELECTED_LABELS = ["PhaseAware", "TrendFollowing", "MeanReversion"]


def _shade_spans(
    ax: plt.Axes, dates: pd.Series, mask: pd.Series, color: str, alpha: float
) -> None:
    """Shade contiguous True runs in *mask* on *ax* using axvspan."""
    in_run = False
    start = None
    for i in range(len(mask)):
        m = bool(mask.iloc[i])
        if m and not in_run:
            in_run = True
            start = dates.iloc[i]
        if in_run and (not m or i == len(mask) - 1):
            end = dates.iloc[i]
            ax.axvspan(start, end, color=color, alpha=alpha)
            in_run = False


def _compute_drawdown(equity: pd.Series) -> pd.Series:
    rm = equity.cummax()
    return (equity / rm) - 1.0


# ---------------------------------------------------------------------------
# Public plotting utilities
# ---------------------------------------------------------------------------


def plot_equity_vs_spikes(
    df: pd.DataFrame,
    title: str = "",
    vol_col: str = "atr_pct",
) -> plt.Axes:
    """2-panel plot: equity curves with spike shading (top) + drawdown (bottom).

    Parameters
    ----------
    df : DataFrame
        Equity debug frame with columns ``equity_baseline``, ``equity_dynamic``,
        ``date``, ``spike``, ``near_spike``.  Optional: ``atr_pct``, ``vol_thr``,
        ``near_thr``.
    title : str
        Figure title placed on the top panel.
    vol_col : str
        Column name for the volatility overlay (plotted on a secondary y-axis).

    Returns
    -------
    plt.Axes
        The top (equity) axes object.
    """
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

    x = df["date"] if "date" in df.columns else np.arange(len(df))
    eq_b = df["equity_baseline"].astype(float)
    eq_d = df["equity_dynamic"].astype(float)
    dd_b = _compute_drawdown(eq_b)
    dd_d = _compute_drawdown(eq_d)

    spike = df["spike"].astype(bool) if "spike" in df.columns else pd.Series(False, index=df.index)
    near_spike = (
        df["near_spike"].astype(bool)
        if "near_spike" in df.columns
        else pd.Series(False, index=df.index)
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    if "date" in df.columns:
        _shade_spans(ax1, df["date"], near_spike, "#f4a261", 0.15)
        _shade_spans(ax1, df["date"], spike, "#e63946", 0.18)
        _shade_spans(ax2, df["date"], near_spike, "#f4a261", 0.15)
        _shade_spans(ax2, df["date"], spike, "#e63946", 0.18)

    ax1.plot(x, eq_b.values, label="Baseline", color="#457b9d", linewidth=1.8)
    ax1.plot(x, eq_d.values, label="Dynamic", color="#1d3557", linewidth=1.8)
    ax1.set_title(title)
    ax1.set_ylabel("Equity")
    ax1.legend(loc="upper left")

    ax2.plot(
        x, dd_b.values * 100.0, label="Baseline DD", color="#457b9d", linewidth=1.3, alpha=0.9
    )
    ax2.plot(
        x, dd_d.values * 100.0, label="Dynamic DD", color="#1d3557", linewidth=1.3, alpha=0.9
    )
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax2.legend(loc="lower left")

    if vol_col in df.columns:
        axv = ax1.twinx()
        axv.plot(
            x,
            df[vol_col].astype(float),
            color="#2a9d8f",
            alpha=0.25,
            linewidth=1.0,
            label=vol_col,
        )
        axv.set_ylabel(vol_col, color="#2a9d8f")
        if "vol_thr" in df.columns and df["vol_thr"].notna().any():
            axv.axhline(
                float(df["vol_thr"].dropna().iloc[0]),
                color="#e63946",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label="spike threshold",
            )
        if "near_thr" in df.columns and df["near_thr"].notna().any():
            axv.axhline(
                float(df["near_thr"].dropna().iloc[0]),
                color="#f4a261",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label="near-spike threshold",
            )

    fig.tight_layout()
    return ax1


def plot_selected_timeline(df: pd.DataFrame, title: str = "") -> plt.Axes:
    """Plot selected-strategy timeline from an equity debug DataFrame.

    Expects ``selected_type`` (or ``selected``) column plus optional ``spike``,
    ``near_spike``, ``atr_pct``, ``vol_thr``, ``near_thr``.

    Returns the primary axes object.
    """
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

    sel_col = "selected_type" if "selected_type" in df.columns else "selected"
    df["_sel_code"] = df[sel_col].map(_SELECTED_MAP).astype(float)

    spike = df["spike"].astype(bool) if "spike" in df.columns else pd.Series(False, index=df.index)
    near_spike = (
        df["near_spike"].astype(bool)
        if "near_spike" in df.columns
        else pd.Series(False, index=df.index)
    )

    sw = df[sel_col] != df[sel_col].shift(1)
    sw.iloc[0] = False

    x = df["date"] if "date" in df.columns else np.arange(len(df))

    fig, ax = plt.subplots(figsize=(14, 5))

    if "date" in df.columns:
        _shade_spans(ax, df["date"], near_spike, "#f4a261", 0.15)
        _shade_spans(ax, df["date"], spike, "#e63946", 0.18)

    ax.step(x, df["_sel_code"], where="post", linewidth=2, color="black", label="selected")
    if sw.any():
        ax.scatter(
            x[sw.values],
            df.loc[sw, "_sel_code"],
            s=14,
            color="#1d3557",
            alpha=0.8,
            label="switch",
        )

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(_SELECTED_LABELS)
    ax.set_ylim(-0.3, 2.3)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Selected policy")

    vol_col = next((c for c in ["atr_pct", "vol"] if c in df.columns), None)
    if vol_col is not None:
        ax2 = ax.twinx()
        ax2.plot(
            x, df[vol_col].astype(float), color="#457b9d", alpha=0.35, linewidth=1.2, label=vol_col
        )
        ax2.set_ylabel(vol_col)
        if "vol_thr" in df.columns and df["vol_thr"].notna().any():
            ax2.axhline(
                float(df["vol_thr"].dropna().iloc[0]),
                color="#e63946",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
            )
        if "near_thr" in df.columns and df["near_thr"].notna().any():
            ax2.axhline(
                float(df["near_thr"].dropna().iloc[0]),
                color="#f4a261",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
            )

    handles, labels_ = ax.get_legend_handles_labels()
    uniq: dict = {}
    for h, lb in zip(handles, labels_):
        if lb not in uniq:
            uniq[lb] = h
    ax.legend(list(uniq.values()), list(uniq.keys()), loc="upper left")

    fig.tight_layout()
    return ax
