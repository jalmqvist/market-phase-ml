#!/usr/bin/env python3
"""
analyze_sentiment_by_phase.py — Regime-conditioned sentiment analysis.

Joins the canonical sentiment dataset produced by market-sentiment-ml
with hourly market-phase labels computed by market-phase-ml, then
summarizes contrarian returns by regime under a pre-registered
persistent-extreme condition.

The analysis is run **twice**, once for each regime definition:

1. ``MarketPhaseDetector`` — the original market-phase-ml detector
   (ATR%-vs-rolling-median volatility, ADX > 25 trend threshold).
2. ``MT4-style`` — legacy MT4-inspired detector
   (ATR(10)/ATR(100) relative volatility, ADX(14) > 20 trend).

Separate output CSVs are written to ``results/sentiment/`` for each
detector.

Additionally, for the MT4-style detector, winsorized-by-horizon and
JPY-vs-non-JPY stratified summaries are produced:

- ``results/sentiment/sentiment_phase_summary_winsor0p5_by_jpy__mt4style.csv``
- ``results/sentiment/sentiment_phase_summary_winsor0p5_by_jpy_pairavg__mt4style.csv``
- ``results/sentiment/sentiment_phase_summary_winsor0p5_per_pair__mt4style.csv``
- ``results/sentiment/sentiment_phase_jpy_diff_bootstrap__mt4style.csv``

Usage
-----
    python analyze_sentiment_by_phase.py

Assumptions
-----------
- A sibling directory ``../market-sentiment-ml/`` exists.
- The sentiment pipeline has already been run so that these files exist:
    ../market-sentiment-ml/data/output/master_research_dataset_core.csv
    ../market-sentiment-ml/data/output/DATASET_MANIFEST.json
- Broker H1 FX CSVs (e.g. USDJPY_H1.csv) are in:
    ../market-sentiment-ml/data/input/fx/
- Output CSVs are written to ``results/sentiment/``.
- See ``docs/sentiment_analysis.md`` for full documentation.

Pre-registered filter
---------------------
    abs_sentiment >= 70  AND  extreme_streak_70 >= 3

Horizons analysed: 12 bars and 48 bars (contrarian returns).
Bootstrap resamples: B=2000 (block bootstrap by ISO week).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ── project root (one level up from analysis/) ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure project root is importable (for running from analysis/ dir)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── project imports ──────────────────────────────────────────────────
from src.phases import MarketPhaseDetector
from src.mt4_regimes import detect_mt4_regimes, get_detector_description
from src.sentiment_loader import load_broker_h1_prices_from_dir

# ── paths (sibling repo) ────────────────────────────────────────────
SENTIMENT_REPO = PROJECT_ROOT / ".." / "market-sentiment-ml"
CORE_DATASET = SENTIMENT_REPO / "data" / "output" / "master_research_dataset_core.csv"
MANIFEST_PATH = SENTIMENT_REPO / "data" / "output" / "DATASET_MANIFEST.json"
PRICE_DIR = SENTIMENT_REPO / "data" / "input" / "fx"

# ── output ───────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "results" / "sentiment"
OUTPUT_CSV_MPHASE = RESULTS_DIR / "sentiment_phase_summary_core__mphasedetector.csv"
OUTPUT_CSV_MT4 = RESULTS_DIR / "sentiment_phase_summary_core__mt4style.csv"
OUTPUT_CSV_WINSOR_JPY_MT4 = (
    RESULTS_DIR / "sentiment_phase_summary_winsor0p5_by_jpy__mt4style.csv"
)
OUTPUT_CSV_WINSOR_JPY_PAIRAVG_MT4 = (
    RESULTS_DIR / "sentiment_phase_summary_winsor0p5_by_jpy_pairavg__mt4style.csv"
)
OUTPUT_CSV_PER_PAIR_MT4 = (
    RESULTS_DIR / "sentiment_phase_summary_winsor0p5_per_pair__mt4style.csv"
)
OUTPUT_CSV_JPY_DIFF_BOOT_MT4 = (
    RESULTS_DIR / "sentiment_phase_jpy_diff_bootstrap__mt4style.csv"
)

# ── winsorization parameters ────────────────────────────────────────
WINSOR_LO_QUANTILE = 0.005
WINSOR_HI_QUANTILE = 0.995

# ── analysis parameters ─────────────────────────────────────────────
MIN_ABS_SENTIMENT = 70
MIN_EXTREME_STREAK = 3
HORIZONS = [12, 48]
BOOTSTRAP_N = 2000
BOOTSTRAP_CI = 0.95
BOOTSTRAP_SEED = 42

# ── sentiment columns we need ───────────────────────────────────────
SENTIMENT_COLS = [
    "pair",
    "entry_time",
    "abs_sentiment",
    "extreme_streak_70",
    "contrarian_ret_12b",
    "contrarian_ret_48b",
]


# =====================================================================
# 1. Validation helpers
# =====================================================================

def validate_manifest(manifest_path: Path) -> dict:
    """Read DATASET_MANIFEST.json and assert schema_version == '1.0'."""
    if not manifest_path.exists():
        sys.exit(
            f"ERROR: manifest not found at {manifest_path}\n"
            "Run the market-sentiment-ml pipeline first."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    version = manifest.get("schema_version")
    if version != "1.0":
        sys.exit(
            f"ERROR: expected schema_version '1.0', got '{version}'"
        )
    return manifest


# =====================================================================
# 2. Data loading
# =====================================================================

def load_sentiment_dataset(path: Path) -> pd.DataFrame:
    """Load the core sentiment CSV, keeping only analysis columns."""
    if not path.exists():
        sys.exit(
            f"ERROR: core dataset not found at {path}\n"
            "Run the market-sentiment-ml pipeline first."
        )
    df = pd.read_csv(path, usecols=SENTIMENT_COLS, parse_dates=["entry_time"])
    # Ensure tz-naive (strip timezone if present, values stay UTC)
    if df["entry_time"].dt.tz is not None:
        df["entry_time"] = df["entry_time"].dt.tz_localize(None)
    return df


def compute_phases_for_prices(
    prices: pd.DataFrame,
    phase_fn: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    min_bars: int = 300,
) -> pd.DataFrame:
    """
    Apply *phase_fn* to each pair's H1 data and return a
    (pair, timestamp, phase) lookup table.

    Parameters
    ----------
    prices : pd.DataFrame
        Combined price data (all pairs) with columns including
        ``pair``, ``timestamp``, and OHLCV.
    phase_fn : callable
        A function that takes a single-pair DataFrame (with OHLCV
        columns) and returns a DataFrame with at least a ``phase``
        column.
    min_bars : int
        Skip pairs with fewer bars than this (warm-up guard).
    """
    frames: list[pd.DataFrame] = []

    for pair, grp in prices.groupby("pair"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        if len(grp) < min_bars:
            print(f"  ⚠ {pair}: only {len(grp)} bars — skipping phase detection")
            continue
        phased = phase_fn(grp)
        phased = phased[["timestamp", "phase", "pair"]].copy()
        frames.append(phased)

    if not frames:
        sys.exit("ERROR: no pairs had enough data for phase detection.")

    return pd.concat(frames, ignore_index=True)


# ── Phase function wrappers ──────────────────────────────────────────

def _mphasedetector_phases(df: pd.DataFrame) -> pd.DataFrame:
    """Apply MarketPhaseDetector (default params) to a single pair."""
    detector = MarketPhaseDetector()
    return detector.detect_phases(df)


def _mt4style_phases(df: pd.DataFrame) -> pd.DataFrame:
    """Apply MT4-style regime detector to a single pair."""
    return detect_mt4_regimes(df)


# =====================================================================
# 3. Join + filter
# =====================================================================

def join_phases_to_sentiment(
    sentiment: pd.DataFrame,
    phase_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """
    Exact join on (pair, entry_time == timestamp) to attach phase
    labels to each sentiment event.
    """
    phase_lookup = phase_lookup.rename(columns={"timestamp": "entry_time"})
    merged = sentiment.merge(
        phase_lookup[["pair", "entry_time", "phase"]],
        on=["pair", "entry_time"],
        how="left",
    )
    return merged


def apply_preregistered_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only events where abs_sentiment >= 70 AND
    extreme_streak_70 >= 3.
    """
    mask = (
        (df["abs_sentiment"] >= MIN_ABS_SENTIMENT)
        & (df["extreme_streak_70"] >= MIN_EXTREME_STREAK)
    )
    return df.loc[mask].copy()


# =====================================================================
# 4. Block bootstrap (by ISO week)
# =====================================================================

def _week_block_bootstrap_ci(
    values: np.ndarray,
    week_labels: np.ndarray,
    stat_fn,
    n_boot: int = BOOTSTRAP_N,
    ci: float = BOOTSTRAP_CI,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """
    Block-bootstrap CI by resampling whole ISO weeks.

    Parameters
    ----------
    values : array of return observations
    week_labels : array of week identifiers (same length as *values*)
    stat_fn : callable(array) -> scalar  (e.g. np.mean)
    n_boot : number of bootstrap resamples
    ci : confidence level (e.g. 0.95)
    seed : random seed

    Returns
    -------
    (ci_lo, ci_hi)
    """
    rng = np.random.default_rng(seed)
    unique_weeks = np.unique(week_labels)
    n_weeks = len(unique_weeks)

    # Need at least 3 unique weeks for meaningful resampling variance
    if n_weeks < 3:
        return (np.nan, np.nan)

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sampled_weeks = rng.choice(unique_weeks, size=n_weeks, replace=True)
        # Gather all observations belonging to the sampled weeks
        mask = np.isin(week_labels, sampled_weeks)
        boot_sample = values[mask]
        if len(boot_sample) == 0:
            boot_stats[i] = np.nan
        else:
            boot_stats[i] = stat_fn(boot_sample)

    alpha = (1 - ci) / 2
    lo = np.nanpercentile(boot_stats, alpha * 100)
    hi = np.nanpercentile(boot_stats, (1 - alpha) * 100)
    return (lo, hi)


# =====================================================================
# 5. Summary statistics
# =====================================================================


def _clean_finite(values) -> np.ndarray:
    """Convert *values* to float64 and keep only finite entries."""
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _safe_trimmed_mean(arr: np.ndarray) -> float:
    """Trimmed mean (1 % each tail) with safe fallback for small n."""
    if len(arr) == 0:
        return np.nan
    if len(arr) < 3:
        return float(np.mean(arr))
    return float(scipy_stats.trim_mean(arr, proportiontocut=0.01))


def _warn_if_inconsistent(
    trimmed_mean: float,
    p05: float,
    p95: float,
    *,
    label: str,
) -> None:
    """Emit a warning when the trimmed mean looks implausible."""
    bound = max(abs(p05), abs(p95))
    if bound > 0 and abs(trimmed_mean) > 10 * bound:
        warnings.warn(
            f"trimmed_mean_1pct ({trimmed_mean:+.6f}) is > 10× "
            f"max(|p05|,|p95|) ({bound:.6f}) for {label}",
            stacklevel=2,
        )


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (phase × horizon), compute mean, median contrarian return,
    count, bootstrap 95 % CI of the mean, quantiles, and trimmed mean.
    """
    # Add ISO-week label for block bootstrap
    df = df.copy()
    df["_iso_week"] = (
        df["entry_time"].dt.isocalendar().year.astype(str)
        + "-W"
        + df["entry_time"].dt.isocalendar().week.astype(str).str.zfill(2)
    )

    rows: list[dict] = []
    phases = sorted(df["phase"].dropna().unique())

    for phase in phases:
        phase_df = df.loc[df["phase"] == phase]
        for h in HORIZONS:
            col = f"contrarian_ret_{h}b"
            vals = phase_df[col].dropna()
            if vals.empty:
                continue
            arr = _clean_finite(vals)
            if len(arr) == 0:
                continue
            week_labels = phase_df.loc[vals.index, "_iso_week"].to_numpy()
            # Keep week_labels aligned after finite filtering
            finite_mask = np.isfinite(
                np.asarray(vals.to_numpy(), dtype=float)
            )
            week_labels = week_labels[finite_mask]
            ci_lo, ci_hi = _week_block_bootstrap_ci(
                arr, week_labels, np.mean
            )

            # Quantiles
            p01, p05, p50, p95, p99 = np.percentile(arr, [1, 5, 50, 95, 99])

            # Trimmed mean (drop top/bottom 1 %)
            trimmed_mean = _safe_trimmed_mean(arr)

            label = f"phase={phase}, h={h}b"
            _warn_if_inconsistent(trimmed_mean, p05, p95, label=label)

            rows.append({
                "phase": phase,
                "horizon_bars": h,
                "n_events": len(arr),
                "mean_contrarian_ret": float(np.mean(arr)),
                "median_contrarian_ret": float(np.median(arr)),
                "ci_lo_95": ci_lo,
                "ci_hi_95": ci_hi,
                "p01": p01,
                "p05": p05,
                "p50": p50,
                "p95": p95,
                "p99": p99,
                "trimmed_mean_1pct": trimmed_mean,
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# =====================================================================
# 5b. Winsorization + JPY stratified summaries
# =====================================================================

def compute_winsor_bounds(
    df: pd.DataFrame,
) -> dict[int, tuple[float, float]]:
    """
    Compute global winsorization bounds per horizon.

    For each horizon *h*, compute quantiles at ``WINSOR_LO_QUANTILE``
    and ``WINSOR_HI_QUANTILE`` across the full filtered dataset.

    Returns
    -------
    dict mapping horizon → (lo, hi)
    """
    bounds: dict[int, tuple[float, float]] = {}
    for h in HORIZONS:
        col = f"contrarian_ret_{h}b"
        vals = df[col].dropna()
        if vals.empty:
            bounds[h] = (np.nan, np.nan)
        else:
            lo = np.percentile(vals, WINSOR_LO_QUANTILE * 100)
            hi = np.percentile(vals, WINSOR_HI_QUANTILE * 100)
            bounds[h] = (lo, hi)
    return bounds


def add_winsorized_columns(
    df: pd.DataFrame,
    bounds: dict[int, tuple[float, float]],
) -> pd.DataFrame:
    """
    Add winsorized return columns ``contrarian_ret_{h}b_w`` to *df*
    by clipping raw contrarian returns to the global [lo, hi] bounds.
    """
    df = df.copy()
    for h in HORIZONS:
        lo, hi = bounds[h]
        raw_col = f"contrarian_ret_{h}b"
        df[f"contrarian_ret_{h}b_w"] = df[raw_col].clip(lower=lo, upper=hi)
    return df


def compute_jpy_stratified_summary(
    df: pd.DataFrame,
    bounds: dict[int, tuple[float, float]],
) -> pd.DataFrame:
    """
    Group by (phase, is_jpy, horizon_bars) and compute winsorized +
    robust statistics.

    Columns produced:
        phase, is_jpy, horizon_bars, n_events,
        winsor_mean_contrarian_ret, median_contrarian_ret,
        trimmed_mean_1pct, p01, p05, p50, p95, p99,
        winsor_lo, winsor_hi
    """
    df = df.copy()
    df["is_jpy"] = df["pair"].str.endswith("-jpy")

    rows: list[dict] = []
    phases = sorted(df["phase"].dropna().unique())

    for phase in phases:
        for is_jpy in [True, False]:
            subset = df.loc[(df["phase"] == phase) & (df["is_jpy"] == is_jpy)]
            if subset.empty:
                continue
            for h in HORIZONS:
                raw_col = f"contrarian_ret_{h}b"
                win_col = f"contrarian_ret_{h}b_w"
                raw_vals = subset[raw_col].dropna()
                win_vals = subset.loc[raw_vals.index, win_col]
                if raw_vals.empty:
                    continue

                raw_arr = _clean_finite(raw_vals)
                win_arr = _clean_finite(win_vals)
                if len(raw_arr) == 0:
                    continue
                p01, p05, p50, p95, p99 = np.percentile(
                    raw_arr, [1, 5, 50, 95, 99]
                )
                trimmed_mean = _safe_trimmed_mean(raw_arr)
                lo, hi = bounds[h]

                jpy_label = "JPY" if is_jpy else "non-JPY"
                label = f"phase={phase}, is_jpy={jpy_label}, h={h}b"
                _warn_if_inconsistent(trimmed_mean, p05, p95, label=label)

                rows.append({
                    "phase": phase,
                    "is_jpy": is_jpy,
                    "horizon_bars": h,
                    "n_events": len(raw_arr),
                    "winsor_mean_contrarian_ret": float(np.mean(win_arr)),
                    "median_contrarian_ret": float(np.median(raw_arr)),
                    "trimmed_mean_1pct": trimmed_mean,
                    "p01": p01,
                    "p05": p05,
                    "p50": p50,
                    "p95": p95,
                    "p99": p99,
                    "winsor_lo": lo,
                    "winsor_hi": hi,
                })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def compute_pair_equal_weighted_summary(
    df: pd.DataFrame,
    bounds: dict[int, tuple[float, float]],
) -> pd.DataFrame:
    """
    For each (phase, is_jpy, horizon), compute the winsorized mean
    per pair, then average those pair means equally.

    Columns produced:
        phase, is_jpy, horizon_bars, n_pairs, total_events,
        mean_of_pair_means, median_of_pair_means
    """
    df = df.copy()
    df["is_jpy"] = df["pair"].str.endswith("-jpy")

    rows: list[dict] = []
    phases = sorted(df["phase"].dropna().unique())

    for phase in phases:
        for is_jpy in [True, False]:
            subset = df.loc[(df["phase"] == phase) & (df["is_jpy"] == is_jpy)]
            if subset.empty:
                continue
            for h in HORIZONS:
                win_col = f"contrarian_ret_{h}b_w"
                # per-pair winsor means
                pair_means = (
                    subset.dropna(subset=[win_col])
                    .groupby("pair")[win_col]
                    .mean()
                )
                if pair_means.empty:
                    continue
                total_events = int(
                    subset[win_col].notna().sum()
                )
                rows.append({
                    "phase": phase,
                    "is_jpy": is_jpy,
                    "horizon_bars": h,
                    "n_pairs": len(pair_means),
                    "total_events": total_events,
                    "mean_of_pair_means": pair_means.mean(),
                    "median_of_pair_means": pair_means.median(),
                })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# =====================================================================
# 5c. Per-pair summary (MT4-style only)
# =====================================================================

def compute_per_pair_summary(
    df: pd.DataFrame,
    bounds: dict[int, tuple[float, float]],
) -> pd.DataFrame:
    """
    Group by (pair, is_jpy, phase, horizon_bars) and compute per-pair
    winsorized + robust statistics.

    Columns produced:
        pair, is_jpy, phase, horizon_bars, n_events,
        winsor_mean_contrarian_ret, median_contrarian_ret,
        trimmed_mean_1pct, p05, p95
    """
    df = df.copy()
    df["is_jpy"] = df["pair"].str.endswith("-jpy")

    rows: list[dict] = []

    for (pair, phase), subset in df.groupby(["pair", "phase"], dropna=False):
        if subset.empty:
            continue
        is_jpy = bool(subset["is_jpy"].iloc[0])
        for h in HORIZONS:
            raw_col = f"contrarian_ret_{h}b"
            win_col = f"contrarian_ret_{h}b_w"
            raw_vals = subset[raw_col].dropna()
            if raw_vals.empty:
                continue
            win_vals = subset.loc[raw_vals.index, win_col]
            raw_arr = _clean_finite(raw_vals)
            win_arr = _clean_finite(win_vals)
            if len(raw_arr) == 0:
                continue

            trimmed_mean = _safe_trimmed_mean(raw_arr)
            p05, p95 = np.percentile(raw_arr, [5, 95])
            lo, hi = bounds[h]

            label = f"pair={pair}, phase={phase}, h={h}b"
            _warn_if_inconsistent(trimmed_mean, p05, p95, label=label)

            rows.append({
                "pair": pair,
                "is_jpy": is_jpy,
                "phase": phase,
                "horizon_bars": h,
                "n_events": len(raw_arr),
                "winsor_mean_contrarian_ret": float(np.mean(win_arr)),
                "median_contrarian_ret": float(np.median(raw_arr)),
                "trimmed_mean_1pct": float(trimmed_mean),
                "p05": float(p05),
                "p95": float(p95),
            })

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["pair", "phase", "horizon_bars"]
    ).reset_index(drop=True)


# =====================================================================
# 5d. Bootstrap CIs for JPY minus non-JPY difference (MT4-style only)
# =====================================================================

def compute_jpy_diff_bootstrap(
    df: pd.DataFrame,
    bounds: dict[int, tuple[float, float]],
    n_boot: int = BOOTSTRAP_N,
    ci: float = BOOTSTRAP_CI,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """
    For each (phase, horizon), compute the pair-equal-weighted
    winsor mean difference: mean_pair(JPY) − mean_pair(non-JPY).

    Uses a block bootstrap by ISO week to preserve autocorrelation.
    Within each resample, per-pair winsor means are recomputed, then
    equal-weight averaged across JPY and non-JPY groups.

    Columns produced:
        phase, horizon_bars, delta_jpy_minus_other,
        ci_lo_95, ci_hi_95, n_boot, n_weeks,
        n_jpy_pairs, n_other_pairs
    """
    df = df.copy()
    df["is_jpy"] = df["pair"].str.endswith("-jpy")
    df["_iso_week"] = (
        df["entry_time"].dt.isocalendar().year.astype(str)
        + "-W"
        + df["entry_time"].dt.isocalendar().week.astype(str).str.zfill(2)
    )

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    phases = sorted(df["phase"].dropna().unique())

    for phase in phases:
        phase_df = df.loc[df["phase"] == phase]
        for h in HORIZONS:
            win_col = f"contrarian_ret_{h}b_w"
            sub = phase_df.dropna(subset=[win_col]).copy()
            if sub.empty:
                continue

            jpy_pairs = sorted(sub.loc[sub["is_jpy"], "pair"].unique())
            other_pairs = sorted(sub.loc[~sub["is_jpy"], "pair"].unique())
            if not jpy_pairs or not other_pairs:
                continue

            unique_weeks = sub["_iso_week"].unique()
            n_weeks = len(unique_weeks)
            if n_weeks < 3:
                continue

            def _compute_delta(data: pd.DataFrame) -> float:
                """Pair-equal-weighted JPY minus non-JPY winsor mean."""
                pair_means = data.groupby(["is_jpy", "pair"])[win_col].mean()
                level_vals = pair_means.index.get_level_values(0)
                jpy_mean = (
                    pair_means.loc[True].mean()
                    if True in level_vals
                    else np.nan
                )
                other_mean = (
                    pair_means.loc[False].mean()
                    if False in level_vals
                    else np.nan
                )
                if np.isnan(jpy_mean) or np.isnan(other_mean):
                    return np.nan
                return jpy_mean - other_mean

            observed_delta = _compute_delta(sub)

            boot_stats = np.empty(n_boot)
            for i in range(n_boot):
                sampled_weeks = rng.choice(
                    unique_weeks, size=n_weeks, replace=True
                )
                # Build resampled dataframe by selecting all rows in
                # sampled weeks (duplicates included for repeated weeks)
                indices: list[np.ndarray] = []
                week_to_idx = sub.groupby("_iso_week").indices
                for w in sampled_weeks:
                    if w in week_to_idx:
                        indices.append(week_to_idx[w])
                if not indices:
                    boot_stats[i] = np.nan
                    continue
                boot_idx = np.concatenate(indices)
                boot_sample = sub.iloc[boot_idx]
                boot_stats[i] = _compute_delta(boot_sample)

            alpha = (1 - ci) / 2
            ci_lo = float(np.nanpercentile(boot_stats, alpha * 100))
            ci_hi = float(np.nanpercentile(boot_stats, (1 - alpha) * 100))

            rows.append({
                "phase": phase,
                "horizon_bars": h,
                "delta_jpy_minus_other": float(observed_delta),
                "ci_lo_95": ci_lo,
                "ci_hi_95": ci_hi,
                "n_boot": n_boot,
                "n_weeks": n_weeks,
                "n_jpy_pairs": len(jpy_pairs),
                "n_other_pairs": len(other_pairs),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["horizon_bars", "phase"]
    ).reset_index(drop=True)


# =====================================================================
# 6. Console report
# =====================================================================

def print_console_summary(
    detector_name: str,
    n_sentiment: int,
    n_after_join: int,
    n_filtered: int,
    summary: pd.DataFrame,
) -> None:
    """Print a compact console summary for one detector."""
    match_rate = n_after_join / n_sentiment * 100 if n_sentiment else 0
    print("\n" + "=" * 64)
    print(f"  Sentiment × Phase Analysis — {detector_name}")
    print("=" * 64)
    print(f"  Sentiment events (core):      {n_sentiment:>7,}")
    print(f"  Matched to phase:             {n_after_join:>7,}  "
          f"({match_rate:.1f}%)")
    print(f"  After filter (extreme):       {n_filtered:>7,}")
    print("-" * 64)

    if summary.empty:
        print("  No events survived filtering.")
    else:
        for _, row in summary.iterrows():
            print(
                f"  {row['phase']:<14s}  h={int(row['horizon_bars']):>2d}b  "
                f"n={int(row['n_events']):>5d}  "
                f"mean={row['mean_contrarian_ret']:+.5f}  "
                f"med={row['median_contrarian_ret']:+.5f}  "
                f"trim1%={row['trimmed_mean_1pct']:+.5f}  "
                f"95%CI=[{row['ci_lo_95']:+.5f}, {row['ci_hi_95']:+.5f}]"
            )
    print("=" * 64 + "\n")


def print_comparison_summary(
    summaries: dict[str, pd.DataFrame],
) -> None:
    """Print a side-by-side comparison of phase counts and headline means."""
    print("\n" + "╔" + "═" * 62 + "╗")
    print("║" + "  Detector Comparison (12-bar horizon)".center(62) + "║")
    print("╚" + "═" * 62 + "╝")

    # Build a compact table: phase | n(det1) mean(det1) | n(det2) mean(det2) …
    detector_names = list(summaries.keys())
    phases = sorted(
        set().union(*(s["phase"].unique() for s in summaries.values() if not s.empty))
    )

    # Header
    header = f"  {'phase':<14s}"
    for name in detector_names:
        short = name[:16]
        header += f"  {'n':>6s}  {'mean':>10s}  {'trim1%':>10s}"
    print(header)
    sub = f"  {'':─<14s}"
    for name in detector_names:
        sub += f"  {'':─>6s}  {'':─>10s}  {'':─>10s}"
    print(sub)

    for phase in phases:
        line = f"  {phase:<14s}"
        for name in detector_names:
            s = summaries[name]
            row = s.loc[(s["phase"] == phase) & (s["horizon_bars"] == 12)]
            if row.empty:
                line += f"  {'—':>6s}  {'—':>10s}  {'—':>10s}"
            else:
                r = row.iloc[0]
                line += (
                    f"  {int(r['n_events']):>6d}"
                    f"  {r['mean_contrarian_ret']:>+10.5f}"
                    f"  {r['trimmed_mean_1pct']:>+10.5f}"
                )
        print(line)

    print()


def print_winsor_bounds(
    bounds: dict[int, tuple[float, float]],
) -> None:
    """Print winsorization bounds per horizon."""
    print("\n" + "╔" + "═" * 62 + "╗")
    print("║" + "  Winsorization Bounds (0.5% tails)".center(62) + "║")
    print("╚" + "═" * 62 + "╝")
    for h in sorted(bounds):
        lo, hi = bounds[h]
        print(f"  horizon {h:>2d}b:  lo = {lo:+.8f}  hi = {hi:+.8f}")
    print()


def print_jpy_stratified_table(
    summary: pd.DataFrame,
) -> None:
    """Print a compact JPY vs non-JPY table for MT4-style, 12 bars."""
    print("╔" + "═" * 62 + "╗")
    print("║" + "  MT4-style: JPY vs non-JPY (12-bar, winsor mean)".center(62) + "║")
    print("╚" + "═" * 62 + "╝")

    sub = summary.loc[summary["horizon_bars"] == 12].copy()
    if sub.empty:
        print("  No data for 12-bar horizon.")
        print()
        return

    header = (
        f"  {'phase':<14s}  {'is_jpy':>6s}  {'n':>6s}"
        f"  {'winsor_mean':>12s}  {'median':>12s}  {'trim1%':>12s}"
    )
    print(header)
    print(f"  {'':─<14s}  {'':─>6s}  {'':─>6s}"
          f"  {'':─>12s}  {'':─>12s}  {'':─>12s}")

    for _, row in sub.sort_values(["phase", "is_jpy"]).iterrows():
        jpy_label = "JPY" if row["is_jpy"] else "other"
        print(
            f"  {row['phase']:<14s}  {jpy_label:>6s}  {int(row['n_events']):>6d}"
            f"  {row['winsor_mean_contrarian_ret']:>+12.6f}"
            f"  {row['median_contrarian_ret']:>+12.6f}"
            f"  {row['trimmed_mean_1pct']:>+12.6f}"
        )
    print()


def print_bootstrap_delta_table(
    boot_df: pd.DataFrame,
) -> None:
    """Print a compact table of bootstrap JPY-minus-other deltas."""
    print("╔" + "═" * 62 + "╗")
    print("║" + "  Bootstrap: JPY − non-JPY (pair-equal-weighted)".center(62) + "║")
    print("╚" + "═" * 62 + "╝")

    if boot_df.empty:
        print("  No bootstrap results.")
        print()
        return

    header = (
        f"  {'phase':<14s}  {'h':>3s}"
        f"  {'delta':>12s}  {'ci_lo_95':>12s}  {'ci_hi_95':>12s}"
        f"  {'n_boot':>6s}"
    )
    print(header)
    print(f"  {'':─<14s}  {'':─>3s}"
          f"  {'':─>12s}  {'':─>12s}  {'':─>12s}"
          f"  {'':─>6s}")

    for _, row in boot_df.iterrows():
        print(
            f"  {row['phase']:<14s}  {int(row['horizon_bars']):>3d}"
            f"  {row['delta_jpy_minus_other']:>+12.6f}"
            f"  {row['ci_lo_95']:>+12.6f}"
            f"  {row['ci_hi_95']:>+12.6f}"
            f"  {int(row['n_boot']):>6d}"
        )
    print()


# =====================================================================
# 7. Main
# =====================================================================

def _run_single_detector(
    detector_name: str,
    sentiment: pd.DataFrame,
    prices: pd.DataFrame,
    phase_fn: Callable[[pd.DataFrame], pd.DataFrame],
    output_csv: Path,
    n_sentiment: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a single detector pipeline: compute phases → join → filter →
    summary → write CSV.  Returns (summary, filtered) DataFrames.
    """
    print(f"\n{'─' * 64}")
    print(f"  Detector: {detector_name}")
    print(f"{'─' * 64}")

    # Compute phases
    print(f"  Computing phases …")
    phase_lookup = compute_phases_for_prices(prices, phase_fn)
    print(f"    {len(phase_lookup):,} phase-labeled bars")

    # Join + filter
    print(f"  Joining phases to sentiment events …")
    n_before_join = len(sentiment)
    merged = join_phases_to_sentiment(sentiment, phase_lookup)

    # Nice-to-have: assert no row multiplication on join
    assert len(merged) == n_before_join, (
        f"Join produced row multiplication: {n_before_join} → {len(merged)}"
    )

    n_matched = int(merged["phase"].notna().sum())

    # Nice-to-have: print matched rate
    match_rate = n_matched / n_before_join * 100 if n_before_join else 0
    print(f"    matched rate: {n_matched:,}/{n_before_join:,} "
          f"({match_rate:.1f}%)")

    filtered = apply_preregistered_filter(merged.loc[merged["phase"].notna()])
    n_filtered = len(filtered)

    # Compute stats + output
    print(f"  Computing summary statistics …")
    summary = compute_summary(filtered)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    print(f"  → saved to {output_csv}")

    print_console_summary(detector_name, n_sentiment, n_matched,
                          n_filtered, summary)
    return summary, filtered


def main() -> None:
    print("─" * 64)
    print("  analyze_sentiment_by_phase.py")
    print("─" * 64)

    # 1. Validate manifest
    print("\n[1/4] Validating dataset manifest …")
    manifest = validate_manifest(MANIFEST_PATH)
    print(f"  schema_version: {manifest['schema_version']}")

    # 2. Load sentiment data
    print("[2/4] Loading core sentiment dataset …")
    sentiment = load_sentiment_dataset(CORE_DATASET)
    n_sentiment = len(sentiment)
    print(f"  {n_sentiment:,} events loaded, "
          f"{sentiment['pair'].nunique()} pairs")

    # 3. Load broker H1 prices
    print("[3/4] Loading broker H1 prices …")
    prices = load_broker_h1_prices_from_dir(PRICE_DIR)
    print(f"  {len(prices):,} bars loaded, "
          f"{prices['pair'].nunique()} pairs")

    # 4. Run both detectors
    print("[4/4] Running regime-conditioned analysis …")

    detector = MarketPhaseDetector()
    mphase_desc = (
        f"MarketPhaseDetector (ADX({detector.adx_period})>"
        f"{detector.adx_trend_threshold}, "
        f"ATR%({detector.atr_period}) vs {detector.vol_rolling_window}-bar "
        f"rolling median)"
    )

    detectors = [
        ("MarketPhaseDetector", mphase_desc, _mphasedetector_phases,
         OUTPUT_CSV_MPHASE),
        ("MT4-style", get_detector_description(), _mt4style_phases,
         OUTPUT_CSV_MT4),
    ]

    summaries: dict[str, pd.DataFrame] = {}
    filtered_data: dict[str, pd.DataFrame] = {}
    for name, desc, phase_fn, output_csv in detectors:
        print(f"\n  ▶ {desc}")
        summary, filtered = _run_single_detector(
            name, sentiment, prices, phase_fn, output_csv, n_sentiment,
        )
        summaries[name] = summary
        filtered_data[name] = filtered

    # 5. Comparison summary
    print_comparison_summary(summaries)

    # 6. Winsorized + JPY-stratified analysis (MT4-style)
    mt4_filtered = filtered_data.get("MT4-style")
    if mt4_filtered is not None and not mt4_filtered.empty:
        print("[+] Running winsorized JPY-stratified analysis (MT4-style) …")

        # Nice-to-have: JPY pair list discovered
        all_pairs = sorted(mt4_filtered["pair"].unique())
        jpy_pairs = [p for p in all_pairs if p.endswith("-jpy")]
        other_pairs = [p for p in all_pairs if not p.endswith("-jpy")]
        print(f"  JPY pairs ({len(jpy_pairs)}): {', '.join(jpy_pairs)}")
        print(f"  Non-JPY pairs ({len(other_pairs)}): {len(other_pairs)} pairs")

        # Compute global winsor bounds
        winsor_bounds = compute_winsor_bounds(mt4_filtered)
        print_winsor_bounds(winsor_bounds)

        # Add winsorized columns
        mt4_win = add_winsorized_columns(mt4_filtered, winsor_bounds)

        # JPY-stratified summary
        jpy_summary = compute_jpy_stratified_summary(mt4_win, winsor_bounds)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        jpy_summary.to_csv(OUTPUT_CSV_WINSOR_JPY_MT4, index=False)
        print(f"  → saved to {OUTPUT_CSV_WINSOR_JPY_MT4}")
        print_jpy_stratified_table(jpy_summary)

        # Pair-equal-weighted summary
        pairavg_summary = compute_pair_equal_weighted_summary(
            mt4_win, winsor_bounds,
        )
        pairavg_summary.to_csv(OUTPUT_CSV_WINSOR_JPY_PAIRAVG_MT4, index=False)
        print(f"  → saved to {OUTPUT_CSV_WINSOR_JPY_PAIRAVG_MT4}")

        # Per-pair summary (Requirement 1)
        per_pair_summary = compute_per_pair_summary(mt4_win, winsor_bounds)
        per_pair_summary.to_csv(OUTPUT_CSV_PER_PAIR_MT4, index=False)
        print(f"  → saved to {OUTPUT_CSV_PER_PAIR_MT4}")

        # Bootstrap CIs for JPY minus non-JPY difference (Requirement 2)
        print(f"  Running bootstrap (B={BOOTSTRAP_N}) for JPY−other delta …")
        boot_df = compute_jpy_diff_bootstrap(
            mt4_win, winsor_bounds,
            n_boot=BOOTSTRAP_N, ci=BOOTSTRAP_CI, seed=BOOTSTRAP_SEED,
        )
        boot_df.to_csv(OUTPUT_CSV_JPY_DIFF_BOOT_MT4, index=False)
        print(f"  → saved to {OUTPUT_CSV_JPY_DIFF_BOOT_MT4}")
        print_bootstrap_delta_table(boot_df)

        # Migration notice for old paths
        old_results = PROJECT_ROOT / "results"
        print(f"\n  ℹ Output directory has moved: results/ → results/sentiment/")
        print(f"    All sentiment CSVs are now written to {RESULTS_DIR}")
    else:
        print("[!] MT4-style filtered data empty; skipping JPY analysis.")


if __name__ == "__main__":
    main()
