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

Separate output CSVs are written to ``results/`` for each detector.

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
- Output CSVs are written to:
    results/sentiment_phase_summary_core__mphasedetector.csv
    results/sentiment_phase_summary_core__mt4style.csv

Pre-registered filter
---------------------
    abs_sentiment >= 70  AND  extreme_streak_70 >= 3

Horizons analysed: 12 bars and 48 bars (contrarian returns).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ── project imports ──────────────────────────────────────────────────
from src.phases import MarketPhaseDetector
from src.mt4_regimes import detect_mt4_regimes, get_detector_description
from src.sentiment_loader import load_broker_h1_prices_from_dir

# ── paths (sibling repo) ────────────────────────────────────────────
SENTIMENT_REPO = Path(__file__).resolve().parent / ".." / "market-sentiment-ml"
CORE_DATASET = SENTIMENT_REPO / "data" / "output" / "master_research_dataset_core.csv"
MANIFEST_PATH = SENTIMENT_REPO / "data" / "output" / "DATASET_MANIFEST.json"
PRICE_DIR = SENTIMENT_REPO / "data" / "input" / "fx"

# ── output ───────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_CSV_MPHASE = RESULTS_DIR / "sentiment_phase_summary_core__mphasedetector.csv"
OUTPUT_CSV_MT4 = RESULTS_DIR / "sentiment_phase_summary_core__mt4style.csv"

# ── analysis parameters ─────────────────────────────────────────────
MIN_ABS_SENTIMENT = 70
MIN_EXTREME_STREAK = 3
HORIZONS = [12, 48]
BOOTSTRAP_N = 1000
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
            arr = vals.to_numpy()
            week_labels = phase_df.loc[vals.index, "_iso_week"].to_numpy()
            ci_lo, ci_hi = _week_block_bootstrap_ci(
                arr, week_labels, np.mean
            )

            # Quantiles
            p01, p05, p50, p95, p99 = np.percentile(arr, [1, 5, 50, 95, 99])

            # Trimmed mean (drop top/bottom 1 %)
            trimmed = scipy_stats.trim_mean(arr, proportiontocut=0.01)

            rows.append({
                "phase": phase,
                "horizon_bars": h,
                "n_events": len(arr),
                "mean_contrarian_ret": np.mean(arr),
                "median_contrarian_ret": np.median(arr),
                "ci_lo_95": ci_lo,
                "ci_hi_95": ci_hi,
                "p01": p01,
                "p05": p05,
                "p50": p50,
                "p95": p95,
                "p99": p99,
                "trimmed_mean_1pct": trimmed,
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


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
) -> pd.DataFrame:
    """
    Run a single detector pipeline: compute phases → join → filter →
    summary → write CSV.  Returns the summary DataFrame.
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
    merged = join_phases_to_sentiment(sentiment, phase_lookup)
    n_matched = int(merged["phase"].notna().sum())
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
    return summary


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
    for name, desc, phase_fn, output_csv in detectors:
        print(f"\n  ▶ {desc}")
        summaries[name] = _run_single_detector(
            name, sentiment, prices, phase_fn, output_csv, n_sentiment,
        )

    # 5. Comparison summary
    print_comparison_summary(summaries)


if __name__ == "__main__":
    main()
