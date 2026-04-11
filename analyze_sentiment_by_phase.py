#!/usr/bin/env python3
"""
analyze_sentiment_by_phase.py — Regime-conditioned sentiment analysis.

Joins the canonical sentiment dataset produced by market-sentiment-ml
with hourly market-phase labels computed by market-phase-ml, then
summarizes contrarian returns by regime under a pre-registered
persistent-extreme condition.

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
- Output CSV is written to ``results/sentiment_phase_summary_core.csv``.

Pre-registered filter
---------------------
    abs_sentiment >= 70  AND  extreme_streak_70 >= 3

Horizons analysed: 12 bars and 48 bars (contrarian returns).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── project imports ──────────────────────────────────────────────────
from src.phases import MarketPhaseDetector
from src.sentiment_loader import load_broker_h1_prices_from_dir

# ── paths (sibling repo) ────────────────────────────────────────────
SENTIMENT_REPO = Path(__file__).resolve().parent / ".." / "market-sentiment-ml"
CORE_DATASET = SENTIMENT_REPO / "data" / "output" / "master_research_dataset_core.csv"
MANIFEST_PATH = SENTIMENT_REPO / "data" / "output" / "DATASET_MANIFEST.json"
PRICE_DIR = SENTIMENT_REPO / "data" / "input" / "fx"

# ── output ───────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_CSV = RESULTS_DIR / "sentiment_phase_summary_core.csv"

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


def compute_phases_for_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Run MarketPhaseDetector on each pair's H1 data and return a
    (pair, timestamp, phase) lookup table.
    """
    detector = MarketPhaseDetector()
    frames: list[pd.DataFrame] = []

    for pair, grp in prices.groupby("pair"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        if len(grp) < 300:
            print(f"  ⚠ {pair}: only {len(grp)} bars — skipping phase detection")
            continue
        phased = detector.detect_phases(grp)
        phased = phased[["timestamp", "phase", "pair"]].copy()
        frames.append(phased)

    if not frames:
        sys.exit("ERROR: no pairs had enough data for phase detection.")

    return pd.concat(frames, ignore_index=True)


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
    count, and bootstrap 95 % CI of the mean.
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
            rows.append({
                "phase": phase,
                "horizon_bars": h,
                "n_events": len(arr),
                "mean_contrarian_ret": np.mean(arr),
                "median_contrarian_ret": np.median(arr),
                "ci_lo_95": ci_lo,
                "ci_hi_95": ci_hi,
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# =====================================================================
# 6. Console report
# =====================================================================

def print_console_summary(
    n_sentiment: int,
    n_after_join: int,
    n_filtered: int,
    summary: pd.DataFrame,
) -> None:
    """Print a compact console summary."""
    match_rate = n_after_join / n_sentiment * 100 if n_sentiment else 0
    print("\n" + "=" * 64)
    print("  Sentiment × Phase Analysis — Summary")
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
                f"95%CI=[{row['ci_lo_95']:+.5f}, {row['ci_hi_95']:+.5f}]"
            )
    print("=" * 64 + "\n")


# =====================================================================
# 7. Main
# =====================================================================

def main() -> None:
    print("─" * 64)
    print("  analyze_sentiment_by_phase.py")
    print("─" * 64)

    # 1. Validate manifest
    print("\n[1/6] Validating dataset manifest …")
    manifest = validate_manifest(MANIFEST_PATH)
    print(f"  schema_version: {manifest['schema_version']}")

    # 2. Load sentiment data
    print("[2/6] Loading core sentiment dataset …")
    sentiment = load_sentiment_dataset(CORE_DATASET)
    n_sentiment = len(sentiment)
    print(f"  {n_sentiment:,} events loaded, "
          f"{sentiment['pair'].nunique()} pairs")

    # 3. Load broker H1 prices
    print("[3/6] Loading broker H1 prices …")
    prices = load_broker_h1_prices_from_dir(PRICE_DIR)
    print(f"  {len(prices):,} bars loaded, "
          f"{prices['pair'].nunique()} pairs")

    # 4. Compute phases
    print("[4/6] Computing market phases …")
    phase_lookup = compute_phases_for_prices(prices)
    print(f"  {len(phase_lookup):,} phase-labeled bars")

    # 5. Join + filter
    print("[5/6] Joining phases to sentiment events …")
    merged = join_phases_to_sentiment(sentiment, phase_lookup)
    n_matched = merged["phase"].notna().sum()
    filtered = apply_preregistered_filter(merged.loc[merged["phase"].notna()])
    n_filtered = len(filtered)

    # 6. Compute stats + output
    print("[6/6] Computing summary statistics …")
    summary = compute_summary(filtered)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"  → saved to {OUTPUT_CSV}")

    print_console_summary(n_sentiment, n_matched, n_filtered, summary)


if __name__ == "__main__":
    main()
