# Sentiment Analysis — Regime-Conditioned Contrarian Returns

This document describes the **regime-conditioned sentiment analysis** pipeline
implemented in `analysis/analyze_sentiment_by_phase.py`. It produces summary
statistics of **contrarian returns** following extreme sentiment readings,
stratified by market regime (phase) and currency group (JPY vs non-JPY).

---

## Data Inputs

### Sibling Repository

The analysis depends on artifacts from the sibling repo
[`market-sentiment-ml`](https://github.com/jalmqvist/market-sentiment-ml),
which must be checked out alongside this repo:

```
parent/
├── market-phase-ml/          ← this repo
└── market-sentiment-ml/      ← sibling repo
    └── data/
        ├── output/
        │   ├── master_research_dataset_core.csv          (unfiltered dataset)
        │   ├── analysis/
        │   │   └── master_research_dataset_core_cleaned.csv  (pair-quality filtered; recommended)
        │   └── DATASET_MANIFEST.json                     (schema version = "1.0")
        └── input/
            └── fx/                                       (broker H1 OHLCV CSVs)
```

### Datasets

Two versions of the sentiment dataset are available.  The **cleaned
dataset is recommended** for all inference and production analyses.

| Dataset | Path (relative to `market-sentiment-ml/`) | Description |
|---------|-------------------------------------------|-------------|
| **cleaned** *(recommended)* | `data/output/analysis/master_research_dataset_core_cleaned.csv` | Pair-quality filtered. Excludes pairs with corrupted price data (eur-mxn, gbp-zar). Use this for all inference. |
| **core** | `data/output/master_research_dataset_core.csv` | Full unfiltered dataset. Includes all pairs. May trigger trimmed-mean inconsistency warnings for corrupted pairs. |

> **Note on data-quality issues in the unfiltered core dataset:**
> Pairs `eur-mxn` and `gbp-zar` exhibit implausible return behavior
> (e.g. `trimmed_mean_1pct` values orders of magnitude larger than
> the p05–p95 range).  This is consistent with broken bars, stale-to-jump
> transitions, or decimal/feed errors in the underlying price history.
> The `_warn_if_inconsistent` guard in the script will emit
> `UserWarning: trimmed_mean_1pct … is > 10× max(|p05|,|p95|)` for
> these groups when running on the unfiltered dataset.
> The cleaned dataset removes these pairs and is the basis for all
> results reported below.

### Key Files

| File | Description |
|------|-------------|
| `master_research_dataset_core_cleaned.csv` | **Recommended.** Pair-quality-filtered sentiment dataset with columns: `pair`, `entry_time`, `abs_sentiment`, `extreme_streak_70`, `contrarian_ret_12b`, `contrarian_ret_48b` |
| `master_research_dataset_core.csv` | Unfiltered sentiment dataset (same columns). |
| `DATASET_MANIFEST.json` | Must contain `"schema_version": "1.0"` |
| `fx/*.csv` | Broker-exported H1 OHLCV files (e.g. `USDJPY_H1.csv`) with columns: `time_utc`, `open`, `high`, `low`, `close`, `tick_volume` |

Pair format is lowercase `xxx-yyy` (e.g. `usd-jpy`, `eur-usd`).
Timestamps are tz-naive UTC.

---

## Pre-Registered Extreme Filter

Only events meeting **both** conditions are analysed:

```
abs_sentiment >= 70   AND   extreme_streak_70 >= 3
```

- `abs_sentiment`: absolute value of the net sentiment reading (0–100 scale)
- `extreme_streak_70`: number of consecutive bars the sentiment has remained
  at or above the 70-threshold

This filters to **persistent extreme** sentiment events, excluding transient
spikes.

---

## Regime Detectors

Two regime detectors are available. Both classify each bar into one of four
phases:

| Phase | Volatility | Trend |
|-------|-----------|-------|
| `HV_Trend` | High | Trending |
| `HV_Ranging` | High | Ranging |
| `LV_Trend` | Low | Trending |
| `LV_Ranging` | Low | Ranging |

### 1. MarketPhaseDetector (D1-native)

- **Volatility:** ATR% vs rolling median ATR% (252-bar window ≈ 1 year)
- **Trend:** ADX(14) > 25
- Designed for daily (D1) data; applied to H1 via the same logic

### 2. MT4-style (H1-native) — **primary for this analysis**

- **Volatility:** ATR(10) / ATR(100) relative ratio
- **Trend:** ADX(14) > 20
- Designed specifically for hourly (H1) data
- Lower ADX threshold captures more "trending" bars

> **Focus:** All winsorized, per-pair, and bootstrap outputs use the MT4-style
> detector exclusively, since the sentiment data is H1-native.

---

## Winsorization Definition

**Level:** 0.5% tails (q0.005 – q0.995)

**Scope:** Global by horizon, computed on the **post-filter extreme dataset**
(after applying `abs_sentiment >= 70` and `extreme_streak_70 >= 3`).

For each horizon `h ∈ {12, 48}`:

1. Compute `lo_h = quantile(contrarian_ret_{h}b, 0.005)` across all
   post-filter events
2. Compute `hi_h = quantile(contrarian_ret_{h}b, 0.995)` across all
   post-filter events
3. Define `contrarian_ret_{h}b_w = clip(contrarian_ret_{h}b, lo_h, hi_h)`

### Why Global-by-Horizon (Not Per-Phase)

Clipping within each phase would normalize away genuine differences in tail
behavior between regimes — which is itself part of what a regime label
captures. Global clipping preserves regime-level tail differences while
controlling for pathological data-quality spikes.

---

## JPY Definition

JPY pairs are defined as pairs where JPY is the **quote currency**:

```python
is_jpy = pair.endswith("-jpy")
```

In the broker dataset, JPY only appears as the quote currency (e.g. `usd-jpy`,
`eur-jpy`, `gbp-jpy`), never as the base.

---

## Output CSVs

All outputs are written to `results/sentiment/`.

### Core Summaries (Both Detectors)

| File | Description |
|------|-------------|
| `sentiment_phase_summary_core__mphasedetector.csv` | Phase × horizon summary using MarketPhaseDetector |
| `sentiment_phase_summary_core__mt4style.csv` | Phase × horizon summary using MT4-style detector |

**Columns:** `phase`, `horizon_bars`, `n_events`, `mean_contrarian_ret`,
`median_contrarian_ret`, `ci_lo_95`, `ci_hi_95`, `p01`, `p05`, `p50`, `p95`,
`p99`, `trimmed_mean_1pct`

### Winsorized JPY-Stratified (MT4-style Only)

| File | Description |
|------|-------------|
| `sentiment_phase_summary_winsor0p5_by_jpy__mt4style.csv` | Event-weighted summary by phase × is_jpy × horizon |
| `sentiment_phase_summary_winsor0p5_by_jpy_pairavg__mt4style.csv` | Pair-equal-weighted summary (average of per-pair winsor means) |
| `sentiment_phase_summary_winsor0p5_per_pair__mt4style.csv` | Per-pair breakdown by pair × is_jpy × phase × horizon |
| `sentiment_phase_jpy_diff_bootstrap__mt4style.csv` | Bootstrap CIs for pair-equal-weighted JPY − non-JPY difference |

### Per-Pair Artifact Columns

`pair`, `is_jpy`, `phase`, `horizon_bars`, `n_events`,
`winsor_mean_contrarian_ret`, `median_contrarian_ret`, `trimmed_mean_1pct`,
`p05`, `p95`

### Bootstrap Artifact Columns

`phase`, `horizon_bars`, `delta_jpy_minus_other`, `ci_lo_95`, `ci_hi_95`,
`n_boot`, `n_weeks`, `n_jpy_pairs`, `n_other_pairs`

The bootstrap uses **block resampling by ISO week** (resample whole weeks with
replacement) to preserve within-week autocorrelation. Default: B=2000
resamples.

---

## Interpretation Guidelines

### Mean vs Median vs Trimmed Mean

| Statistic | Sensitivity to Tails | When to Use |
|-----------|---------------------|-------------|
| **Winsorized mean** | Low (tails clipped) | Primary headline statistic — robust to outliers while preserving sign and magnitude |
| **Median** | None | Useful for confirming direction; if median and winsor mean agree, the effect is not driven by extremes |
| **Trimmed mean (1%)** | Low (top/bottom 1% removed) | Cross-check for winsorized mean; slightly different tail handling |
| **Raw mean** | High | Shown in core summaries for transparency; can be dominated by a few tail events |

### Why Winsorization?

Raw means in regime-conditioned FX returns are heavily influenced by a small
number of extreme events (e.g. news-driven spikes). This can cause:

- Sign flips between phases that are driven by 2–3 outliers
- Apparent "trend regime" effects that are really just fat tails

Winsorization at 0.5% tails removes the most extreme ~1% of observations
(0.5% each tail) while keeping all rows. This produces estimates that are:

- Still "mean-like" (preserve sign and relative magnitude)
- Robust to data-quality issues and genuine extreme events
- Easy to explain and reproduce

### Pair-Equal-Weighting (Why It Matters)

The event-weighted summary pools all events, which means heavily-traded pairs
(e.g. USD-JPY) dominate the average. The pair-equal-weighted summary:

1. Computes a winsorized mean **per pair** within each phase × horizon
2. Averages those pair means with **equal weight**

This answers "are JPY pairs generally special?" rather than "is USD-JPY
special?".

### Bootstrap CIs

The bootstrap CI for the JPY − non-JPY difference uses the same pair-equal-
weighted statistic:

1. Within each bootstrap resample (by ISO week), recompute per-pair winsor
   means
2. Average across JPY pairs and across non-JPY pairs (equal weight)
3. Compute delta = mean(JPY) − mean(non-JPY)
4. Report the 2.5th and 97.5th percentiles of the bootstrap distribution

If the 95% CI excludes zero, the JPY vs non-JPY difference is statistically
significant at the 5% level (under block-bootstrap assumptions).

---

## How to Reproduce

### Prerequisites

1. Clone both repos side by side:
   ```bash
   git clone https://github.com/jalmqvist/market-phase-ml.git
   git clone https://github.com/jalmqvist/market-sentiment-ml.git
   ```

2. Run the sentiment pipeline in `market-sentiment-ml` first (produces
   `master_research_dataset_core.csv`).

3. Install dependencies:
   ```bash
   cd market-phase-ml
   pip install -r requirements.txt
   ```

### Run the Analysis

From the `market-phase-ml` root:

```bash
python analyze_sentiment_by_phase.py
```

By default, the script uses the **cleaned dataset** if it exists
(`market-sentiment-ml/data/output/analysis/master_research_dataset_core_cleaned.csv`),
otherwise falls back to the core dataset.

#### Dataset selection options

```bash
# Use cleaned dataset (default when available; recommended)
python analyze_sentiment_by_phase.py --dataset cleaned

# Use the unfiltered core dataset
python analyze_sentiment_by_phase.py --dataset core

# Override with an explicit path
python analyze_sentiment_by_phase.py --dataset-path /path/to/custom_dataset.csv
```

> **Recommendation:** Always use `--dataset cleaned` (the default) for
> inference.  The core dataset includes `eur-mxn` and `gbp-zar` which
> have corrupted price data and will produce `UserWarning` messages about
> `trimmed_mean_1pct` inconsistencies.

This runs the top-level wrapper which delegates to
`analysis/analyze_sentiment_by_phase.py`. Alternatively, you can run the
analysis module directly:

```bash
python -m analysis.analyze_sentiment_by_phase
```

### Key Console Tables

The script prints several formatted tables:

1. **Detector comparison** (12-bar horizon, both detectors)
2. **Winsorization bounds** (per horizon)
3. **JPY vs non-JPY** (12-bar winsor mean by phase)
4. **Bootstrap: JPY − non-JPY** (pair-equal-weighted delta with 95% CI)

### Output Files

All CSVs are written to `results/sentiment/`:

```bash
ls results/sentiment/*.csv
```

---

## Findings so far (as of 2026-04-12)

> **Note:** This is exploratory research, not a trading claim. All results
> should be treated as working hypotheses subject to further validation.

### Methodology recap

- **Winsorization:** 0.5 % tails, global-by-horizon (q0.005 / q0.995
  computed on the full post-filter extreme dataset, then applied
  identically to every pair and phase).
- **Bootstrap:** Pair-equal-weighted block bootstrap by ISO week
  (B = 2 000). For each resample, per-pair winsor means are computed and
  then averaged with equal weight across JPY-quote pairs and separately
  across non-JPY pairs. The delta Δ = mean(JPY) − mean(non-JPY) is the
  test statistic.

### JPY − non-JPY bootstrap deltas (MT4-style)

#### 12-bar horizon

| Phase | Δ (JPY − other) | 95 % CI |
|-------|-----------------|---------|
| HV_Ranging | +0.000229 | [−0.000292, +0.000743] |
| HV_Trend | +0.000277 | [−0.000092, +0.000663] |
| LV_Ranging | +0.000548 | [+0.000273, +0.000829] |
| LV_Trend | +0.000307 | [−0.000018, +0.000687] |

#### 48-bar horizon

| Phase | Δ (JPY − other) | 95 % CI |
|-------|-----------------|---------|
| HV_Ranging | +0.000908 | [−0.000324, +0.002140] |
| HV_Trend | +0.000554 | [−0.000610, +0.001690] |
| LV_Ranging | +0.001571 | [+0.000670, +0.002435] |
| LV_Trend | +0.001273 | [+0.000288, +0.002305] |

### Interpretation

- **Low-volatility regimes:** The JPY − non-JPY difference is strongest
  and statistically robust (95 % CI excludes zero) in **LV_Ranging** at
  both 12 and 48 bars, and in **LV_Trend** at 48 bars. At 12 bars,
  LV_Trend is borderline (CI just barely crosses zero).
- **High-volatility regimes:** The point estimates are positive but the
  confidence intervals include zero in all HV buckets. The evidence for
  a JPY contrarian edge in high-volatility conditions is **inconclusive**.
- **Per-pair consistency (48-bar LV regimes):** 7 out of 8 JPY-quote
  pairs have a positive winsor mean in both LV_Ranging and LV_Trend at
  48 bars. The single negative pair differs between the two phases
  (cad-jpy in LV_Ranging; gbp-jpy in LV_Trend), suggesting the overall
  effect is broad-based rather than driven by one or two pairs.
- The 8 JPY-quote pairs in the dataset are: aud-jpy, cad-jpy, chf-jpy,
  eur-jpy, gbp-jpy, nzd-jpy, sgd-jpy, usd-jpy.
