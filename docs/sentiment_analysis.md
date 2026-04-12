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
        │   ├── master_research_dataset_core.csv   (canonical sentiment dataset)
        │   └── DATASET_MANIFEST.json              (schema version = "1.0")
        └── input/
            └── fx/                                (broker H1 OHLCV CSVs)
```

### Key Files

| File | Description |
|------|-------------|
| `master_research_dataset_core.csv` | Canonical sentiment dataset with columns: `pair`, `entry_time`, `abs_sentiment`, `extreme_streak_70`, `contrarian_ret_12b`, `contrarian_ret_48b` |
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

This runs the top-level wrapper which delegates to
`analysis/analyze_sentiment_by_phase.py`.

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
