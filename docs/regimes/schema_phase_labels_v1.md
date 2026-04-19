# Common phase output schema (phase_labels_v1)

This schema standardizes outputs across regime detectors so downstream joins are consistent.

## Required keys
- `schema_version`: string, e.g. `phase_labels_v1`
- `detector_id`: string (e.g. `mphasedetector_d1native`, `mt4style_h1native`)
- `timeframe`: `H1` or `D1`
- `pair`: normalized `xxx-yyy`
- `timestamp`: bar open UTC

## Time semantics & join contract (important)

- `timestamp` is the **bar open time** (start of the bar) in **UTC**.
- When exported to CSV/Parquet, datetime columns may be timezone-naive, but must be **interpreted as UTC**.
- Cross-repo join contract (H1):
  - When joining to `market-sentiment-ml` artifacts keyed by `entry_time`, use an **exact join**:
    - `(pair, timestamp) == (pair, entry_time)`
  - `timestamp` must refer to the same H1 bar open as `entry_time` (no bar-close timestamps).

## Required canonical fields

- `phase`: `HV_Trend`, `LV_Trend`, `HV_Ranging`, `LV_Ranging`, or `Unknown`
- `is_high_vol`: bool
- `is_trending`: bool
- `adx`: float

## Recommended volatility interpretability fields
- `vol_metric_name`: `atr_pct` or `rel_atr_ratio`
- `vol_metric_value`: float
- `vol_threshold`: float (rolling median for ATR% detector; 1.0 for MT4-style)

## Optional detector-specific diagnostics (namespaced)
MarketPhaseDetector:
- `mph_atr`
- `mph_atr_pct`
- `mph_atr_pct_median`
- `mph_stop_atr_mult`

MT4-style:
- `mt4_atr_short`
- `mt4_atr_long`
- `mt4_rel_vol`
- `mt4_sma200`

---

## Usage guidelines

This schema is designed to support multi-timeframe workflows.

Typical usage:

- **D1 labels**
  - used as regime/context features
- **H1 labels**
  - used as signal-level features or diagnostics

When combining timeframes:

- ensure that joins are performed on matching timestamps
- avoid using the same detector output as both:
  - a feature and a regime (prevents circular logic)

The schema intentionally separates:

- classification (`phase`)
- interpretable components (`is_high_vol`, `is_trending`)
- detector-specific diagnostics

This allows flexible use across modeling and strategy layers.