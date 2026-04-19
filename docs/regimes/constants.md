# Regime detection constants (contract defaults)

## Versioning & purpose

This file is the single source of truth for the **default regime detector and D1-from-H1 aggregation parameters** used in `market-phase-ml`. If any parameter changes, update this file, bump a detector/config version identifier, and record that version in all generated outputs (e.g., manifests/metadata). This preserves reproducibility and makes cross-repo experiments comparable.

---

## 1) Common conventions

- All timestamps refer to **bar open** times in **UTC**.
- When written to CSV/Parquet, datetimes may be timezone-naive but must be interpreted as UTC.
- Phase label set (4-phase + Unknown):
  - `HV_Trend`, `LV_Trend`, `HV_Ranging`, `LV_Ranging`, `Unknown`

---

## 2) MarketPhaseDetector (D1-native)

**Implementation:** `src/phases.py` → `MarketPhaseDetector`

### Trend (ADX)
- `adx_period = 14`
- `adx_trend_threshold = 25.0`
- `trending = (adx > adx_trend_threshold)`

### Volatility (ATR% vs rolling median)
- `atr_period = 14`
- `atr_pct = ATR(atr_period) / Close * 100`
- `vol_rolling_window = 252`
- `rolling_median_min_periods = vol_rolling_window // 2`  (i.e. 126)
- `is_high_vol = (atr_pct >= rolling_median(atr_pct, window=vol_rolling_window, min_periods=rolling_median_min_periods))`

### Phase mapping
- `HV_Trend`    if `is_high_vol & trending`
- `LV_Trend`    if `~is_high_vol & trending`
- `HV_Ranging`  if `is_high_vol & ~trending`
- `LV_Ranging`  if `~is_high_vol & ~trending`

### Stop multipliers (if used downstream)
- `HV_Trend: 2.0`
- `LV_Trend: 1.0`
- `HV_Ranging: 2.0`
- `LV_Ranging: 0.5`

---

## 3) MT4-style detector (H1-native)

**Implementation:** `src/mt4_regimes.py` → `detect_mt4_regimes`

### Volatility (relative ATR ratio)
- `atr_short_period = 10`
- `atr_long_period = 100`
- `rel_atr_ratio = ATR(10) / ATR(100)`
- `is_high_vol = (rel_atr_ratio >= 1.0)`

Warm-up behavior (contract):
- If `ATR(100)` is missing and `rel_atr_ratio` is NaN, treat as **high volatility** (conservative).

### Trend (ADX)
- `adx_period = 14`
- `adx_trend_threshold = 20.0`
- `trending = (adx > adx_trend_threshold)`

### SMA direction (informational only)
- `sma_period = 200`
- SMA direction is not included in the 4-phase label.

### Phase mapping
Same as MarketPhaseDetector.

---

## 4) D1 derived from H1 (aggregation defaults)

**Spec:** `docs/regimes/d1_derived_from_h1.md`

- `day_boundary = 00:00 UTC`
- `min_valid_hours = 12`
- Daily label: **mode** of valid H1 phases
- Tie-break priority order (conservative):
  1. `HV_Trend`
  2. `LV_Trend`
  3. `HV_Ranging`
  4. `LV_Ranging`

Recommended diagnostics:
- `pct_trending`, `pct_high_vol`, `phase_mode_share`
- `n_total_hours`, `n_valid_hours`, `n_unknown_hours`
- optional: `trend_flip_count`, `vol_flip_count`

---