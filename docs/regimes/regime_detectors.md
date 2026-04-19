# Regime detectors (H1 vs D1)

This document describes the **two regime detectors currently implemented** in `market-phase-ml`,
including exact formulas, thresholds, and output columns.

## Shared phase label scheme (4-phase)

Both detectors classify each bar into one of:

| Phase        | Volatility | Trend    |
| ------------ | ---------- | -------- |
| `HV_Trend`   | High       | Trending |
| `HV_Ranging` | High       | Ranging  |
| `LV_Trend`   | Low        | Trending |
| `LV_Ranging` | Low        | Ranging  |

Direction (up/down) is intentionally excluded from the regime label.

---

## Timeframe roles (important)

The two detectors are designed for different roles:

### D1 detector (MarketPhaseDetector)

- Intended for **macro regime classification**
- Produces stable, low-noise signals
- Suitable for:
  - conditioning models
  - filtering trades
  - defining market context

### H1 detector (MT4-style)

- Intended for **micro structure analysis**
- More reactive and sensitive to short-term changes
- Suitable for:
  - feature construction
  - execution timing
  - short-horizon conditioning

### Important

H1 and D1 regimes are **not interchangeable**.

Any analysis must explicitly define:

- which timeframe provides the regime
- which timeframe provides the signal

---

## Input assumptions (important)

- Each detector operates on a **single pair** time series that is **sorted chronologically**.
- Indicators (ATR/ADX, rolling statistics) have warm-up periods. Early bars may contain:
  - missing intermediate values, or
  - unstable classifications
- Downstream consumers should treat `phase == "Unknown"` as non-actionable and handle it explicitly.

---

## Detector comparison (definitions + columns)

| Dimension                          | MarketPhaseDetector (D1-native)                              | MT4-style detector (H1-native)                               |
| ---------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Implementation                     | `src/phases.py` → `MarketPhaseDetector.detect_phases(df)`    | `src/mt4_regimes.py` → `detect_mt4_regimes(df)`              |
| Intended timescale                 | D1 (may be applied to H1 via same logic)                     | H1 (MT4-inspired)                                            |
| Required input columns             | `Open`, `High`, `Low`, `Close`, `Volume`                     | `Open`, `High`, `Low`, `Close`                               |
| Output columns added (exact names) | `atr`, `atr_pct`, `adx`, `high_vol`, `trending`, `phase`, `stop_atr_mult` | `atr_short`, `atr_long`, `rel_vol`, `high_volatility`, `adx`, `sma200`, `trending`, `phase` |
| Trend metric                       | ADX                                                          | ADX                                                          |
| ADX period                         | 14 (default)                                                 | 14                                                           |
| Trend threshold                    | Trending if `adx > 25.0` (default)                           | Trending if `adx > 20.0`                                     |
| Volatility metric                  | `atr_pct = ATR(14) / Close * 100` (defaults)                 | `rel_vol = ATR(10) / ATR(100)`                               |
| HV rule                            | `high_vol = atr_pct >= rolling_median(atr_pct, window=252, min_periods=126)` | `high_volatility = rel_vol >= 1.0` (warm-up: treat NaN rel_vol as high-vol) |
| Phase mapping                      | HV/LV × Trending/Ranging                                     | Same mapping                                                 |

---

## Trend and volatility definitions

Trend and volatility definitions differ between detectors by design:

- **D1 detector**
  - Trend: ADX > 25
  - Volatility: ATR% vs rolling median (long-term normalization)

- **H1 detector**
  - Trend: ADX > 20
  - Volatility: ATR(10) / ATR(100) (short vs long comparison)

These differences reflect the different timescales:

- D1 captures **sustained structural conditions**
- H1 captures **local expansions and contractions**

Users should not assume equivalence between H1 and D1 labels.

---

## Notes

### Warm-up behavior (MT4-style)
During warm-up when `ATR(100)` is missing, the MT4-style detector treats the bar as high-volatility.

### Stop multiplier (MarketPhaseDetector)
`MarketPhaseDetector` also outputs `stop_atr_mult`, which maps phase to ATR stop multipliers:
- `HV_Trend`: 2.0
- `LV_Trend`: 1.0
- `HV_Ranging`: 2.0
- `LV_Ranging`: 0.5