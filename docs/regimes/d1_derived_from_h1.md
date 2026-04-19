# D1 derived regimes from H1 (aggregation spec)

This document defines how to derive a **daily (D1) regime view** from canonical
**hourly (H1) phase labels**.

---

## Purpose

This specification defines how to construct a **D1 regime view** from H1 phase labels.

This is used when:

- working with H1-native datasets (e.g. sentiment features)
- requiring a **daily regime context** without introducing external data

The derived D1 regime serves as a **context layer**, while H1 signals remain the execution layer.

## Day boundary

- Day boundary is **00:00 UTC**.
- A D1 record for `date_utc` summarizes H1 bars with timestamps:
  - `date_utc 00:00` through `date_utc 23:00` (bar open timestamps)
- No bars are invented; only timestamps present in the price series are used.

## Inputs (H1 phase table)
Minimum required columns:
- `pair`
- `timestamp` (H1 bar open UTC)
- `phase` in {`HV_Trend`,`LV_Trend`,`HV_Ranging`,`LV_Ranging`,`Unknown`}
- `is_high_vol` (bool) and `is_trending` (bool), or derivable from `phase`

## Valid-hour definition
- Valid if `phase` is one of the four real phases.
- `phase == "Unknown"` is invalid for aggregation counts.

## Minimum coverage
- `min_valid_hours` default: **12**
- If `n_valid_hours < min_valid_hours`, then:
  - `phase_d1 = "Unknown"`
  - `is_trending_d1 = null`
  - `is_high_vol_d1 = null`
  - still output diagnostics counts

## Primary daily label: mode + deterministic tie-break
If coverage passes:
1. Count hours in each phase.
2. Choose `phase_d1` as the phase with the highest count (mode).
3. If tie, break ties using priority order (conservative):
   1) `HV_Trend`
   2) `LV_Trend`
   3) `HV_Ranging`
   4) `LV_Ranging`

## Daily continuous diagnostics (recommended outputs)
Compute over valid hours:
- `pct_trending = mean(is_trending == True)`
- `pct_high_vol = mean(is_high_vol == True)`
- `phase_mode_share = max_phase_count / n_valid_hours`

Optional:
- `trend_flip_count` (count of transitions in `is_trending`)
- `vol_flip_count` (count of transitions in `is_high_vol`)

## Derived daily booleans (recommended)

When coverage passes, daily booleans can be derived from valid-hour proportions:

- `is_trending_d1 = (pct_trending >= 0.5)`
- `is_high_vol_d1 = (pct_high_vol >= 0.5)`

## Output schema (D1 derived) — normative

Keys:
- `pair` (string)
- `date_utc` (date)

Core:
- `phase_d1` (string): one of `HV_Trend`, `LV_Trend`, `HV_Ranging`, `LV_Ranging`, `Unknown`
- `is_trending_d1` (bool, nullable)
- `is_high_vol_d1` (bool, nullable)

Diagnostics:
- `pct_trending` (float, nullable)
- `pct_high_vol` (float, nullable)
- `phase_mode_share` (float, nullable)
- `n_total_hours` (int)
- `n_valid_hours` (int)
- `n_unknown_hours` (int)
- optional: `trend_flip_count` (int), `vol_flip_count` (int)

### Nullability rules

- If `n_valid_hours < min_valid_hours`:
  - `phase_d1 = "Unknown"`
  - `is_trending_d1 = null`
  - `is_high_vol_d1 = null`
  - `pct_trending`, `pct_high_vol`, `phase_mode_share` may still be computed from valid hours **if any exist**; otherwise null.

---

## Interpretation

The derived D1 regime should be interpreted as:

- a **summary of intraday structure**
- not identical to a D1-native detector, but consistent with H1 inputs

It is particularly useful when:

- signals are defined at H1 resolution
- regime conditioning is required without mixing data sources