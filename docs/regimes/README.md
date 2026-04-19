# Regime detection documentation

This folder defines the **canonical regime detection contracts** used by `market-phase-ml`.

It covers:

- The two built-in regime detectors:
  - **MarketPhaseDetector** (D1-native; ATR% vs rolling median, ADX>25)
  - **MT4-style detector** (H1-native; ATR(10)/ATR(100), ADX>20)
- A deterministic **D1 derived** regime view computed from canonical H1 phase labels.
- A common, detector-agnostic output schema for phase labels.

## Files

- `regime_detectors.md` — exact definitions, thresholds, computed columns, and a side-by-side comparison table.
- `d1_derived_from_h1.md` — daily aggregation spec (00:00 UTC boundary, mode + tie-break rules, coverage rules).
- `schema_phase_labels_v1.md` — proposed common phase output schema (keys + canonical columns + optional detector-specific diagnostics).

## Design principles

- **H1 is canonical** when joining to H1-native datasets (e.g. sentiment features keyed by `entry_time`).
- Detectors must not invent bars; they operate on the timestamps present in the price series.
- Any D1 view derived from H1 must be deterministic and include diagnostics (coverage, dominance).

---

## Quick integration recipe

### Joining phase labels to H1-native datasets (e.g. sentiment features)

- Compute phase labels on the **same H1 price series** used to define `entry_time`.
- Join using an **exact match** on bar open timestamps:
  - `(pair, entry_time)` from `market-sentiment-ml`
  - `(pair, timestamp)` from `market-phase-ml` phase labels

### Using daily (D1) context with H1 signals

- Derive D1 regimes from H1 using `d1_derived_from_h1.md` (00:00 UTC boundary).
- Join the derived D1 context back onto H1 rows using `date_utc = floor(entry_time to UTC date)`.

---

## Why two detectors and timeframes?

This project includes two regime detectors:

- **MarketPhaseDetector (D1-native)** — originally developed for a daily (D1) trading system
- **MT4-style detector (H1-native)** — derived from an earlier MT4 system designed for intraday (H1) trading

The existence of two detectors reflects a deliberate design choice rather than inconsistency.

### Historical context

The D1 detector originates from a prior trading system where regime classification
was designed for daily decision-making. The H1 detector was later introduced when
working with sentiment data, which is naturally aligned to hourly price series.

During development, it became clear that:

- **Different timescales require different regime definitions**
  - D1: stable, macro structure
  - H1: reactive, micro structure
- A single unified definition would either:
  - be too noisy (if H1-based), or
  - too slow (if D1-based)

### Current design

The system now explicitly separates roles:

- **D1 regimes** → market context (slow, stable)
- **H1 regimes / features** → signal construction (fast, reactive)

This enables multi-timescale modeling:


D1 regime (context)
↓
H1 signals (sentiment, behavioral features)
↓
decision layer


### Guideline

- Use **D1 regimes for conditioning and filtering**
- Use **H1 features for signal generation**

The two detectors are therefore complementary, not redundant.