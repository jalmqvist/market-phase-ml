# Regime detection design rationale

This document explains why the project uses:

- multiple detectors
- multiple timeframes

## Key idea

Markets exhibit structure at multiple timescales:

- slow (macro regimes)
- fast (micro structure)

Attempting to model both with a single definition leads to:

- noise (if too fast)
- lag (if too slow)

## Solution

Use:

- D1 → regime (context)
- H1 → signal (execution)

## Outcome

This enables:

- cleaner signal extraction
- better interpretability
- reduced overfitting

This design emerged from:

- earlier MT4-based trading systems (H1)
- later D1-based regime modeling
- integration with H1 sentiment datasets

The current system formalizes this into a consistent architecture.

## Concrete examples

### Example 1 — H1 join (execution layer)

An H1 signal row at:

- `pair = eur-usd`
- `entry_time = 2026-04-19 13:00:00` (UTC, bar open)

should join to H1 phase labels where:

- `pair = eur-usd`
- `timestamp = 2026-04-19 13:00:00` (UTC, bar open)

### Example 2 — D1 derived context (00:00 UTC boundary)

A derived daily label for:

- `pair = eur-usd`
- `date_utc = 2026-04-19`

summarizes all available H1 phase bars with timestamps:

- `2026-04-19 00:00:00` through `2026-04-19 23:00:00` (UTC, bar opens)

No hours are invented; only timestamps present in the price series are used.