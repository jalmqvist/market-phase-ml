# DL Surface Integration

This document describes how `market-phase-ml` consumes the consolidated DL
signal cube produced by the sibling repository
[`market-sentiment-ml`](https://github.com/jalmqvist/market-sentiment-ml).

---

## Overview

`market-sentiment-ml` trains Deep-Learning (DL) models on H1 FX data, grouped
by *behavioral regime*.  After training it exports a consolidated Parquet file
(the "DL signal cube") where each row is a timestamped signal for one
(pair, surface) combination.  `market-phase-ml` can read that cube and inject
the DL signal columns into its feature-assembly pipeline.

The integration is **off by default** (`DL_SIGNALS_ENABLED=False`), so all
existing pipelines continue to run unchanged without the sibling repo present.

---

## Sibling-repo requirement

The feature is designed for the standard side-by-side layout:

```
workspace/
  market-phase-ml/       ← this repo
  market-sentiment-ml/   ← sibling repo
```

The default cube path resolves to:

```
../market-sentiment-ml/data/output/dl_signals/dl_signals_h1_v1.parquet
```

You can override the path via the `DL_SIGNALS_CUBE_PATH` environment variable
or by editing `src/dl_config.py`.

---

## Generating the DL cube in market-sentiment-ml

Run the inference + consolidation pipeline in `market-sentiment-ml`:

```bash
# Inside market-sentiment-ml/
python scripts/export_dl_predictions.py    # per-run prediction artifacts
python scripts/build_dl_signal_cube.py     # consolidated cube + manifest
```

This produces:

```
market-sentiment-ml/data/output/dl_signals/
  dl_signals_h1_v1.parquet          ← consolidated cube
  DL_SIGNAL_MANIFEST_h1_v1.json     ← manifest
```

### Cube schema

| Column               | Type          | Notes                                      |
|----------------------|---------------|--------------------------------------------|
| `pair`               | string        | lowercase `xxx-yyy` (e.g. `eur-usd`)       |
| `entry_time`         | datetime64    | tz-naive UTC, hourly aligned               |
| `model`              | string        | e.g. `lstm`                                |
| `target_horizon`     | Int64         | number of H1 bars                          |
| `feature_set`        | string        | e.g. `price_trend`                         |
| `dl_regime`          | string        | `HVTF`, `LVTF`, `HVR`, or `LVR`           |
| `signal_strength`    | float64       | `2 * pred_prob_up - 1`, range `[-1, +1]`  |
| `dl_confidence`      | float64       | *(optional)* model confidence              |
| `pred_prob_up`       | float64       | *(optional)* raw up-probability `[0, 1]`  |
| `prediction_timestamp` | datetime64  | *(optional)* per-row inference timestamp  |

**Uniqueness contract:** `(pair, entry_time, model, target_horizon, feature_set, dl_regime)`

---

## Enabling the integration

### 1. Set environment variables

```bash
export DL_SIGNALS_ENABLED=true
# Optional overrides (defaults shown):
export DL_SIGNALS_CUBE_PATH=../market-sentiment-ml/data/output/dl_signals/dl_signals_h1_v1.parquet
export DL_SURFACE_MODEL=lstm
export DL_SURFACE_TARGET_HORIZON=24
export DL_SURFACE_FEATURE_SET=price_trend
export DL_SURFACE_REGIME=HVTF   # one of: HVTF, LVTF, HVR, LVR
```

### 2. Or edit `src/dl_config.py` directly

```python
DL_SIGNALS_ENABLED = True

DL_SIGNAL_SURFACE = {
    "model": "lstm",
    "target_horizon": 24,      # number of H1 bars
    "feature_set": "price_trend",
    "dl_regime": "HVTF",       # required; exact-match surface selection
}
```

---

## Surface selection semantics

`dl_regime` is **required** and must match exactly one regime.  This is
intentional: DL findings are regime-conditional, so treating regimes as
interchangeable would erase the main behavioural insight.

Valid MSML-DL regimes and their MPML taxonomy equivalents:

| MSML-DL  | MPML equiv | Description                       |
|----------|------------|-----------------------------------|
| `HVTF`   | `HVTF`     | High-Volatility Trend-Following   |
| `LVTF`   | `LVTF`     | Low-Volatility Trend-Following    |
| `HVR`    | `HVMR`     | High-Volatility Mean-Reversion    |
| `LVR`    | `LVMR`     | Low-Volatility Mean-Reversion     |

The `mpml_regime_equiv` column in the loader output carries this mapping for
informational purposes.  It must **not** be used as an ML feature.

---

## Feature group: `dl_signal`

When DL signals are attached, the following columns are available in the
`dl_signal` feature group (`features/registry.py`):

| Feature column       | Source column      | Notes                          |
|----------------------|--------------------|--------------------------------|
| `dl_signal_strength` | `signal_strength`  | renamed to avoid collision     |
| `dl_confidence`      | `dl_confidence`    | NaN if absent from cube        |
| `pred_prob_up`       | `pred_prob_up`     | NaN if absent from cube        |

The `baseline_plus_dl` experiment in `features/experiments.py` combines the
`baseline` and `dl_signal` groups and is automatically included when
`DL_SIGNALS_ENABLED=True`.

---

## Running the experiments

```bash
# Default (DL disabled — unchanged behaviour):
python features/run_experiments.py

# With DL signals:
DL_SIGNALS_ENABLED=true python features/run_experiments.py
```

---

## Sanity-check script

Run the loader validation suite (no real artifact required):

```bash
python scripts/validate_dl_surface.py
```

To also validate a real cube:

```bash
python scripts/validate_dl_surface.py \
  --cube-path ../market-sentiment-ml/data/output/dl_signals/dl_signals_h1_v1.parquet \
  --model lstm \
  --target-horizon 24 \
  --feature-set price_trend \
  --dl-regime HVTF
```

---

## Loader API

```python
from pathlib import Path
from src.dl_surface_loader import load_dl_surface, empty_dl_surface_df
from src.dl_config import DL_SIGNALS_ENABLED, DL_SIGNALS_CUBE_PATH, DL_SIGNAL_SURFACE

# Load a surface (returns empty DF on failure when strict=False)
surface_df = load_dl_surface(
    cube_path=DL_SIGNALS_CUBE_PATH,
    surface=DL_SIGNAL_SURFACE,
    strict=False,   # warn + return empty DF on any failure
)

# Safe empty fallback (consistent dtypes, never None)
empty = empty_dl_surface_df()
```

### Attaching signals to a feature DataFrame

```python
from features.assembler import attach_dl_signals

# df must have 'pair' and 'entry_time' columns
df = attach_dl_signals(df, surface_df)
# dl_signal_strength, dl_confidence, pred_prob_up now available in df
# rows without a matching DL signal get NaN
```

---

## Safety invariants

- `mpml_regime_equiv` and `prediction_timestamp` are carried in the loader
  output for informational/diagnostic use but must **not** be passed as ML
  features.
- Missing DL columns are filled with `NaN` (not 0) so that downstream
  models can distinguish "no signal" from "neutral signal".
- The join is a left-join keyed on `(pair, entry_time)`.  Rows in the main
  DataFrame that have no matching DL signal silently receive `NaN`, ensuring
  deterministic behavior.
- `strict=True` is recommended for production runs where a missing or corrupt
  cube should be treated as a hard error.
