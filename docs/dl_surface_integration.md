# DL surface integration (v1)

This document describes how `market-phase-ml` consumes **row-level DL prediction artifacts** produced by the upstream repo `market-sentiment-ml`.

Key design principles:

- Integration is **artifact-based**, not code-based.
- `market-phase-ml` remains fully functional when DL artifacts are missing (opt-in, off by default).
- The interface is a **semantic surface selector** (stable) rather than training internals (unstable).
- No implicit aggregation/ensembling in v1: one selector → one surface → one interpretation.

---

## Overview: what is being consumed?

`market-sentiment-ml` produces per-run artifacts under:

- `market-sentiment-ml/data/output/dl_predictions/<run_id>.parquet`
- `market-sentiment-ml/data/output/dl_predictions/<run_id>.manifest.json`

Each parquet contains a timestamped time series of DL predictions at H1 resolution for a single (model, target_horizon, feature_set, dl_regime) surface.

In v1, `market-phase-ml` consumes **one per-run parquet** (one surface) and joins it into the feature frame.

> **Note:** Consolidation into a multi-surface cube is a future extension; v1 does not require it.

---

## Recommended local layout

Clone the repos side-by-side:

```text
~/projects/
  market-sentiment-ml/
  market-phase-ml/
```

This allows `market-phase-ml` to reference DL artifacts by relative path.

---

## Generating the DL artifact (market-sentiment-ml side)

Run the training/inference pipeline inside `market-sentiment-ml`:

```bash
# Inside market-sentiment-ml/
python -m research.deep_learning.train \
  --dataset-version 1.3.2 \
  --pairs EURUSD \
  --regime HVTF \
  --target-horizon 24   # produces per-run prediction artifact
```

This writes one file per run:

- `data/output/dl_predictions/<run_id>.parquet`
- `data/output/dl_predictions/<run_id>.manifest.json`

Example run ID: `mlp__HVTF__24__price_trend__20260510T182643Z`

---

## DL surface identity (stable interface)

A **surface** is the operational identity of a DL signal stream:

- `model`
- `target_horizon` (**number of bars**; numeric)
- `feature_set`
- `dl_regime` (producer taxonomy: `HVTF | LVTF | HVR | LVR`)

This is treated as *signal identity*, not provenance. In v1, `dl_regime` must be explicitly specified and must be one of `HVTF | LVTF | HVR | LVR`.

`main.py` does **not** support `dl_regime="all"` in v1. If `dl_regime` is `"all"` (or invalid), the run logs a warning and skips DL attachment (baseline behavior).

> **For H1 DL artifacts**:
> **target_horizon=24 means 24 hourly bars (~24h), NOT D1 bars.**

---

## Expected parquet schema (upstream naming)

A per-run parquet must contain these columns:

| Column           | Type       | Notes                                           |
|------------------|------------|-------------------------------------------------|
| `pair`           | string     | lowercase `xxx-yyy` (e.g. `eur-usd`)            |
| `entry_time`     | datetime64 | tz-naive UTC, H1-aligned                        |
| `signal_strength`| float64    | `2 * pred_prob_up - 1`, range `[-1, +1]`        |
| `model`          | string     | e.g. `mlp`                                     |
| `target_horizon` | Int64      | number of H1 bars                               |
| `feature_set`    | string     | e.g. `price_trend`                              |
| `dl_regime`      | string     | `HVTF`, `LVTF`, `HVR`, or `LVR`                |
| `pred_prob_up`   | float64    | *(optional)* raw up-probability `[0, 1]`        |
| `dl_confidence`  | float64    | *(optional)* model confidence                   |
| `prediction_timestamp` | datetime64 | *(optional)* per-row inference timestamp  |

**Uniqueness contract:** `(pair, entry_time, model, target_horizon, feature_set, dl_regime)`

---

## MPML feature naming (`dl_*` prefix)

After loading through `src.dl_surface_loader`, all DL columns are renamed with a `dl_` prefix to avoid name collisions with native MPML features and to make provenance clear:

| MPML feature column       | Upstream source column   | Notes                      |
|---------------------------|--------------------------|----------------------------|
| `dl_signal_strength`      | `signal_strength`        | always present             |
| `dl_pred_prob_up`         | `pred_prob_up`           | present when in parquet    |
| `dl_confidence`           | `dl_confidence`          | present when in parquet    |
| `dl_prediction_timestamp` | `prediction_timestamp`   | present when in parquet    |

> The upstream parquet uses `pred_prob_up`; MPML always accesses it as `dl_pred_prob_up`.
> Never use the unprefixed `pred_prob_up` name inside MPML frames.

---

## Configuration

### Option 1 — environment variables

```bash
export DL_SIGNALS_ENABLED=true
# Optional overrides (defaults shown):
export DL_PREDICTION_ARTIFACT_PATH=../market-sentiment-ml/data/output/dl_predictions/
export DL_SURFACE_MODEL=mlp
export DL_SURFACE_TARGET_HORIZON=24
export DL_SURFACE_FEATURE_SET=price_trend
export DL_SURFACE_REGIME=HVTF   # one of: HVTF, LVTF, HVR, LVR
```

### Option 2 — edit `src/dl_config.py` directly

```python
DL_SIGNALS_ENABLED = True

DL_SIGNAL_SURFACE = {
    "model": "mlp",
    "target_horizon": 24,      # bars, not hours
    "feature_set": "q0.5",
    "dl_regime": "HVTF",       # required; exact-match surface selection
}

DL_PREDICTION_ARTIFACT_PATH = (
    "../market-sentiment-ml/data/output/dl_predictions/"
    "mlp__HVTF__24__price_trend__20260510T182643Z.parquet"
)
```

---

## Loader behavior (contract)

1. **Fail closed by default**
   - `DL_SIGNALS_ENABLED=False`: return empty DF immediately.
   - Artifact path missing or corrupt: warn and return empty DF (`strict=False`) or raise (`strict=True`).

2. **Validate invariants**
   - Required columns exist.
   - `entry_time` is tz-naive UTC and H1-aligned.
   - `pair` is normalized `xxx-yyy`.
   - No duplicate `(pair, entry_time)`.
   - Monotone `entry_time` per `pair`.
   - `pred_prob_up in [0, 1]` and `signal_strength in [-1, +1]`.

3. **Surface selection is exact-match**
   - Rows are filtered on `model`, `target_horizon`, `feature_set`, `dl_regime`.
   - Mismatches raise (strict) or warn + empty (non-strict).

4. **Normalize to MPML conventions**
   - `entry_time` -> `timestamp` (internal join key).
   - `signal_strength` -> `dl_signal_strength`.
   - `pred_prob_up` -> `dl_pred_prob_up` (when present).
   - `prediction_timestamp` -> `dl_prediction_timestamp` (when present).

5. **Left-join into the feature frame**
   - Keyed on `(pair, timestamp)`.
   - Rows without a matching DL signal receive `NaN` (not 0).

---

## Regime taxonomy (producer vs consumer)

| MSML-DL  | MPML equiv | Description                       |
|----------|------------|-----------------------------------|
| `HVTF`   | `HVTF`     | High-Volatility Trend-Following   |
| `LVTF`   | `LVTF`     | Low-Volatility Trend-Following    |
| `HVR`    | `HVMR`     | High-Volatility Mean-Reversion    |
| `LVR`    | `LVMR`     | Low-Volatility Mean-Reversion     |

The loader adds an `mpml_regime_equiv` column for informational/diagnostic use.  It must **not** be used as an ML feature (it would introduce leakage).

---

## Feature group: `dl_signal`

The `dl_signal` feature group (defined in `features/registry.py`) makes the following columns available to experiments when DL signals are attached:

| Feature column       | Description                          |
|----------------------|--------------------------------------|
| `dl_signal_strength` | Directional signal `[-1, +1]`        |
| `dl_pred_prob_up`    | Raw up-probability `[0, 1]`          |
| `dl_confidence`      | Model confidence (optional)          |

The `baseline_plus_dl` experiment combines the `baseline` and `dl_signal` groups and is included automatically when `DL_SIGNALS_ENABLED=True`.

---

## Running the experiments

```bash
# Default (DL disabled — unchanged behaviour):
python features/run_experiments.py

# With DL signals:
DL_SIGNALS_ENABLED=true python features/run_experiments.py
```

---

## Loader API

```python
from pathlib import Path
from src.dl_surface_loader import load_dl_surface, empty_dl_surface_df
from src.dl_config import (
    DL_SIGNALS_ENABLED,
    DL_SIGNAL_SURFACE,
    resolve_dl_prediction_artifact_path,
)

artifact_path = resolve_dl_prediction_artifact_path()
if artifact_path is None:
    surface_df = empty_dl_surface_df()
else:
    surface_df = load_dl_surface(
        cube_path=artifact_path,
        surface=DL_SIGNAL_SURFACE,
        strict=False,   # warn + return empty DF on any failure
    )
```

### Attaching H1 signals to a feature DataFrame

```python
from features.assembler import attach_dl_signals

# df must have 'pair' and 'entry_time' columns
df = attach_dl_signals(df, surface_df)
# dl_signal_strength, dl_pred_prob_up, dl_confidence now available in df
# rows without a matching DL signal get NaN
```

---

## D1 aggregation layer (`src/dl_daily_features.py`) and operational attachment in `main.py`

### Overview

The D1 aggregation layer converts H1 DL signal rows into **daily** features suitable for joining to the D1 regime/trading pipeline.

`main.py` now operationally attaches these D1 features immediately after `process_pair(...)` in the processed-data pipeline when DL mode is enabled.

### No-leakage semantics

For a D1 prediction at the **start** of trading day D (timestamp 00:00 UTC), features are aggregated from H1 bars whose `entry_time` falls on calendar day **D-1** only (all bars strictly before D 00:00 UTC).

```
H1 bar at 2024-01-04 xx:00  -->  trading_day = 2024-01-05  (D1 bar that may use it)
```

No H1 bar from day D is ever used to compute features for the D1 prediction at the start of day D.

### Produced daily feature columns

| Column                 | Type    | Description                                           |
|------------------------|---------|-------------------------------------------------------|
| `dl_signal_mean_24h`   | float64 | Mean of `dl_signal_strength` over all H1 bars in D-1 |
| `dl_signal_std_24h`    | float64 | Std  of `dl_signal_strength` (ddof=1)                 |
| `dl_signal_last`       | float64 | Last (23:00 UTC) `dl_signal_strength` value           |
| `dl_signal_abs_mean`   | float64 | Mean of `|dl_signal_strength|`                        |
| `dl_signal_flip_count` | float64 | Number of sign changes in `dl_signal_strength`        |

> `dl_signal_flip_count` uses `float64` (not `int64`) so that `NaN` is
> representable after left-joins for days without DL coverage.

### End-to-end example

**Step 1 — MSML: export the per-run DL artifact** (see [Generating the DL artifact](#generating-the-dl-artifact-market-sentiment-ml-side) above)

**Step 2 — MPML: compute D1 daily features**

```python
from pathlib import Path
from src.dl_daily_features import load_and_aggregate_d1

daily_df = load_and_aggregate_d1(
    artifact_path=Path(
        "../market-sentiment-ml/data/output/dl_predictions/"
        "mlp__HVTF__24__price_trend__20260510T182643Z.parquet"
    ),
    surface={
        "model": "mlp",
        "target_horizon": 24,
        "feature_set": "price_trend",
        "dl_regime": "HVTF",
    },
    strict=False,
)
# daily_df: DataFrame keyed by (pair, trading_day) with the 5 dl_signal_* columns
print(daily_df.head())
#         pair  trading_day  dl_signal_mean_24h  dl_signal_std_24h  ...
# 0  eur-usd   2024-01-02        0.12               0.08           ...
```

**Step 3 — how `main.py` consumes D1 daily features (operational v1)**

```python
# In main.py:
from src.dl_daily_features import load_and_aggregate_d1
from src.dl_config import (
    DL_SIGNALS_ENABLED, DL_SIGNAL_SURFACE,
    resolve_dl_prediction_artifact_path,
)

if DL_SIGNALS_ENABLED:
    artifact_path = resolve_dl_prediction_artifact_path()
    if artifact_path is not None:
        daily_df = load_and_aggregate_d1(artifact_path, DL_SIGNAL_SURFACE)

        # d1_df is keyed by (pair, timestamp) where timestamp = start-of-day UTC.
        d1_df = d1_df.merge(
            daily_df.rename(columns={"trading_day": "timestamp"}),
            on=["pair", "timestamp"],
            how="left",
        )
        # dl_signal_mean_24h ... dl_signal_flip_count are optional columns.
```

Per pair, `main.py` logs:

- attached DL columns
- per-column coverage (% non-null)
- first/last non-null timestamps
- row counts before/after join and after feature-mask filtering + retention ratio

Additionally, lightweight integrity assertions guard:

- monotonic `trading_day` in D1 features
- no duplicate `(pair, trading_day)` rows before join
- no row multiplication across the left join

---

## Sanity-check script

Run the loader validation suite (no real artifact required):

```bash
python scripts/validate_dl_surface.py
```

To also validate a real artifact:

```bash
python scripts/validate_dl_surface.py \
  --cube-path ../market-sentiment-ml/data/output/dl_predictions/mlp__HVTF__24__price_trend__20260510T182643Z.parquet \
  --model mlp \
  --target-horizon 24 \
  --feature-set price_trend \
  --dl-regime HVTF
```

---

## Safety invariants

- `mpml_regime_equiv` and `dl_prediction_timestamp` are carried in the loader output for informational/diagnostic use but must **not** be passed as ML features.
- Missing DL columns are filled with `NaN` (not 0) so that downstream models can distinguish "no signal" from "neutral signal".
- The H1 join is a left-join keyed on `(pair, timestamp)`.  Rows without a matching DL signal receive `NaN`, ensuring deterministic behavior.
- `strict=True` is recommended for production runs where a missing or corrupt artifact should be a hard error.
- The loader always renames `pred_prob_up` -> `dl_pred_prob_up` to avoid collisions with native MPML features.

---

## Troubleshooting

### Artifact exists but loader returns empty

Common causes:

- `pair` format mismatch (must be lowercase `xxx-yyy`).
- `entry_time` timezone or alignment issues (must be tz-naive UTC, H1-aligned).
- Duplicates on `(pair, entry_time)` in the artifact.
- `DL_SIGNAL_SURFACE` does not exactly match the parquet's identity columns.

### Artifact path is stale

Per-run artifacts are per-run; point `DL_PREDICTION_ARTIFACT_PATH` at the correct `<run_id>.parquet` from your most recent DL training run.

---

## Future extensions (out of scope for v1)

- `dl_regime=None` meaning "all regimes" (multi-regime aggregation).
- Multi-surface ensembles / regime blending.
- Calibration (isotonic, Platt, etc.).
- Consolidated multi-surface cube (aggregating across runs/surfaces).

When those are introduced, they should be explicit orchestration rather than overloaded selector semantics.
