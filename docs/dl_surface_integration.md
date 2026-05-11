# DL surface integration (v1)

This document describes how `market-phase-ml` consumes **row-level DL prediction artifacts** produced by the upstream repo `market-sentiment-ml`.

Key design principles:

- Integration is **artifact-based**, not code-based.
- `market-phase-ml` remains fully functional when DL artifacts are missing (opt-in).
- The interface is a **semantic surface selector** (stable) rather than training internals (unstable).
- No implicit aggregation/ensembling in v1: one selector → one surface → one interpretation.

---

## Overview: what is being consumed?

`market-sentiment-ml` produces per-run artifacts under:

- `market-sentiment-ml/data/output/dl_predictions/<run_id>.parquet`
- `market-sentiment-ml/data/output/dl_predictions/<run_id>.manifest.json`

Each parquet contains a timestamped time series of DL predictions at H1 resolution.

In v1, `market-phase-ml` consumes **one per-run parquet** (one surface) and joins it into the feature frame.

(Consolidation into a multi-surface cube is a later step; v1 does not require a cube.)

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

## DL surface identity (stable interface)

A **surface** is the operational identity of a DL signal stream:

- `model`
- `target_horizon` (**number of bars**; numeric)
- `feature_set`
- `dl_regime` (producer taxonomy: `HVTF | LVTF | HVR | LVR`)

This is treated as *signal identity*, not provenance.

### V1 requirement: `dl_regime` is required

In v1, `dl_regime` must be explicitly specified. Do **not** allow `dl_regime=None` to mean “all regimes”.

Reason: the DL work is regime-conditional and mixing regimes implicitly creates ambiguous semantics.

---

## Configuration (v1)

### Minimal recommended config shape

Configure one surface selection dict:

```python
DL_SIGNAL_SURFACE = {
    "model": "mlp",
    "target_horizon": 24,      # bars
    "feature_set": "price_trend",
    "dl_regime": "HVTF",
}
```

Enable integration and point at a per-run parquet artifact:

```python
DL_SIGNALS_ENABLED = True

DL_PREDICTION_ARTIFACT_PATH = (
  "../market-sentiment-ml/data/output/dl_predictions/"
  "mlp__HVTF__24__price_trend__20260510T182643Z.parquet"
)
```

Notes:

- `target_horizon` is **bars**, not “hours”. This keeps the identity frequency-agnostic for future non-H1 exports.
- The artifact path can be absolute or relative. Relative paths are recommended for side-by-side repo usage.

---

## Expected parquet schema (what the loader requires)

A per-run parquet must contain these columns:

- `pair` (lowercase `xxx-yyy`)
- `entry_time` (tz-naive UTC; H1-aligned)
- `pred_prob_up` (float `[0,1]`)
- `signal_strength` (float `2*pred_prob_up - 1`, range `[-1,+1]`)
- identity columns:
  - `model`, `target_horizon` (bars), `feature_set`, `dl_regime`

Optional columns (if present, can be passed through):
- `confidence`
- `pred_direction`
- `prediction_timestamp`

---

## Loader behavior (contract)

A correct v1 loader should:

1. **Fail closed by default**
   - If `DL_SIGNALS_ENABLED` is false: return empty DF.
   - If enabled but artifact path missing: warn and return empty DF (or raise in strict mode).

2. **Validate invariants**
   - required columns exist
   - `entry_time` tz-naive UTC and H1-aligned
   - `pair` normalized `xxx-yyy`
   - no duplicate `(pair, entry_time)`
   - monotonic `entry_time` per `pair`
   - `pred_prob_up ∈ [0,1]`, `signal_strength ∈ [-1,+1]`

3. **Surface selection is exact-match**
   - The loaded parquet should match the configured `DL_SIGNAL_SURFACE` on:
     - `model`, `target_horizon`, `feature_set`, `dl_regime`
   - If the parquet includes identity columns that conflict with the config, raise (strict) or warn+empty (non-strict).

4. **Normalize to MPML timestamp convention**
   - rename `entry_time` → `timestamp` internally

5. **Join into the feature frame**
   - left-join DL columns onto the MPML base frame keyed by `(pair, timestamp)`
   - keep missing values explicit (NaN) unless you deliberately define a fill policy

### Recommended feature column names in MPML

To avoid collisions and to keep provenance clear:

- `dl_pred_prob_up`
- `dl_signal_strength`
- optional:
  - `dl_confidence`
  - `dl_pred_direction`
  - `dl_prediction_timestamp`

---

## Regime taxonomy notes (producer vs consumer)

Producer (`market-sentiment-ml`) DL regimes:

- `HVTF`, `LVTF`, `HVR`, `LVR`

`market-phase-ml` uses a different naming convention for mean-reversion regimes.

If you need a mapped column for internal comparisons, do it explicitly and keep the producer regime intact:

- `HVR` → `HVMR`
- `LVR` → `LVMR`
- `HVTF` → `HVTF`
- `LVTF` → `LVTF`

Recommended: add a separate column such as `mpml_regime_equiv` rather than overwriting `dl_regime`.

---

## Future extensions (explicit, out of scope for v1)

These are intentionally **not** part of v1 surface selection:

- `dl_regime=None` meaning “all regimes”
- regime blending / adaptive mixing
- calibration systems (isotonic/Platt/etc)
- multi-surface ensembles

When those are introduced, they should be explicit orchestration, e.g.:

```python
surface_a = load_dl_surface(...)
surface_b = load_dl_surface(...)
ensemble = combine_surfaces([surface_a, surface_b], method="...")
```

rather than overloading the selector semantics.

---

## Troubleshooting

### Artifact exists but loader returns empty
Common causes:

- `pair` format mismatch (must be `xxx-yyy` lowercase)
- `entry_time` timezone or alignment issues (must be tz-naive UTC, H1-aligned)
- duplicates on `(pair, entry_time)` in the artifact
- `DL_SIGNAL_SURFACE` does not exactly match the parquet’s identity columns

### Artifact path is stale
Since per-run artifacts are per run, you must point to the correct `<run_id>.parquet` produced by your most recent DL run, or define a small “latest artifact resolver” utility (optional and out of scope for v1).

---

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

The default artifact path resolves to:

```
../market-sentiment-ml/data/output/dl_predictions/
```

By default, `market-phase-ml` resolves this directory to the newest
`*.parquet` file (single-surface per-run export).

You can override the path via `DL_PREDICTION_ARTIFACT_PATH`
(or legacy `DL_SIGNALS_CUBE_PATH`) in `src/dl_config.py`.

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
export DL_PREDICTION_ARTIFACT_PATH=../market-sentiment-ml/data/output/dl_predictions/
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

| Feature column       | Source column      | Notes                                    |
|----------------------|--------------------|------------------------------------------|
| `dl_signal_strength` | `signal_strength`  | renamed to avoid collision               |
| `dl_confidence`      | `dl_confidence`    | NaN if absent from cube                  |
| `dl_pred_prob_up`    | `pred_prob_up`     | renamed to `dl_*` prefix; NaN if absent  |

All DL feature column names carry the `dl_` prefix to avoid name collisions
with MPML native features and to make provenance clear.

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
from src.dl_config import (
    DL_SIGNALS_ENABLED,
    DL_PREDICTION_ARTIFACT_PATH,
    DL_SIGNAL_SURFACE,
    resolve_dl_prediction_artifact_path,
)

# Load a surface (returns empty DF on failure when strict=False)
artifact_path = resolve_dl_prediction_artifact_path()
if artifact_path is None:
    surface_df = empty_dl_surface_df()
else:
    surface_df = load_dl_surface(
        cube_path=artifact_path,
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
# dl_signal_strength, dl_confidence, dl_pred_prob_up now available in df
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
- All DL feature column names carry the `dl_` prefix to avoid collisions with
  native MPML features. The loader always renames `pred_prob_up` →
  `dl_pred_prob_up` and `prediction_timestamp` → `dl_prediction_timestamp`.

---

## D1 aggregation layer (`src/dl_daily_features.py`)

### Overview

The D1 aggregation layer converts H1 DL signal rows into daily features that
can be joined to the D1 regime/trading pipeline.  It is implemented in
`src/dl_daily_features.py` and is **not** integrated into `main.py` yet — it
is available as a standalone utility for future integration.

### No-leakage semantics

For a D1 prediction at the **start** of trading day D (timestamp 00:00 UTC),
features are aggregated from H1 bars whose `entry_time` falls on calendar day
**D − 1** only (strictly before D 00:00 UTC).

```
H1 bar at 2024-01-04 xx:00  →  trading_day = 2024-01-05  (D1 prediction day)
```

### Produced daily feature columns

| Column                | Description                                              |
|-----------------------|----------------------------------------------------------|
| `dl_signal_mean_24h`  | Mean of `dl_signal_strength` over all H1 bars in D−1    |
| `dl_signal_std_24h`   | Std  of `dl_signal_strength` (ddof=1)                   |
| `dl_signal_last`      | Last (23:00 UTC) `dl_signal_strength` value              |
| `dl_signal_abs_mean`  | Mean of `|dl_signal_strength|`                           |
| `dl_signal_flip_count`| Number of sign changes in `dl_signal_strength`          |

### End-to-end example

**Step 1 — market-sentiment-ml: export the per-run DL artifact**

```bash
# Inside market-sentiment-ml/
python scripts/export_dl_predictions.py
# Produces: data/output/dl_predictions/<run_id>.parquet
```

**Step 2 — market-phase-ml: run D1 aggregation**

```python
from pathlib import Path
from src.dl_daily_features import load_and_aggregate_d1

daily_df = load_and_aggregate_d1(
    artifact_path=Path(
        "../market-sentiment-ml/data/output/dl_predictions/"
        "lstm__HVTF__24__price_trend__20260510T182643Z.parquet"
    ),
    surface={
        "model": "lstm",
        "target_horizon": 24,
        "feature_set": "price_trend",
        "dl_regime": "HVTF",
    },
    strict=False,
)
# daily_df: DataFrame keyed by (pair, trading_day) with 5 dl_signal_* columns
print(daily_df.head())
#         pair  trading_day  dl_signal_mean_24h  dl_signal_std_24h  ...
# 0  eur-usd   2024-01-02        0.12               0.08           ...
```

**Step 3 — how main.py would consume D1 daily features (future integration)**

> **Note:** the code below is illustrative only — `main.py` does not currently
> import or call `dl_daily_features`.  The integration point is documented
> here to show where and how it would be wired in when ready.

```python
# In main.py (future, when DL daily features are enabled):
from src.dl_daily_features import load_and_aggregate_d1
from src.dl_config import (
    DL_SIGNALS_ENABLED, DL_SIGNAL_SURFACE,
    resolve_dl_prediction_artifact_path,
)

if DL_SIGNALS_ENABLED:
    artifact_path = resolve_dl_prediction_artifact_path()
    if artifact_path is not None:
        daily_df = load_and_aggregate_d1(artifact_path, DL_SIGNAL_SURFACE)

        # d1_df is the D1 DataFrame keyed by (pair, timestamp) where
        # timestamp is the start-of-day D1 prediction timestamp.
        d1_df = d1_df.merge(
            daily_df.rename(columns={"trading_day": "timestamp"}),
            on=["pair", "timestamp"],
            how="left",
        )
        # dl_signal_mean_24h … dl_signal_flip_count are now optional columns
        # in d1_df and will be picked up by PhaseMLExperiment.get_feature_columns()
        # and PhaseMLPredictor._get_feature_cols() when DL_SIGNALS_ENABLED=True.
```
