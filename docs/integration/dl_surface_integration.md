# DL surface integration contract (MPML consumer, schema v2)

`market-phase-ml` treats the DL parquet artifact as a validated API boundary.
Artifacts are validated before any DL data is accepted or attached.

## Centralized schema constants

MPML uses `schemas/dl_artifact_schema.py` as the single source of truth:

- `DL_SCHEMA_VERSION = "2.0.0"`
- `DL_TIMESTAMP_COL = "timestamp"`
- `DL_AVAILABLE_TS_COL = "prediction_available_timestamp"`
- `DL_GENERATED_TS_COL = "prediction_generated_timestamp"`
- `DL_ARTIFACT_CREATED_COL = "artifact_created_timestamp"`
- `DL_PAIR_COL = "pair"`

## Causal semantics (strict)

MPML enforces causality with:

- `prediction_available_timestamp <= timestamp`

Only `prediction_available_timestamp` is used for causal validation.
`prediction_generated_timestamp` and `artifact_created_timestamp` are **not**
used for causality checks.

## Validation layer

`src/dl_surface_loader.py` implements `validate_dl_artifact(df, metadata)` and
runs it before surface filtering or joins.

Validation checks:

- required columns present
- schema version present and compatible (v2 major)
- pair normalization (`xxx-yyy`)
- no duplicate `(pair, timestamp)`
- monotonic timestamp ordering within pair/surface
- timezone consistency across timestamp fields
- causal ordering (`prediction_available_timestamp <= timestamp`)
- null rejection in required columns

Invalid artifacts raise `ValueError` (fail fast).

## Attachment behavior

In `main.py::attach_dl_features`:

- DL data is loaded through `load_and_aggregate_d1(..., strict=True)` so
  artifact contract violations fail loudly.
- D1 features continue to join on MPML bar timestamp.
- If an artifact is valid but yields no timestamp overlap or no per-pair
  matches, MPML keeps baseline behavior (graceful no-coverage fallback).

## Analysing DL-enabled runs

Use the analysis framework v2 to inspect DL coverage and performance
across runs:

```bash
python analysis/pipeline.py results_archive/
```

The generated `report.md` includes:

- DL coverage per pair (from `vol_guard_diagnostics` or log fallback)
- Sentiment ON vs OFF walkforward deltas (DL-enabled vs baseline)
- Selector uplift: does DL-gated routing improve OOS Sharpe?

See [`docs/research/analysis_framework_v2.md`](../research/analysis_framework_v2.md) for full documentation.
