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

## Runtime `experiment_surface` canonical emission

`src/experiment_surface_runtime.py::build_runtime_experiment_surface(...)`
must emit canonical semantics directly in `run_manifest.json`:

- `sentiment_surface`
  - `price_trend` → `sentiment`
  - `trend_vol_only` → `no_sentiment`
  - `dl_enabled=false` → `none`
- `training_pair_family`
  - inferred from explicit metadata first
  - otherwise inferred from artifact provenance (e.g. `persistent_*`, `reactive_*`)
- `evaluation_pair_family`
  - inferred from explicit metadata first
  - otherwise inferred from runtime `ACTIVE_PAIRS` cohort membership
- `imputation_awareness`
  - `missing_indicators_enabled=false` → `blind`
  - `missing_indicators_enabled=true` → `aware`

Canonical V5 analysis assumes these fields are present and already resolved at
runtime emission (no post-hoc semantic repair).

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

## Behavioral-conditional strategy performance

### Overview

When behavioral/DL runtime is enabled (`dl_runtime_enabled=True`), MPML
annotates each standalone strategy trade with a `behavioral_eligible` flag
and generates a dedicated behavioral-conditional performance artifact:

    results/strategy_behavioral_performance__dl_enabled.csv

This artifact is **additive** — it does not replace or modify any existing
walk-forward or strategy artifact.

### Semantic distinction

| Concept | Definition |
|---|---|
| **Unconditional strategy performance** | Complete strategy execution on the full fold population. Reported in `walkforward_results_*.csv`. |
| **Behavioral-conditional performance** | Subset of trades whose **ENTRY observation** was active for the selected behavioral surface/state. Reported in `strategy_behavioral_performance__dl_enabled.csv`. |

### Attribution semantics

A trade is *behaviorally eligible* (`behavioral_eligible == True`) when its
**entry observation** is state-active for the selected behavioral
surface/state.  State-active means at least one `D1_FEATURE_COLS` value is
non-null for that bar (the established MPML DL coverage mask).

An eligible trade **retains its complete realized lifecycle and P&L**, even
if its exit occurs after the behavioral state becomes inactive.  Only the
**entry bar** determines eligibility.  Ineligible entry trades are excluded
from the conditional performance statistics.

The strategy itself is not modified.  No signals are zeroed, no positions are
force-closed at state boundaries, no trade is truncated.

### Normal vs explicit-strategy modes

The same reporting/aggregation code is used for both modes.  The schema is
identical:

| Mode | Strategy population in artifact |
|---|---|
| Normal behavioral run | All strategies produced by normal strategy evaluation |
| `--strategy TF1` | TF1 only |
| `--strategy TF1 --strategy TF4` | TF1 and TF4 only |

### Artifact generation rule

    behavioral/DL runtime enabled → generate behavioral-performance artifact
    behavioral/DL runtime disabled → no behavioral-performance artifact

The `--strategy` flag does **not** determine whether the artifact is generated.

### Artifact schema (`strategy_behavioral_performance__dl_enabled.csv`)

| Column | Type | Description |
|---|---|---|
| `behavioral_surface_id` | str | Active behavioral surface (e.g. `reactive_jpy`) |
| `behavioral_state_id` | str | Active behavioral state (e.g. `JPY_CONSENSUS_YOUNG`) |
| `pair` | str | Currency pair |
| `fold` | str/int | Walk-forward fold identifier |
| `strategy_id` | str | Strategy identifier (e.g. `TF1`, `MR42`) |
| `eligible_trades` | int | Trades with `behavioral_eligible == True` |
| `total_pnl` | float | Sum of P&L for eligible trades |
| `mean_trade_return` | float | Mean of `pnl_pct` for eligible trades |
| `median_trade_return` | float | Median of `pnl_pct` for eligible trades |
| `std_trade_return` | float | Sample std (ddof=1) of `pnl_pct`; NaN for 1 trade |
| `win_rate` | float | Fraction of eligible trades with `pnl_pct > 0` |
| `wins` | int | Count of eligible trades with `pnl_pct > 0` |

**Note on omitted metrics:** Conditional Sharpe ratio and maximum drawdown are
intentionally excluded.  The eligible-trade sub-population does not preserve
the temporal ordering required to derive a correct time-series drawdown, and a
naïve Sharpe computed from eligible-trade returns is not comparable to the
unconditional walk-forward Sharpe (which is computed over the equity curve).
These may be added in a later iteration once a correct definition is agreed upon.



> **Status: resolved (infrastructure fix, not a research result)**

Behavioral Surface evaluation of the existing strategy universe (e.g. `reactive_jpy`
with `TF1` or `MR5`) was previously blocked by an evaluation-scope compatibility
defect in `resolve_evaluation_scope`.

The defect caused the function to reject any valid strategy against the
`reactive_jpy` surface because no strategy declared `reactive_jpy` as a native
strategy capability (`supported_surfaces`).  This was an architectural mismatch:
strategy capability and behavioral-surface conditioning are independent concepts
and must not be conflated.

The fix separates evaluation-scope resolution from intrinsic strategy capability:

- **Strategy capability** (`supported_surfaces`) describes what a strategy is
  intrinsically implemented to execute.
- **Behavioral Surface / State** describes the market population on which the
  strategy is *evaluated*.
- **Evaluation Scope** determines which strategies participate in a particular
  experiment.

A valid registered strategy may now participate in any valid registered
Behavioral Surface experiment without needing to declare that surface as a
native capability.  Validation of unknown strategy IDs, unknown surface IDs,
and cross-surface state IDs is preserved.

This is an infrastructure correction.  It does not constitute a research result
and does not imply any conclusion about behavioral-surface-conditioned strategy
performance.

