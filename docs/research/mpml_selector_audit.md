# MPML Selector Audit

Focused audit of the MPML selector stack covering feature-schema integrity, runtime routing behavior, and diagnostics surfaces.

## Scope audited

- `src/models.py`
  - `validate_feature_schema()`
  - `apply_optional_feature_imputation()`
  - `build_training_matrix()`
  - `StrategySelector.train()`
  - `StrategySelector._prepare_inference_matrix()`
  - `StrategySelector.predict_proba()`
- `src/strategies.py`
  - `StrategySelector_Dynamic.generate_signals()`
- `main.py`
  - `missing_indicators_enabled` runtime wiring
  - `EXPORT_SELECTOR_STATE_TIMELINE` diagnostics export
- `tests/test_selector_schema.py`
  - schema hard-fail and awareness/blind-mode regression coverage

## Executive summary

- Selector schema handling is now fail-fast and deterministic.
- Training/inference parity is explicitly enforced by frozen `feature_schema_` and strict schema checks.
- Optional DL feature handling is robust to partial/absent coverage via deterministic imputation and optional missing indicators.
- Runtime routing logic includes confidence gating, hysteresis/min-hold controls, and volatility-spike guard overrides.
- Main residual risk remains behavioral: selector may learn DL availability timing through missing indicators in sparse-overlap regimes.

## Key findings

### 1) Hard-fail schema contract is implemented

- `validate_feature_schema()` raises `RuntimeError` on:
  - missing columns
  - extra columns
  - ordering-only mismatches
- Inference no longer relies on silent fallback when schema drift occurs.

### 2) Training/inference alignment is deterministic

- `StrategySelector.train()` freezes exact training column order into `feature_schema_` before any re-sorting.
- `_prepare_inference_matrix()` rebuilds features from non-indicator base cols, regenerates indicators deterministically, validates exact schema equality, and reindexes to `feature_schema_` before scaling.

### 3) Optional DL features are handled explicitly

- `build_training_matrix()` separates required vs optional features, drops rows only on required-feature/label invalidity, and imputes optional DL fields.
- Optional DL columns can emit `*_missing` indicators (aware mode) or skip them (blind mode) via `missing_indicators_enabled`.
- Awareness diagnostics (`[AWARENESS]`) report mode and number of indicator columns.

### 4) Leakage-sensitive DL metadata is excluded from selector features

- `_DL_LEAKAGE_GUARD_COLS` excludes `dl_regime`, `mpml_regime_equiv`, and prediction timestamp fields from model features.

### 5) Runtime selector routing behavior is policy-driven and layered

- `StrategySelector_Dynamic.generate_signals()`:
  - uses selector probabilities for class decision (`TrendFollowing`, `MeanReversion`, `PhaseAware`)
  - applies probability-margin gating (`p_margin`)
  - applies hysteresis (`tau_enter`/`tau_exit`)
  - enforces `min_hold` and optional `max_hold` resets
  - applies volatility guard with USD-quote override to force TF under spikes
- `ValueError` from selector inference (e.g., required-feature NaN) is treated as legitimate fallback; `RuntimeError` schema mismatches are intentionally not swallowed.

### 6) Selector-state observability exists for diagnostics

- `main.py` can export per-bar selector timeline (`selector_state_timeline.csv`) when `EXPORT_SELECTOR_STATE_TIMELINE=true`.
- This supports downstream conditional diagnostics in the analysis framework.

## Validation evidence

- `tests/test_selector_schema.py` verifies:
  - strict schema equality behavior
  - deterministic indicator generation
  - no sklearn feature-name mismatch warning regressions
  - aware/blind schema divergence as intended
  - no silent continuation under ordering drift

## Residual risks and watch items

1. **Availability-learning risk:** missing indicators can encode DL coverage timing in sparse regimes.
2. **Policy sensitivity:** routing behavior can materially shift with threshold/hysteresis/hold hyperparameters.
3. **Volatility guard asymmetry:** USD-quote override may produce pair-group behavior divergence that should be monitored in per-pair diagnostics.

## Recommended follow-ups

- Keep both aware and blind selector variants in comparative experiments where DL overlap is sparse.
- Track selector-state metrics (switch density, confidence collapse, transition windows) alongside aggregate uplift.
- Treat schema mismatch exceptions as integrity incidents and investigate immediately (do not downgrade to warnings).
