# MPML Leakage & Causality Audit

Canonical reference for temporal integrity, fold isolation, selector integrity, and DL integration safety in `market-phase-ml`.

## Scope audited

Primary code paths audited:

- `main.py`
  - `attach_dl_features()`
  - `generate_walkforward_folds_by_pos()`
  - `_build_causal_selector_training_data()`
  - walk-forward loop (`RUN_WALKFORWARD`)
  - tau/policy sweep loops
- `src/models.py`
  - `PhaseMLPredictor.fit_predict()`
  - `StrategyPerformanceTracker.compute_strategy_returns()`
  - `StrategySelector.train()`
  - `build_training_matrix()`
- `src/dl_surface_loader.py`
  - `load_dl_surface()`
- `src/dl_daily_features.py`
  - `compute_d1_features()`
  - `load_and_aggregate_d1()`
- `src/phases.py`
  - `MarketPhaseDetector.calculate_atr_pct()`
  - `MarketPhaseDetector.classify_volatility()`
  - `MarketPhaseDetector.detect_phases()`
- `src/data.py`
  - `MarketDataPipeline.prepare()`
- `src/cache.py`
  - `save_cache()`
  - `load_cache()`

## Executive summary

- **Confirmed safe after audit/fixes**
  - DL surface loading and D1 aggregation are causal by construction.
  - Walk-forward fold boundaries now fail fast on overlap.
  - Phase predictor retraining now asserts `train_end_ts < inference_ts`.
  - Selector fold training was tightened to use **train-fold-contained** label horizons only.
  - Non-causal selector equity `bfill()` was removed.
  - DL merge diagnostics now assert causal prediction timestamps and report merge hit-rate / lag.

- **Confirmed suspicious before fix**
  - Walk-forward selector training in `main.py` previously built training labels from `train_end_pos + LABEL_HORIZON_BARS`, which allowed strategy-ranking labels to extend into the future test fold.
  - `StrategyPerformanceTracker.compute_strategy_returns()` previously used `.ffill().bfill()`, which could back-propagate future equity into earlier rows when alignment gaps existed.

- **Still unresolved / monitor closely**
  - Sparse-DL missingness indicators created by `apply_optional_feature_imputation()` can encode DL availability timing.
  - In-sample selector training/backtests (`main.py` sections `4b/4c/4d/4e`) are not scientific OOS evidence.
  - Cache keys are caller-defined; integrity depends on every caller encoding all relevant runtime knobs.

## Confirmed-safe components

### 1) Walk-forward fold generation

**Files/functions**

- `main.py: generate_walkforward_folds_by_pos()`
- `main.py: _window_diagnostics()`

**Reasoning**

- Fold boundaries are date-derived, then snapped to bar positions.
- The audit added fail-fast window validation:
  - `train_start_ts <= train_end_ts`
  - `test_start_ts <= test_end_ts`
  - `train_end_ts < test_start_ts`
- Each fold now records `gap_days` and `overlap_days`; overlap must remain zero.

### 2) Phase predictor walk-forward retraining

**Files/functions**

- `src/models.py: PhaseMLPredictor.fit_predict()`

**Reasoning**

- Training slice is `X.iloc[train_start:train_end]`; inference is at bar `i`; target is bar `i+1`.
- Audit instrumentation now logs:
  - train start / end timestamps
  - inference timestamp
  - target timestamp
- New assertions enforce:
  - `train_end_ts < inference_ts`
  - `inference_ts <= target_ts`
- Existing DL coverage diagnostics remain active per retrain window.

### 3) DL artifact loading and D1 aggregation

**Files/functions**

- `src/dl_surface_loader.py: load_dl_surface()`
- `src/dl_daily_features.py: compute_d1_features()`

**Reasoning**

- `load_dl_surface()` exact-matches one surface and validates:
  - required schema
  - hourly alignment
  - duplicate `(pair, entry_time)` rejection
  - monotone timestamps
  - numeric range checks
- `compute_d1_features()` maps H1 rows from day `D-1` onto trading day `D`, so D1 features only become visible on the next day.
- No `merge_asof`, nearest join, centered window, or backfill is used in the DL ingestion path.

### 4) Regime logic

**Files/functions**

- `src/phases.py: MarketPhaseDetector.classify_volatility()`
- `src/phases.py: MarketPhaseDetector.detect_phases()`

**Reasoning**

- Volatility split uses trailing `rolling(...).median()` with no centered window.
- Trend classification uses current-bar ADX only.
- No future-looking smoothing or centered volatility logic was found in phase detection.

## Confirmed suspicious components

### 1) Selector fold-label leakage — **fixed**

**Previous code path**

- `main.py` walk-forward / tau-sweep / policy-sweep loops
- `StrategyPerformanceTracker.compute_strategy_returns()`

**Leakage mechanism**

- The prior fold code built `df_for_labels` up to `train_end_pos + LABEL_HORIZON_BARS`.
- `compute_strategy_returns()` labels each training row using future strategy return over the next `window_days`.
- That meant late train-fold rows could rank strategies using equity moves from the future test fold.

**Fix**

- Added `main.py: _build_causal_selector_training_data()`.
- Selector training data is now built only from bars whose full ranking horizon is contained inside the train fold.
- Effective selector training end is now:
  - `selector_train_end_pos = train_end_pos - LABEL_HORIZON_BARS`
- Diagnostics now print:
  - raw fold train/eval windows
  - effective selector training window
  - ranking timestamp range

### 2) Selector equity backward fill — **fixed**

**Previous code path**

- `src/models.py: StrategyPerformanceTracker.compute_strategy_returns()`

**Leakage mechanism**

- `.ffill().bfill()` on aligned equity curves could use future equity to populate earlier missing rows.

**Fix**

- Removed `.bfill()`.
- Added causal contract comment.
- Added fail-fast assertion if leading NaNs remain after causal `ffill()`.

### 3) DL timestamp visibility — **hardened**

**Code path**

- `main.py: attach_dl_features()`

**Reasoning**

- If upstream parquet includes `dl_prediction_timestamp`, MPML now asserts:
  - `dl_prediction_timestamp <= timestamp`
- Added H1 prediction lag diagnostics.
- Added daily merge hit-rate and `dl_merge_lag_days` diagnostics.
- Added non-causal merge assertion if merged DL feature age is `< 1 day`.

## Sparse DL coverage audit

**Files/functions**

- `src/models.py: build_training_matrix()`
- `src/models.py: apply_optional_feature_imputation()`

**Findings**

- Optional DL features are imputed to `0.0`.
- Missing-indicator columns are added for optional DL features.
- Existing diagnostics already report:
  - `dl_coverage_pct`
  - `effective_training_samples`
  - per-column missingness
  - optional-collapse warnings

**Risk status**

- **Unresolved risk:** missing-indicator columns can let models learn *DL availability timing* rather than behavioral signal content.
- This is especially important because actual DL overlap is sparse.
- No hard proof of leakage was found, but this remains one of the highest-priority residual research risks.

## Selector integrity audit

**Files/functions**

- `src/models.py: StrategySelector.train()`
- `src/strategies.py: StrategySelector_Dynamic.generate_signals()`
- `main.py: _build_causal_selector_training_data()`

**Findings**

- `StrategySelector.train()` now logs the exact training date range used.
- Walk-forward/tau/policy folds now log selector train/eval windows explicitly.
- `StrategySelector_Dynamic.generate_signals()` uses only the current-row feature slice during inference.
- `reindex(columns=selector.feature_cols)` remains schema-tolerant for sparse DL cases.

**Residual concern**

- In-sample selector training/backtesting in `main.py` sections `4b`, `4c`, `4d`, `4e` still exists and is useful for debugging, but should not be treated as causal OOS evidence.

## Cached state / reuse audit

**Files/functions**

- `src/cache.py`
- callers in `main.py`

**Findings**

- `main.py` currently executes `clear_cache()` at module import time, so cache reuse is effectively disabled unless that line is removed.
- `processed_data` cache key includes:
  - raw data hash
  - detector params
  - DL enabled/disabled
  - DL surface
  - DL artifact path
- ML phase prediction cache key includes predictor params including `min_dl_coverage_pct`.
- Walk-forward selector state is rebuilt per fold; no per-fold selector cache was found.
- Fresh TF/MR strategy dicts are created via `_make_strategy_dicts()` per fold to reduce state reuse.

**Residual concern**

- `src/cache.py` itself is generic and does not enforce semantic completeness of cache keys.
- `main.py` import-time `clear_cache()` is a strong side effect; it reduces stale-cache leakage risk but also means cache behavior is not purely configuration-driven.
- Any future cached selector/fold artifact must include pair universe, DL settings, selector config, and label horizon explicitly.

## Future-sensitive grep audit

Search scope: all `**/*.py`.

### Findings

- `shift(-1)`
  - `src/data.py` — `next_return` target construction; acceptable as label generation only.
  - `src/models.py` — phase predictor target/evaluation; acceptable when confined to labels.
- `center=True`
  - no matches.
- `bfill()`
  - no matches in Python code after audit fix.
- `expanding()`
  - `src/strategies.py`, `src/visualization.py` for rolling max / drawdown; analytics only, not selector feature construction.
- `qcut()`
  - no matches.
- `merge_asof` / `nearest`
  - no matches.

## Fail-fast invariants added

- `main.py: _window_diagnostics()`
  - `assert train_end_ts < test_start_ts`
- `main.py: _build_causal_selector_training_data()`
  - selector effective train end must be before test start
  - ranking end must not exceed selector effective train end
- `main.py: attach_dl_features()`
  - `dl_prediction_timestamp <= timestamp`
  - merged DL feature age must be at least one day
- `src/models.py: PhaseMLPredictor.fit_predict()`
  - `train_end_ts < inference_ts`
  - `inference_ts <= target_ts`
- `src/models.py: StrategyPerformanceTracker.compute_strategy_returns()`
  - no leading NaNs after causal equity alignment
- `src/models.py: StrategySelector.train()`
  - training dates must be monotone increasing

## Diagnostics added

- Fold train/test timestamp ranges and gap/overlap diagnostics
- Selector train/eval window diagnostics
- Selector ranking timestamp-range diagnostics
- H1 DL prediction lag diagnostics
- DL merge hit-rate diagnostics
- DL merge lag distribution diagnostics
- Phase predictor retrain-window timestamp diagnostics

## Remaining TODO items

1. Add fold-level CSV fields for selector ranking window start/end if downstream reporting needs them, not just logs.
2. Evaluate whether optional DL missing-indicator columns should be disabled, bucketed, or separately ablated for scientific runs.
3. Add dedicated regression tests for:
   - selector train-fold containment
   - non-causal DL prediction timestamps
   - non-causal merge lag
   - fold overlap assertions
4. Treat in-sample selector comparisons as debugging only in downstream writeups.
5. If any future fold/selector caching is introduced, require cache-key completeness assertions at write/read time.
