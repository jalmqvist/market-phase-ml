# MPML Audit: Phase-as-Feature Learning Dataset Trace

## A) Pipeline Trace (`main.py` → `results_ml__dl_enabled.csv`)

1. `main.py:1439` — `main(...)` entrypoint
2. `main.py:1731` — `pipeline.run(...)` loads/prepares market data (`src/data.py`)
3. `main.py:1791` — `process_pair(pair_name, df, detector)`:
   - `main.py:867` `detector.detect_phases(df)` (`src/phases.py`)
   - `main.py:870` `engineer_features(df)` (lag/rolling ML features)
   - `main.py:873` `df = df.dropna()`
4. `main.py:1794` — optional `attach_dl_features(...)` (adds `dl_signal_*` columns when DL enabled)
5. `main.py:1887-1903` — ML experiment loop per pair:
   - `PhaseMLExperiment.run_baseline(df)`
   - `PhaseMLExperiment.run_phase_features(df)`  ← **Phase-as-Feature rows originate here**
   - `PhaseMLExperiment.run_phase_models(df, ...)`
   - `experiment.compare_results()` returns per-pair table with `Model` rows
6. `main.py:1915-1921` — concat all pair tables, then write:
   - `_with_mode_tag('results/results_ml.csv', '__dl_enabled')`
   - output file: `results_ml__dl_enabled.csv`

`Model = "Phase as Feature"` row text is set in `src/models.py:1232` (`compare_results`).

---

## B) Dataset Schema (immediately before Phase-as-Feature training)

### Source frame passed into experiment
- Variable: `df` from `processed_data[pair_name]` (`main.py:1887`)
- Metadata:
  - pair via dict key `pair_name`
  - also stored as `df.attrs["pair_name"]` (`main.py:1888`)
- Time key: dataframe index (`DatetimeIndex`, timestamp)

### Target
- Column: `next_direction_binary`
- Defined in `src/data.py:280-284` as:
  - `next_return = returns.shift(-1)`
  - `next_direction_binary = (next_return > 0).astype(int)`

### Phase-as-Feature matrix creation
- `src/models.py:1103-1133` (`run_phase_features`)
- `X = prepare_features(df, include_phase=True)` (`src/models.py:1116`)
  - Base numeric features selected by `get_feature_columns` (`src/models.py:933-967`)
  - Plus one-hot phase columns from:
    - `pd.get_dummies(df['phase'], prefix='phase')` (`src/models.py:987-990`)
- `y = df['next_direction_binary']` (`src/models.py:1117`)
- `build_training_matrix(...)` (`src/models.py:1120`, implementation at `src/models.py:246-363`) applies required-column masking and optional DL imputation.

### Last dataframe before `fit()`
- In `_cross_validate(...)` (`src/models.py:1007-1055`):
  - split matrices: `X_train`, `X_test`, `y_train`, `y_test` (`src/models.py:1031-1034`)
  - scaled arrays: `X_train_scaled`, `X_test_scaled` (`src/models.py:1038-1039`)
  - train call: `model.fit(X_train_scaled, y_train)` (`src/models.py:1043`)

### Feature columns (non-DL baseline set)
From `src/data.py`, `src/phases.py`, and `main.py:488-531`, then filtered by `EXCLUDE_COLS`:
- Core indicators: `atr`, `atr_pct`, `adx`, `plus_di`, `minus_di`, `rsi`
- Lags: `return_lag_{1,2,3,5,10}`, `adx_lag_{1,2,3,5,10}`
- Rolling stats: `return_mean_{5,10,20}`, `return_std_{5,10,20}`, `return_skew_{5,10,20}`
- Derived: `di_spread`, `di_ratio`, `returns_recent`, `volatility_recent`
- Optional DL columns (when attached):  
  `dl_signal_mean_24h`, `dl_signal_std_24h`, `dl_signal_last`, `dl_signal_abs_mean`, `dl_signal_flip_count` (`src/dl_daily_features.py:71-77`)
- Phase-as-feature only adds one-hot columns such as:  
  `phase_HV_Trend`, `phase_LV_Trend`, `phase_HV_Ranging`, `phase_LV_Ranging` (presence depends on data)

---

## C) Answers to Questions 1–7

### 1) Where is the target variable created?
`target definition = next_direction_binary = 1 if next_return > 0 else 0`

- File: `src/data.py`
- Function: `MarketDataPipeline.prepare`
- Lines: `280-284`

---

### 2) Where is the feature matrix created?
`X matrix created in: PhaseMLExperiment.prepare_features(...)`

- File: `src/models.py`
- Functions:
  - `get_feature_columns` (`933-967`) — chooses numeric non-excluded columns
  - `prepare_features` (`968-993`) — builds `X`, optionally appends phase dummies
  - `run_phase_features` (`1103-1133`) — Phase-as-Feature experiment path
  - `build_training_matrix` (`246-363`) — final training matrix shaping/imputation

---

### 3) How is phase information added (Phase-as-Feature)?
- File: `src/models.py`
- Function: `PhaseMLExperiment.prepare_features(include_phase=True)`
- Implementation: `pd.get_dummies(df['phase'], prefix='phase')` (`987-990`)

`phase feature column name = phase_<label>`  
`phase values = HV_Trend, LV_Trend, HV_Ranging, LV_Ranging` (from `src/phases.py:244-249`)  
`encoding = one-hot`

---

### 4) What data exists immediately before model training?
Immediately before `fit()`:
- `X_train` / `y_train` inside `_cross_validate` (`src/models.py:1031-1034`)
- Then scaled to `X_train_scaled` (`1038`) and passed to `fit` (`1043`)

Schema at that point:
- `X_train`: numeric matrix from `build_training_matrix`, including optional phase dummy columns and optional DL/missing-indicator columns
- `y_train`: binary target (`next_direction_binary`)
- Row count: per pair, equals `n_samples` reported in results (`compare_results` uses `len(X)` from `_cross_validate`; `src/models.py:1054`)

---

### 5) Is phase available alongside timestamp and pair?
`YES`

Before training (in `main.py` loop):
- File: `main.py`
- Function: `main`
- Dataframe: `df = processed_data[pair_name]` (`1887`)

Availability:
- `pair`: dict key `pair_name`, also `df.attrs["pair_name"]` (`1888`)
- `timestamp`: dataframe index (`DatetimeIndex`)
- `phase`: `df['phase']` (from `detect_phases`)
- `target`: `df['next_direction_binary']` (from `MarketDataPipeline.prepare`)

---

### 6) What distinguishes Baseline vs Phase-as-Feature?
Implementation difference in `src/models.py`:

- **Baseline** (`run_baseline`, `1069-1101`)
  - `X = prepare_features(df, include_phase=False)`
  - `y = df['next_direction_binary']`

- **Phase-as-Feature** (`run_phase_features`, `1103-1133`)
  - `X = prepare_features(df, include_phase=True)` ← adds phase one-hot columns
  - `y = df['next_direction_binary']`

So target and base dataset are the same; only `include_phase=True` changes feature columns.

---

### 7) What distinguishes Phase-as-Feature vs Separate Phase Models?
Implementation difference in `src/models.py`:

- **Phase-as-Feature** (`run_phase_features`)
  - One training flow over full dataset
  - Phase provided as one-hot features
  - Single model/CV process over all rows

- **Separate Phase Models** (`run_phase_models`, `1138-1206`)
  - Starts from same base feature construction (`include_phase=False`)
  - Splits data by `phase_mask = phases == phase` (`1176`)
  - Trains/evaluates separate model per phase subset
  - Aggregates per-phase accuracies into one summary row in `compare_results`

So: **same source dataset, then filtered into multiple phase-specific training loops** for Separate Phase Models.

---

## D) Suggested Analysis Hook (no training-logic changes)

Easiest hook:
- Location: `main.py:1887-1889` inside ML loop
- Object: `df = processed_data[pair_name]`

At this point you already have:
- `pair` (loop key / `df.attrs["pair_name"]`)
- `timestamp` (index)
- `phase` (`df['phase']`)
- `target` (`df['next_direction_binary']`)

For a standalone analysis script without code changes, use the persisted processed CSVs written at `main.py:1806-1807`:
- `data/processed/<PAIR>.csv`
- reconstruct `pair` from filename and read `timestamp`, `phase`, `next_direction_binary`.
