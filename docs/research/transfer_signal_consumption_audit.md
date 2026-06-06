# Audit: Transfer Signal Consumption (`persistent_to_reactive_HVR_sentiment_aware`)

Date: 2026-06-06

## Scope and evidence status

I could verify the runtime wiring end-to-end in code, but I could not find the concrete run outputs for `persistent_to_reactive_HVR_sentiment_aware` in this clone (no matching `run_manifest_*.json`, parquet, or `selector_state_timeline__dl_enabled.csv` under `/tmp/workspace/jalmqvist/market-phase-ml`).

---

## 1) Transfer Surface Attachment

### Artifact parquet → DL attachment

- DL artifact path is resolved from `DL_PREDICTION_ARTIFACT_PATH` and passed into runtime surface construction and DL join path (`/tmp/workspace/jalmqvist/market-phase-ml/main.py:1463-1553`, `:563-849`).
- Attachment uses `attach_dl_features()` and joins D1 DL features per `(pair, timestamp)` (`main.py:563-713`).

### DL feature count and names

- Canonical transferred DL feature set is fixed at **5** columns (`/tmp/workspace/jalmqvist/market-phase-ml/src/dl_daily_features.py:71-77`):
  - `dl_signal_mean_24h`
  - `dl_signal_std_24h`
  - `dl_signal_last`
  - `dl_signal_abs_mean`
  - `dl_signal_flip_count`
- Runtime writes `manifest["dl"]["dl_feature_count"]` and `manifest["dl"]["dl_feature_columns"]` after processing (`main.py:1838-1842`).

### Attachment row counts

- Attachment row evidence is emitted in logs per pair:
  - `rows_with_any_dl=<n>/<N>` (`main.py:714-724`)
  - `dl_merge_hit_rate_pct=<pct>` (`main.py:726-728`)
- For this specific run ID, these concrete counts are **not available in-repo** (log/run folder absent).

---

## 2) Transfer Signal Reachability

### Phase predictor feature matrix

- Phase predictor auto-includes numeric `dl_*` columns when DL is enabled (`/tmp/workspace/jalmqvist/market-phase-ml/src/models.py:414-427`, `:447-460`).
- Optional DL columns are passed into `build_training_matrix()` with imputation/missing-indicator handling (`models.py:246-351`, `:1081-1129`).

### Selector feature matrix

- Selector base features are `REQUIRED_FEATURE_COLS` plus available DL D1 columns (`models.py:28-36`, `:1586-1614`).
- DL leakage guard excludes metadata/timestamp leakage fields (`models.py:40-47`, `:1591-1595`).

### Runtime inference

- Runtime routing calls `selector.predict_proba(df.loc[df.index[[i]]])` bar-by-bar (`/tmp/workspace/jalmqvist/market-phase-ml/src/strategies.py:1819-1834`).
- Selector inference rebuilds feature matrix including optional DL columns and validates schema equality (`models.py:1713-1761`).

---

## 3) Selector Timeline Provenance (`selector_state_timeline__dl_enabled.csv`)

- `selected_strategy` is produced by `StrategySelector_Dynamic.generate_signals()` as `current_type` and returned as `selected_s` (`/tmp/workspace/jalmqvist/market-phase-ml/src/strategies.py:1928-1964`).
- In walk-forward, timeline rows are built from `selected_s` and written to `results/selector_state_timeline.csv` (+ `__dl_enabled` mode tag) (`/tmp/workspace/jalmqvist/market-phase-ml/main.py:2708-2710`, `:2863-2883`, `:3026-3029`).
- At this stage, transferred DL features are available in `df_test` and explicitly checked via `_dl_cols_fold` / `dl_available` (`main.py:2826-2839`, `:2867-2879`).
- Therefore selector decisions **can depend on transferred DL signals** (via `predict_proba` input features), subject to per-fold DL overlap.

---

## 4) Transfer Semantics (requested run)

For `persistent_to_reactive_HVR_sentiment_aware`, runtime semantics resolve as:

- `training_pair_family`: **persistent**
- `evaluation_pair_family`: **reactive**
- `artifact source family`: inferred from artifact naming pattern `persistent_to_reactive*`
- `evaluation cohort`: inferred from `ACTIVE_PAIRS` membership (reactive basket)

Evidence:

- Transfer artifact pattern mapping (`/tmp/workspace/jalmqvist/market-phase-ml/src/experiment_surface_runtime.py:19-24`, `:152-170`, `:345-360`)
- Explicit test coverage for `persistent_to_reactive_*_HVR` mapping (`/tmp/workspace/jalmqvist/market-phase-ml/tests/test_runtime_experiment_surface.py:232-262`)
- Cohort constants and `ACTIVE_PAIRS` family inference (`experiment_surface_runtime.py:12-13`, `:173-185`)

---

## 5) Topology Interpretation

`persistent_to_reactive ≈ reactive baseline` can happen from two mechanisms:

- **A) Target retraining effects (strong):** MPML retrains phase/selector models on target-universe data every run/fold (`main.py:1900-1904`, `:2649-2657`; `models.py:435+`, `:1556+`).
- **B) Actual DL transfer effects (conditional):** transferred DL can influence features and routing, but only where overlap exists (`main.py:714-735`, `:2826-2840`, `:2877`).

Given missing concrete run artifacts in this clone, the most defensible interpretation is:

- Do **not** treat small `distance(transfer, target)` as adaptation proof by itself.
- Require overlap-conditioned evidence (e.g., timeline/fold deltas by `dl_overlap_state`) to separate A vs B.

**Confidence:** **Medium** (high confidence in wiring/semantics, medium confidence in this run’s realized DL effect without its logs/artifacts).

---

## Deliverable: Transfer Semantics (one paragraph)

A `persistent_to_reactive_HVR_sentiment_aware` run measures how an MSML DL surface trained on the persistent family is consumed inside MPML while MPML evaluates on the reactive target cohort.  
The observed outcome mixes two effects: transferred DL signal availability/content from the source artifact, and target-universe retraining dynamics from MPML’s fold-local model fitting.  
Because of that mixture, causal transfer claims should rely on overlap-conditioned evidence, not aggregate similarity to the target baseline alone.

## Deliverable: Transfer Signal Data Flow

MSML artifact parquet  
→ `attach_dl_features()` D1 join (`main.py`)  
→ MPML feature matrices via `build_training_matrix()` (`src/models.py`)  
→ selector training (`StrategySelector.train`)  
→ selector timeline export (`selected_s` → `selector_state_timeline__dl_enabled.csv`)

## Deliverable: Interpretation Guidance

`distance(transfer, target)` is only weak evidence of adaptation unless accompanied by evidence that DL-active bars/folds changed selector behavior relative to DL-missing bars/folds under the same target retraining setup.
