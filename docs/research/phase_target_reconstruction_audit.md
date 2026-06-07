# MPML Audit: Phase→Target Reconstruction Without Family Runs

## Executive Summary

**Yes — the `pair, timestamp, phase, next_direction_binary` dataset can be reconstructed directly from source code + broker CSV input data, without rerunning the 16 family experiments and without DL parquet artifacts.**

Reason:
- `next_direction_binary` is created in `MarketDataPipeline.prepare(...)` (`src/data.py:240-289`).
- `phase` is created in `MarketPhaseDetector.detect_phases(...)` (`src/phases.py:169-220`).
- Both are created before optional DL attachment (`main.py:1793-1800`).

## Pipeline Diagram

```text
main.py::__main__ → main(...)
  ↓
MarketDataPipeline.run(...)                          (src/data.py:349-450)
  ├─ download(...)                                   (src/data.py:187-238)
  ├─ prepare(...)  -> next_direction_binary created  (src/data.py:240-289)
  └─ engineer(...)                                   (src/data.py:291-347)
  ↓
process_pair(...)                                    (main.py:851-886)
  ├─ detector.detect_phases(...) -> phase created    (main.py:867, src/phases.py:169-220)
  ├─ engineer_features(...)                          (main.py:488-531, called at 870)
  └─ dropna                                          (main.py:873)
  ↓
optional attach_dl_features(...)                     (main.py:1793-1800, fn at 563-849)
  ↓
PhaseMLExperiment.run_* training                     (main.py:1891-1903, src/models.py:1069+)
```

---

## 1) Pipeline Ordering (call chain + file/function names)

1. Entry: `main.py` `if __name__ == '__main__'` → `main(...)` (`main.py:3546-3589`)
2. Load/prepare base data: `pipeline.run(pairs=ALL_PAIRS)` (`main.py:1731`)
   - `MarketDataPipeline.download(...)` (`src/data.py:187-238`)
   - `MarketDataPipeline.prepare(...)` (`src/data.py:240-289`)
     - creates `next_return`, `next_direction`, `next_direction_binary` (`src/data.py:280-284`)
   - `MarketDataPipeline.engineer(...)` (`src/data.py:291-347`)
3. Per pair processing: `process_pair(...)` (`main.py:851-886`)
   - `detector.detect_phases(df)` (`main.py:867`; implementation `src/phases.py:169-220`)
   - `engineer_features(df)` (`main.py:870`; function `main.py:488-531`)
4. Optional DL join: `attach_dl_features(...)` (`main.py:1793-1800`; function `main.py:563-849`)
5. Model training: `PhaseMLExperiment.run_baseline/run_phase_features/run_phase_models` (`main.py:1891-1903`; methods in `src/models.py:1069+`)

**Conclusion for Q1:** `phase` and `next_direction_binary` both exist before DL attachment.

---

## 2) DL Dependency Audit

### Influence point A — `DL_SIGNALS_ENABLED`

- Definition: (`main.py:109`); runtime flag copied into `dl_runtime_enabled` (`main.py:1454`)
- Controls whether `attach_dl_features(...)` is called (`main.py:1793`)
- Also controls feature-column gating in modeling (`src/models.py:963-964`, `main.py:1993`)

Effect on required fields:
- `phase`: **No**
- `next_direction_binary`: **No**
- `timestamps`: **No** (no timestamp rewrite by flag itself)
- `pair identity`: **No**

### Influence point B — `attach_dl_features(...)`

- Called only when DL runtime enabled (`main.py:1793-1800`)
- Left-joins DL D1 features by `(pair, timestamp)` (`main.py:708-713`)
- Keeps row count invariant (`main.py:737-741`)
- Rebuilds original index and returns frame (`main.py:847-849`)
- Adds/removes only DL feature columns (`D1_FEATURE_COLS`), with fallback to unchanged frame on no overlap / no artifact (`main.py:586-611`, `702-707`, `730-735`, `776-781`)

Effect on required fields:
- `phase`: **No direct change**
- `next_direction_binary`: **No direct change**
- `timestamps`: **No semantic change** (temporary normalization for join key, restored original index)
- `pair identity`: **No semantic change** (temporary join key column dropped before return)

### Influence point C — `DL_PREDICTION_ARTIFACT_PATH`

- Read from env (`main.py:1463`) and used to resolve artifact path (`main.py:1467-1475`)
- Also defined in config module (`src/dl_config.py:47-85`)
- Determines which DL parquet is read by `attach_dl_features(...)` via `load_and_aggregate_d1(...)` (`main.py:604-608`)

Effect on required fields:
- `phase`: **No**
- `next_direction_binary`: **No**
- `timestamps`: **No** (only DL join coverage/values vary)
- `pair identity`: **No**

---

## 3) Reconstruction Feasibility

**Answer: YES.**

Justification:
- `next_direction_binary` derives from broker-price-derived returns in `prepare(...)` (`src/data.py:280-284`).
- `phase` derives from OHLCV via `detect_phases(...)` (`src/phases.py:169-220`).
- Both are computed before optional DL join (`main.py:1791-1794`).
- Required output fields are available without parquet artifacts.

---

## 4) Minimal Reproduction Path

Shortest code path from source:

1. **Load broker CSV + aggregate D1 OHLCV**
   - `BrokerCSVLoader.load(...)`  
   - File: `src/data_sources/broker_csv_loader.py:40-89`
2. **Create target**
   - `MarketDataPipeline.prepare(...)`  
   - File: `src/data.py:240-289`
3. **Create phase**
   - `MarketPhaseDetector.detect_phases(...)`  
   - File: `src/phases.py:169-220`
4. **Export columns**
   - `pair` (from pair loop key / filename mapping),
   - `timestamp` (DataFrame index),
   - `phase`,
   - `next_direction_binary`

No `attach_dl_features(...)` call is required.

---

## 5) Family Dependence Audit

`phase` and `next_direction_binary` are **not** derived from:
- persistent/reactive family assignment,
- transfer artifacts,
- family parquet files.

They are derived from market OHLCV data only:
- `next_direction_binary` from `Close` returns (`src/data.py:274-284`)
- `phase` from `Open/High/Low/Close/Volume` (`src/phases.py:174-220`)

Family constructs are configured at experiment/run orchestration level (for example `ACTIVE_PAIRS` and run matrix scripting in `run_v5_full_matrix.sh`) and do not define target/phase formulas.

---

## 6) Research Recommendation

Preferred path: **Option B — write a lightweight reconstruction script**.

Why:
- Required dataset fields are upstream of DL and model training.
- Re-running 16 family experiments is unnecessary for this specific `phase → next_direction_binary` analysis.
- Lightweight reconstruction is faster, cheaper, and directly aligned with the audit question.

## Final Recommendation

**Proceed with lightweight reconstruction.**
