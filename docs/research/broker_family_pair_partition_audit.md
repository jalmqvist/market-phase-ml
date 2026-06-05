# Audit: Origin of Persistent/Reactive Pair Partition

Date: 2026-06-05

## Root Cause

The pair partition originates in the experiment launcher’s explicit cohort definitions (`PERSISTENT_PAIRS` / `REACTIVE_PAIRS`) combined with `ACTIVE_PAIRS` filtering in `main.py`, which restricts `raw_data` before ML evaluation and CSV export.

## Findings by Audit Question

### 1) Where broker families are defined

#### Explicit cohort definitions (launcher)
- `/tmp/workspace/jalmqvist/market-phase-ml/run_v5_full_matrix.sh:14`
  - `PERSISTENT_PAIRS="EURUSD,GBPUSD,NZDUSD,EURGBP,EURAUD"`
- `/tmp/workspace/jalmqvist/market-phase-ml/run_v5_full_matrix.sh:15`
  - `REACTIVE_PAIRS="USDJPY,EURJPY,GBPJPY,EURCHF,USDCHF"`
- `/tmp/workspace/jalmqvist/market-phase-ml/run_v5_full_matrix.sh:45`
  - `export ACTIVE_PAIRS="${active_pairs}"`

#### Family semantics metadata (runtime surface inference)
- `/tmp/workspace/jalmqvist/market-phase-ml/src/experiment_surface_runtime.py:12-13`
  - `_PERSISTENT_PAIRS` / `_REACTIVE_PAIRS` constants with the same 5+5 split
- `/tmp/workspace/jalmqvist/market-phase-ml/src/experiment_surface_runtime.py:173-185`
  - `_infer_eval_family_from_active_pairs()` infers family labels from `ACTIVE_PAIRS`

#### Family transfer naming support
- `/tmp/workspace/jalmqvist/market-phase-ml/src/experiment_surface_runtime.py:19-24`
  - transfer artifact patterns (`persistent_to_reactive`, `reactive_to_persistent`)

### 2) Is pair assignment explicit?

Yes. The exact observed baskets are hard-coded:
- Launcher: `/tmp/workspace/jalmqvist/market-phase-ml/run_v5_full_matrix.sh:14-16`
- Runtime family constants: `/tmp/workspace/jalmqvist/market-phase-ml/src/experiment_surface_runtime.py:12-13`

### 3) Is pair assignment inferred?

Not for universe partitioning. The actual universe restriction is set-membership against `ACTIVE_PAIRS`:
- `/tmp/workspace/jalmqvist/market-phase-ml/main.py:141-150` parses env var to set `ACTIVE_PAIRS`
- `/tmp/workspace/jalmqvist/market-phase-ml/main.py:153-183` filters pairs via `if str(key).upper() in ACTIVE_PAIRS`

There is JPY/USD grouping logic in `main.py` (e.g., diagnostics/volatility guard metadata), but it does not build the persistent/reactive universe split:
- `/tmp/workspace/jalmqvist/market-phase-ml/main.py:319-334` (`usd_role`, `is_jpy_pair`, `major_minor_group`)
- `/tmp/workspace/jalmqvist/market-phase-ml/main.py:345-375` diagnostics usage

### 4) Trace `results_ml__dl_enabled.csv` backwards

#### Write site
- `/tmp/workspace/jalmqvist/market-phase-ml/main.py:1920-1921`
  - `ml_results_path = _with_mode_tag('results/results_ml.csv', dl_mode_tag)`
  - `ml_combined.to_csv(ml_results_path, index=False)`
- `/tmp/workspace/jalmqvist/market-phase-ml/main.py:1512`
  - `dl_mode_tag = "__dl_enabled" if dl_runtime_enabled else "__baseline"`
- `/tmp/workspace/jalmqvist/market-phase-ml/main.py:555-561`
  - `_with_mode_tag(...)` appends the mode suffix to filename

#### Dataframe feed
- `/tmp/workspace/jalmqvist/market-phase-ml/main.py:1915-1919`
  - `ml_combined = pd.concat([df.assign(Pair=pair) for pair, df in ml_results_all.items()])`
- `/tmp/workspace/jalmqvist/market-phase-ml/main.py:1886-1904`
  - `ml_results_all[pair_name] = experiment.compare_results()` for each `pair_name in sorted(processed_data.keys())`

#### Universe source and filter point
- `/tmp/workspace/jalmqvist/market-phase-ml/main.py:1731`
  - `raw_data = pipeline.run(pairs=ALL_PAIRS)` (full configured universe input)
- `/tmp/workspace/jalmqvist/market-phase-ml/main.py:1734`
  - `raw_data = filter_pair_universe(raw_data)` (**partition applied here**)
- `/tmp/workspace/jalmqvist/market-phase-ml/main.py:1844-1845`
  - `processed_data` built by iterating `sorted(raw_data.keys())`
- `/tmp/workspace/jalmqvist/market-phase-ml/main.py:1886`
  - ML loop iterates `sorted(processed_data.keys())`

#### Call chain
`run_v5_full_matrix.sh` (`ACTIVE_PAIRS` export)  
→ `main.py` module init (`ACTIVE_PAIRS_ENV` parse)  
→ `filter_pair_universe(raw_data)` at `main.py:1734`  
→ `processed_data` pair iteration (`main.py:1844+`)  
→ `ml_results_all` (`main.py:1886-1904`)  
→ `ml_combined` (`main.py:1915-1919`)  
→ `results_ml__dl_enabled.csv` write (`main.py:1920-1921`)

### 5) ACTIVE_PAIRS / equivalent filtering

Yes, broker-family style runs set `ACTIVE_PAIRS` explicitly in launcher:
- `/tmp/workspace/jalmqvist/market-phase-ml/run_v5_full_matrix.sh:45`
  - `export ACTIVE_PAIRS="${active_pairs}"`
- Called with cohort-specific pair strings for persistent and reactive runs:
  - persistent calls: `/tmp/workspace/jalmqvist/market-phase-ml/run_v5_full_matrix.sh:95-105`
  - reactive calls: `/tmp/workspace/jalmqvist/market-phase-ml/run_v5_full_matrix.sh:111-121`

## Evidence Summary (exact partition)

- Persistent cohort constants:
  - `EURUSD, GBPUSD, NZDUSD, EURGBP, EURAUD`
  - `/tmp/workspace/jalmqvist/market-phase-ml/run_v5_full_matrix.sh:14`
  - `/tmp/workspace/jalmqvist/market-phase-ml/src/experiment_surface_runtime.py:12`

- Reactive cohort constants:
  - `USDJPY, EURJPY, GBPJPY, EURCHF, USDCHF`
  - `/tmp/workspace/jalmqvist/market-phase-ml/run_v5_full_matrix.sh:15`
  - `/tmp/workspace/jalmqvist/market-phase-ml/src/experiment_surface_runtime.py:13`

## Impact

Because filtering happens before feature/ML/backtest/walk-forward loops, all downstream artifacts inherit the partition for that run:

- `results_ml__dl_enabled.csv`
  - generated from filtered `processed_data` universe (`main.py:1886`, `1915-1921`)
- `results_ml_backtest__dl_enabled.csv`
  - generated from ML backtest loop over filtered pairs (`main.py:1955`, `2110-2111`)
- `results_per_pair__dl_enabled.csv`
  - `save_results(...)` receives `hardcoded_results` built from filtered loops (`main.py:2138`, `2338-2343`, `1045-1077`)
- `selector_state_timeline__dl_enabled.csv`
  - timeline rows are populated inside walk-forward pair/fold loops over filtered `processed_data` and exported (`main.py:2595`, `3026-3028`)
- Family topology analyses
  - inherit partition indirectly because they consume run outputs/manifests produced from filtered universes.

## Confidence

**High**

Why:
1. The exact 5+5 baskets are explicitly hard-coded in launcher/runtime constants.
2. `ACTIVE_PAIRS` is explicitly exported by launcher and consumed by `main.py` pair filtering before ML and backtest outputs.
3. The write path for `results_ml__dl_enabled.csv` is directly downstream of loops over filtered `processed_data.keys()`.

## Note on `experiments/broker_family_regimes/...`

The referenced `experiments/broker_family_regimes/...` paths were not present in this repository clone, but the same partition mechanism is clearly implemented in the checked-in launcher/runtime code above.
