# DL-enabled artifact bias audit

## Scope

Audited artifacts:

- `results_ml__dl_enabled.csv`
- `results_ml_backtest__dl_enabled.csv`
- `results_per_pair__dl_enabled.csv`
- `results_summary__dl_enabled.csv`
- `results_majors__dl_enabled.csv`
- `results_minors__dl_enabled.csv`

`__dl_enabled` suffix is produced by `dl_mode_tag = "__dl_enabled"` when DL runtime is enabled (`/tmp/workspace/jalmqvist/market-phase-ml/main.py:1512`) and appended by `_with_mode_tag(...)` (`/tmp/workspace/jalmqvist/market-phase-ml/main.py:555-561`).

---

## 1) Exact CSV write sites

- `results_ml__dl_enabled.csv`  
  `ml_combined.to_csv(ml_results_path, index=False)` at `/tmp/workspace/jalmqvist/market-phase-ml/main.py:1921` (path from `results/results_ml.csv` + mode tag at `main.py:1920`).

- `results_ml_backtest__dl_enabled.csv`  
  `ml_df.to_csv(ml_backtest_path, index=False)` at `/tmp/workspace/jalmqvist/market-phase-ml/main.py:2111` (path from `results/results_ml_backtest.csv` + mode tag at `main.py:2110`).

- `results_per_pair__dl_enabled.csv`  
  `per_pair_df.to_csv(per_pair_path, index=False)` at `/tmp/workspace/jalmqvist/market-phase-ml/main.py:1076`.

- `results_majors__dl_enabled.csv`  
  `majors_summary.to_csv(majors_path, index=False)` at `/tmp/workspace/jalmqvist/market-phase-ml/main.py:1082`.

- `results_minors__dl_enabled.csv`  
  `minors_summary.to_csv(minors_path, index=False)` at `/tmp/workspace/jalmqvist/market-phase-ml/main.py:1087`.

- `results_summary__dl_enabled.csv`  
  `combined_summary.to_csv(summary_path, index=False)` at `/tmp/workspace/jalmqvist/market-phase-ml/main.py:1096`.

---

## 2) Backward trace and strategy path per artifact

## A. `results_ml__dl_enabled.csv`

Source flow:

1. Per pair: `PhaseMLExperiment` runs `run_baseline`, `run_phase_features`, `run_phase_models` and stores `compare_results()` output (`/tmp/workspace/jalmqvist/market-phase-ml/main.py:1891-1903`).
2. Per-pair ML result DataFrames are concatenated into `ml_combined` (`main.py:1915-1919`).
3. `ml_combined` is written at `main.py:1921`.

Strategy object used:

- None (ML experiment object is `PhaseMLExperiment`, not trading strategy) (`/tmp/workspace/jalmqvist/market-phase-ml/src/models.py:844`, `src/models.py:1069-1101`, `src/models.py:1103-1136`, `src/models.py:1138-1206`).

Direct `generate_signals()` on TF/MR/PhaseAware/StrategySelector_Dynamic:

- No.

## B. `results_ml_backtest__dl_enabled.csv`

Source flow:

1. `run_backtests(...)` is called with ML-predicted phases (`/tmp/workspace/jalmqvist/market-phase-ml/main.py:2054-2064`).
2. Only `PhaseAware_TF4_MR42` result is extracted (`main.py:2066-2068`).
3. Rows are built into `ml_df` and written (`main.py:2090-2111`).

Strategy object used:

- Selected output strategy: `PhaseAwareStrategy("TF4", "MR42")` via `run_backtests` (`/tmp/workspace/jalmqvist/market-phase-ml/src/strategies.py:1618-1621`).

Direct `generate_signals()` on TF/MR/PhaseAware/StrategySelector_Dynamic:

- Yes on TF1–TF5 (`src/strategies.py:1578-1584`, `src/strategies.py:1596-1599`).
- Yes on MR1/MR2/MR32/MR42/MR5 (`src/strategies.py:1585-1590`, `src/strategies.py:1602-1605`).
- Yes on `PhaseAwareStrategy` (`src/strategies.py:1618-1621`).
- No on `StrategySelector_Dynamic` in this path.

## C. `results_per_pair__dl_enabled.csv`, `results_majors__dl_enabled.csv`, `results_minors__dl_enabled.csv`, `results_summary__dl_enabled.csv`

Source flow:

1. `all_pair_results` is built from hardcoded backtests (`run_backtests(..., tf_strategy_name="TF4", mr_strategy_name="MR42", use_atr_sizing=False)`) (`/tmp/workspace/jalmqvist/market-phase-ml/main.py:2147-2157`).
2. Hardcoded subset extracted into `hardcoded_results` (`main.py:2207-2221`).
3. Group summaries computed by `aggregate_backtest_results(...)` (`main.py:2301-2306`, `main.py:889-962`).
4. `save_results(hardcoded_results, majors_hardcoded, minors_hardcoded, ...)` writes the 4 files (`main.py:2338-2343`, `main.py:1045-1097`).

Strategy object used:

- `hardcoded_results` contains TF1–TF5, MR1/MR2/MR32/MR42/MR5, and `PhaseAware_TF4_MR42` from `run_backtests` (`/tmp/workspace/jalmqvist/market-phase-ml/src/strategies.py:1578-1621`).

Direct `generate_signals()` on TF/MR/PhaseAware/StrategySelector_Dynamic:

- Yes on TF1–TF5 (`src/strategies.py:1596-1599`).
- Yes on MR1/MR2/MR32/MR42/MR5 (`src/strategies.py:1602-1605`).
- Yes on `PhaseAwareStrategy` (`src/strategies.py:1618-1621`).
- No on `StrategySelector_Dynamic` for these writes (dynamic test starts later at `main.py:2357-2359`).

---

## 3) Required pre-write path checks

Checked paths:

- volatility guard
- `_usd_role()`
- USD-quote TrendFollowing override
- `no_mr` behavior
- selector confidence gating
- selector hysteresis
- selector min-hold/max-hold logic

All of these are inside `StrategySelector_Dynamic.generate_signals(...)` (`/tmp/workspace/jalmqvist/market-phase-ml/src/strategies.py:1743-1965`), including:

- confidence/margin gating (`src/strategies.py:1849-1868`)
- hysteresis (`src/strategies.py:1854-1866`)
- vol guard (`src/strategies.py:1882-1897`)
- `_usd_role()` + USD-quote override (`src/strategies.py:1734-1741`, `src/strategies.py:1886-1891`)
- `no_mr` behavior (`src/strategies.py:1896-1897`)
- min-hold/max-hold (`src/strategies.py:1901-1904`, `src/strategies.py:1917-1919`)

First `StrategySelector_Dynamic.generate_signals(...)` call in `main.py` occurs at `/tmp/workspace/jalmqvist/market-phase-ml/main.py:2412`, after:

- `results_ml` write (`main.py:1921`)
- `results_ml_backtest` write (`main.py:2111`)
- `save_results(...)` writes for per_pair/majors/minors/summary (`main.py:2338`, `main.py:1076-1096`)

So those specific selector/vol-guard/USD-role/hysteresis/min-hold/max-hold code paths do **not** execute before any of the six audited writes.

---

## 4) JPY-specific logic check

Backtest artifacts use per-pair pip values:

- `PIP_VALUES_BY_PAIRNAME` built from `PIP_VALUES` (`/tmp/workspace/jalmqvist/market-phase-ml/main.py:1614-1617`).
- `pip_value` passed into backtests at `main.py:2052` and `main.py:2140`.
- `PIP_VALUES` defines JPY pairs as `0.01` pip (`/tmp/workspace/jalmqvist/market-phase-ml/src/data.py:41`, `src/data.py:47`, `src/data.py:53`, `src/data.py:55`, `src/data.py:56`).

Therefore JPY-specific pip handling affects backtest-derived artifacts, but not the pure ML comparison artifact (`results_ml__dl_enabled.csv`).

---

## 5) Required summary table

| Artifact | Selector used? | Vol guard used? | PhaseAware used? | USD-role logic used? | JPY logic used? |
|---|---|---|---|---|---|
| `results_ml__dl_enabled.csv` | No | No | No | No | No |
| `results_ml_backtest__dl_enabled.csv` | No (`StrategySelector_Dynamic` not used) | No | Yes (`PhaseAware_TF4_MR42`) | No | Yes (pip mapping applied in backtest) |
| `results_per_pair__dl_enabled.csv` | No (`StrategySelector_Dynamic` not used) | No | Yes (includes `PhaseAware_TF4_MR42` among strategies) | No | Yes (backtest metrics from pip mapping) |
| `results_summary__dl_enabled.csv` | No (`StrategySelector_Dynamic` not used) | No | Yes (aggregates strategy set including PhaseAware) | No | Yes (aggregates backtest metrics) |
| `results_majors__dl_enabled.csv` | No (`StrategySelector_Dynamic` not used) | No | Yes (aggregated from hardcoded strategy backtests) | No | Yes (group includes JPY majors where present) |
| `results_minors__dl_enabled.csv` | No (`StrategySelector_Dynamic` not used) | No | Yes (aggregated from hardcoded strategy backtests) | No | Yes (group can include JPY minors) |

---

## 6) Earliest artifact free of selector/vol-guard/USD-role/JPY-routing bias

Earliest qualifying artifact is:

- **`results_ml__dl_enabled.csv`** (`/tmp/workspace/jalmqvist/market-phase-ml/main.py:1921`)

Reason:

- Produced before any `StrategySelector_Dynamic` execution (`main.py:2412` first call).
- Not produced by backtester or pip-value path (no `pip_value` usage in ML experiment path).
- Does not invoke `PhaseAwareStrategy` or TF/MR signal generation (`src/models.py:1069-1206`).
