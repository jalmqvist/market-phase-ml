# Market Phase–Based Strategy Selection (Regime-Aware Time Series Gating)

![Python](https://img.shields.io/badge/Python-3.10+-blue)![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)![XGBoost](https://img.shields.io/badge/XGBoost-latest-green)![License](https://img.shields.io/badge/license-MIT-blue)

---

## TL;DR

A regime-aware **time-series ML pipeline** that combines:

- **Rule-based regime labeling** (market “phases”), and
- an **XGBoost gating model** (mixture-of-experts)

…to route between expert policies (**trend-following** vs **mean-reversion**) with **leakage-safe walk-forward evaluation** and reproducible CSV artifacts.

> Goal: demonstrate end-to-end ML engineering + experimental discipline on a non-stationary time series decision problem (not “promise profits”).

---

## Notebook walkthrough (recommended)

After you run `python main.py` and the run-directory artifacts are generated, you can open the notebook:

- `notebooks/01_regime_gating_walkforward.ipynb`

It is intentionally “read-only” on the artifacts: it loads the CSV outputs + selected debug exports and produces the main figures (fold distributions, per-pair breakdown, and a fold-level case stu[...]

---

## Why this project

This repo is intentionally built as an end-to-end ML engineering project for **non-stationary time series**:

- Mixture-of-experts style **policy routing** (gating model)
- A realistic **leakage-safe evaluation** design (walk-forward)
- Reproducible experimentation (CSV artifacts + caching)
- Failure-mode driven iteration (volatility guard, max-hold reset)
- Code that’s structured like a small production pipeline (data → features → labels → model → evaluation)

Trading is simply the “toy domain”; the underlying pattern generalizes to many real ML systems where the best decision policy depends on context (e.g., demand forecasting regimes, anomaly handling[...]

---

## Quickstart

```bash
pip install -r requirements.txt
python main.py --experiment-variant A
```

By default, each invocation now writes into a dedicated immutable run directory:

`results_archive/<generation>_<variant>__<timestamp>/`

Variant is the canonical runtime identity and must be selected explicitly:

```bash
# CLI overrides env when both are set
python main.py --experiment-variant A
python main.py --experiment-variant B
python main.py --experiment-variant C
python main.py --experiment-variant D
python main.py --experiment-variant E
python main.py --experiment-variant F

# env fallback
export EXPERIMENT_VARIANT=B
python main.py
```

Canonical variant matrix (source of truth: `experiment_semantics.EXPERIMENT_VARIANTS`):

| Variant | Generation | Sentiment | Missing Indicators |
|---|---|---|---|
| A | gen1 | ON | OFF |
| B | gen1 | OFF | OFF |
| C | gen2 | ON | ON |
| D | gen2 | OFF | ON |
| E | gen1 | ON | ON |
| F | gen2 | ON | OFF |

You can override the run output path explicitly:

```bash
python main.py --output-dir /absolute/path/to/my_run_dir
```

Key outputs (written inside that run directory):

- `ablation_summary_aggregate.csv` — headline in-sample ablation numbers
- `ablation_summary_per_pair.csv` — per-pair breakdown
- `walkforward_results_summary.csv` — walk-forward (OOS) summary vs baseline
- `walkforward_results_per_pair.csv` — walk-forward per-pair deltas
- `walkforward_results_per_fold.csv` — walk-forward deltas per fold (debuggable)
- `walkforward_tau_sweep_summary.csv` — τ sweep summary (optional)
- `walkforward_policy_sweep_summary.csv` — policy sweep summary (optional)
- `logs/run_summary.log` — deterministic runtime metadata log
- `analysis/report.md` — auto-generated analysis report for this run
- `analysis/comparisons.json` — auto-generated factor and cohort comparisons

The notebook (`notebooks/01_regime_gating_walkforward.ipynb`) reads these artifacts to generate figures and fold-level case studies.

> Note: expensive sweeps (τ/policy sweeps) are gated behind flags in `main.py`.

---

## Analysis framework (single-command, v2)

After archiving your run outputs, analyse them with one command:

```bash
python analysis/pipeline.py results/evidence/    # analyse archived run(s)
python analysis/pipeline.py results_archive/     # multi-run archive
```

This automatically:

- Discovers run directories with strict manifest-centric provenance rules
- Builds canonical run identities that include generation, variant, timestamp, and archive path context
- Parses CSV outputs, one canonical run manifest, and (as fallback) log files
- Validates provenance/semantic/manifest integrity (duplicates, malformed manifests, incomplete cohorts)
- Generates normalised summary JSON per run
- Generates sentiment/generation comparisons and generalized factor-conditioned cohorts
- Renders a unified markdown report

Outputs are written to `analysis/output/` by default:

| File | Description |
|---|---|
| `summaries/<canonical_run_id>.summary.json` | Per-run summary with stable canonical identity |
| `comparisons.json` | Cross-run comparison tables |
| `report.md` | Human-readable report with validation and diagnostics sections |

For full documentation see [`docs/research/analysis_framework_v2.md`](docs/research/analysis_framework_v2.md).

Run identity format (example):

- semantic: `gen1_A__20260521T131739Z`
- canonical (unique): `gen1_A__20260521T131739Z__fp_gen1_A`

Where:

- `gen1/gen2` = generation label from canonical semantics
- `A/B/C/D/E/F` = experiment variant semantics
- timestamp = canonical manifest timestamp (fallback: run_id timestamp, else `unknown_ts`)
- archive suffix = discovered directory identity to prevent collisions

Integrity notes:

- run discovery trusts exactly one run manifest (`run_manifest.json` preferred; legacy `run_manifest_*.json` supported)
- each run directory must contain exactly one manifest
- semantic attribution is **manifest `experiment`-driven only** — no inference from folder names, DL flags, or runtime state

Experiment metadata (factor-first, canonical):

| Field | Meaning |
|---------|---------|
| `experiment.run_family` | Comparison ontology version (`factorial_v1`) |
| `experiment.generation` | Legacy generation label (backward compatibility) |
| `experiment.variant` | Canonical variant label (`A`..`F`) |
| `experiment.factors` | Source of truth for cohort filtering and comparisons |

Each run manifest must contain an explicit `experiment` block:

```json
{
  "experiment": {
    "run_family": "factorial_v1",
    "generation": "gen1",
    "variant": "B",
    "sentiment_enabled": false,
    "missing_indicators_enabled": false,
    "factors": {
      "dl_enabled": false,
      "sentiment_enabled": false,
      "missing_indicators_enabled": false,
      "msml_regime": "LVTF",
      "overlap_only": false,
      "selector_enabled": true
    },
    "semantic_label": "Gen1_B",
    "legacy_semantics": false,
    "semantics_version": 3
  }
}
```

Each new run manifest also emits an explicit canonical `experiment_surface` block
from runtime artifact introspection (parquet metadata and artifact sidecar, with
configuration fallbacks), for parquet/training attribution in analysis:

```json
{
  "experiment_surface": {
    "surface_semantics_version": 5,
    "surface_source": "artifact_introspection",
    "training_pair_family": "persistent",
    "evaluation_pair_family": "persistent",
    "sentiment_surface": false,
    "feature_surface": "trend_vol_only",
    "artifact_source": "path/to/artifact.parquet",
    "dl_enabled": true,
    "selector_enabled": true,
    "overlap_only": false,
    "msml_regime": "LVTF",
    "target_horizon": 24,
    "artifact_model": "mlp"
  }
}
```

Analysis still reports run `surface_source="manifest"` when this v5 block is present.

### Baseline no-DL runs

Use:

```bash
DL_SIGNALS_ENABLED=false python main.py --experiment-variant B
```

The run remains fully analyzable and will be grouped under `factors.dl_enabled=false`.

### Factor-conditioned comparisons

Cross-run analysis is factor-conditioned (generation/sentiment/missing-indicators) and does not assume only four variants.  
Comparisons derive from `experiment.factors` + `experiment_semantics.EXPERIMENT_VARIANTS`, not from folder names or DL flags.

### Alternative MSML regimes

MSML regime metadata is now explicit and comparison-ready:

```bash
MSML_REGIME=HTF python main.py --experiment-variant C
MSML_REGIME=LV  python main.py --experiment-variant A
```

This only updates run metadata/cohorting in this cleanup PR; strategy logic is unchanged.

Run the analysis test suite with:

```bash
python -m unittest tests/test_parsers.py tests/test_comparisons.py -v
```

---

## DL integration quickstart (contract v2)

`market-sentiment-ml` exports per-run H1 DL prediction artifacts, and `market-phase-ml` consumes one selected surface, validates the artifact contract, aggregates it to D1, and feeds those numeric features into the existing pipeline.

```bash
# 1) In market-sentiment-ml/: export one surface artifact
python -m research.deep_learning.train \
  --dataset-version 1.3.2 \
  --pairs EURUSD \
  --regime HVTF \
  --target-horizon 24

# Output example:
# data/output/dl_predictions/mlp__HVTF__24__price_trend__20260510T182643Z.parquet
# data/output/dl_predictions/mlp__HVTF__24__price_trend__20260510T182643Z.manifest.json
```

```bash
# 2) In market-phase-ml/: point to that parquet and enable DL integration
export DL_PREDICTION_ARTIFACT_PATH=../market-sentiment-ml/data/output/dl_predictions/mlp__HVTF__24__price_trend__20260510T182643Z.parquet
export DL_SIGNALS_ENABLED=true
python main.py
```

Contract + behavior notes:

- single-surface only per run (no ensembles / multi-surface aggregation)
- explicit `dl_regime` required (`HVTF|LVTF|HVR|LVR`), no `"all"` support
- strict schema contract: MPML requires a compatible `schema_version` (v2 major), required columns, duplicate-free `(pair, timestamp)`, monotonic timestamps, and causal ordering
- timestamp semantics: causality is enforced only with `prediction_available_timestamp <= timestamp` (not `prediction_generated_timestamp` or `artifact_created_timestamp`)
- invalid artifacts fail fast; valid artifacts with no matching join coverage still fall back gracefully
- no guarantee of performance improvement; treat as an experimental feature layer

---

## Full-Pipeline DL Propagation (infrastructure capability expansion)

> This is an **infra-only** change. It does not alter strategy logic, backtester semantics, transaction costs, or signal generation.

When `DL_SIGNALS_ENABLED=true`, DL feature columns now flow through the **entire** MPML pipeline:

| Pipeline stage | DL columns now present |
|---|---|
| `attach_dl_features()` | ✓ (v1 — unchanged) |
| `PhaseMLExperiment` / `PhaseMLPredictor` | ✓ (v1 — unchanged) |
| `StrategyPerformanceTracker` | ✓ **(new in v2)** — DL columns propagated into training rows |
| `StrategySelector.train()` | ✓ **(new in v2)** — DL columns included in feature set automatically |
| `StrategySelector_Dynamic.generate_signals()` | ✓ **(new in v2)** — DL-aware diagnostics available |
| Walk-forward experiments | ✓ **(new in v2)** — DL columns preserved in fold slices |
| Ablation experiments | ✓ **(new in v2)** — same selector as walk-forward |

### Schema-tolerant feature access (v2 stabilization)

Downstream stages are now **schema-tolerant**: they safely handle DataFrames
where DL columns may be absent (e.g. pair with zero DL artifact coverage,
baseline runs, or MLP vs LSTM column differences).

Key patterns:

- `StrategySelector.predict()` / `predict_proba()` use `reindex` to fill
  missing columns with `NaN` instead of raising `KeyError`; existing `NaN`
  guards return a `PhaseAware` fallback.
- `StrategySelector_Dynamic.generate_signals()` uses the same `reindex`
  pattern when building the per-bar feature row.
- `safe_existing_columns(df, cols)` helper in `main.py` and `src/models.py`
  for any additional projection that must tolerate optional DL columns.

### Backward compatibility

When `DL_SIGNALS_ENABLED=false` (the default), the pipeline behaves **identically** to the previous baseline. No new required env vars, no breaking CLI changes.

### Startup diagnostics

After each pair's DL attachment, MPML logs the detected DL feature surface:

```
[DL FEATURE SURFACE] EURUSD: columns=['dl_signal_mean_24h', 'dl_signal_std_24h', ...] count=5
```

At key downstream stages, `[DL PIPELINE]` log lines report the active DL configuration:

```
[DL PIPELINE] strategy-selector training: dl_enabled=True dl_cols=[...] pairs=[...]
[DL PIPELINE] aggregation: dl_enabled=True pairs=[...] hardcoded_pairs=[...]
[DL PIPELINE] final export: dl_enabled=True dl_cols=[...] mode_tag=__dl_enabled
```

Enable verbose per-call selector diagnostics with `DL_DEBUG_VERBOSE=True` in `main.py`.

### Manifest

The run manifest now includes `dl_feature_columns` and `dl_feature_count` in the `dl` section.

For full details, see [`docs/dl_surface_integration.md`](docs/dl_surface_integration.md).

---

## What is a “fold” (walk-forward terminology)

A **fold** is one out-of-sample (OOS) evaluation step in a walk-forward backtest.

In each fold:
1. Train the gating model on an **expanding** historical window (e.g., the prior 7 years).
2. Test on the **next** time window (e.g., the next 6 months).
3. Advance the window and repeat.

So “361 folds” means 361 sequential train/test evaluations over time. Each fold produces a separate OOS result, which makes failure modes inspectable instead of hiding them in one aggregate number.

---

## System overview (high-level architecture)

**Pipeline:**

1. Download OHLCV (Yahoo Finance via `yfinance`)
2. Feature engineering (trend/volatility/momentum + “recent” features)
3. **Regime labeling** (rule-based market phase)
4. Backtest expert strategies (TF suite, MR suite, PhaseAware router)
5. Build supervised dataset: features → best strategy type label (horizon-based)
6. Train **StrategySelector** (XGBoost) with leakage-safe preprocessing
7. Run **StrategySelector_Dynamic** in the backtester (mixture-of-experts routing)
8. Export results to CSV + (optionally) generate figures

This is a **gating model** that decides which expert policy to execute at each time step.

---

## Key results (how to read these)

- “Dynamic − PhaseAware(TF4/MR42)” means: *out-of-sample performance difference* between the ML-gated routing and the rule-based baseline.
- In-sample ablations are primarily for debugging/intuition; the main generalization check is the walk-forward evaluation.
- Max DD (%) is stored as a **negative** number (e.g., **−30%**).
- “Max DD Δ (Dynamic − Baseline)” is therefore:
  - **positive** = Dynamic had a *less negative* drawdown → **better**
  - **negative** = Dynamic had a more negative drawdown → worse

### A) In-sample ablation (A0–A3)

Across **14 FX pairs** using ~20 years of daily data:

| Variant                                    | Description                                                  | Avg Return | Avg Sharpe | Avg Max DD | Pairs |
| ------------------------------------------ | ------------------------------------------------------------ | ---------: | ---------: | ---------: | ----: |
| A0_TF4                                     | Fixed policy (TrendFollowing only)                           |      -3.3% |     -0.047 |       -25% |    14 |
| A1_MR42                                    | Fixed policy (MeanReversion only)                            |       +20% |     +0.074 |       -42% |    14 |
| A2_PhaseAware_TF4_MR42                     | Rule-based routing using detected regimes                    |       +22% |      +0.14 |       -29% |    14 |
| A3_DynamicSelector_tau0.62_exit0.57_hold10 | **ML gating** with confidence threshold + hysteresis + min-hold |       +57% |      +0.22 |       -31% |    14 |

Artifacts:
- `results/ablation_summary_per_pair.csv`
- `results/ablation_summary_aggregate.csv`
- `results/dynamic_selector_results_per_pair.csv`
- `results/baseline_vs_dynamic_comparison.csv`

---

### B) Walk-forward evaluation (out-of-sample)

Walk-forward setup (example):
- Train: 7y (expanding)
- Test: 6m
- Step: 6m
- Horizon: 20 bars

The dynamic selector uses:
- confidence gating + hysteresis:
  - τ_enter = 0.62
  - τ_exit = 0.57
- minimum hold:
  - min_hold_bars = 10

**Headline result (14 pairs, 361 folds) vs PhaseAware(TF4/MR42):**

| Metric (Dynamic − PhaseAware TF4/MR42) |               Value |
| -------------------------------------- | ------------------: |
| Avg Return Δ                           |          **+0.19%** |
| Avg Sharpe Δ                           |          **+0.084** |
| Avg Max DD Δ                           |           **-0.17** |
| Folds with Sharpe improvement          | **192 / 361 (53%)** |

Outputs:
- `results/walkforward_results_per_fold.csv`
- `results/walkforward_results_per_pair.csv`
- `results/walkforward_results_summary.csv`

> Note: The exact numbers in this README are snapshots from the current default configuration.
> Re-run `python main.py` to reproduce the latest artifacts on your machine.

---

## Practical failure modes and mitigations (why the extra guards exist)

Mixture-of-experts gating can fail in predictable ways on non-stationary time series. This project includes two lightweight mitigations that were added after inspecting walk-forward fold failures (debug plots + per-fold CSVs).

### C) Volatility guard (leakage-safe)

**Problem:** the gating model can select mean-reversion during volatility spikes, producing large drawdowns.

**Mitigation:** a per-fold, leakage-safe volatility guard:
- Feature: ATR% (`atr_pct`)
- Threshold: per-fold training quantile `q` (computed using *only the training slice*)
- Default action on trigger: `no_mr` (block MR selections when volatility is extreme)
- Extra safety: **USD-quote override** — on spike bars, force `TrendFollowing` for USD-quote pairs

**Current best config found so far (walk-forward):**
- `VOL_GUARD_Q = 0.80`
- `VOL_GUARD_MODE = "no_mr"`
- USD-quote override: force `TrendFollowing` on spike bars

This is designed to be global + group-aware (not per-pair tuned).

---

### D) Time-based reset (max-hold)

**Problem:** with hysteresis + min-hold, the selector can get “stuck” in one non-default expert (TrendFollowing/MeanReversion) long after conditions change.

**Mitigation:** a simple time-based reset:
- `max_hold_bars`: after N consecutive bars in a non-PhaseAware state, force a reset back to `PhaseAware`.
- To avoid cutting winners / interrupting live trades, the reset is applied **only when flat** (i.e., when the executed position is 0, using the same previous-bar signal convention as the backtester).

**Current default (D1):**
- `max_hold_bars = 60`
- reset only when flat

We chose 60 bars as a conservative default after a small grid search (5–60): similar Sharpe uplift to shorter holds, with less drawdown penalty.

---

### E) Evidence for group-aware gating (instead of per-pair tuning)

Pair-specific parameter tuning can improve metrics but is intrusive and may overfit. A middle ground is **group-aware gating** based on simple market-structure categories (JPY vs non-JPY, USD role, major vs minor).

When we aggregated walk-forward deltas for the `q=0.80` volatility guard run with the USD-quote override, we observed strong group-level differences (means shown; two decimals):

#### Majors vs minors
| Group | Return Δ | Sharpe Δ | Max DD Δ |
| ----- | -------: | -------: | -------: |
| Major |   +0.011 |   +0.063 |    -0.35 |
| Minor |    +0.38 |    +0.11 |   +0.013 |

#### JPY vs non-JPY
| Group   | Return Δ | Sharpe Δ | Max DD Δ |
| ------- | -------: | -------: | -------: |
| JPY     |    +0.31 |   +0.073 |    +0.42 |
| non-JPY |    +0.15 |   +0.089 |    -0.41 |

#### USD role (base/quote)
| Group     | Return Δ | Sharpe Δ | Max DD Δ |
| --------- | -------: | -------: | -------: |
| USD-base  |    +0.16 |   +0.093 |   -0.011 |
| USD-quote |    -0.11 |   +0.040 |    -0.61 |
| No-USD    |    +0.38 |    +0.11 |   +0.013 |

Interpretation:
- JPY pairs show large drawdown improvements (tail-risk suppression) but limited Sharpe uplift.
- USD-quote majors are a challenging bucket: a group-aware volatility action (force TF on spike bars) materially improves drawdowns versus simpler guards.
- Crosses (“No-USD”) benefit strongly on average.

These findings motivate keeping a **global** threshold `q` while using **group-conditioned** guard actions rather than bespoke per-pair thresholds.

---

### F) Confidence gating: τ sweep (optional experiment)

A global τ sweep evaluates the trade-off between:
- **coverage** (how many bars the selector is “confident” enough to override PhaseAware)
- and performance

Example sweep (14 pairs, 361 folds; run33, rounded):
- τ=0.60 → Avg Sharpe Δ **+0.019**, Avg Return Δ **+0.17%**, confident bars **~52%**
- τ=0.62 → Avg Sharpe Δ **+0.072**, Avg Return Δ **+0.21%**, confident bars **~49%**
- τ=0.70 → Avg Sharpe Δ **+0.045**, Avg Return Δ **+0.14%**, confident bars **~38%**

This project’s current default uses **τ_enter=0.62** with hysteresis (τ_exit=0.57) plus a 10-bar minimum hold.

Outputs:
- `results/walkforward_tau_sweep_per_fold.csv`
- `results/walkforward_tau_sweep_summary.csv`

---

## Phase detection (rule-based regime labeling)

Markets are classified into four phases using two dimensions:

### 1) Volatility (ATR%)
- Compute ATR as a % of price (ATR%)
- Compare ATR% to a rolling median (252 bars ≈ 1 trading year):
  - **High Volatility (HV):** ATR% ≥ rolling median ATR%
  - **Low Volatility (LV):** ATR% < rolling median ATR%

### 2) Trend strength (ADX)
- **Trending:** ADX(14) > 25
- **Ranging:** ADX(14) ≤ 25

This yields: `HV_Trend`, `LV_Trend`, `HV_Ranging`, `LV_Ranging`.

---

## Regime artifacts (versioned research outputs)

This repo can export a versioned D1 regime label artifact for downstream projects (e.g. `market-sentiment-ml`).

### Build
```bash
python scripts/build_phase_labels_d1.py
```

### Outputs
- `data/output/regimes/phase_labels_d1.parquet` — D1 phase labels (UTC day buckets at 00:00)
- `data/output/regimes/phase_labels_d1_gap_report.csv` — full vendor gap report (pair, prev_timestamp, timestamp, gap_days)
- `data/output/regimes/phase_labels_d1_gap_summary.csv` — per-pair gap summary statistics

### Join contract (H1 consumers)
Downstream H1 datasets should join using:
- `(pair, floor(entry_time to UTC day)) == (pair, timestamp)`

### Transparency-only gap policy (important)
Yahoo FX data may contain historical gaps (e.g. Aug 2008). This pipeline does **not** fill, interpolate, forward-fill, or insert synthetic rows. Missing days remain missing; downstream consumers must handle missing regimes explicitly.

---

## Expert strategies (the “experts” in mixture-of-experts)

### Trend Following Suite (TF1–TF5)
All TF strategies use crossover detection to avoid immediate re-entry whipsaws.

| Strategy | Entry Logic                              | Exit Logic                     |
| -------- | ---------------------------------------- | ------------------------------ |
| TF1      | Close outside LWMA ± σ×StdDev band       | Close crosses back inside band |
| TF2      | Donchian channel breakout                | Trailing channel exit          |
| TF3      | SMA(9) crosses SMA(26)                   | Opposite crossover             |
| TF4      | LWMA(40) slope + Stochastic extreme      | Trailing stochastic exit       |
| TF5      | Bollinger Band breakout (σ=1.0, 20 bars) | Revert through center          |

### Mean Reversion Suite (MR1–MR5)
All MR strategies use explicit stop-loss and take-profit series.

| Strategy | Entry Logic                        | Exit Logic                  |
| -------- | ---------------------------------- | --------------------------- |
| MR1      | Fade LWMA ± σ×StdDev               | 2% SL / 2% TP               |
| MR2      | Stochastic extreme + momentum turn | Stochastic crosses 50       |
| MR3      | RSI extremes                       | 1% SL / 3% TP               |
| MR32     | RSI extreme + MA(200) filter       | RSI crosses 60/40 + 2.5% SL |
| MR42     | BB(20,2) breakout + ADX<20         | 2.5% SL / 1.25% TP          |
| MR5      | BB(20,2) breakout                  | 2% SL / 2% TP               |

### Rule-based routing: PhaseAware
Routes each bar to either a TF or MR strategy based on the detected regime:
- `HV_Trend, LV_Trend` → TrendFollowing expert
- `HV_Ranging, LV_Ranging` → MeanReversion expert

All TF×MR combinations are backtested automatically.

---

## Machine learning: StrategySelector (gating model)

### What it predicts
A supervised classifier predicts which **strategy type** is most likely to perform best over the next fixed horizon:

- `TrendFollowing`
- `MeanReversion`
- `PhaseAware` (rule-based router)

### Features (examples)
- ADX, ATR%, RSI, DI+/DI-
- recent returns and recent volatility features (computed using prior bars only to avoid subtle leakage/mismatch)

### Training & evaluation (leakage-safe)
- Model: XGBoost classifier
- Preprocessing uses a scaler fitted only on training data
- Walk-forward evaluation is used as a primary generalization check

### Online use in backtesting
`StrategySelector_Dynamic`:
- predicts a strategy type per bar
- uses precomputed expert signals (performance optimization)
- applies confidence gating + hysteresis + minimum hold to reduce churn
- executes the selected expert’s signal on that bar

---

## Backtest assumptions & cost model

**Execution model**
- Daily OHLCV data
- Enter at the *next bar close* after a signal is generated
- No pyramiding / no partial closes

**Transaction costs (defaults)**
- Spread: **1.0 pip**
- Slippage: **0.5 pip**
- Commission: **$0 per trade**
- Costs are applied at **both entry and exit** (round-trip cost modeled as a fraction of price)

**Position sizing**
- Two sizing modes are supported:
  - **Hardcoded multipliers** (`use_atr_sizing=False`): signal magnitude encodes a size multiplier (legacy mode, used for the main ablations shown in this README)
  - **ATR constant-risk sizing** (`use_atr_sizing=True`): targets a constant fraction of equity risked per trade (default `risk_pct=1%`)

See implementation details in `src/strategies.py` (`Backtester`).

---

## Run metadata (for reproducibility)

- **Data source:** Yahoo Finance (`yfinance`)
- **Instruments:** 14 FX pairs (7 majors + 7 minors)
- **Bar size:** Daily (D1)
- **Date range (typical):** 2005-01-01 to 2024-12-31
- **All results generated locally** via `python main.py`.

---

## Reproducibility

### Run
```bash
# optional (default: 42)
export EXPERIMENT_SEED=42
export EXPERIMENT_VARIANT=A

# CLI overrides env when both are set
python main.py --experiment-seed 42 --experiment-variant A
python main.py --experiment-seed 42 --experiment-variant F

# or use env/default resolution
python main.py
```

### Outputs
- Run-owned directory under `results_archive/<generation>_<variant>__<timestamp>/`
- Canonical `run_manifest.json` includes:
  - `reproducibility.experiment_seed`
  - `reproducibility.numpy_seed`
  - `reproducibility.python_random_seed`
  - `reproducibility.torch_seed` (when torch is available)
  - `feature_ordering.phase_predictor_by_pair`
  - `feature_ordering.strategy_selector_by_pair`

### Deterministic guarantees
- Global RNG seeding is applied for Python `random`, NumPy, and torch (best-effort if installed).
- All ML feature column lists are finalized in sorted order before train/test slicing, `X/y` extraction, model fitting, and feature-importance reporting.
- Selector tie-breaking and fold generation use deterministic ordering rules.
- Filesystem discovery and artifact selection use sorted iteration/tie-breaks.
- XGBoost defaults are pinned to deterministic CPU settings (`random_state`/`seed`, `n_jobs=1`, `tree_method="exact"`).

### Exact reproducibility procedure
```bash
export EXPERIMENT_SEED=42
python main.py --output-dir /absolute/path/to/results/run1

export EXPERIMENT_SEED=42
python main.py --output-dir /absolute/path/to/results/run2

diff /absolute/path/to/results/run1/walkforward_results_per_pair__dl_enabled.csv \
     /absolute/path/to/results/run2/walkforward_results_per_pair__dl_enabled.csv
```

Expected outcome: no material differences. If analysis is run on both archives, the report will also warn on missing seed metadata or mismatched feature ordering for runs sharing the same seed.

### Caveats
- Perfect bitwise equality can still vary across hardware/BLAS/CUDA/PyTorch versions.
- If torch is not installed, torch-specific seeds are omitted (other deterministic controls still apply).
- Walk-forward pipelines amplify small ordering bugs quickly; unsorted feature columns or filesystem iteration can become different CV folds, selector decisions, and Sharpe paths.

### Caching
The pipeline uses caching to speed up iteration. If you want a clean run, clear cached files (see `src/cache.py` and the `clear_cache(...)` calls in `main.py`).

---

## Project structure

```
market-phase-ml/
├── main.py
├── notebooks/
│   └── 01_regime_gating_walkforward.ipynb		# Jupyter notebook
├── src/
│   ├── data.py              # data pipeline (download/prepare)
│   ├── phases.py            # regime labeling (rule-based)
│   ├── strategies.py        # expert strategies + routers + backtester helpers
│   ├── models.py            # StrategySelector training/inference
│   ├── cache.py             # caching utilities
│   └── visualization.py     # plotting
├── results/                 # CSV artifacts (ablations, comparisons, walk-forward)
└── figures/                 # plots (optional)
```

---

## Engineering highlights
- end-to-end ML pipeline for a non-stationary time series decision problem
- leakage-safe preprocessing for inference consistency (train/inference feature alignment)
- walk-forward evaluation + experiment sweeps gated behind flags
- ablation design (fixed policy vs rule gating vs ML gating)
- performance optimization (precomputing expert signals; vectorized probability inference)
- reproducible artifacts (CSV outputs) suitable for CI and reporting

---

## Limitations & future work
- Explicit modeling of **switching costs / churn** (transaction-cost sensitivity, regime change penalties)
- Probability calibration (Platt / isotonic) and uncertainty-aware gating
- Additional gating approaches (contextual bandits, online learning)
- Further improvements to drawdown behavior (risk targeting / volatility scaling)
- Group-aware gating and guard actions (JPY vs non-JPY, USD role) to improve robustness without per-pair tuning

---

## Background
This project is based on a trading system I originally implemented in MQL4 (MetaTrader) and later reworked into a Python research/engineering pipeline for reproducible experimentation.

---

## About the author

I’m Jonas Almqvist — a Data Scientist / ML Engineer with a PhD and 15+ years of applied computational research.

- LinkedIn: https://linkedin.com/in/jalmqvist
- GitHub: https://github.com/jalmqvist

---

## License
MIT License

---

## Disclaimer
This repository is for educational and research purposes only. It is not financial advice.
Past performance does not guarantee future results.
