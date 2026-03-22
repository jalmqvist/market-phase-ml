# Market Phase–Based Strategy Selection (Regime-Aware Time Series Gating)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-green)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## TL;DR

A regime-aware **time-series ML pipeline** that combines:

- **Rule-based regime labeling** (market “phases”), and
- an **XGBoost gating model** (mixture-of-experts)

…to route between expert policies (**trend-following** vs **mean-reversion**) with **leakage-safe walk-forward evaluation** and reproducible CSV artifacts.

> Goal: demonstrate end-to-end ML engineering + experimental discipline on a non-stationary time series decision problem (not “promise profits”).

---

## Notebook walkthrough (recommended)

After you run `python main.py` and the `results/*.csv` artifacts are generated, you can open the notebook:

- `notebooks/01_regime_gating_walkforward.ipynb`

It is intentionally “read-only” on the artifacts: it loads the CSV outputs + selected debug exports and produces the main figures (fold distributions, per-pair breakdown, and a fold-level case study showing volatility spikes + strategy selection).

---

## Why this project

This repo is intentionally built as an end-to-end ML engineering project for **non-stationary time series**:

- Mixture-of-experts style **policy routing** (gating model)
- A realistic **leakage-safe evaluation** design (walk-forward)
- Reproducible experimentation (CSV artifacts + caching)
- Failure-mode driven iteration (volatility guard, max-hold reset)
- Code that’s structured like a small production pipeline (data → features → labels → model → evaluation)

Trading is simply the “toy domain”; the underlying pattern generalizes to many real ML systems where the best decision policy depends on context (e.g., demand forecasting regimes, anomaly handling, recommender exploration/exploitation, etc.).

---

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

Key outputs (written to `results/`):
- `ablation_summary_aggregate.csv` — headline in-sample ablation numbers
- `ablation_summary_per_pair.csv` — per-pair breakdown
- `walkforward_results_summary.csv` — walk-forward (OOS) summary vs baseline
- `walkforward_results_per_pair.csv` — walk-forward per-pair deltas
- `walkforward_results_per_fold.csv` — walk-forward deltas per fold (debuggable)
- `walkforward_tau_sweep_summary.csv` — τ sweep summary (optional)
- `walkforward_policy_sweep_summary.csv` — policy sweep summary (optional)

The notebook (`notebooks/01_regime_gating_walkforward.ipynb`) reads these artifacts to generate figures and fold-level case studies.

> Note: expensive sweeps (τ/policy sweeps) are gated behind flags in `main.py`.

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
| A0_TF4                                     | Fixed policy (TrendFollowing only)                           |     -3.31% |      -0.05 |    -25.28% |    14 |
| A1_MR42                                    | Fixed policy (MeanReversion only)                            |    +20.38% |      +0.07 |    -41.82% |    14 |
| A2_PhaseAware_TF4_MR42                     | Rule-based routing using detected regimes                    |    +22.01% |      +0.14 |    -29.13% |    14 |
| A3_DynamicSelector_tau0.62_exit0.57_hold10 | **ML gating** with confidence threshold + hysteresis + min-hold |    +56.17% |      +0.21 |    -31.61% |    14 |

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
| Major |    +0.07 |    +0.09 |    -0.30 |
| Minor |    +0.34 |    +0.05 |    +0.08 |

#### JPY vs non-JPY
| Group   | Return Δ | Sharpe Δ | Max DD Δ |
| ------- | -------: | -------: | -------: |
| JPY     |    +0.34 |    +0.01 |    +0.53 |
| non-JPY |    +0.15 |    +0.10 |    -0.37 |

#### USD role (base/quote)
| Group     | Return Δ | Sharpe Δ | Max DD Δ |
| --------- | -------: | -------: | -------: |
| USD-base  |    +0.28 |    +0.15 |    +0.03 |
| USD-quote |    -0.09 |    +0.05 |    -0.55 |
| No-USD    |    +0.34 |    +0.05 |    +0.08 |

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
- τ=0.60 → Avg Sharpe Δ **+0.02**, Avg Return Δ **+0.17%**, confident bars **~52%**
- τ=0.62 → Avg Sharpe Δ **+0.07**, Avg Return Δ **+0.21%**, confident bars **~49%**
- τ=0.70 → Avg Sharpe Δ **+0.05**, Avg Return Δ **+0.14%**, confident bars **~38%**

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
python main.py
```

### Outputs
- `results/` — CSV tables including ablations, comparisons, and walk-forward artifacts
- `figures/` — generated charts (optional / may evolve)

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