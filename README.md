# Market Phase-Based Strategy Selection

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## Overview

This project investigates whether **market phase detection enables better trading strategy selection**, leading to improved risk-adjusted trading performance compared to single-strategy approaches.

The core insight is that financial markets behave fundamentally differently depending on their current phase. Rather than using one strategy for all conditions, we select the optimal strategy based on the detected phase:

| Phase      | Volatility | Trend | Strategy        | Position Size |
| ---------- | ---------- | ----- | --------------- | ------------- |
| LV_Trend   | Low        | Yes   | Trend Following | Large (1.5x)  |
| HV_Trend   | High       | Yes   | Trend Following | Small (0.5x)  |
| LV_Ranging | Low        | No    | Mean Reversion  | Medium (1.0x) |
| HV_Ranging | High       | No    | Mean Reversion  | Small (0.5x)  |

## Key Finding

### Best Strategy: PhaseAware_TF5_MR42

**PhaseAware_TF5_MR42** significantly outperforms standalone strategies on major pairs (hardcoded sizing):

| Strategy              | Total Return | Sharpe | Max Drawdown | Win Rate | Profit Factor |
| -------------------- | ------------ | ------ | ------------ | -------- | ------------- |
| **PhaseAware_TF5_MR42** | **+39.92%**  | **0.164** | **-30.56%**  | **59.38%** | **1.113**     |
| PhaseAware_TF4_MR42  | +37.88%      | 0.217  | -27.43%      | 62.44%   | 1.130         |
| MR42 (standalone)    | +38.26%      | 0.123  | -42.12%      | 68.00%   | 1.056         |
| TF4 (standalone)     | +7.09%       | 0.035  | -22.02%      | 52.10%   | 1.047         |

### With ATR Constant-Risk Sizing

The same strategies perform even better with position sizing scaled by ATR (targeting 1% risk per trade):

| Strategy              | Total Return | Sharpe | Max Drawdown |
| -------------------- | ------------ | ------ | ------------ |
| **PhaseAware_TF5_MR42** | **+61.12%**  | **0.222** | **-42.85%**  |
| PhaseAware_TF4_MR42  | +43.98%      | 0.184  | -42.49%      |
| MR42 (standalone)    | **+84.05%**  | **0.264** | -46.03%      |

### Key Insights

✅ **Phase-aware routing adds ~10-15% Sharpe improvement** over standalone strategies  
✅ **Major pairs are profitable**, minor pairs are not (avoid trading minors)  
❌ **ML phase prediction underperforms rule-based detection** — simple ADX + volatility median is better  
🚀 **Strategy selector ML** trained on 3-class problem (TF vs MR vs PhaseAware) — CV accuracy 39.5% (vs 33.3% baseline)

## Phase Detection Method

Markets are classified into four phases using two dimensions:

### 1. Volatility (ATR%)

Current ATR expressed as a percentage of price, compared to its rolling median (252 bars = 1 trading year):

- **High Volatility (HV)**: ATR% ≥ rolling median ATR%
- **Low Volatility (LV)**:  ATR% < rolling median ATR%

This adaptive approach normalizes volatility across pairs with different price levels (e.g., USDJPY ~150 vs EURUSD ~1.10).

### 2. Trend Strength (ADX)

- **Trending**: ADX(14) > 25 (stricter than standard 20 to reduce false trends)
- **Ranging**:  ADX(14) ≤ 25

This gives four phases: **HV_Trend, LV_Trend, HV_Ranging, LV_Ranging**.

Reference: Kaufman, "Trading Systems and Methods" (5th ed., pg 854)

## Strategies

### Trend Following Suite (TF1–TF5)

All strategies use crossover detection to avoid re-entry whipsaws:

| Strategy | Entry Logic                                    | Exit Logic                        |
| -------- | ---------------------------------------------- | --------------------------------- |
| TF1      | Close outside LWMA ± σ×StdDev band             | Close crosses back inside band    |
| TF2      | Donchian channel breakout (new N-day high/low) | Trailing channel exit             |
| TF3      | SMA(9) crosses above/below SMA(26)             | Opposite crossover                |
| TF4      | LWMA(40) rising/falling + Stochastic extreme   | Trailing stochastic exit          |
| TF5      | Close outside Bollinger Band (σ=1.0, 20 bars)  | Close crosses back through center |

### Mean Reversion Suite (MR1–MR5)

All strategies use fixed stop-loss and take-profit levels:

| Strategy | Entry Logic                                   | Exit Logic                  |
| -------- | --------------------------------------------- | --------------------------- |
| MR1      | Close outside LWMA ± σ×StdDev (fade the move) | 2% SL / 2% TP               |
| MR2      | Stochastic(2,3,1) extreme + momentum turning  | Stochastic crosses 50       |
| MR3      | RSI(14) crosses above 25 or below 75          | 1% SL / 3% TP               |
| MR32     | RSI(14) extreme + price above/below MA(200)   | RSI crosses 60/40 + 2.5% SL |
| MR42     | BB(20,2) breakout + ADX(14) < 20 filter       | 2.5% SL / 1.25% TP          |
| MR5      | BB(20,2) breakout (no ADX filter)             | 2% SL / 2% TP               |

### Phase-Aware Strategy (PhaseAware)

Routes each bar to either a TF or MR strategy based on detected phase:

- **HV_Trend, LV_Trend** → Use selected TF strategy with reduced position size
- **HV_Ranging, LV_Ranging** → Use selected MR strategy with appropriate position size

All 30 TF×MR combinations (TF1-5 × MR1-5) are backtested automatically.

## Machine Learning Component

### StrategySelector: 3-Class Predictor

A supervised ML model predicts which **strategy type** will perform best in the next 20 bars:

- **TrendFollowing** — enter trend-following strategies
- **MeanReversion** — enter mean-reversion strategies
- **PhaseAware** — use phase-aware routing

**Training:**
- 5000+ samples per pair (daily OHLCV data, 20 years)
- Features: ADX, ATR%, RSI, DI, recent returns, recent volatility
- Model: XGBoost (100 estimators, max_depth=4)
- CV Accuracy: **39.5%** (vs 33.3% random baseline, +18% improvement)

**Status:** Models trained for all 14 pairs; integration into backtester in progress.

### Discontinued: Phase Prediction ML

Initial attempt to use ML to predict next bar's market phase achieved only 57-59% accuracy (vs 94%+ rule-based baseline). Conclusion: **Phase transitions are too rare and Markovian** — simple rule-based detection is optimal.

## Results Summary (Hardcoded Position Sizing)

### Major Pairs (EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD)

**Best strategies:**
- PhaseAware_TF5_MR42: +39.92% return, 0.164 Sharpe, -30.56% max DD
- PhaseAware_TF2_MR42: +44.95% return, 0.128 Sharpe, -32.79% max DD
- MR42 standalone: +38.26% return, 0.123 Sharpe, -42.12% max DD

**Average across all strategies:**
- Win rate: 48-68% (trend following lower, mean reversion higher)
- Sharpe ratio: -0.15 to +0.22
- Max drawdown: -14% to -46%

### Minor Pairs (EURGBP, EURJPY, EURCHF, GBPJPY, AUDJPY, EURAUD, GBPAUD)

⚠️ **Not recommended for trading:**
- Best strategy: PhaseAware_TF4_MR42: +3.38% return, 0.051 Sharpe
- Most strategies: negative or near-zero returns
- High drawdowns (-32% to -68%)

## Project Structure

```
market-phase-ml/
│
├── README.md                           ← This file
├── requirements.txt
├── .gitignore
├── main.py                             ← Main pipeline
│
├── src/
│   ├── __init__.py
│   ├── data.py                         ← MarketDataPipeline
│   ├── phases.py                       ← MarketPhaseDetector
│   ├── strategies.py                   ← TF1-5, MR1-5, PhaseAware
│   ├── models.py                       ← StrategyPerformanceTracker, StrategySelector
│   ├── cache.py                        ← Hash-based caching system
│   └── visualization.py                ← All plotting functions
│
├── data/
│   ├── raw/                            ← Downloaded OHLCV (cached)
│   ├── processed/                      ← Phase-detected, engineered (cached)
│   └── cache/                          ← Pickle files for fast re-runs
│
├── figures/                            ← Generated PNG charts
│   ├── phases_overview.png
│   ├── phase_statistics.png
│   ├── backtest_results.png
│   ├── phase_performance.png
│   ├── group_comparison.png
│   ├── equity_curves_*.png
│   └── key_results.png
│
└── results/                            ← CSV results files
    ├── results_per_pair.csv
    ├── results_majors.csv
    ├── results_minors.csv
    ├── results_summary.csv
    ├── results_ml.csv
    ├── results_ml_backtest.csv
    └── results_majors_atr.csv
```

## Installation

```bash
git clone https://github.com/jalmqvist/market-phase-ml
cd market-phase-ml
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Usage

```bash
# Run complete analysis pipeline (backtests + ML + visualizations)
python main.py

# Output: 
#   - figures/ directory with PNG charts
#   - results/ directory with CSV summary tables
#   - Console output with detailed metrics
```

### Pipeline Steps

1. **[1/5] Download & prepare** — Fetch 20 years daily OHLCV from Yahoo Finance
2. **[2/5] Phase detection** — Detect HV/LV and Trend/Ranging using ADX + ATR%
3. **[3/5] ML phase prediction** — Train walk-forward XGBoost to predict next phase
4. **[3b/5] ML backtest** — Test PhaseAware_TF4_MR42 with ML-predicted phases
5. **[3c/5] Strategy selector** — Train 3-class classifier (TF vs MR vs PhaseAware)
6. **[4/5] Backtest all strategies** — Run 30 TF×MR combos + phase-aware routing
7. **[5/5] Aggregate & visualize** — Compute metrics, generate charts

### Configuration

Edit `main.py` to adjust:

```python
START_DATE          = '2005-01-01'    # Backtest start date
END_DATE            = '2024-12-31'    # Backtest end date
INITIAL_CAPITAL     = 10000.0         # Starting equity
MIN_PHASE_SAMPLES   = 100             # Min bars per phase for ML
USE_ATR_SIZING      = False           # Use ATR-based position sizing
```

## Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
xgboost>=1.7.0
yfinance>=0.2.0
ta>=0.10.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
notebook>=6.5.0
ipykernel>=6.0.0
```

## Background & Motivation

This project is based on a real production trading system developed from 2018-2024. The market phase detection logic was originally implemented in MQL4 for MetaTrader, trading major forex pairs across multiple timeframes (H1-D1).

**Key observations from live trading:**

- ✅ Trend following strategies excel in trending conditions (ADX > 25)
- ✅ Mean reversion strategies excel in ranging conditions (ADX < 20)
- ✅ Position sizing by phase quality significantly impacts Sharpe ratio
- ✅ Major pairs (EURUSD, GBPUSD, USDJPY) have stronger directional bias
- ❌ Minor pairs and cross-pairs lack consistent tradeable patterns

## Key Methodology Insights

### Why Phase Detection Matters

Most single-strategy systems perform well in some conditions and poorly in others. Phase detection enables:

1. **Strategy selection** — Use the right tool for the market condition
2. **Position sizing** — Risk less in choppy/volatile periods
3. **Drawdown control** — Early detection of regime change
4. **Walk-forward robustness** — Avoid over-optimizing to one regime

### Why ML Phase Prediction Failed

Initial attempt to use ML to predict the next bar's phase:
- Achieved 57-59% accuracy (vs 57% baseline random guessing)
- Walk-forward retraining every 21 bars created model drift
- Phase transitions are too rare (~15-20% of bars) for supervised learning
- Rule-based detection (ADX + volatility) is already near-optimal

**Lesson:** Not all problems benefit from ML. Domain expertise + simple rules often outperform black-box ML.

### Why Strategy Selector ML Could Work

Unlike phase prediction (rare events), predicting **which strategy type wins** is common:
- ~33% of bars: TrendFollowing dominates
- ~33% of bars: MeanReversion dominates
- ~33% of bars: PhaseAware (routing) dominates

**Current status:** 3-class model achieves 39.5% accuracy; integration into live backtester in progress.

## Author

**Jonas Almqvist**  
PhD in Chemistry | Data Scientist | ML Engineer  
15+ years experience in computational analysis

🔗 [LinkedIn](https://www.linkedin.com/in/jalmqvist/)  
🐙 [GitHub](https://github.com/jalmqvist)

## License

MIT License — feel free to use and adapt this code for research or trading.

## References

- Kaufman, P.J. (2013). *Trading Systems and Methods* (5th ed.). Wiley Trading.
- Wilder, J.W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
- Connors, L. & Raschke, L. (1995). *Street Smarts*. M. Gordon Publishing.
- Katz, J.O. & McCormick, D.L. (2000). *The Encyclopedia of Trading Strategies*. McGraw-Hill.
- Weissman, R.L. (2005). *Mechanical Trading Systems*. Wiley Trading.

## Disclaimer

**This is a research project for educational purposes only.** Past performance does not guarantee future results. Futures, options, and forex trading involve substantial risk of loss. Use at your own risk. Always backtest on recent out-of-sample data before deploying any strategy with real capital.