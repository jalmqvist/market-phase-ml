# Market Phase-Based Strategy Selection

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## Overview

This project investigates whether **market phase detection
enables better trading strategy selection**, leading to
improved risk-adjusted trading performance compared to
single-strategy approaches.

The core insight is that financial markets behave
fundamentally differently depending on their current
phase. Rather than using one strategy for all conditions,
we select the optimal strategy based on the detected phase:

| Phase      | Volatility | Trend | Strategy        | Position Size |
| ---------- | ---------- | ----- | --------------- | ------------- |
| LV_Trend   | Low        | Yes   | Trend Following | Large         |
| HV_Trend   | High       | Yes   | Trend Following | Small         |
| LV_Ranging | Low        | No    | Mean Reversion  | Medium        |
| HV_Ranging | High       | No    | Mean Reversion  | Small         |

## Key Hypothesis

> Market phase detection enables optimal **strategy selection**:
> trend-following methods work best in trending conditions,
> while mean-reversion methods work best in ranging conditions.
> Furthermore, phase-aware routing improves **risk-adjusted**
> performance even when raw returns are comparable to
> single-strategy approaches.

## Key Finding

> **PhaseAware_TF4_MR42 is the best performing strategy
> combination on both major and minor forex pairs.**
>
> Critically, PhaseAware_TF4_MR42 outperforms MR42 standalone
> not on raw return, but on **risk-adjusted return**:
>
> | Strategy            | Total Return | Sharpe | Max Drawdown | Win Rate | Profit Factor |
> | ------------------- | ------------ | ------ | ------------ | -------- | ------------- |
> | PhaseAware_TF4_MR42 | +37.88%      | 0.217  | -27.43%      | 62.44%   | 1.130         |
> | MR42 (standalone)   | +38.26%      | 0.123  | -42.12%      | 68.00%   | 1.056         |
> | TF4 (standalone)    | +7.09%       | 0.035  | -22.02%      | 52.10%   | 1.047         |
>
> PhaseAware achieves a **77% higher Sharpe ratio** and
> **35% lower max drawdown** than MR42 standalone, while
> maintaining comparable total return. This is meaningful
> evidence that phase-aware strategy routing adds value
> beyond simply running the best individual strategy.
>
> Results based on 14 forex pairs (7 majors, 7 minors),
> daily D1 data, hardcoded position sizing.

## Phase Detection Method

Markets are classified into four phases using two dimensions:

### 1. Volatility (ATR%)

Current ATR expressed as a percentage of price, compared
to its rolling median:

- **High Volatility**: ATR% > rolling median ATR%
- **Low Volatility**:  ATR% ≤ rolling median ATR%

### 2. Trend Strength (ADX)

- **Trending**: ADX(14) > 20
- **Ranging**:  ADX(14) ≤ 20

This gives four phases: HV_Trend, LV_Trend, HV_Ranging, LV_Ranging.

Reference: Kaufman, "Trading Systems and Methods" (pg 854)

## Strategies

### Trend Following Suite (TF1–TF5)

| Strategy | Entry Logic                                    | Exit Logic                        |
| -------- | ---------------------------------------------- | --------------------------------- |
| TF1      | Close outside LWMA ± σ×StdDev band             | Close crosses back inside band    |
| TF2      | Donchian channel breakout (new N-day high/low) | Trailing channel exit             |
| TF3      | SMA(9) crosses SMA(26)                         | Opposite crossover                |
| TF4      | LWMA(40) direction + Stochastic(5,3,1) extreme | Trailing stochastic exit          |
| TF5      | Close outside Bollinger Band (σ=1.0, 20 bars)  | Close crosses back through center |

### Mean Reversion Suite (MR1–MR5)

| Strategy | Entry Logic                                   | Exit Logic                  |
| -------- | --------------------------------------------- | --------------------------- |
| MR1      | Close outside LWMA ± σ×StdDev band            | 2% SL / 2% TP               |
| MR2      | Stochastic(2,3,1) < 15 rising OR > 85 falling | Stochastic crosses 50       |
| MR3      | RSI(14) crosses above 25 or below 75          | 1% SL / 3% TP               |
| MR32     | RSI(14) extreme + price above/below MA(200)   | RSI crosses 60/40 + 2.5% SL |
| MR42     | BB(20,2) breakout + ADX(14) < 20 filter       | 2.5% SL / 1.25% TP          |
| MR5      | BB(20,2) breakout (no ADX filter)             | 2% SL / 2% TP               |

### Phase-Aware Strategy (PhaseAware)

Routes each bar to either a TF or MR strategy based on
the detected market phase, with position sizing scaled
by phase risk profile. All 30 TF×MR combinations are
tested automatically.

## Machine Learning Component

A supervised ML model (`models.py`) is trained to
**predict the next bar's market phase** using features
derived from price action and technical indicators.

Rather than labeling the current bar's phase (which is
already known), the model learns to anticipate **regime
transitions** — for example, recognizing early signs that
a trending market is about to transition into a ranging one.

**How it helps:**
- Enter trend-following trades *earlier* in a new trend
- Avoid deploying mean-reversion strategies into the tail
  end of a ranging phase that is about to break out
- Reduce whipsaw trades caused by late phase detection

The ML model's predicted phases are used by PhaseAwareStrategy
to route signals, making it straightforward to measure whether
ML-predicted phases outperform rule-based phase classification.




## Project Structure
```
market-phase-ml/
│
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
│
├── src/
│   ├── __init__.py
│   ├── data.py                ← MarketDataPipeline
│   ├── phases.py              ← MarketPhaseDetector
│   ├── features.py            ← Feature engineering
│   ├── strategies.py          ← TF, MR, PhaseAware strategies
│   ├── backtester.py          ← Backtester, TradeResult
│   └── visualization.py       ← All plotting functions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_phase_analysis.ipynb
│   ├── 03_strategy_comparison.ipynb
│   └── 04_results_analysis.ipynb
│
├── figures/
│   ├── phases_overview.png
│   ├── phase_statistics.png
│   ├── model_comparison.png
│   ├── backtest_results.png
│   └── phase_performance.png
│
└── data/
    ├── raw/
    └── processed/
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
# Run complete analysis
python main.py

# Or run individual notebooks in PyCharm:
# notebooks/01_data_exploration.ipynb
# notebooks/02_phase_analysis.ipynb
# notebooks/03_strategy_comparison.ipynb
# notebooks/04_results_analysis.ipynb
```
#Requirements
```bash
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

This project is based on a real production trading system developed and operated from 2018-2024. The market phase detection logic was originally implemented in MQL4 for MetaTrader, trading major forex pairs (EURUSD, GBPUSD, USDJPY, USDCHF) across multiple timeframes (H1-D1).

Key observations from live trading:

- Trend following strategies perform poorly in high volatility, non-trending conditions
- Mean reversion strategies perform poorly in strong trending conditions
- Position sizing based on phase quality significantly impacts overall performance
- Major pairs (EURUSD, GBPUSD) tend to trend more cleanly than minor pairs

## Author

**Jonas Almqvist**
PhD in Chemistry | Data Scientist | ML Engineer
15+ years experience in computational analysis 🔗 [LinkedIn](https://www.linkedin.com/in/jalmqvist/) 🐙 [GitHub](https://github.com/jalmqvist/market-phase-ml)

## License

MIT License - feel free to use and adapt this code.

## References

- Kaufman, P.J. (2013). *Trading Systems and Methods* (5th ed.). Wiley Trading.
- Wilder, J.W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
- Connors, L. & Raschke, L. (1995). *Street Smarts*. M. Gordon Publishing.
- Katz, J.O. & McCormick, D.L. (2000). *The Encyclopedia of Trading Strategies*. McGraw-Hill.
- Weissman, R.L. (2005). *Mechanical Trading Systems*. Wiley Trading.
