# Market Phase-Based Strategy Selection

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## Overview

This project investigates whether **market phase detection 
enables better trading strategy selection**, leading to 
improved trading performance compared to single-strategy 
approaches.

The core insight is that financial markets behave 
fundamentally differently depending on their current 
phase. Rather than using one strategy for all conditions, 
we select the optimal strategy based on detected phase:

| Phase | Volatility | Trend | Strategy | Position Size |
|-------|-----------|-------|----------|---------------|
| LV_Trend_Up/Down | Low | Yes | Trend Following | +50% |
| HV_Consolidation | High | No | Mean Reversion | +50% |
| HV_Trend_Up/Down | High | Yes | Trend Following | -50% |
| LV_Consolidation | Low | No | No Trade | 0% |
| Pullback phases | Any | Yes* | Mean Reversion | Variable |
| Rangebound | Any | Yes | Reduced TF | -50% |

## Key Hypothesis

> Market phase detection does not directly improve 
> prediction accuracy. Instead, it enables optimal 
> **strategy selection**: trend-following methods work 
> best in low-volatility trending conditions, while 
> mean-reversion methods work best in high-volatility 
> consolidating conditions.

## Phase Detection Method

Markets are classified using two dimensions:

### 1. Relative Volatility (ATR Ratio)
RV = ATR(10) / ATR(100) RV >= 1.0 → High Volatility RV < 1.0 → Low Volatility

Reference: Kaufman, "Trading Systems and Methods" (pg 854)

### 2. Trend Detection
For D1 timeframe: Trending: ADX(14) > 20 Direction: +DI > -DI → Uptrend -DI > +DI → Downtrend Consolidation: ADX(14) <= 20

For shorter timeframes: Trending: Close > MA(200) → Uptrend Close < MA(200) → Downtrend Pullback: Uptrend but -DI > +DI (or vice versa)

## Strategies Compared

### Baseline A: Always Trend Following
- Entry: +DI crosses above -DI (long)
- Entry: -DI crosses above +DI (short)
- Exit: Opposing DI cross or ADX drops below threshold

### Baseline B: Always Mean Reversion
- Entry: RSI < 30 (long - oversold)
- Entry: RSI > 70 (short - overbought)
- Exit: RSI crosses 50

### Phase-Aware Strategy Selection
- Selects TF or MR based on current market phase
- Adjusts position size based on phase quality
- Avoids trading in unfavorable conditions


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
dataclasses>=0.6
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
15+ years experience in computational analysis
🔗 LinkedIn

## License

MIT License - feel free to use and adapt this code.

## References

- Kaufman, P.J. (2013). *Trading Systems and Methods* (5th ed.). Wiley Trading.
- Wilder, J.W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
