
# MPML REFERENCE BENCHMARK

**Architecture**: mlp  
**Experiments**: 12  
**Baseline**: No-DL PhaseAware (aggregate)  
**Target pairs**: EURJPY, GBPJPY, USDJPY (Reactive-JPY family)  

> Δ values are walk-forward OOS deltas vs no-DL baseline.  
> `+` = positive Sharpe uplift.  
> For ΔDD: **smaller = better** (less drawdown).  
> All values rounded to 3 decimals for readability.

## 1. Uplift Matrix — ΔRet, ΔSh, and ΔDD per State and Pair
| Architecture | Behavioral Surface | Feature Set | State | ΔRet EURJPY | ΔRet GBPJPY | ΔRet USDJPY | ΔSh EURJPY | ΔSh GBPJPY | ΔSh USDJPY | ΔDD EURJPY | ΔDD GBPJPY | ΔDD USDJPY | Mean ΔSh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp | Consensus Lifecycle Surface | price_trend | JPY_CONSENSUS_YOUNG  |   0.67  |   0.12  |   0.54  |  0.271+  |  0.088+  |  0.140+  |  -0.27  |  -0.30  |  -0.16  |  0.167 |
| mlp | Consensus Lifecycle Surface | price_trend | JPY_CONSENSUS_MATURING  |   0.70  |  -0.09  |   0.48  |  0.281+  |  0.034+  |  0.122+  |  -0.29  |  -0.47  |  -0.08  |  0.146 |
| mlp | Consensus Lifecycle Surface | price_trend | JPY_CONSENSUS_MATURE  |   0.79  |  -0.00  |   0.74  |  0.290+  |  0.060+  |  0.203+  |  -0.20  |  -0.37  |   0.07  |  0.185 |
| mlp | Consensus Lifecycle Surface | price_trend | JPY_NON_EXTREME  |   0.81  |   0.16  |   0.40  |  0.296+  |  0.078+  |  0.084+  |  -0.24  |  -0.35  |  -0.25  |  0.153 |
| mlp | Trend / Volatility Surface | price_trend | LVTF  |   0.77  |  -0.11  |   0.69  |  0.313+  |  0.022+  |  0.171+  |  -0.30  |  -0.46  |  -0.04  |  0.169 |
| mlp | Trend / Volatility Surface | price_trend | HVTF  |   0.94  |  -0.02  |   0.42  |  0.351+  |  0.055+  |  0.062+  |  -0.17  |  -0.36  |  -0.21  |  0.156 |
| mlp | Trend / Volatility Surface | price_trend | LVR  |   0.67  |  -0.05  |   0.69  |  0.263+  |  0.043+  |  0.190+  |  -0.36  |  -0.38  |  -0.03  |  0.165 |
| mlp | Trend / Volatility Surface | price_trend | HVR  |   0.76  |   0.11  |   0.55  |  0.289+  |  0.080+  |  0.160+  |  -0.23  |  -0.31  |  -0.15  |  0.176 |
| mlp | Trend / Volatility Surface | trend_vol_only | LVTF  |   0.74  |  -0.06  |   0.39  |  0.314+  |  0.048+  |  0.101+  |  -0.32  |  -0.40  |  -0.24  |  0.154 |
| mlp | Trend / Volatility Surface | trend_vol_only | HVTF  |   0.71  |   0.04  |   0.35  |  0.272+  |  0.067+  |  0.056+  |  -0.24  |  -0.36  |  -0.28  |  0.132 |
| mlp | Trend / Volatility Surface | trend_vol_only | LVR  |   0.69  |  -0.03  |   0.69  |  0.275+  |  0.046+  |  0.183+  |  -0.27  |  -0.38  |   0.08  |  0.168 |
| mlp | Trend / Volatility Surface | trend_vol_only | HVR  |   0.70  |   0.02  |   0.61  |  0.259+  |  0.065+  |  0.179+  |  -0.19  |  -0.36  |  -0.10  |  0.167 |


## 2. Internal MPML Improvement — Dynamic Selector
> Dynamic selector improvement over the static PhaseAware baseline.
> All 14 pairs shown. Target pairs: EURJPY, GBPJPY, USDJPY.

### Consensus Lifecycle Surface — JPY_CONSENSUS_YOUNG — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |   7.38 |  0.050 |  -6.24 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   3.92 |  0.085 |  -3.21 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  26.61 |  0.098 |  -0.16 |


### Consensus Lifecycle Surface — JPY_CONSENSUS_MATURING — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |   9.34 |  0.060 |  -6.61 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   2.17 |  0.068 |  -5.18 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |   3.45 |  0.002 |  -8.89 |


### Consensus Lifecycle Surface — JPY_CONSENSUS_MATURE — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  25.58 |  0.135 |  -4.67 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   5.66 |  0.096 |  -2.53 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  30.84 |  0.112 |  -1.75 |


### Consensus Lifecycle Surface — JPY_NON_EXTREME — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  15.75 |  0.091 |  -6.37 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   6.35 |  0.098 |  -0.08 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  29.35 |  0.113 |   0.73 |


### Trend / Volatility Surface — LVTF — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  19.78 |  0.110 |  -3.33 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   2.25 |  0.072 |  -3.70 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  35.32 |  0.131 |  -0.46 |


### Trend / Volatility Surface — LVTF — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  10.14 |  0.064 |  -6.09 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   5.65 |  0.096 |  -1.35 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  14.29 |  0.050 |  -1.79 |


### Trend / Volatility Surface — HVTF — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  19.58 |  0.108 |  -5.00 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   8.76 |  0.118 |  -1.08 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  34.55 |  0.126 |   0.49 |


### Trend / Volatility Surface — HVTF — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  21.37 |  0.117 |  -7.76 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   6.21 |  0.100 |  -0.61 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  23.92 |  0.085 |  -1.92 |


### Trend / Volatility Surface — LVR — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |   5.48 |  0.040 |  -8.63 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   2.75 |  0.074 |  -1.64 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  31.97 |  0.122 |   2.56 |


### Trend / Volatility Surface — LVR — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  16.01 |  0.092 |  -4.96 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   7.77 |  0.110 |   0.48 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  35.91 |  0.135 |   0.61 |


### Trend / Volatility Surface — HVR — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |   9.11 |  0.059 |  -5.85 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   6.10 |  0.100 |  -4.85 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |   8.03 |  0.022 |  -7.43 |


### Trend / Volatility Surface — HVR — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  16.84 |  0.096 |  -7.31 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |  10.64 |  0.130 |  -0.80 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  14.35 |  0.047 |  -3.27 |


**(* = Reactive-JPY target pair)**

## 3. Target Family vs Negative Controls
> Control pair averages (mean across all 12 experiments)
> Separation: mean ΔSh (target) minus mean ΔSh (controls)

| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |   1.23 |  0.292 |  -0.02 |
| AUDUSD |  -0.55 | -0.082 |  -1.23 |
| EURAUD |   0.58 |  0.116 |  -0.62 |
| EURCHF |   0.16 |  0.030 |  -0.31 |
| EURGBP |  -0.29 | -0.061 |  -0.45 |
| EURUSD |  -1.24 | -0.305 |  -1.04 |
| GBPAUD |  -0.38 | -0.108 |  -1.34 |
| GBPUSD |  -0.78 | -0.133 |  -0.91 |
| NZDUSD |  -0.23 | -0.020 |  -1.13 |
| USDCAD |  -0.03 |  0.003 |  -0.51 |
| USDCHF |  -0.23 | -0.074 |  -0.35 |


### Separation Summary (mean ΔSh: target vs controls)
| State | Feature Set | Target ΔSh | Control ΔSh | Separation |
|---|---|---|---|---|
| JPY_CONSENSUS_YOUNG | price_trend |  0.167 | -0.031 |  0.198 |
| JPY_CONSENSUS_MATURING | price_trend |  0.146 | -0.031 |  0.177 |
| JPY_CONSENSUS_MATURE | price_trend |  0.185 | -0.031 |  0.216 |
| JPY_NON_EXTREME | price_trend |  0.153 | -0.031 |  0.184 |
| LVTF | price_trend |  0.169 | -0.031 |  0.200 |
| HVTF | price_trend |  0.156 | -0.031 |  0.187 |
| LVR | price_trend |  0.165 | -0.031 |  0.196 |
| HVR | price_trend |  0.176 | -0.031 |  0.207 |
| LVTF | trend_vol_only |  0.154 | -0.031 |  0.186 |
| HVTF | trend_vol_only |  0.132 | -0.031 |  0.163 |
| LVR | trend_vol_only |  0.168 | -0.031 |  0.199 |
| HVR | trend_vol_only |  0.167 | -0.031 |  0.198 |


## 4. Behavioral Family Comparison — Reactive-JPY
> Compares the two Behavioral Surfaces of the Reactive-JPY family.
> Metric: mean walk-forward ΔSharpe across surface states.
> Trend/Volatility is split by feature set.

| Surface / Feature Set | EURJPY | GBPJPY | USDJPY | Mean |
|---|---|---|---|---|
| Consensus Lifecycle  [price_trend]  |  0.284  |  0.065  |  0.137  |  0.162 |
| Trend / Volatility  [price_trend]  |  0.304  |  0.050  |  0.146  |  0.166 |
| Trend / Volatility  [trend_vol_only]  |  0.280  |  0.057  |  0.130  |  0.155 |


### Per-experiment breakdown
| Surface | State | Feature Set | EURJPY | GBPJPY | USDJPY | Mean |
|---|---|---|---|---|---|---|
| cLife | JPY_CONSENSUS_YOUNG | price_trend  |  0.271  |  0.088  |  0.140  |  0.167 |
| cLife | JPY_CONSENSUS_MATURING | price_trend  |  0.281  |  0.034  |  0.122  |  0.146 |
| cLife | JPY_CONSENSUS_MATURE | price_trend  |  0.290  |  0.060  |  0.203  |  0.185 |
| cLife | JPY_NON_EXTREME | price_trend  |  0.296  |  0.078  |  0.084  |  0.153 |
| tVol | LVTF | price_trend  |  0.313  |  0.022  |  0.171  |  0.169 |
| tVol | HVTF | price_trend  |  0.351  |  0.055  |  0.062  |  0.156 |
| tVol | LVR | price_trend  |  0.263  |  0.043  |  0.190  |  0.165 |
| tVol | HVR | price_trend  |  0.289  |  0.080  |  0.160  |  0.176 |
| tVol | LVTF | trend_vol_only  |  0.314  |  0.048  |  0.101  |  0.154 |
| tVol | HVTF | trend_vol_only  |  0.272  |  0.067  |  0.056  |  0.132 |
| tVol | LVR | trend_vol_only  |  0.275  |  0.046  |  0.183  |  0.168 |
| tVol | HVR | trend_vol_only  |  0.259  |  0.065  |  0.179  |  0.167 |


---
Generated by `compare_to_baseline.py` — MPML Stage 3 OOS validator.
Validated against VALIDATION_SPEC_JPY.md (frozen June 2026).
Report format: Markdown — optimized for GitHub, Jupyter, VS Code, Obsidian.
