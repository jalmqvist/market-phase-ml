
    # MPML REFERENCE BENCHMARK

    **Architecture**: m, l, p  
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
| mlp | Trend / Volatility Surface | price_trend | LVTF  |   0.51  |  -0.12  |   0.50  |  0.243+  |  0.021+  |  0.121+  |  -0.42  |  -0.45  |  -0.09  |  0.128 |
| mlp | Trend / Volatility Surface | price_trend | HVTF  |   0.84  |   0.09  |   0.40  |  0.303+  |  0.086+  |  0.070+  |  -0.28  |  -0.30  |  -0.13  |  0.153 |
| mlp | Trend / Volatility Surface | price_trend | LVR  |   0.81  |  -0.07  |   0.65  |  0.304+  |  0.032+  |  0.158+  |  -0.19  |  -0.40  |   0.01  |  0.165 |
| mlp | Trend / Volatility Surface | price_trend | HVR  |   0.78  |   0.02  |   0.65  |  0.294+  |  0.055+  |  0.183+  |  -0.17  |  -0.33  |  -0.13  |  0.177 |
| mlp | Trend / Volatility Surface | trend_vol_only | LVTF  |   0.73  |  -0.04  |   0.57  |  0.292+  |  0.027+  |  0.162+  |  -0.24  |  -0.34  |  -0.06  |  0.161 |
| mlp | Trend / Volatility Surface | trend_vol_only | HVTF  |   0.64  |   0.11  |   0.47  |  0.262+  |  0.077+  |  0.122+  |  -0.27  |  -0.31  |  -0.21  |  0.154 |
| mlp | Trend / Volatility Surface | trend_vol_only | LVR  |   0.80  |  -0.14  |   0.70  |  0.302+  | -0.010  |  0.180+  |  -0.22  |  -0.41  |  -0.07  |  0.157 |
| mlp | Trend / Volatility Surface | trend_vol_only | HVR  |   0.72  |   0.02  |   0.56  |  0.271+  |  0.059+  |  0.167+  |  -0.23  |  -0.34  |  -0.11  |  0.165 |


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
| EURJPY * |  19.93 |  0.110 |  -6.08 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   6.94 |  0.107 |  -1.93 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  48.60 |  0.184 |   1.84 |


### Trend / Volatility Surface — LVTF — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  28.26 |  0.146 |  -4.32 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |  12.07 |  0.138 |   0.54 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  34.04 |  0.129 |   0.78 |


### Trend / Volatility Surface — HVTF — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  11.87 |  0.072 |  -7.31 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   8.13 |  0.113 |  -1.77 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  42.30 |  0.156 |   2.99 |


### Trend / Volatility Surface — HVTF — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  16.47 |  0.094 |  -6.04 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   4.97 |  0.091 |  -2.50 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  30.81 |  0.114 |  -0.37 |


### Trend / Volatility Surface — LVR — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  12.56 |  0.076 | -10.56 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   9.92 |  0.123 |  -0.81 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  27.31 |  0.100 |  -2.16 |


### Trend / Volatility Surface — LVR — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  20.17 |  0.112 |  -7.88 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   7.80 |  0.110 |  -1.40 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  35.57 |  0.132 |  -0.14 |


### Trend / Volatility Surface — HVR — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  11.09 |  0.069 |  -6.09 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   4.12 |  0.085 |  -4.04 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  45.01 |  0.166 |  -1.63 |


### Trend / Volatility Surface — HVR — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  26.49 |  0.139 |  -4.26 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |  13.36 |  0.145 |   2.39 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |   5.30 |  0.010 |  -7.01 |


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
| LVTF | price_trend |  0.128 | -0.031 |  0.159 |
| HVTF | price_trend |  0.153 | -0.031 |  0.184 |
| LVR | price_trend |  0.165 | -0.031 |  0.196 |
| HVR | price_trend |  0.177 | -0.031 |  0.208 |
| LVTF | trend_vol_only |  0.161 | -0.031 |  0.192 |
| HVTF | trend_vol_only |  0.154 | -0.031 |  0.185 |
| LVR | trend_vol_only |  0.157 | -0.031 |  0.188 |
| HVR | trend_vol_only |  0.165 | -0.031 |  0.196 |


## 4. Behavioral Family Comparison — Reactive-JPY
> Compares the two Behavioral Surfaces of the Reactive-JPY family.
> Metric: mean walk-forward ΔSharpe across surface states.
> Trend/Volatility is split by feature set.

| Surface / Feature Set | EURJPY | GBPJPY | USDJPY | Mean |
|---|---|---|---|---|
| Consensus Lifecycle  [price_trend]  |  0.284  |  0.065  |  0.137  |  0.162 |
| Trend / Volatility  [price_trend]  |  0.286  |  0.049  |  0.133  |  0.156 |
| Trend / Volatility  [trend_vol_only]  |  0.282  |  0.038  |  0.158  |  0.159 |


### Per-experiment breakdown
| Surface | State | Feature Set | EURJPY | GBPJPY | USDJPY | Mean |
|---|---|---|---|---|---|---|
| cLife | JPY_CONSENSUS_YOUNG | price_trend  |  0.271  |  0.088  |  0.140  |  0.167 |
| cLife | JPY_CONSENSUS_MATURING | price_trend  |  0.281  |  0.034  |  0.122  |  0.146 |
| cLife | JPY_CONSENSUS_MATURE | price_trend  |  0.290  |  0.060  |  0.203  |  0.185 |
| cLife | JPY_NON_EXTREME | price_trend  |  0.296  |  0.078  |  0.084  |  0.153 |
| tVol | LVTF | price_trend  |  0.243  |  0.021  |  0.121  |  0.128 |
| tVol | HVTF | price_trend  |  0.303  |  0.086  |  0.070  |  0.153 |
| tVol | LVR | price_trend  |  0.304  |  0.032  |  0.158  |  0.165 |
| tVol | HVR | price_trend  |  0.294  |  0.055  |  0.183  |  0.177 |
| tVol | LVTF | trend_vol_only  |  0.292  |  0.027  |  0.162  |  0.161 |
| tVol | HVTF | trend_vol_only  |  0.262  |  0.077  |  0.122  |  0.154 |
| tVol | LVR | trend_vol_only  |  0.302  | -0.010  |  0.180  |  0.157 |
| tVol | HVR | trend_vol_only  |  0.271  |  0.059  |  0.167  |  0.165 |


---
Generated by `compare_to_baseline.py` — MPML Stage 3 OOS validator.
Validated against VALIDATION_SPEC_JPY.md (frozen June 2026).
Report format: Markdown — optimized for GitHub, Jupyter, VS Code, Obsidian.

    # MPML REFERENCE BENCHMARK

    **Architecture**: m, l, p  
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
| mlp | Trend / Volatility Surface | price_trend | LVTF  |   0.51  |  -0.12  |   0.50  |  0.243+  |  0.021+  |  0.121+  |  -0.42  |  -0.45  |  -0.09  |  0.128 |
| mlp | Trend / Volatility Surface | price_trend | HVTF  |   0.84  |   0.09  |   0.40  |  0.303+  |  0.086+  |  0.070+  |  -0.28  |  -0.30  |  -0.13  |  0.153 |
| mlp | Trend / Volatility Surface | price_trend | LVR  |   0.81  |  -0.07  |   0.65  |  0.304+  |  0.032+  |  0.158+  |  -0.19  |  -0.40  |   0.01  |  0.165 |
| mlp | Trend / Volatility Surface | price_trend | HVR  |   0.78  |   0.02  |   0.65  |  0.294+  |  0.055+  |  0.183+  |  -0.17  |  -0.33  |  -0.13  |  0.177 |
| mlp | Trend / Volatility Surface | trend_vol_only | LVTF  |   0.73  |  -0.04  |   0.57  |  0.292+  |  0.027+  |  0.162+  |  -0.24  |  -0.34  |  -0.06  |  0.161 |
| mlp | Trend / Volatility Surface | trend_vol_only | HVTF  |   0.64  |   0.11  |   0.47  |  0.262+  |  0.077+  |  0.122+  |  -0.27  |  -0.31  |  -0.21  |  0.154 |
| mlp | Trend / Volatility Surface | trend_vol_only | LVR  |   0.80  |  -0.14  |   0.70  |  0.302+  | -0.010  |  0.180+  |  -0.22  |  -0.41  |  -0.07  |  0.157 |
| mlp | Trend / Volatility Surface | trend_vol_only | HVR  |   0.72  |   0.02  |   0.56  |  0.271+  |  0.059+  |  0.167+  |  -0.23  |  -0.34  |  -0.11  |  0.165 |


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
| EURJPY * |  19.93 |  0.110 |  -6.08 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   6.94 |  0.107 |  -1.93 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  48.60 |  0.184 |   1.84 |


### Trend / Volatility Surface — LVTF — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  28.26 |  0.146 |  -4.32 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |  12.07 |  0.138 |   0.54 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  34.04 |  0.129 |   0.78 |


### Trend / Volatility Surface — HVTF — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  11.87 |  0.072 |  -7.31 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   8.13 |  0.113 |  -1.77 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  42.30 |  0.156 |   2.99 |


### Trend / Volatility Surface — HVTF — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  16.47 |  0.094 |  -6.04 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   4.97 |  0.091 |  -2.50 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  30.81 |  0.114 |  -0.37 |


### Trend / Volatility Surface — LVR — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  12.56 |  0.076 | -10.56 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   9.92 |  0.123 |  -0.81 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  27.31 |  0.100 |  -2.16 |


### Trend / Volatility Surface — LVR — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  20.17 |  0.112 |  -7.88 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   7.80 |  0.110 |  -1.40 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  35.57 |  0.132 |  -0.14 |


### Trend / Volatility Surface — HVR — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  11.09 |  0.069 |  -6.09 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   4.12 |  0.085 |  -4.04 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  45.01 |  0.166 |  -1.63 |


### Trend / Volatility Surface — HVR — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  26.49 |  0.139 |  -4.26 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |  13.36 |  0.145 |   2.39 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |   5.30 |  0.010 |  -7.01 |


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
| LVTF | price_trend |  0.128 | -0.031 |  0.159 |
| HVTF | price_trend |  0.153 | -0.031 |  0.184 |
| LVR | price_trend |  0.165 | -0.031 |  0.196 |
| HVR | price_trend |  0.177 | -0.031 |  0.208 |
| LVTF | trend_vol_only |  0.161 | -0.031 |  0.192 |
| HVTF | trend_vol_only |  0.154 | -0.031 |  0.185 |
| LVR | trend_vol_only |  0.157 | -0.031 |  0.188 |
| HVR | trend_vol_only |  0.165 | -0.031 |  0.196 |


## 4. Behavioral Family Comparison — Reactive-JPY
> Compares the two Behavioral Surfaces of the Reactive-JPY family.
> Metric: mean walk-forward ΔSharpe across surface states.
> Trend/Volatility is split by feature set.

| Surface / Feature Set | EURJPY | GBPJPY | USDJPY | Mean |
|---|---|---|---|---|
| Consensus Lifecycle  [price_trend]  |  0.284  |  0.065  |  0.137  |  0.162 |
| Trend / Volatility  [price_trend]  |  0.286  |  0.049  |  0.133  |  0.156 |
| Trend / Volatility  [trend_vol_only]  |  0.282  |  0.038  |  0.158  |  0.159 |


### Per-experiment breakdown
| Surface | State | Feature Set | EURJPY | GBPJPY | USDJPY | Mean |
|---|---|---|---|---|---|---|
| cLife | JPY_CONSENSUS_YOUNG | price_trend  |  0.271  |  0.088  |  0.140  |  0.167 |
| cLife | JPY_CONSENSUS_MATURING | price_trend  |  0.281  |  0.034  |  0.122  |  0.146 |
| cLife | JPY_CONSENSUS_MATURE | price_trend  |  0.290  |  0.060  |  0.203  |  0.185 |
| cLife | JPY_NON_EXTREME | price_trend  |  0.296  |  0.078  |  0.084  |  0.153 |
| tVol | LVTF | price_trend  |  0.243  |  0.021  |  0.121  |  0.128 |
| tVol | HVTF | price_trend  |  0.303  |  0.086  |  0.070  |  0.153 |
| tVol | LVR | price_trend  |  0.304  |  0.032  |  0.158  |  0.165 |
| tVol | HVR | price_trend  |  0.294  |  0.055  |  0.183  |  0.177 |
| tVol | LVTF | trend_vol_only  |  0.292  |  0.027  |  0.162  |  0.161 |
| tVol | HVTF | trend_vol_only  |  0.262  |  0.077  |  0.122  |  0.154 |
| tVol | LVR | trend_vol_only  |  0.302  | -0.010  |  0.180  |  0.157 |
| tVol | HVR | trend_vol_only  |  0.271  |  0.059  |  0.167  |  0.165 |


---
Generated by `compare_to_baseline.py` — MPML Stage 3 OOS validator.
Validated against VALIDATION_SPEC_JPY.md (frozen June 2026).
Report format: Markdown — optimized for GitHub, Jupyter, VS Code, Obsidian.
