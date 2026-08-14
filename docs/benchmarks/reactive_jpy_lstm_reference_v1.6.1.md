
    # MPML REFERENCE BENCHMARK

    **Architecture**: l, s, t, m  
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
| lstm | Consensus Lifecycle Surface | price_trend | JPY_CONSENSUS_YOUNG  |   0.64  |   0.08  |   0.53  |  0.237+  |  0.081+  |  0.144+  |  -0.35  |  -0.33  |  -0.13  |  0.154 |
| lstm | Consensus Lifecycle Surface | price_trend | JPY_CONSENSUS_MATURING  |   0.62  |   0.03  |   0.70  |  0.254+  |  0.053+  |  0.187+  |  -0.34  |  -0.33  |  -0.02  |  0.165 |
| lstm | Consensus Lifecycle Surface | price_trend | JPY_CONSENSUS_MATURE  |   0.73  |   0.06  |   0.65  |  0.273+  |  0.058+  |  0.176+  |  -0.26  |  -0.29  |   0.05  |  0.169 |
| lstm | Consensus Lifecycle Surface | price_trend | JPY_NON_EXTREME  |   0.66  |   0.01  |   0.58  |  0.267+  |  0.052+  |  0.147+  |  -0.30  |  -0.39  |  -0.08  |  0.155 |
| lstm | Trend / Volatility Surface | price_trend | LVTF  |   0.62  |   0.02  |   0.54  |  0.249+  |  0.055+  |  0.129+  |  -0.33  |  -0.39  |  -0.11  |  0.145 |
| lstm | Trend / Volatility Surface | price_trend | HVTF  |   0.47  |   0.01  |   0.39  |  0.189+  |  0.065+  |  0.119+  |  -0.37  |  -0.33  |  -0.31  |  0.124 |
| lstm | Trend / Volatility Surface | price_trend | LVR  |   0.81  |   0.05  |   0.65  |  0.306+  |  0.076+  |  0.182+  |  -0.21  |  -0.35  |  -0.02  |  0.188 |
| lstm | Trend / Volatility Surface | price_trend | HVR  |   0.81  |   0.02  |   0.52  |  0.301+  |  0.055+  |  0.141+  |  -0.20  |  -0.37  |  -0.12  |  0.166 |
| lstm | Trend / Volatility Surface | trend_vol_only | LVTF  |   0.70  |  -0.20  |   0.50  |  0.286+  |  0.011+  |  0.117+  |  -0.37  |  -0.52  |  -0.05  |  0.138 |
| lstm | Trend / Volatility Surface | trend_vol_only | HVTF  |   0.74  |  -0.01  |   0.48  |  0.278+  |  0.063+  |  0.103+  |  -0.18  |  -0.37  |  -0.23  |  0.148 |
| lstm | Trend / Volatility Surface | trend_vol_only | LVR  |   0.78  |  -0.06  |   0.54  |  0.306+  |  0.032+  |  0.143+  |  -0.24  |  -0.39  |   0.00  |  0.160 |
| lstm | Trend / Volatility Surface | trend_vol_only | HVR  |   0.66  |   0.00  |   0.71  |  0.263+  |  0.070+  |  0.199+  |  -0.28  |  -0.38  |  -0.13  |  0.177 |


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
| EURJPY * |   8.23 |  0.054 |  -7.03 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |  18.46 |  0.177 |   5.49 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  -0.25 | -0.015 | -12.12 |


### Consensus Lifecycle Surface — JPY_CONSENSUS_MATURING — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |   4.11 |  0.032 | -10.90 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   9.37 |  0.119 |  -1.10 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |   7.64 |  0.020 |  -7.47 |


### Consensus Lifecycle Surface — JPY_CONSENSUS_MATURE — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  16.62 |  0.095 |  -7.43 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   7.51 |  0.109 |  -0.47 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  55.32 |  0.203 |   1.82 |


### Consensus Lifecycle Surface — JPY_NON_EXTREME — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  15.14 |  0.088 |  -8.66 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   1.61 |  0.065 |  -3.74 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  25.21 |  0.096 |   2.68 |


### Trend / Volatility Surface — LVTF — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  19.99 |  0.110 |  -5.00 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   3.83 |  0.080 |  -3.00 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  24.61 |  0.091 |   0.20 |


### Trend / Volatility Surface — LVTF — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  30.30 |  0.155 |  -3.57 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   5.06 |  0.088 |  -3.25 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  35.19 |  0.135 |   1.78 |


### Trend / Volatility Surface — HVTF — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  26.01 |  0.136 |  -7.19 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   9.65 |  0.122 |   0.39 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  62.35 |  0.223 |   4.21 |


### Trend / Volatility Surface — HVTF — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  32.50 |  0.164 |  -3.32 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   4.16 |  0.083 |  -2.44 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  30.28 |  0.110 |   2.37 |


### Trend / Volatility Surface — LVR — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |   8.39 |  0.055 |  -9.11 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |  16.57 |  0.165 |   5.04 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  18.64 |  0.065 |  -2.24 |


### Trend / Volatility Surface — LVR — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  20.91 |  0.114 |  -5.61 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   4.40 |  0.081 |  -2.39 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  10.90 |  0.034 |  -1.33 |


### Trend / Volatility Surface — HVR — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  19.21 |  0.107 |  -4.94 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   4.15 |  0.084 |  -3.10 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  22.00 |  0.078 |  -0.43 |


### Trend / Volatility Surface — HVR — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  10.58 |  0.066 |  -6.99 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |  17.08 |  0.169 |   4.81 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  59.60 |  0.214 |   2.96 |


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
| JPY_CONSENSUS_YOUNG | price_trend |  0.154 | -0.031 |  0.185 |
| JPY_CONSENSUS_MATURING | price_trend |  0.165 | -0.031 |  0.196 |
| JPY_CONSENSUS_MATURE | price_trend |  0.169 | -0.031 |  0.200 |
| JPY_NON_EXTREME | price_trend |  0.155 | -0.031 |  0.186 |
| LVTF | price_trend |  0.145 | -0.031 |  0.176 |
| HVTF | price_trend |  0.124 | -0.031 |  0.155 |
| LVR | price_trend |  0.188 | -0.031 |  0.219 |
| HVR | price_trend |  0.166 | -0.031 |  0.197 |
| LVTF | trend_vol_only |  0.138 | -0.031 |  0.169 |
| HVTF | trend_vol_only |  0.148 | -0.031 |  0.179 |
| LVR | trend_vol_only |  0.160 | -0.031 |  0.191 |
| HVR | trend_vol_only |  0.177 | -0.031 |  0.208 |


## 4. Behavioral Family Comparison — Reactive-JPY
> Compares the two Behavioral Surfaces of the Reactive-JPY family.
> Metric: mean walk-forward ΔSharpe across surface states.
> Trend/Volatility is split by feature set.

| Surface / Feature Set | EURJPY | GBPJPY | USDJPY | Mean |
|---|---|---|---|---|
| Consensus Lifecycle  [price_trend]  |  0.258  |  0.061  |  0.163  |  0.161 |
| Trend / Volatility  [price_trend]  |  0.262  |  0.063  |  0.143  |  0.156 |
| Trend / Volatility  [trend_vol_only]  |  0.283  |  0.044  |  0.141  |  0.156 |


### Per-experiment breakdown
| Surface | State | Feature Set | EURJPY | GBPJPY | USDJPY | Mean |
|---|---|---|---|---|---|---|
| cLife | JPY_CONSENSUS_YOUNG | price_trend  |  0.237  |  0.081  |  0.144  |  0.154 |
| cLife | JPY_CONSENSUS_MATURING | price_trend  |  0.254  |  0.053  |  0.187  |  0.165 |
| cLife | JPY_CONSENSUS_MATURE | price_trend  |  0.273  |  0.058  |  0.176  |  0.169 |
| cLife | JPY_NON_EXTREME | price_trend  |  0.267  |  0.052  |  0.147  |  0.155 |
| tVol | LVTF | price_trend  |  0.249  |  0.055  |  0.129  |  0.145 |
| tVol | HVTF | price_trend  |  0.189  |  0.065  |  0.119  |  0.124 |
| tVol | LVR | price_trend  |  0.306  |  0.076  |  0.182  |  0.188 |
| tVol | HVR | price_trend  |  0.301  |  0.055  |  0.141  |  0.166 |
| tVol | LVTF | trend_vol_only  |  0.286  |  0.011  |  0.117  |  0.138 |
| tVol | HVTF | trend_vol_only  |  0.278  |  0.063  |  0.103  |  0.148 |
| tVol | LVR | trend_vol_only  |  0.306  |  0.032  |  0.143  |  0.160 |
| tVol | HVR | trend_vol_only  |  0.263  |  0.070  |  0.199  |  0.177 |


---
Generated by `compare_to_baseline.py` — MPML Stage 3 OOS validator.
Validated against VALIDATION_SPEC_JPY.md (frozen June 2026).
Report format: Markdown — optimized for GitHub, Jupyter, VS Code, Obsidian.

    # MPML REFERENCE BENCHMARK

    **Architecture**: l, s, t, m  
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
| lstm | Consensus Lifecycle Surface | price_trend | JPY_CONSENSUS_YOUNG  |   0.64  |   0.08  |   0.53  |  0.237+  |  0.081+  |  0.144+  |  -0.35  |  -0.33  |  -0.13  |  0.154 |
| lstm | Consensus Lifecycle Surface | price_trend | JPY_CONSENSUS_MATURING  |   0.62  |   0.03  |   0.70  |  0.254+  |  0.053+  |  0.187+  |  -0.34  |  -0.33  |  -0.02  |  0.165 |
| lstm | Consensus Lifecycle Surface | price_trend | JPY_CONSENSUS_MATURE  |   0.73  |   0.06  |   0.65  |  0.273+  |  0.058+  |  0.176+  |  -0.26  |  -0.29  |   0.05  |  0.169 |
| lstm | Consensus Lifecycle Surface | price_trend | JPY_NON_EXTREME  |   0.66  |   0.01  |   0.58  |  0.267+  |  0.052+  |  0.147+  |  -0.30  |  -0.39  |  -0.08  |  0.155 |
| lstm | Trend / Volatility Surface | price_trend | LVTF  |   0.62  |   0.02  |   0.54  |  0.249+  |  0.055+  |  0.129+  |  -0.33  |  -0.39  |  -0.11  |  0.145 |
| lstm | Trend / Volatility Surface | price_trend | HVTF  |   0.47  |   0.01  |   0.39  |  0.189+  |  0.065+  |  0.119+  |  -0.37  |  -0.33  |  -0.31  |  0.124 |
| lstm | Trend / Volatility Surface | price_trend | LVR  |   0.81  |   0.05  |   0.65  |  0.306+  |  0.076+  |  0.182+  |  -0.21  |  -0.35  |  -0.02  |  0.188 |
| lstm | Trend / Volatility Surface | price_trend | HVR  |   0.81  |   0.02  |   0.52  |  0.301+  |  0.055+  |  0.141+  |  -0.20  |  -0.37  |  -0.12  |  0.166 |
| lstm | Trend / Volatility Surface | trend_vol_only | LVTF  |   0.70  |  -0.20  |   0.50  |  0.286+  |  0.011+  |  0.117+  |  -0.37  |  -0.52  |  -0.05  |  0.138 |
| lstm | Trend / Volatility Surface | trend_vol_only | HVTF  |   0.74  |  -0.01  |   0.48  |  0.278+  |  0.063+  |  0.103+  |  -0.18  |  -0.37  |  -0.23  |  0.148 |
| lstm | Trend / Volatility Surface | trend_vol_only | LVR  |   0.78  |  -0.06  |   0.54  |  0.306+  |  0.032+  |  0.143+  |  -0.24  |  -0.39  |   0.00  |  0.160 |
| lstm | Trend / Volatility Surface | trend_vol_only | HVR  |   0.66  |   0.00  |   0.71  |  0.263+  |  0.070+  |  0.199+  |  -0.28  |  -0.38  |  -0.13  |  0.177 |


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
| EURJPY * |   8.23 |  0.054 |  -7.03 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |  18.46 |  0.177 |   5.49 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  -0.25 | -0.015 | -12.12 |


### Consensus Lifecycle Surface — JPY_CONSENSUS_MATURING — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |   4.11 |  0.032 | -10.90 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   9.37 |  0.119 |  -1.10 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |   7.64 |  0.020 |  -7.47 |


### Consensus Lifecycle Surface — JPY_CONSENSUS_MATURE — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  16.62 |  0.095 |  -7.43 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   7.51 |  0.109 |  -0.47 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  55.32 |  0.203 |   1.82 |


### Consensus Lifecycle Surface — JPY_NON_EXTREME — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  15.14 |  0.088 |  -8.66 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   1.61 |  0.065 |  -3.74 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  25.21 |  0.096 |   2.68 |


### Trend / Volatility Surface — LVTF — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  19.99 |  0.110 |  -5.00 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   3.83 |  0.080 |  -3.00 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  24.61 |  0.091 |   0.20 |


### Trend / Volatility Surface — LVTF — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  30.30 |  0.155 |  -3.57 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   5.06 |  0.088 |  -3.25 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  35.19 |  0.135 |   1.78 |


### Trend / Volatility Surface — HVTF — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  26.01 |  0.136 |  -7.19 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   9.65 |  0.122 |   0.39 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  62.35 |  0.223 |   4.21 |


### Trend / Volatility Surface — HVTF — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  32.50 |  0.164 |  -3.32 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   4.16 |  0.083 |  -2.44 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  30.28 |  0.110 |   2.37 |


### Trend / Volatility Surface — LVR — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |   8.39 |  0.055 |  -9.11 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |  16.57 |  0.165 |   5.04 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  18.64 |  0.065 |  -2.24 |


### Trend / Volatility Surface — LVR — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  20.91 |  0.114 |  -5.61 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   4.40 |  0.081 |  -2.39 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  10.90 |  0.034 |  -1.33 |


### Trend / Volatility Surface — HVR — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  19.21 |  0.107 |  -4.94 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   4.15 |  0.084 |  -3.10 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  22.00 |  0.078 |  -0.43 |


### Trend / Volatility Surface — HVR — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  10.58 |  0.066 |  -6.99 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |  17.08 |  0.169 |   4.81 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  59.60 |  0.214 |   2.96 |


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
| JPY_CONSENSUS_YOUNG | price_trend |  0.154 | -0.031 |  0.185 |
| JPY_CONSENSUS_MATURING | price_trend |  0.165 | -0.031 |  0.196 |
| JPY_CONSENSUS_MATURE | price_trend |  0.169 | -0.031 |  0.200 |
| JPY_NON_EXTREME | price_trend |  0.155 | -0.031 |  0.186 |
| LVTF | price_trend |  0.145 | -0.031 |  0.176 |
| HVTF | price_trend |  0.124 | -0.031 |  0.155 |
| LVR | price_trend |  0.188 | -0.031 |  0.219 |
| HVR | price_trend |  0.166 | -0.031 |  0.197 |
| LVTF | trend_vol_only |  0.138 | -0.031 |  0.169 |
| HVTF | trend_vol_only |  0.148 | -0.031 |  0.179 |
| LVR | trend_vol_only |  0.160 | -0.031 |  0.191 |
| HVR | trend_vol_only |  0.177 | -0.031 |  0.208 |


## 4. Behavioral Family Comparison — Reactive-JPY
> Compares the two Behavioral Surfaces of the Reactive-JPY family.
> Metric: mean walk-forward ΔSharpe across surface states.
> Trend/Volatility is split by feature set.

| Surface / Feature Set | EURJPY | GBPJPY | USDJPY | Mean |
|---|---|---|---|---|
| Consensus Lifecycle  [price_trend]  |  0.258  |  0.061  |  0.163  |  0.161 |
| Trend / Volatility  [price_trend]  |  0.262  |  0.063  |  0.143  |  0.156 |
| Trend / Volatility  [trend_vol_only]  |  0.283  |  0.044  |  0.141  |  0.156 |


### Per-experiment breakdown
| Surface | State | Feature Set | EURJPY | GBPJPY | USDJPY | Mean |
|---|---|---|---|---|---|---|
| cLife | JPY_CONSENSUS_YOUNG | price_trend  |  0.237  |  0.081  |  0.144  |  0.154 |
| cLife | JPY_CONSENSUS_MATURING | price_trend  |  0.254  |  0.053  |  0.187  |  0.165 |
| cLife | JPY_CONSENSUS_MATURE | price_trend  |  0.273  |  0.058  |  0.176  |  0.169 |
| cLife | JPY_NON_EXTREME | price_trend  |  0.267  |  0.052  |  0.147  |  0.155 |
| tVol | LVTF | price_trend  |  0.249  |  0.055  |  0.129  |  0.145 |
| tVol | HVTF | price_trend  |  0.189  |  0.065  |  0.119  |  0.124 |
| tVol | LVR | price_trend  |  0.306  |  0.076  |  0.182  |  0.188 |
| tVol | HVR | price_trend  |  0.301  |  0.055  |  0.141  |  0.166 |
| tVol | LVTF | trend_vol_only  |  0.286  |  0.011  |  0.117  |  0.138 |
| tVol | HVTF | trend_vol_only  |  0.278  |  0.063  |  0.103  |  0.148 |
| tVol | LVR | trend_vol_only  |  0.306  |  0.032  |  0.143  |  0.160 |
| tVol | HVR | trend_vol_only  |  0.263  |  0.070  |  0.199  |  0.177 |


---
Generated by `compare_to_baseline.py` — MPML Stage 3 OOS validator.
Validated against VALIDATION_SPEC_JPY.md (frozen June 2026).
Report format: Markdown — optimized for GitHub, Jupyter, VS Code, Obsidian.
