
    # MPML REFERENCE BENCHMARK

    **Architecture**: mlp  
    **Experiments**: 17  
    **Baseline**: No-DL PhaseAware (aggregate)  
    **Target pairs**: experiment-specific; matrix columns cover the union of target populations (EURAUD, EURGBP, EURUSD, GBPUSD, NZDUSD)  

    > Δ values are walk-forward OOS deltas vs no-DL baseline.  
    > `+` = positive Sharpe uplift.  
    > For ΔDD: **smaller = better** (less drawdown).  
    > All values rounded to 3 decimals for readability.
    
## 1. Uplift Matrix — ΔRet, ΔSh, and ΔDD per State and Pair
| Architecture | Behavioral Surface | Feature Set | State | ΔRet EURAUD | ΔRet EURGBP | ΔRet EURUSD | ΔRet GBPUSD | ΔRet NZDUSD | ΔSh EURAUD | ΔSh EURGBP | ΔSh EURUSD | ΔSh GBPUSD | ΔSh NZDUSD | ΔDD EURAUD | ΔDD EURGBP | ΔDD EURUSD | ΔDD GBPUSD | ΔDD NZDUSD | Mean ΔSh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_LL  |   0.98  |  -0.29  |  -1.36  |  -0.12  |  -0.26  |  0.248+  | -0.099  | -0.357  |  0.146+  | -0.045  |  -0.58  |  -0.66  |  -1.06  |  -0.81  |  -1.09  | -0.021 |
| mlp | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_LM  |   0.99  |  -0.33  |  -1.54  |  -0.18  |  -0.42  |  0.249+  | -0.125  | -0.425  |  0.134+  | -0.095  |  -0.57  |  -0.68  |  -1.08  |  -0.86  |  -1.32  | -0.052 |
| mlp | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_LH  |   0.95  |  -0.27  |  -1.46  |  -0.20  |  -0.38  |  0.236+  | -0.083  | -0.352  |  0.098+  | -0.057  |  -0.57  |  -0.64  |  -1.13  |  -0.86  |  -1.24  | -0.032 |
| mlp | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_ML  |   1.10  |  -0.19  |  -1.25  |  -0.15  |  -0.37  |  0.278+  | -0.035  | -0.298  |  0.161+  | -0.049  |  -0.55  |  -0.57  |  -1.02  |  -0.90  |  -1.25  |  0.011 |
| mlp | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_MM  |   1.01  |  -0.30  |  -1.44  |  -0.11  |  -0.24  |  0.244+  | -0.098  | -0.379  |  0.177+  | -0.039  |  -0.56  |  -0.64  |  -1.12  |  -0.89  |  -1.26  | -0.019 |
| mlp | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_MH  |   0.99  |  -0.23  |  -1.67  |  -0.10  |  -0.18  |  0.236+  | -0.069  | -0.455  |  0.180+  | -0.032  |  -0.60  |  -0.66  |  -1.28  |  -0.92  |  -1.19  | -0.028 |
| mlp | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_HL  |   0.90  |  -0.26  |  -1.44  |  -0.33  |   0.02  |  0.220+  | -0.081  | -0.373  |  0.050+  |  0.061+  |  -0.64  |  -0.60  |  -1.10  |  -1.02  |  -1.21  | -0.025 |
| mlp | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_HM  |   0.98  |  -0.26  |  -1.39  |  -0.10  |  -0.01  |  0.251+  | -0.078  | -0.309  |  0.141+  | -0.025  |  -0.56  |  -0.64  |  -1.05  |  -0.92  |  -1.04  | -0.004 |
| mlp | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_HH  |   1.07  |  -0.31  |  -1.52  |  -0.10  |  -0.38  |  0.281+  | -0.100  | -0.366  |  0.165+  | -0.062  |  -0.53  |  -0.68  |  -1.25  |  -0.88  |  -1.34  | -0.016 |
| mlp | Trend / Volatility Surface | price_trend | LVTF  |   0.96  |  -0.28  |  -1.20  |   0.03  |  -0.32  |  0.236+  | -0.074  | -0.243  |  0.178+  | -0.076  |  -0.57  |  -0.62  |  -0.90  |  -0.73  |  -1.38  |  0.004 |
| mlp | Trend / Volatility Surface | price_trend | HVTF  |   1.04  |  -0.26  |  -1.26  |  -0.41  |  -0.31  |  0.252+  | -0.082  | -0.295  |  0.063+  | -0.043  |  -0.56  |  -0.58  |  -0.97  |  -1.00  |  -1.31  | -0.021 |
| mlp | Trend / Volatility Surface | price_trend | LVR  |   0.88  |  -0.24  |  -1.44  |  -0.41  |  -0.23  |  0.216+  | -0.062  | -0.367  |  0.056+  | -0.031  |  -0.60  |  -0.64  |  -1.07  |  -1.05  |  -1.22  | -0.037 |
| mlp | Trend / Volatility Surface | price_trend | HVR  |   1.05  |  -0.29  |  -1.39  |  -0.14  |  -0.38  |  0.276+  | -0.104  | -0.331  |  0.157+  | -0.060  |  -0.60  |  -0.66  |  -1.08  |  -0.87  |  -1.33  | -0.012 |
| mlp | Trend / Volatility Surface | trend_vol_only | LVTF  |   1.20  |  -0.24  |  -1.26  |  -0.15  |  -0.13  |  0.300+  | -0.083  | -0.316  |  0.145+  | -0.022  |  -0.49  |  -0.60  |  -1.04  |  -0.93  |  -1.20  |  0.005 |
| mlp | Trend / Volatility Surface | trend_vol_only | HVTF  |   1.07  |  -0.24  |  -1.43  |  -0.22  |  -0.20  |  0.256+  | -0.074  | -0.362  |  0.102+  | -0.031  |  -0.56  |  -0.64  |  -1.14  |  -0.90  |  -1.16  | -0.022 |
| mlp | Trend / Volatility Surface | trend_vol_only | LVR  |   0.98  |  -0.33  |  -1.31  |  -0.20  |  -0.22  |  0.246+  | -0.112  | -0.317  |  0.102+  | -0.043  |  -0.59  |  -0.68  |  -1.02  |  -0.83  |  -1.29  | -0.025 |
| mlp | Trend / Volatility Surface | trend_vol_only | HVR  |   0.97  |  -0.29  |  -1.48  |   0.03  |  -0.17  |  0.236+  | -0.109  | -0.309  |  0.215+  | -0.020  |  -0.60  |  -0.68  |  -1.19  |  -0.77  |  -1.25  |  0.002 |


## 2. Internal MPML Improvement — Dynamic Selector
> Dynamic selector improvement over the static PhaseAware baseline.
> All 14 baseline-universe pairs shown. Target membership is defined by the experiment's FX pair family.

### Persistent Commitment Lifecycle Surface — PERSISTENT_LL — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 117.64 |  0.281 |   3.64 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * |  -8.97 | -0.099 |  -8.45 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -20.31 | -0.089 | -16.00 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |  15.50 |  0.101 |  -1.14 |
| NZDUSD * |  92.86 |  0.213 |   6.85 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Persistent Commitment Lifecycle Surface — PERSISTENT_LM — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * |  98.33 |  0.237 |  -2.65 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * | -19.68 | -0.160 |  -9.96 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -28.10 | -0.154 | -20.29 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |  14.79 |  0.097 |   0.92 |
| NZDUSD * | 165.47 |  0.358 |  17.26 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Persistent Commitment Lifecycle Surface — PERSISTENT_LH — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 118.20 |  0.278 |   5.43 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * |  -8.91 | -0.099 |  -7.99 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * |  -4.04 |  0.016 |   0.88 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |  10.29 |  0.075 |  -1.10 |
| NZDUSD * | 178.20 |  0.379 |   9.28 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Persistent Commitment Lifecycle Surface — PERSISTENT_ML — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 115.57 |  0.284 |   4.37 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * | -15.12 | -0.133 |  -8.73 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -31.22 | -0.184 | -23.13 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |   2.27 |  0.033 |  -2.94 |
| NZDUSD * | 212.22 |  0.413 |  12.20 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Persistent Commitment Lifecycle Surface — PERSISTENT_MM — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 147.04 |  0.332 |   7.21 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * |  -5.52 | -0.082 |  -7.85 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -29.86 | -0.168 | -22.00 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |   7.83 |  0.062 |  -0.80 |
| NZDUSD * | 272.01 |  0.497 |  11.63 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Persistent Commitment Lifecycle Surface — PERSISTENT_MH — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 120.53 |  0.289 |  -0.24 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * | -10.65 | -0.108 |  -8.29 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -25.69 | -0.133 | -20.00 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |  16.29 |  0.104 |  -0.20 |
| NZDUSD * | 101.97 |  0.239 |   1.19 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Persistent Commitment Lifecycle Surface — PERSISTENT_HL — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 144.10 |  0.351 |  -4.47 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * |  -4.99 | -0.080 |  -7.92 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -18.19 | -0.077 | -16.81 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |  10.23 |  0.075 |  -2.77 |
| NZDUSD * | 194.51 |  0.397 |  10.54 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Persistent Commitment Lifecycle Surface — PERSISTENT_HM — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 176.61 |  0.387 |   4.07 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * | -16.43 | -0.141 |  -7.81 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -16.06 | -0.063 | -11.11 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |  10.45 |  0.076 |  -0.79 |
| NZDUSD * | 330.73 |  0.561 |  13.71 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Persistent Commitment Lifecycle Surface — PERSISTENT_HH — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 128.89 |  0.306 |  -0.76 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * | -18.23 | -0.151 |  -9.25 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -23.29 | -0.112 | -14.41 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |  13.19 |  0.089 |  -1.34 |
| NZDUSD * | 134.73 |  0.293 |   5.39 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Trend / Volatility Surface — LVTF — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 154.83 |  0.349 |   5.20 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * |  -2.32 | -0.064 |  -5.69 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -28.19 | -0.152 | -24.43 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |  16.84 |  0.107 |  -0.15 |
| NZDUSD * | 158.29 |  0.329 |   1.88 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Trend / Volatility Surface — LVTF — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 230.10 |  0.474 |   6.95 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * |  -0.31 | -0.056 |  -6.74 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -27.31 | -0.146 | -17.24 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |  14.70 |  0.097 |   0.16 |
| NZDUSD * | 122.10 |  0.278 |   4.67 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Trend / Volatility Surface — HVTF — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 211.66 |  0.448 |   5.74 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * | -12.83 | -0.120 |  -7.90 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -27.01 | -0.145 | -20.51 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |   4.87 |  0.046 |  -1.05 |
| NZDUSD * | 146.71 |  0.312 |   8.82 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Trend / Volatility Surface — HVTF — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 141.51 |  0.326 |   3.23 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * |  -7.00 | -0.091 |  -8.21 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -32.22 | -0.190 | -21.82 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |  11.28 |  0.080 |   0.06 |
| NZDUSD * | 177.63 |  0.376 |  -0.05 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Trend / Volatility Surface — LVR — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 167.31 |  0.377 |   5.98 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * | -11.70 | -0.117 |  -9.35 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -28.88 | -0.159 | -19.60 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |  14.07 |  0.094 |  -2.07 |
| NZDUSD * | 123.72 |  0.289 |  10.95 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Trend / Volatility Surface — LVR — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 162.12 |  0.386 |  -0.93 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * | -17.34 | -0.145 |  -9.76 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * |  -8.61 | -0.016 |  -6.12 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |   3.55 |  0.040 |  -1.23 |
| NZDUSD * |  73.72 |  0.186 |  -5.13 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Trend / Volatility Surface — HVR — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 161.16 |  0.340 |   2.54 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * | -11.77 | -0.114 |  -8.73 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -19.79 | -0.088 | -14.31 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |   9.58 |  0.071 |  -0.70 |
| NZDUSD * | 144.96 |  0.305 |  13.18 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Trend / Volatility Surface — HVR — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 143.23 |  0.325 |   3.76 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * | -12.58 | -0.120 |  -8.19 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -18.06 | -0.074 | -10.25 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |  17.92 |  0.112 |  -0.88 |
| NZDUSD * | 113.27 |  0.255 |   5.06 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


## 3. Target Family vs Negative Controls
> Control statistics are summarized separately for each FX pair family.
> Target pairs are defined by the experiment's pair family; all other baseline-universe pairs are controls.
> Separation = mean target ΔSharpe minus mean control ΔSharpe.

### Control FX pairs — outside the Persistent target family
> These 9 FX pairs are outside the Persistent target population and serve as negative controls.

| Pair | Mean ΔReturn | Mean ΔSharpe | Mean ΔDD |
|---|---:|---:|---:|
| AUDJPY |  -0.14 | -0.228 |  -0.74 |
| AUDUSD |  -1.01 | -0.278 |  -0.91 |
| EURCHF |   0.44 |  0.326 |  -0.30 |
| EURJPY |   0.25 |  0.010 |  -0.18 |
| GBPAUD |   0.57 |  0.102 |  -1.17 |
| GBPJPY |  -0.43 | -0.139 |  -0.53 |
| USDCAD |   0.20 |  0.094 |  -0.28 |
| USDCHF |   0.44 |  0.212 |  -0.42 |
| USDJPY |   0.12 | -0.018 |  -0.39 |

#### Target vs negative-control separation
> Target ΔSh = mean across 5 target pairs (EURAUD, EURGBP, EURUSD, GBPUSD, NZDUSD).  Control ΔSh = mean across 9 control pairs (AUDJPY, AUDUSD, EURCHF, EURJPY, GBPAUD, GBPJPY, USDCAD, USDCHF, USDJPY).  > Separation = target ΔSh minus control ΔSh.
| State | Behavioral Surface | Feature Set | Target ΔSh | Control ΔSh | Separation |
|---|---|---|---:|---:|---:|
| PERSISTENT_HH | Persistent Commitment Lifecycle Surface | price_trend | -0.016 |  0.009 | -0.025 |
| PERSISTENT_HL | Persistent Commitment Lifecycle Surface | price_trend | -0.025 |  0.009 | -0.034 |
| PERSISTENT_HM | Persistent Commitment Lifecycle Surface | price_trend | -0.004 |  0.009 | -0.013 |
| PERSISTENT_LH | Persistent Commitment Lifecycle Surface | price_trend | -0.032 |  0.009 | -0.041 |
| PERSISTENT_LL | Persistent Commitment Lifecycle Surface | price_trend | -0.021 |  0.009 | -0.030 |
| PERSISTENT_LM | Persistent Commitment Lifecycle Surface | price_trend | -0.052 |  0.009 | -0.061 |
| PERSISTENT_MH | Persistent Commitment Lifecycle Surface | price_trend | -0.028 |  0.009 | -0.037 |
| PERSISTENT_ML | Persistent Commitment Lifecycle Surface | price_trend |  0.011 |  0.009 |  0.002 |
| PERSISTENT_MM | Persistent Commitment Lifecycle Surface | price_trend | -0.019 |  0.009 | -0.028 |
| HVR | Trend / Volatility Surface | price_trend | -0.012 |  0.009 | -0.021 |
| HVTF | Trend / Volatility Surface | price_trend | -0.021 |  0.009 | -0.030 |
| LVR | Trend / Volatility Surface | price_trend | -0.037 |  0.009 | -0.046 |
| LVTF | Trend / Volatility Surface | price_trend |  0.004 |  0.009 | -0.005 |
| HVR | Trend / Volatility Surface | trend_vol_only |  0.002 |  0.009 | -0.006 |
| HVTF | Trend / Volatility Surface | trend_vol_only | -0.022 |  0.009 | -0.031 |
| LVR | Trend / Volatility Surface | trend_vol_only | -0.025 |  0.009 | -0.034 |
| LVTF | Trend / Volatility Surface | trend_vol_only |  0.005 |  0.009 | -0.004 |


## 4. Behavioral Surface Comparison
> Compares Behavioral Surfaces within each FX pair family present in the benchmark archive.
> Metric: mean walk-forward ΔSharpe across that family's evaluated target pairs.
> Trend/Volatility is split by feature set.


### Persistent
| Surface / Feature Set | EURAUD | EURGBP | EURUSD | GBPUSD | NZDUSD | Mean |
|---|---|---|---|---|---|---|
| Persistent Commitment Lifecycle  [price_trend]  |  0.249  | -0.085  | -0.368  |  0.139  | -0.038  | -0.021 |
| Trend / Volatility  [price_trend]  |  0.245  | -0.081  | -0.309  |  0.114  | -0.052  | -0.017 |
| Trend / Volatility  [trend_vol_only]  |  0.259  | -0.094  | -0.326  |  0.141  | -0.029  | -0.010 |

#### Per-experiment breakdown
| Surface | State | Feature Set | EURAUD | EURGBP | EURUSD | GBPUSD | NZDUSD | Mean |
|---|---|---|---|---|---|---|---|---|
| pLife | PERSISTENT_LL | price_trend  |  0.248  | -0.099  | -0.357  |  0.146  | -0.045  | -0.021 |
| pLife | PERSISTENT_LM | price_trend  |  0.249  | -0.125  | -0.425  |  0.134  | -0.095  | -0.052 |
| pLife | PERSISTENT_LH | price_trend  |  0.236  | -0.083  | -0.352  |  0.098  | -0.057  | -0.032 |
| pLife | PERSISTENT_ML | price_trend  |  0.278  | **-0.035**  | -0.298  |  0.161  | -0.049  | **0.011** |
| pLife | PERSISTENT_MM | price_trend  |  0.244  | -0.098  | -0.379  |  0.177  | -0.039  | -0.019 |
| pLife | PERSISTENT_MH | price_trend  |  0.236  | -0.069  | -0.455  |  0.180  | -0.032  | -0.028 |
| pLife | PERSISTENT_HL | price_trend  |  0.220  | -0.081  | -0.373  |  0.050  | **0.061**  | -0.025 |
| pLife | PERSISTENT_HM | price_trend  |  0.251  | -0.078  | -0.309  |  0.141  | -0.025  | -0.004 |
| pLife | PERSISTENT_HH | price_trend  |  0.281  | -0.100  | -0.366  |  0.165  | -0.062  | -0.016 |
| tVol | LVTF | price_trend  |  0.236  | -0.074  | **-0.243**  |  0.178  | -0.076  |  0.004 |
| tVol | HVTF | price_trend  |  0.252  | -0.082  | -0.295  |  0.063  | -0.043  | -0.021 |
| tVol | LVR | price_trend  |  0.216  | -0.062  | -0.367  |  0.056  | -0.031  | -0.037 |
| tVol | HVR | price_trend  |  0.276  | -0.104  | -0.331  |  0.157  | -0.060  | -0.012 |
| tVol | LVTF | trend_vol_only  | **0.300**  | -0.083  | -0.316  |  0.145  | -0.022  |  0.005 |
| tVol | HVTF | trend_vol_only  |  0.256  | -0.074  | -0.362  |  0.102  | -0.031  | -0.022 |
| tVol | LVR | trend_vol_only  |  0.246  | -0.112  | -0.317  |  0.102  | -0.043  | -0.025 |
| tVol | HVR | trend_vol_only  |  0.236  | -0.109  | -0.309  | **0.215**  | -0.020  |  0.002 |
> **Bold** = highest ΔSharpe in that numerical column across the per-experiment rows.



---
Generated by `compare_to_baseline.py` — MPML Stage 3 OOS validator.
Validated against the MPML benchmark validation contract.
Report format: Markdown — optimized for GitHub, Jupyter, VS Code, Obsidian.
