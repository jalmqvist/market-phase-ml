
    # MPML REFERENCE BENCHMARK

    **Architecture**: lstm  
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
| lstm | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_LL  |   0.94  |  -0.32  |  -1.52  |  -0.06  |  -0.19  |  0.229+  | -0.115  | -0.427  |  0.163+  | -0.024  |  -0.58  |  -0.65  |  -1.07  |  -0.81  |  -1.27  | -0.035 |
| lstm | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_LM  |   1.09  |  -0.35  |  -1.58  |  -0.19  |  -0.23  |  0.276+  | -0.122  | -0.399  |  0.136+  |  0.007+  |  -0.56  |  -0.69  |  -1.23  |  -0.88  |  -1.29  | -0.020 |
| lstm | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_LH  |   0.98  |  -0.28  |  -1.41  |  -0.22  |  -0.25  |  0.251+  | -0.082  | -0.345  |  0.122+  | -0.052  |  -0.60  |  -0.63  |  -1.09  |  -0.88  |  -1.17  | -0.021 |
| lstm | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_ML  |   0.92  |  -0.27  |  -1.76  |  -0.12  |  -0.26  |  0.231+  | -0.077  | -0.501  |  0.147+  | -0.037  |  -0.61  |  -0.65  |  -1.33  |  -0.88  |  -1.31  | -0.047 |
| lstm | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_MM  |   0.98  |  -0.26  |  -1.35  |  -0.19  |  -0.47  |  0.251+  | -0.090  | -0.377  |  0.143+  | -0.092  |  -0.56  |  -0.68  |  -1.07  |  -0.89  |  -1.42  | -0.033 |
| lstm | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_MH  |   1.04  |  -0.29  |  -1.54  |  -0.18  |  -0.04  |  0.275+  | -0.100  | -0.383  |  0.132+  |  0.028+  |  -0.57  |  -0.65  |  -1.14  |  -0.85  |  -1.13  | -0.009 |
| lstm | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_HL  |   0.98  |  -0.21  |  -1.35  |  -0.19  |  -0.24  |  0.240+  | -0.063  | -0.292  |  0.132+  | -0.030  |  -0.60  |  -0.61  |  -1.06  |  -0.86  |  -1.28  | -0.002 |
| lstm | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_HM  |   0.90  |  -0.22  |  -1.42  |  -0.14  |  -0.26  |  0.212+  | -0.065  | -0.402  |  0.140+  |  0.009+  |  -0.55  |  -0.64  |  -1.11  |  -0.82  |  -1.31  | -0.021 |
| lstm | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_HH  |   1.08  |  -0.29  |  -1.49  |  -0.05  |  -0.27  |  0.273+  | -0.100  | -0.393  |  0.174+  |  0.019+  |  -0.60  |  -0.69  |  -1.12  |  -0.84  |  -1.16  | -0.005 |
| lstm | Trend / Volatility Surface | price_trend | LVTF  |   0.98  |  -0.37  |  -1.60  |  -0.30  |  -0.05  |  0.222+  | -0.125  | -0.408  |  0.076+  |  0.003+  |  -0.58  |  -0.68  |  -1.22  |  -0.97  |  -1.15  | -0.046 |
| lstm | Trend / Volatility Surface | price_trend | HVTF  |   1.02  |  -0.19  |  -1.42  |  -0.34  |  -0.39  |  0.248+  | -0.050  | -0.353  |  0.067+  | -0.105  |  -0.60  |  -0.60  |  -1.13  |  -0.93  |  -1.19  | -0.039 |
| lstm | Trend / Volatility Surface | price_trend | LVR  |   0.94  |  -0.23  |  -1.42  |  -0.25  |  -0.23  |  0.245+  | -0.065  | -0.400  |  0.120+  | -0.034  |  -0.61  |  -0.68  |  -1.17  |  -0.92  |  -1.24  | -0.027 |
| lstm | Trend / Volatility Surface | price_trend | HVR  |   0.97  |  -0.36  |  -1.40  |   0.04  |  -0.15  |  0.243+  | -0.142  | -0.336  |  0.218+  | -0.013  |  -0.61  |  -0.70  |  -1.07  |  -0.74  |  -1.17  | -0.006 |
| lstm | Trend / Volatility Surface | trend_vol_only | LVTF  |   1.06  |  -0.30  |  -1.28  |  -0.13  |  -0.10  |  0.261+  | -0.101  | -0.289  |  0.143+  | -0.010  |  -0.60  |  -0.67  |  -1.02  |  -0.89  |  -1.10  |  0.001 |
| lstm | Trend / Volatility Surface | trend_vol_only | HVTF  |   0.97  |  -0.20  |  -1.29  |  -0.11  |  -0.36  |  0.247+  | -0.052  | -0.296  |  0.158+  | -0.039  |  -0.60  |  -0.58  |  -1.04  |  -0.81  |  -1.29  |  0.004 |
| lstm | Trend / Volatility Surface | trend_vol_only | LVR  |   0.92  |  -0.25  |  -1.35  |  -0.10  |   0.02  |  0.241+  | -0.075  | -0.288  |  0.176+  |  0.023+  |  -0.63  |  -0.66  |  -1.11  |  -0.89  |  -1.17  |  0.015 |
| lstm | Trend / Volatility Surface | trend_vol_only | HVR  |   0.98  |  -0.36  |  -1.38  |  -0.02  |  -0.20  |  0.253+  | -0.138  | -0.317  |  0.193+  |  0.044+  |  -0.57  |  -0.71  |  -1.09  |  -0.73  |  -1.22  |  0.007 |


## 2. Internal MPML Improvement — Dynamic Selector
> Dynamic selector improvement over the static PhaseAware baseline.
> All 14 baseline-universe pairs shown. Target membership is defined by the experiment's FX pair family.

### Persistent Commitment Lifecycle Surface — PERSISTENT_LL — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 152.04 |  0.354 |   2.78 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * | -13.99 | -0.127 |  -8.41 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * |  -7.80 | -0.004 |  -8.02 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |   6.67 |  0.056 |  -2.52 |
| NZDUSD * | 325.86 |  0.565 |  22.30 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Persistent Commitment Lifecycle Surface — PERSISTENT_LM — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 101.87 |  0.222 |   3.98 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * | -15.92 | -0.139 |  -9.06 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -17.26 | -0.067 | -14.59 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |  10.57 |  0.076 |   0.30 |
| NZDUSD * | 187.02 |  0.382 |  18.29 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Persistent Commitment Lifecycle Surface — PERSISTENT_LH — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 166.67 |  0.374 |   4.66 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * |  -6.81 | -0.090 |  -7.99 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -30.18 | -0.173 | -20.10 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |   4.85 |  0.047 |  -5.66 |
| NZDUSD * | 192.96 |  0.406 |  10.29 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Persistent Commitment Lifecycle Surface — PERSISTENT_ML — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 176.57 |  0.379 |   7.09 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * |  -3.89 | -0.075 |  -7.14 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -37.06 | -0.235 | -26.41 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |  21.11 |  0.127 |   0.50 |
| NZDUSD * | 195.75 |  0.405 |   6.81 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Persistent Commitment Lifecycle Surface — PERSISTENT_MM — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 145.51 |  0.359 |  -0.37 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * |  -2.12 | -0.065 |  -6.87 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -25.58 | -0.133 | -16.98 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |  12.30 |  0.085 |  -0.08 |
| NZDUSD * | 278.42 |  0.502 |  17.34 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Persistent Commitment Lifecycle Surface — PERSISTENT_MH — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 155.06 |  0.337 |   7.14 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * |  -9.76 | -0.104 | -10.31 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -18.77 | -0.082 | -11.64 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |  13.57 |  0.091 |  -0.56 |
| NZDUSD * | 136.83 |  0.295 |   7.46 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Persistent Commitment Lifecycle Surface — PERSISTENT_HL — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 107.72 |  0.259 |   2.45 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * |  -1.85 | -0.064 |  -9.84 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -24.94 | -0.127 | -16.42 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |  13.71 |  0.092 |   0.68 |
| NZDUSD * | 164.83 |  0.337 |   5.02 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Persistent Commitment Lifecycle Surface — PERSISTENT_HM — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 123.82 |  0.301 |   5.33 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * | -15.07 | -0.134 | -10.28 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -28.23 | -0.155 | -18.94 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |   2.04 |  0.031 |  -4.53 |
| NZDUSD * | 161.36 |  0.355 |  16.45 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Persistent Commitment Lifecycle Surface — PERSISTENT_HH — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 154.43 |  0.349 |   3.64 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * |  -4.31 | -0.075 |  -7.12 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -29.30 | -0.165 | -19.97 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |   8.54 |  0.066 |   0.49 |
| NZDUSD * | 113.97 |  0.268 |  12.79 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Trend / Volatility Surface — LVTF — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 143.03 |  0.326 |   5.96 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * |  -9.76 | -0.103 |  -7.94 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * |  -8.33 | -0.012 |  -9.52 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |  14.08 |  0.094 |  -0.74 |
| NZDUSD * | 142.45 |  0.306 |   7.99 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Trend / Volatility Surface — LVTF — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * |  78.43 |  0.188 |  -4.89 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * | -10.99 | -0.111 |  -8.01 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -22.49 | -0.107 | -18.84 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |   0.20 |  0.022 |  -4.90 |
| NZDUSD * | 146.45 |  0.312 |   8.72 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Trend / Volatility Surface — HVTF — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * |  91.26 |  0.225 |  -3.05 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * | -20.11 | -0.162 |  -8.38 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -18.31 | -0.082 | -14.50 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |  17.59 |  0.111 |  -0.55 |
| NZDUSD * | 132.86 |  0.304 |   7.99 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Trend / Volatility Surface — HVTF — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 106.06 |  0.258 |  -4.21 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * | -11.82 | -0.115 |  -7.71 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -29.28 | -0.166 | -20.91 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |   3.36 |  0.039 |  -1.37 |
| NZDUSD * | 152.26 |  0.343 |  15.05 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Trend / Volatility Surface — LVR — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * |  94.01 |  0.228 |  -3.41 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * | -14.08 | -0.128 |  -7.57 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -19.92 | -0.089 | -14.24 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |  11.98 |  0.083 |  -0.09 |
| NZDUSD * | 235.17 |  0.459 |  11.60 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Trend / Volatility Surface — LVR — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 180.35 |  0.417 |   4.48 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * |  -8.29 | -0.094 |  -4.97 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -21.82 | -0.106 | -13.54 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |   6.49 |  0.055 |  -6.03 |
| NZDUSD * | 144.57 |  0.315 |   9.32 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Trend / Volatility Surface — HVR — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 156.64 |  0.360 |   5.05 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * | -12.70 | -0.119 |  -7.63 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -30.17 | -0.173 | -20.86 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |   7.43 |  0.060 |  -1.71 |
| NZDUSD * | 148.57 |  0.332 |   5.83 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |


### Trend / Volatility Surface — HVR — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURAUD * | 200.24 |  0.448 |   4.41 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURGBP * |  -3.98 | -0.077 |  -7.54 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| EURUSD * | -20.08 | -0.087 | -17.33 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| GBPUSD * |   1.57 |  0.029 |  -2.34 |
| NZDUSD * | 127.59 |  0.293 |   5.12 |
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
| PERSISTENT_HH | Persistent Commitment Lifecycle Surface | price_trend | -0.005 |  0.009 | -0.014 |
| PERSISTENT_HL | Persistent Commitment Lifecycle Surface | price_trend | -0.002 |  0.009 | -0.011 |
| PERSISTENT_HM | Persistent Commitment Lifecycle Surface | price_trend | -0.021 |  0.009 | -0.030 |
| PERSISTENT_LH | Persistent Commitment Lifecycle Surface | price_trend | -0.021 |  0.009 | -0.030 |
| PERSISTENT_LL | Persistent Commitment Lifecycle Surface | price_trend | -0.035 |  0.009 | -0.044 |
| PERSISTENT_LM | Persistent Commitment Lifecycle Surface | price_trend | -0.020 |  0.009 | -0.029 |
| PERSISTENT_MH | Persistent Commitment Lifecycle Surface | price_trend | -0.009 |  0.009 | -0.018 |
| PERSISTENT_ML | Persistent Commitment Lifecycle Surface | price_trend | -0.047 |  0.009 | -0.056 |
| PERSISTENT_MM | Persistent Commitment Lifecycle Surface | price_trend | -0.033 |  0.009 | -0.042 |
| HVR | Trend / Volatility Surface | price_trend | -0.006 |  0.009 | -0.015 |
| HVTF | Trend / Volatility Surface | price_trend | -0.039 |  0.009 | -0.048 |
| LVR | Trend / Volatility Surface | price_trend | -0.027 |  0.009 | -0.036 |
| LVTF | Trend / Volatility Surface | price_trend | -0.046 |  0.009 | -0.055 |
| HVR | Trend / Volatility Surface | trend_vol_only |  0.007 |  0.009 | -0.002 |
| HVTF | Trend / Volatility Surface | trend_vol_only |  0.004 |  0.009 | -0.005 |
| LVR | Trend / Volatility Surface | trend_vol_only |  0.015 |  0.009 |  0.006 |
| LVTF | Trend / Volatility Surface | trend_vol_only |  0.001 |  0.009 | -0.008 |


## 4. Behavioral Surface Comparison
> Compares Behavioral Surfaces within each FX pair family present in the benchmark archive.
> Metric: mean walk-forward ΔSharpe across that family's evaluated target pairs.
> Trend/Volatility is split by feature set.


### Persistent
| Surface / Feature Set | EURAUD | EURGBP | EURUSD | GBPUSD | NZDUSD | Mean |
|---|---|---|---|---|---|---|
| Persistent Commitment Lifecycle  [price_trend]  |  0.249  | -0.090  | -0.391  |  0.143  | -0.019  | -0.022 |
| Trend / Volatility  [price_trend]  |  0.239  | -0.095  | -0.374  |  0.120  | -0.037  | -0.029 |
| Trend / Volatility  [trend_vol_only]  |  0.251  | -0.091  | -0.298  |  0.168  |  0.005  |  0.007 |

#### Per-experiment breakdown
| Surface | State | Feature Set | EURAUD | EURGBP | EURUSD | GBPUSD | NZDUSD | Mean |
|---|---|---|---|---|---|---|---|---|
| pLife | PERSISTENT_LL | price_trend  |  0.229  | -0.115  | -0.427  |  0.163  | -0.024  | -0.035 |
| pLife | PERSISTENT_LM | price_trend  | **0.276**  | -0.122  | -0.399  |  0.136  |  0.007  | -0.020 |
| pLife | PERSISTENT_LH | price_trend  |  0.251  | -0.082  | -0.345  |  0.122  | -0.052  | -0.021 |
| pLife | PERSISTENT_ML | price_trend  |  0.231  | -0.077  | -0.501  |  0.147  | -0.037  | -0.047 |
| pLife | PERSISTENT_MM | price_trend  |  0.251  | -0.090  | -0.377  |  0.143  | -0.092  | -0.033 |
| pLife | PERSISTENT_MH | price_trend  |  0.275  | -0.100  | -0.383  |  0.132  |  0.028  | -0.009 |
| pLife | PERSISTENT_HL | price_trend  |  0.240  | -0.063  | -0.292  |  0.132  | -0.030  | -0.002 |
| pLife | PERSISTENT_HM | price_trend  |  0.212  | -0.065  | -0.402  |  0.140  |  0.009  | -0.021 |
| pLife | PERSISTENT_HH | price_trend  |  0.273  | -0.100  | -0.393  |  0.174  |  0.019  | -0.005 |
| tVol | LVTF | price_trend  |  0.222  | -0.125  | -0.408  |  0.076  |  0.003  | -0.046 |
| tVol | HVTF | price_trend  |  0.248  | **-0.050**  | -0.353  |  0.067  | -0.105  | -0.039 |
| tVol | LVR | price_trend  |  0.245  | -0.065  | -0.400  |  0.120  | -0.034  | -0.027 |
| tVol | HVR | price_trend  |  0.243  | -0.142  | -0.336  | **0.218**  | -0.013  | -0.006 |
| tVol | LVTF | trend_vol_only  |  0.261  | -0.101  | -0.289  |  0.143  | -0.010  |  0.001 |
| tVol | HVTF | trend_vol_only  |  0.247  | -0.052  | -0.296  |  0.158  | -0.039  |  0.004 |
| tVol | LVR | trend_vol_only  |  0.241  | -0.075  | **-0.288**  |  0.176  |  0.023  | **0.015** |
| tVol | HVR | trend_vol_only  |  0.253  | -0.138  | -0.317  |  0.193  | **0.044**  |  0.007 |
> **Bold** = highest ΔSharpe in that numerical column across the per-experiment rows.



---
Generated by `compare_to_baseline.py` — MPML Stage 3 OOS validator.
Validated against the MPML benchmark validation contract.
Report format: Markdown — optimized for GitHub, Jupyter, VS Code, Obsidian.
