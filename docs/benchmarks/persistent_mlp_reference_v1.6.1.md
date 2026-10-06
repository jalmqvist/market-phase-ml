
    # MPML REFERENCE BENCHMARK

    **Architecture**: mlp  
    **Experiments**: 17  
    **Baseline**: No-DL PhaseAware (aggregate)  
    **Target pairs**: experiment-specific; matrix columns cover the union of target populations (EURAUD, EURGBP, EURUSD, GBPUSD, NZDUSD)  

    > **Sensitivity mode:** Deltas recomputed from absolute values against current baseline.  
> Δ values are walk-forward OOS deltas vs no-DL baseline.  
    > `+` = positive Sharpe uplift.  
    > For ΔDD: **smaller = better** (less drawdown).  
    > All values rounded to 3 decimals for readability.
    
## 1. Uplift Matrix — ΔRet, ΔSh, and ΔDD per State and Pair
| Architecture | Behavioral Surface | Feature Set | State | ΔRet EURAUD | ΔRet EURGBP | ΔRet EURUSD | ΔRet GBPUSD | ΔRet NZDUSD | ΔSh EURAUD | ΔSh EURGBP | ΔSh EURUSD | ΔSh GBPUSD | ΔSh NZDUSD | ΔDD EURAUD | ΔDD EURGBP | ΔDD EURUSD | ΔDD GBPUSD | ΔDD NZDUSD | Mean ΔSh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_LL  | 117.64  |  -8.97  | -20.31  |  15.50  |  92.86  |  0.281+  | -0.099  | -0.089  |  0.101+  |  0.213+  |   3.64  |  -8.45  | -16.00  |  -1.14  |   6.85  |  0.081 |
| mlp | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_LM  |  98.33  | -19.68  | -28.10  |  14.79  | 165.47  |  0.237+  | -0.160  | -0.154  |  0.097+  |  0.358+  |  -2.65  |  -9.96  | -20.29  |   0.92  |  17.26  |  0.076 |
| mlp | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_LH  | 118.20  |  -8.91  |  -4.04  |  10.29  | 178.20  |  0.278+  | -0.099  |  0.016+  |  0.075+  |  0.379+  |   5.43  |  -7.99  |   0.88  |  -1.10  |   9.28  |  0.130 |
| mlp | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_ML  | 115.57  | -15.12  | -31.22  |   2.27  | 212.22  |  0.284+  | -0.133  | -0.184  |  0.033+  |  0.413+  |   4.37  |  -8.73  | -23.13  |  -2.94  |  12.20  |  0.083 |
| mlp | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_MM  | 147.04  |  -5.52  | -29.86  |   7.83  | 272.01  |  0.332+  | -0.082  | -0.168  |  0.062+  |  0.497+  |   7.21  |  -7.85  | -22.00  |  -0.80  |  11.63  |  0.128 |
| mlp | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_MH  | 120.53  | -10.65  | -25.69  |  16.29  | 101.97  |  0.289+  | -0.108  | -0.133  |  0.104+  |  0.239+  |  -0.24  |  -8.29  | -20.00  |  -0.20  |   1.19  |  0.078 |
| mlp | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_HL  | 144.10  |  -4.99  | -18.19  |  10.23  | 194.51  |  0.351+  | -0.080  | -0.077  |  0.075+  |  0.397+  |  -4.47  |  -7.92  | -16.81  |  -2.77  |  10.54  |  0.133 |
| mlp | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_HM  | 176.61  | -16.43  | -16.06  |  10.45  | 330.73  |  0.387+  | -0.141  | -0.063  |  0.076+  |  0.561+  |   4.07  |  -7.81  | -11.11  |  -0.79  |  13.71  |  0.164 |
| mlp | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_HH  | 128.89  | -18.23  | -23.29  |  13.19  | 134.73  |  0.306+  | -0.151  | -0.112  |  0.089+  |  0.293+  |  -0.76  |  -9.25  | -14.41  |  -1.34  |   5.39  |  0.085 |
| mlp | Trend / Volatility Surface | price_trend | LVTF  | 154.83  |  -2.32  | -28.19  |  16.84  | 158.29  |  0.349+  | -0.064  | -0.152  |  0.107+  |  0.329+  |   5.20  |  -5.69  | -24.43  |  -0.15  |   1.88  |  0.114 |
| mlp | Trend / Volatility Surface | price_trend | HVTF  | 211.66  | -12.83  | -27.01  |   4.87  | 146.71  |  0.448+  | -0.120  | -0.145  |  0.046+  |  0.312+  |   5.74  |  -7.90  | -20.51  |  -1.05  |   8.82  |  0.108 |
| mlp | Trend / Volatility Surface | price_trend | LVR  | 167.31  | -11.70  | -28.88  |  14.07  | 123.72  |  0.377+  | -0.117  | -0.159  |  0.094+  |  0.289+  |   5.98  |  -9.35  | -19.60  |  -2.07  |  10.95  |  0.097 |
| mlp | Trend / Volatility Surface | price_trend | HVR  | 161.16  | -11.77  | -19.79  |   9.58  | 144.96  |  0.340+  | -0.114  | -0.088  |  0.071+  |  0.305+  |   2.54  |  -8.73  | -14.31  |  -0.70  |  13.18  |  0.103 |
| mlp | Trend / Volatility Surface | trend_vol_only | LVTF  | 230.10  |  -0.31  | -27.31  |  14.70  | 122.10  |  0.474+  | -0.056  | -0.147  |  0.097+  |  0.278+  |   6.95  |  -6.74  | -17.24  |   0.16  |   4.67  |  0.129 |
| mlp | Trend / Volatility Surface | trend_vol_only | HVTF  | 141.51  |  -7.00  | -32.22  |  11.28  | 177.63  |  0.326+  | -0.091  | -0.190  |  0.080+  |  0.376+  |   3.23  |  -8.21  | -21.82  |   0.06  |  -0.05  |  0.100 |
| mlp | Trend / Volatility Surface | trend_vol_only | LVR  | 162.12  | -17.34  |  -8.61  |   3.55  |  73.72  |  0.386+  | -0.145  | -0.016  |  0.040+  |  0.186+  |  -0.93  |  -9.76  |  -6.12  |  -1.23  |  -5.13  |  0.090 |
| mlp | Trend / Volatility Surface | trend_vol_only | HVR  | 143.23  | -12.58  | -18.06  |  17.92  | 113.27  |  0.325+  | -0.120  | -0.074  |  0.112+  |  0.255+  |   3.76  |  -8.19  | -10.25  |  -0.88  |   5.06  |  0.100 |


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
| AUDJPY | -15.94 | -0.038 | -14.49 |
| AUDUSD |  41.75 |  0.126 |  -2.64 |
| EURCHF |  16.90 |  0.087 |  -2.08 |
| EURJPY | 164.77 |  0.607 |   9.60 |
| GBPAUD |  15.48 |  0.211 |   6.64 |
| GBPJPY |  26.32 |  0.222 |   5.76 |
| USDCAD |  16.97 |  0.120 |   1.17 |
| USDCHF |  15.47 |  0.140 |  -2.39 |
| USDJPY |  69.15 |  0.251 |   5.31 |

#### Target vs negative-control separation
> Target ΔSh = mean across 5 target pairs (EURAUD, EURGBP, EURUSD, GBPUSD, NZDUSD).  Control ΔSh = mean across 9 control pairs (AUDJPY, AUDUSD, EURCHF, EURJPY, GBPAUD, GBPJPY, USDCAD, USDCHF, USDJPY).  > Separation = target ΔSh minus control ΔSh.
| State | Behavioral Surface | Feature Set | Target ΔSh | Control ΔSh | Separation |
|---|---|---|---:|---:|---:|
| PERSISTENT_HH | Persistent Commitment Lifecycle Surface | price_trend |  0.085 |  0.192 | -0.107 |
| PERSISTENT_HL | Persistent Commitment Lifecycle Surface | price_trend |  0.133 |  0.192 | -0.059 |
| PERSISTENT_HM | Persistent Commitment Lifecycle Surface | price_trend |  0.164 |  0.192 | -0.028 |
| PERSISTENT_LH | Persistent Commitment Lifecycle Surface | price_trend |  0.130 |  0.192 | -0.062 |
| PERSISTENT_LL | Persistent Commitment Lifecycle Surface | price_trend |  0.081 |  0.192 | -0.111 |
| PERSISTENT_LM | Persistent Commitment Lifecycle Surface | price_trend |  0.076 |  0.192 | -0.116 |
| PERSISTENT_MH | Persistent Commitment Lifecycle Surface | price_trend |  0.078 |  0.192 | -0.114 |
| PERSISTENT_ML | Persistent Commitment Lifecycle Surface | price_trend |  0.083 |  0.192 | -0.109 |
| PERSISTENT_MM | Persistent Commitment Lifecycle Surface | price_trend |  0.128 |  0.192 | -0.064 |
| HVR | Trend / Volatility Surface | price_trend |  0.103 |  0.192 | -0.089 |
| HVTF | Trend / Volatility Surface | price_trend |  0.108 |  0.192 | -0.084 |
| LVR | Trend / Volatility Surface | price_trend |  0.097 |  0.192 | -0.095 |
| LVTF | Trend / Volatility Surface | price_trend |  0.114 |  0.192 | -0.078 |
| HVR | Trend / Volatility Surface | trend_vol_only |  0.100 |  0.192 | -0.092 |
| HVTF | Trend / Volatility Surface | trend_vol_only |  0.100 |  0.192 | -0.092 |
| LVR | Trend / Volatility Surface | trend_vol_only |  0.090 |  0.192 | -0.102 |
| LVTF | Trend / Volatility Surface | trend_vol_only |  0.129 |  0.192 | -0.063 |


## 4. Behavioral Surface Comparison
> Compares Behavioral Surfaces within each FX pair family present in the benchmark archive.
> Metric: mean walk-forward ΔSharpe across that family's evaluated target pairs.
> Trend/Volatility is split by feature set.


### Persistent
| Surface / Feature Set | EURAUD | EURGBP | EURUSD | GBPUSD | NZDUSD | Mean |
|---|---|---|---|---|---|---|
| Persistent Commitment Lifecycle  [price_trend]  |  0.305  | -0.117  | -0.107  |  0.079  |  0.372  |  0.106 |
| Trend / Volatility  [price_trend]  |  0.379  | -0.104  | -0.136  |  0.080  |  0.309  |  0.105 |
| Trend / Volatility  [trend_vol_only]  |  0.378  | -0.103  | -0.107  |  0.082  |  0.274  |  0.105 |

#### Per-experiment breakdown
| Surface | State | Feature Set | EURAUD | EURGBP | EURUSD | GBPUSD | NZDUSD | Mean |
|---|---|---|---|---|---|---|---|---|
| pLife | PERSISTENT_LL | price_trend  |  0.281  | -0.099  | -0.089  |  0.101  |  0.213  |  0.081 |
| pLife | PERSISTENT_LM | price_trend  |  0.237  | -0.160  | -0.154  |  0.097  |  0.358  |  0.076 |
| pLife | PERSISTENT_LH | price_trend  |  0.278  | -0.099  | **0.016**  |  0.075  |  0.379  |  0.130 |
| pLife | PERSISTENT_ML | price_trend  |  0.284  | -0.133  | -0.184  |  0.033  |  0.413  |  0.083 |
| pLife | PERSISTENT_MM | price_trend  |  0.332  | -0.082  | -0.168  |  0.062  |  0.497  |  0.128 |
| pLife | PERSISTENT_MH | price_trend  |  0.289  | -0.108  | -0.133  |  0.104  |  0.239  |  0.078 |
| pLife | PERSISTENT_HL | price_trend  |  0.351  | -0.080  | -0.077  |  0.075  |  0.397  |  0.133 |
| pLife | PERSISTENT_HM | price_trend  |  0.387  | -0.141  | -0.063  |  0.076  | **0.561**  | **0.164** |
| pLife | PERSISTENT_HH | price_trend  |  0.306  | -0.151  | -0.112  |  0.089  |  0.293  |  0.085 |
| tVol | LVTF | price_trend  |  0.349  | -0.064  | -0.152  |  0.107  |  0.329  |  0.114 |
| tVol | HVTF | price_trend  |  0.448  | -0.120  | -0.145  |  0.046  |  0.312  |  0.108 |
| tVol | LVR | price_trend  |  0.377  | -0.117  | -0.159  |  0.094  |  0.289  |  0.097 |
| tVol | HVR | price_trend  |  0.340  | -0.114  | -0.088  |  0.071  |  0.305  |  0.103 |
| tVol | LVTF | trend_vol_only  | **0.474**  | **-0.056**  | -0.147  |  0.097  |  0.278  |  0.129 |
| tVol | HVTF | trend_vol_only  |  0.326  | -0.091  | -0.190  |  0.080  |  0.376  |  0.100 |
| tVol | LVR | trend_vol_only  |  0.386  | -0.145  | -0.016  |  0.040  |  0.186  |  0.090 |
| tVol | HVR | trend_vol_only  |  0.325  | -0.120  | -0.074  | **0.112**  |  0.255  |  0.100 |
> **Bold** = highest ΔSharpe in that numerical column across the per-experiment rows.



---
Generated by `compare_to_baseline.py` — MPML Stage 3 OOS validator.
Validated against the MPML benchmark validation contract.
Report format: Markdown — optimized for GitHub, Jupyter, VS Code, Obsidian.
