
    # MPML REFERENCE BENCHMARK

    **Architecture**: lstm  
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
| lstm | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_LL  | 152.04  | -13.99  |  -7.80  |   6.67  | 325.86  |  0.354+  | -0.127  | -0.004  |  0.056+  |  0.565+  |   2.78  |  -8.41  |  -8.02  |  -2.52  |  22.30  |  0.169 |
| lstm | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_LM  | 101.87  | -15.92  | -17.26  |  10.57  | 187.02  |  0.222+  | -0.139  | -0.067  |  0.076+  |  0.382+  |   3.98  |  -9.06  | -14.59  |   0.30  |  18.29  |  0.095 |
| lstm | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_LH  | 166.67  |  -6.81  | -30.18  |   4.85  | 192.96  |  0.374+  | -0.090  | -0.173  |  0.047+  |  0.406+  |   4.66  |  -7.99  | -20.10  |  -5.66  |  10.29  |  0.113 |
| lstm | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_ML  | 176.57  |  -3.89  | -37.06  |  21.11  | 195.75  |  0.379+  | -0.075  | -0.235  |  0.127+  |  0.405+  |   7.09  |  -7.14  | -26.41  |   0.50  |   6.81  |  0.120 |
| lstm | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_MM  | 145.51  |  -2.12  | -25.58  |  12.30  | 278.42  |  0.359+  | -0.065  | -0.133  |  0.085+  |  0.502+  |  -0.37  |  -6.87  | -16.98  |  -0.08  |  17.34  |  0.150 |
| lstm | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_MH  | 155.06  |  -9.76  | -18.77  |  13.57  | 136.83  |  0.337+  | -0.104  | -0.082  |  0.091+  |  0.295+  |   7.14  | -10.31  | -11.64  |  -0.56  |   7.46  |  0.108 |
| lstm | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_HL  | 107.72  |  -1.85  | -24.94  |  13.71  | 164.83  |  0.259+  | -0.064  | -0.127  |  0.092+  |  0.337+  |   2.45  |  -9.84  | -16.42  |   0.68  |   5.02  |  0.099 |
| lstm | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_HM  | 123.82  | -15.07  | -28.23  |   2.04  | 161.36  |  0.301+  | -0.134  | -0.155  |  0.031+  |  0.355+  |   5.33  | -10.28  | -18.94  |  -4.53  |  16.45  |  0.080 |
| lstm | Persistent Commitment Lifecycle Surface | price_trend | PERSISTENT_HH  | 154.43  |  -4.31  | -29.30  |   8.54  | 113.97  |  0.349+  | -0.075  | -0.165  |  0.066+  |  0.268+  |   3.64  |  -7.12  | -19.97  |   0.49  |  12.79  |  0.089 |
| lstm | Trend / Volatility Surface | price_trend | LVTF  | 143.03  |  -9.76  |  -8.33  |  14.08  | 142.45  |  0.326+  | -0.103  | -0.012  |  0.094+  |  0.306+  |   5.96  |  -7.94  |  -9.52  |  -0.74  |   7.99  |  0.122 |
| lstm | Trend / Volatility Surface | price_trend | HVTF  |  91.26  | -20.11  | -18.31  |  17.59  | 132.86  |  0.225+  | -0.162  | -0.082  |  0.111+  |  0.304+  |  -3.05  |  -8.38  | -14.50  |  -0.55  |   7.99  |  0.079 |
| lstm | Trend / Volatility Surface | price_trend | LVR  |  94.01  | -14.08  | -19.92  |  11.98  | 235.17  |  0.228+  | -0.128  | -0.089  |  0.083+  |  0.459+  |  -3.41  |  -7.57  | -14.24  |  -0.09  |  11.60  |  0.110 |
| lstm | Trend / Volatility Surface | price_trend | HVR  | 156.64  | -12.70  | -30.17  |   7.43  | 148.57  |  0.360+  | -0.119  | -0.173  |  0.060+  |  0.332+  |   5.05  |  -7.63  | -20.86  |  -1.71  |   5.83  |  0.092 |
| lstm | Trend / Volatility Surface | trend_vol_only | LVTF  |  78.43  | -10.99  | -22.49  |   0.20  | 146.45  |  0.188+  | -0.111  | -0.107  |  0.022+  |  0.312+  |  -4.89  |  -8.01  | -18.84  |  -4.90  |   8.72  |  0.061 |
| lstm | Trend / Volatility Surface | trend_vol_only | HVTF  | 106.06  | -11.82  | -29.28  |   3.36  | 152.26  |  0.258+  | -0.115  | -0.166  |  0.039+  |  0.343+  |  -4.21  |  -7.71  | -20.91  |  -1.37  |  15.05  |  0.072 |
| lstm | Trend / Volatility Surface | trend_vol_only | LVR  | 180.35  |  -8.29  | -21.82  |   6.49  | 144.57  |  0.417+  | -0.094  | -0.106  |  0.055+  |  0.315+  |   4.48  |  -4.97  | -13.54  |  -6.03  |   9.32  |  0.118 |
| lstm | Trend / Volatility Surface | trend_vol_only | HVR  | 200.24  |  -3.98  | -20.08  |   1.57  | 127.59  |  0.448+  | -0.077  | -0.087  |  0.029+  |  0.293+  |   4.41  |  -7.54  | -17.33  |  -2.34  |   5.12  |  0.121 |


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
| PERSISTENT_HH | Persistent Commitment Lifecycle Surface | price_trend |  0.089 |  0.192 | -0.103 |
| PERSISTENT_HL | Persistent Commitment Lifecycle Surface | price_trend |  0.099 |  0.192 | -0.093 |
| PERSISTENT_HM | Persistent Commitment Lifecycle Surface | price_trend |  0.080 |  0.192 | -0.112 |
| PERSISTENT_LH | Persistent Commitment Lifecycle Surface | price_trend |  0.113 |  0.192 | -0.079 |
| PERSISTENT_LL | Persistent Commitment Lifecycle Surface | price_trend |  0.169 |  0.192 | -0.023 |
| PERSISTENT_LM | Persistent Commitment Lifecycle Surface | price_trend |  0.095 |  0.192 | -0.097 |
| PERSISTENT_MH | Persistent Commitment Lifecycle Surface | price_trend |  0.108 |  0.192 | -0.084 |
| PERSISTENT_ML | Persistent Commitment Lifecycle Surface | price_trend |  0.120 |  0.192 | -0.072 |
| PERSISTENT_MM | Persistent Commitment Lifecycle Surface | price_trend |  0.150 |  0.192 | -0.042 |
| HVR | Trend / Volatility Surface | price_trend |  0.092 |  0.192 | -0.100 |
| HVTF | Trend / Volatility Surface | price_trend |  0.079 |  0.192 | -0.113 |
| LVR | Trend / Volatility Surface | price_trend |  0.110 |  0.192 | -0.082 |
| LVTF | Trend / Volatility Surface | price_trend |  0.122 |  0.192 | -0.070 |
| HVR | Trend / Volatility Surface | trend_vol_only |  0.121 |  0.192 | -0.071 |
| HVTF | Trend / Volatility Surface | trend_vol_only |  0.072 |  0.192 | -0.120 |
| LVR | Trend / Volatility Surface | trend_vol_only |  0.118 |  0.192 | -0.074 |
| LVTF | Trend / Volatility Surface | trend_vol_only |  0.061 |  0.192 | -0.131 |


## 4. Behavioral Surface Comparison
> Compares Behavioral Surfaces within each FX pair family present in the benchmark archive.
> Metric: mean walk-forward ΔSharpe across that family's evaluated target pairs.
> Trend/Volatility is split by feature set.


### Persistent
| Surface / Feature Set | EURAUD | EURGBP | EURUSD | GBPUSD | NZDUSD | Mean |
|---|---|---|---|---|---|---|
| Persistent Commitment Lifecycle  [price_trend]  |  0.326  | -0.097  | -0.127  |  0.075  |  0.391  |  0.113 |
| Trend / Volatility  [price_trend]  |  0.285  | -0.128  | -0.089  |  0.087  |  0.350  |  0.101 |
| Trend / Volatility  [trend_vol_only]  |  0.328  | -0.099  | -0.117  |  0.036  |  0.316  |  0.093 |

#### Per-experiment breakdown
| Surface | State | Feature Set | EURAUD | EURGBP | EURUSD | GBPUSD | NZDUSD | Mean |
|---|---|---|---|---|---|---|---|---|
| pLife | PERSISTENT_LL | price_trend  |  0.354  | -0.127  | **-0.004**  |  0.056  | **0.565**  | **0.169** |
| pLife | PERSISTENT_LM | price_trend  |  0.222  | -0.139  | -0.067  |  0.076  |  0.382  |  0.095 |
| pLife | PERSISTENT_LH | price_trend  |  0.374  | -0.090  | -0.173  |  0.047  |  0.406  |  0.113 |
| pLife | PERSISTENT_ML | price_trend  |  0.379  | -0.075  | -0.235  | **0.127**  |  0.405  |  0.120 |
| pLife | PERSISTENT_MM | price_trend  |  0.359  | -0.065  | -0.133  |  0.085  |  0.502  |  0.150 |
| pLife | PERSISTENT_MH | price_trend  |  0.337  | -0.104  | -0.082  |  0.091  |  0.295  |  0.108 |
| pLife | PERSISTENT_HL | price_trend  |  0.259  | **-0.064**  | -0.127  |  0.092  |  0.337  |  0.099 |
| pLife | PERSISTENT_HM | price_trend  |  0.301  | -0.134  | -0.155  |  0.031  |  0.355  |  0.080 |
| pLife | PERSISTENT_HH | price_trend  |  0.349  | -0.075  | -0.165  |  0.066  |  0.268  |  0.089 |
| tVol | LVTF | price_trend  |  0.326  | -0.103  | -0.012  |  0.094  |  0.306  |  0.122 |
| tVol | HVTF | price_trend  |  0.225  | -0.162  | -0.082  |  0.111  |  0.304  |  0.079 |
| tVol | LVR | price_trend  |  0.228  | -0.128  | -0.089  |  0.083  |  0.459  |  0.110 |
| tVol | HVR | price_trend  |  0.360  | -0.119  | -0.173  |  0.060  |  0.332  |  0.092 |
| tVol | LVTF | trend_vol_only  |  0.188  | -0.111  | -0.107  |  0.022  |  0.312  |  0.061 |
| tVol | HVTF | trend_vol_only  |  0.258  | -0.115  | -0.166  |  0.039  |  0.343  |  0.072 |
| tVol | LVR | trend_vol_only  |  0.417  | -0.094  | -0.106  |  0.055  |  0.315  |  0.118 |
| tVol | HVR | trend_vol_only  | **0.448**  | -0.077  | -0.087  |  0.029  |  0.293  |  0.121 |
> **Bold** = highest ΔSharpe in that numerical column across the per-experiment rows.



---
Generated by `compare_to_baseline.py` — MPML Stage 3 OOS validator.
Validated against the MPML benchmark validation contract.
Report format: Markdown — optimized for GitHub, Jupyter, VS Code, Obsidian.
