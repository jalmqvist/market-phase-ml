
# MPML REFERENCE BENCHMARK

**Architecture**: lstm  
**Experiments**: 12  
**Baseline**: No-DL PhaseAware (aggregate)  
**Target pairs**: EURJPY, GBPJPY, USDJPY (Reactive-JPY family)  

> Δ values are walk-forward OOS deltas vs no-DL baseline.  
> `+` = positive Sharpe uplift.  
> All values rounded to 3 decimals for readability.

## 1. Uplift Matrix — ΔRet and ΔSh per State and Pair
| Architecture | Behavioral Surface | Feature Set | State | ΔRet EURJPY | ΔRet GBPJPY | ΔRet USDJPY | ΔSh EURJPY | ΔSh GBPJPY | ΔSh USDJPY | Mean ΔSh |
|---|---|---|---|---|---|---|---|---|---|---|
| lstm | Consensus Lifecycle Surface | price_trend | JPY_CONSENSUS_YOUNG  |   0.64  |   0.08  |   0.53  |  0.237+  |  0.081+  |  0.144+  |  0.154 |
| lstm | Consensus Lifecycle Surface | price_trend | JPY_CONSENSUS_MATURING  |   0.62  |   0.03  |   0.70  |  0.254+  |  0.053+  |  0.187+  |  0.165 |
| lstm | Consensus Lifecycle Surface | price_trend | JPY_CONSENSUS_MATURE  |   0.73  |   0.06  |   0.65  |  0.273+  |  0.058+  |  0.176+  |  0.169 |
| lstm | Consensus Lifecycle Surface | price_trend | JPY_NON_EXTREME  |   0.66  |   0.01  |   0.58  |  0.267+  |  0.052+  |  0.147+  |  0.155 |
| lstm | Trend / Volatility Surface | price_trend | LVTF  |   0.62  |  -0.16  |   0.33  |  0.240+  |  0.011+  |  0.082+  |  0.111 |
| lstm | Trend / Volatility Surface | price_trend | HVTF  |   0.57  |  -0.09  |   0.69  |  0.232+  |  0.032+  |  0.192+  |  0.152 |
| lstm | Trend / Volatility Surface | price_trend | LVR  |   0.67  |   0.09  |   0.86  |  0.269+  |  0.079+  |  0.233+  |  0.194 |
| lstm | Trend / Volatility Surface | price_trend | HVR  |   0.87  |   0.06  |   0.64  |  0.332+  |  0.070+  |  0.174+  |  0.192 |
| lstm | Trend / Volatility Surface | trend_vol_only | LVTF  |   0.64  |   0.08  |   0.47  |  0.277+  |  0.078+  |  0.128+  |  0.161 |
| lstm | Trend / Volatility Surface | trend_vol_only | HVTF  |   0.70  |  -0.00  |   0.33  |  0.255+  |  0.042+  |  0.081+  |  0.126 |
| lstm | Trend / Volatility Surface | trend_vol_only | LVR  |   0.83  |  -0.11  |   0.62  |  0.323+  |  0.025+  |  0.165+  |  0.171 |
| lstm | Trend / Volatility Surface | trend_vol_only | HVR  |   0.81  |  -0.02  |   0.59  |  0.295+  |  0.051+  |  0.167+  |  0.171 |


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
| EURJPY * |   8.25 |  0.054 |  -8.48 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   4.13 |  0.085 |  -3.95 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  14.25 |  0.048 |  -0.71 |


### Trend / Volatility Surface — LVTF — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  25.64 |  0.136 |  -3.26 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   6.96 |  0.101 |  -1.33 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |   4.26 |  0.005 |  -2.93 |


### Trend / Volatility Surface — HVTF — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  18.06 |  0.102 |  -5.13 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   7.30 |  0.103 |  -2.11 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  46.62 |  0.172 |   1.76 |


### Trend / Volatility Surface — HVTF — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  23.16 |  0.124 |  -4.56 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |  -0.01 |  0.053 |  -4.72 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  25.73 |  0.093 |   1.84 |


### Trend / Volatility Surface — LVR — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |   8.11 |  0.054 |  -5.38 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |  14.64 |  0.154 |   3.62 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  51.01 |  0.190 |   2.96 |


### Trend / Volatility Surface — LVR — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |   7.55 |  0.051 |  -8.82 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   1.49 |  0.067 |  -4.55 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  12.66 |  0.044 |  -2.20 |


### Trend / Volatility Surface — HVR — price_trend
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |  27.40 |  0.142 |  -4.99 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   7.28 |  0.108 |  -3.02 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  10.13 |  0.030 |   0.63 |


### Trend / Volatility Surface — HVR — trend_vol_only
| Pair | ΔReturn | ΔSharpe | ΔDD |
|---|---|---|---|
| AUDJPY |  21.15 |  0.199 |   9.13 |
| AUDUSD | -53.93 | -0.244 | -20.86 |
| EURAUD | 114.97 |  0.257 |   5.14 |
| EURCHF |  27.50 |  0.112 |  -4.61 |
| EURGBP |  29.17 |  0.059 |  -2.31 |
| EURJPY * |   6.43 |  0.045 |  -8.71 |
| EURUSD | -31.27 | -0.182 | -22.54 |
| GBPAUD |  -6.22 |  0.022 |  -8.01 |
| GBPJPY * |   9.18 |  0.118 |  -1.73 |
| GBPUSD |  -5.55 | -0.018 |  -8.93 |
| NZDUSD |  21.27 |  0.049 | -12.80 |
| USDCAD |  36.21 |  0.217 |   0.84 |
| USDCHF |  12.92 |  0.103 |   5.82 |
| USDJPY * |  53.78 |  0.195 |  -0.20 |


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
| LVTF | price_trend |  0.111 | -0.031 |  0.142 |
| HVTF | price_trend |  0.152 | -0.031 |  0.183 |
| LVR | price_trend |  0.194 | -0.031 |  0.225 |
| HVR | price_trend |  0.192 | -0.031 |  0.223 |
| LVTF | trend_vol_only |  0.161 | -0.031 |  0.192 |
| HVTF | trend_vol_only |  0.126 | -0.031 |  0.157 |
| LVR | trend_vol_only |  0.171 | -0.031 |  0.202 |
| HVR | trend_vol_only |  0.171 | -0.031 |  0.202 |


## 4. Behavioral Family Comparison — Reactive-JPY
> Compares the two Behavioral Surfaces of the Reactive-JPY family.
> Metric: mean walk-forward ΔSharpe across surface states.
> Trend/Volatility is split by feature set.

| Surface / Feature Set | EURJPY | GBPJPY | USDJPY | Mean |
|---|---|---|---|---|
| Consensus Lifecycle  [price_trend]  |  0.258  |  0.061  |  0.163  |  0.161 |
| Trend / Volatility  [price_trend]  |  0.268  |  0.048  |  0.170  |  0.162 |
| Trend / Volatility  [trend_vol_only]  |  0.288  |  0.049  |  0.135  |  0.157 |


### Per-experiment breakdown
| Surface | State | Feature Set | EURJPY | GBPJPY | USDJPY | Mean |
|---|---|---|---|---|---|---|
| cLife | JPY_CONSENSUS_YOUNG | price_trend  |  0.237  |  0.081  |  0.144  |  0.154 |
| cLife | JPY_CONSENSUS_MATURING | price_trend  |  0.254  |  0.053  |  0.187  |  0.165 |
| cLife | JPY_CONSENSUS_MATURE | price_trend  |  0.273  |  0.058  |  0.176  |  0.169 |
| cLife | JPY_NON_EXTREME | price_trend  |  0.267  |  0.052  |  0.147  |  0.155 |
| tVol | LVTF | price_trend  |  0.240  |  0.011  |  0.082  |  0.111 |
| tVol | HVTF | price_trend  |  0.232  |  0.032  |  0.192  |  0.152 |
| tVol | LVR | price_trend  |  0.269  |  0.079  |  0.233  |  0.194 |
| tVol | HVR | price_trend  |  0.332  |  0.070  |  0.174  |  0.192 |
| tVol | LVTF | trend_vol_only  |  0.277  |  0.078  |  0.128  |  0.161 |
| tVol | HVTF | trend_vol_only  |  0.255  |  0.042  |  0.081  |  0.126 |
| tVol | LVR | trend_vol_only  |  0.323  |  0.025  |  0.165  |  0.171 |
| tVol | HVR | trend_vol_only  |  0.295  |  0.051  |  0.167  |  0.171 |


---
Generated by `compare_to_baseline.py` — MPML Stage 3 OOS validator.
Validated against VALIDATION_SPEC_JPY.md (frozen June 2026).
Report format: Markdown — optimized for GitHub, Jupyter, VS Code, Obsidian.
