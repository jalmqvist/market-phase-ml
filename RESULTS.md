# Key Results Summary

## In-Sample Ablation (A0–A3)

| Variant | Return | Sharpe | Max DD | Description |
|---|---|---|---|---|
| A0_TF4 | −3.3% | −0.047 | −25% | Trend-following only |
| A1_MR42 | +20% | +0.074 | −42% | Mean-reversion only |
| A2_PhaseAware | +22% | +0.14 | −29% | Rule-based routing |
| **A3_Dynamic** | **+57%** | **+0.22** | **−31%** | **ML gating** |

## Walk-Forward Evaluation (Out-of-Sample)

**Setup:** 14 FX pairs, ~20 years daily data, 361 folds (expanding window)

**Headline:** Dynamic selector vs PhaseAware baseline

- **Avg Sharpe Δ:** +0.084 (53% of folds improved)
- **Avg Return Δ:** +0.19%
- **Avg Max DD Δ:** −0.17% (less negative = better)

**Interpretation:**

- Modest but consistent out-of-sample gains
- Robustness mitigations (volatility guard, max-hold reset) prevent catastrophic failures
- Value concentrated in specific regimes (not uniform across all folds)

## Failure Mode: Volatility Spikes

**Problem:** Gating model selects Mean-Reversion during extreme volatility → whipsaw losses

**Solution:** Per-fold, leakage-safe volatility guard
- ATR% threshold computed on training window only
- Blocks MR selections during spike bars
- Case study: GBPJPY fold 8 (Brexit spike, mid-2016)

## Engineering Highlights

- Leakage-safe walk-forward evaluation (per-fold preprocessing)
- Reproducible CSV artifacts (per-fold, per-pair, summary)
- Fold-level debuggability (equity curves, selection timelines, spike masks)
- Robustness mitigations grounded in failure analysis
- Mixture-of-experts design (hand-crafted experts + ML gating + online guards)
