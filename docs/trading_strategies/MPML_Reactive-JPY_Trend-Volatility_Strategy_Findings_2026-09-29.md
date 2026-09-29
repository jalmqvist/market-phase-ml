# MPML Reactive-JPY Trend-Volatility Strategy Findings

**Status:** Active research record  
**Scope:** Reactive-JPY strategy performance conditioned on the MPML trend-volatility surface  
**Population:** EURJPY, GBPJPY, USDJPY  
**Last updated:** 2026-09-29

---

## 1. Purpose and scope

This document records the MPML strategy research concerning the **trend-volatility surface** for the Reactive-JPY pair family.

The Reactive-JPY population consists of:

- EURJPY
- GBPJPY
- USDJPY

The four MPML trend-volatility regimes are:

- **HVTF** — High-Volatility Trend
- **LVTF** — Low-Volatility Trend
- **HVR** — High-Volatility Range
- **LVR** — Low-Volatility Range

The purpose of this document is to keep the trend-volatility evidence separate from the later **Reactive-JPY consensus-lifecycle behavioral surface**. The two surfaces describe different conditioning dimensions and should not be treated as interchangeable.

The corresponding behavioral-surface findings are maintained separately in:

> `MPML_Reactive-JPY_Consensus-Lifecycle_Strategy_Findings_2026-09-29.md`

This document therefore does **not** incorporate the later `JPY_CONSENSUS_YOUNG`, `JPY_CONSENSUS_MATURING`, `JPY_CONSENSUS_MATURE`, or `JPY_NON_EXTREME` results except where they are mentioned explicitly to establish the boundary between the two research tracks.

---

## 2. Research context

Early MPML research established `TF4 + MR42` as the PhaseAware benchmark. At that stage, the strategy universe was evaluated across the full 14-pair FX universe. TF4 emerged as the strongest overall trend-following strategy and MR42 as the strongest overall mean-reversion strategy.

The subsequent trend-volatility investigation tested whether this global view was sufficient. It found that strategy performance varies materially with both **FX pair family** and **trend-volatility regime**.

The relevant research question became:

> **Is strategy performance sufficiently homogeneous across FX pairs and market regimes for a globally fixed TF4/MR42 combination to remain an appropriate basis for strategy selection?**

For Reactive-JPY, the answer from the baseline research is that the strategy/regime surface differs materially from the Persistent family and should therefore be treated as a population-specific surface rather than as a generic FX surface.

---

## 3. Methodology

### 3.1 Walk-forward evaluation

The analysis uses the existing MPML walk-forward evaluation framework. Performance is evaluated out of sample using individual-strategy execution artifacts.

The primary conditional-performance unit is:

> **strategy × pair × walk-forward fold × entry regime**

This preserves temporal ordering and cross-pair structure rather than treating individual trades as independent observations.

Walk-forward evaluation remains the primary validation mechanism.

### 3.2 Conditional performance

The primary performance measure is mean trade return (`pnl_pct`) conditioned on the trend-volatility regime in which a trade was entered:

> **E[R | strategy, pair/family, regime]**

This asks how a strategy performs when its trades are initiated in a particular regime. It does not assign the complete lifetime of a trade to a single regime.

### 3.3 Operating surface versus performance surface

Two quantities must be kept separate:

**Operating surface**

> `P(regime | strategy)`

Where does a strategy naturally operate?

**Performance surface**

> `E[R | strategy, regime]`

Where does a strategy perform well?

A strategy can naturally generate much of its activity in one regime while performing relatively better in another. This distinction is important when interpreting strategy/regime relationships and when designing a conditional selector.

### 3.4 Pair-level consistency

Family-level averages can conceal differences between constituent pairs. The analysis therefore examines whether the direction of a relationship is shared across individual pairs.

The historical consistency analysis used **25 trades per pair/regime** as the minimum observation threshold. This distinguishes:

1. effects shared by the constituent pairs;
2. family-level effects driven by a subset of pairs; and
3. pair-specific effects that appear in the family aggregate.

Reactive-JPY contains only three pairs, so its family-level evidence is necessarily less independent than the five-pair Persistent family.

---

## 4. Reactive-JPY trend-volatility surface

The historical relative-performance matrix identifies the highest relative-performing strategy in each family/regime cell. For Reactive-JPY the surface is:

| Regime | Highest relative performer | Relative performance |
|---|---|---:|
| **HVTF** | **MR32** | **+44.2** |
| **LVTF** | **TF3** | **+19.6** |
| **HVR** | **MR1** | **+12.7** |
| **LVR** | **TF3** | **+28.8** |

The displayed values are relative mean trade returns compared with each strategy's own Reactive-JPY family-level baseline, multiplied by 100 for readability. They are therefore **relative-performance values, not conventional percentage returns**.

The matrix is descriptive. It identifies the highest relative performer in each cell but does not establish statistical superiority in every cell.

### 4.1 Main observation

Reactive-JPY does not have one strategy that dominates the complete trend-volatility surface.

Instead, the relative leader changes with regime:

- **HVTF:** MR32
- **LVTF:** TF3
- **HVR:** MR1
- **LVR:** TF3

The two trend regimes therefore do not share the same relative leader, and the ranging regimes likewise differ.

The important finding is the existence of a **population-specific strategy/regime structure**, rather than any single cell-level winner being considered a deployable replacement for the PhaseAware representatives.

---

## 5. Reactive-JPY compared with the other FX populations

The broader 14-pair baseline provides useful context for interpreting the Reactive-JPY surface.

| Family | HVTF | LVTF | HVR | LVR |
|---|---|---|---|---|
| **Persistent** | MR42 +29.2 | MR1 +13.4 | TF3 +22.8 | MR3 +37.9 |
| **Reactive-JPY** | **MR32 +44.2** | **TF3 +19.6** | **MR1 +12.7** | **TF3 +28.8** |
| **Reactive-CHF** | MR32 +31.1 | TF2 +36.3 | TF3 +24.2 | TF2 +10.7 |
| **Unclassified** | TF3 +24.5 | TF2 +30.6 | TF5 +12.9 | MR32 +9.4 |

This broader matrix is included only as historical context. The present document remains focused on Reactive-JPY.

The cross-family comparison demonstrates why a single universal strategy/regime surface is inadequate: the relative strategy pattern changes materially between populations.

In particular, MR32 is the relative leader in HVTF for both Reactive-JPY and Reactive-CHF, whereas the Persistent population has MR42 in that cell. TF3 is the relative leader in both Reactive-JPY LVTF and LVR, while other populations show different representatives.

---

## 6. Reactive-JPY versus Persistent structure

The baseline investigation found that the Reactive-JPY strategy/regime surface differs materially from the Persistent family.

One notable Reactive-JPY result is **TF3 in LVR**, which shows positive conditional performance across the three-pair population. Because Reactive-JPY contains only three pairs, this should be interpreted more cautiously than the corresponding five-pair Persistent-family evidence.

The broader conclusion is:

> **Reactive-JPY exhibits a strategy/regime structure that is materially different from the Persistent family.**

This provided the population-specific foundation for the later behavioral-surface investigation.

---

## 7. What the trend-volatility evidence does and does not establish

The trend-volatility investigation supports several conclusions about the structure of the MPML strategy space.

### Established within the historical evaluation

- Strategy performance varies across trend-volatility regimes.
- The pattern varies between FX pair families.
- Reactive-JPY has a distinct strategy/regime surface relative to the Persistent family.
- A strategy's natural operating distribution is not necessarily its strongest conditional-performance environment.
- Pair identity remains an important source of variation even after introducing FX pair families.

### Not established by this analysis

The cell-level relative-performance matrix does **not** establish that MR32, TF3 or MR1 should replace the current PhaseAware representatives.

It also does not establish a causal mechanism explaining why a strategy performs differently in a particular regime.

Finally, the trend-volatility results do not by themselves establish that adding a behavioral surface improves strategy selection. That question belongs to the subsequent behavioral-surface experiments.

---

## 8. Operating surface versus conditional performance

A central methodological result of the baseline research is that operating frequency and conditional performance should not be conflated.

For example, a strategy can naturally generate many trades in a trend regime while having its strongest relative performance in a different regime. The strategy's historical operating distribution therefore cannot, on its own, determine where the strategy should be routed.

This distinction is one of the motivations for a conditional strategy selector: the selector can potentially learn **where a strategy has demonstrated relative suitability**, rather than merely reproducing where that strategy has historically generated the most activity.

The same distinction should be preserved when the trend-volatility surface is later combined with behavioral information.

---

## 9. Relationship to the Reactive-JPY consensus-lifecycle surface

The Reactive-JPY research now contains two distinct conditioning surfaces.

### Trend-volatility surface

> `strategy × pair × trend-volatility regime`

with regimes:

> `HVTF, LVTF, HVR, LVR`

### Consensus-lifecycle behavioral surface

> `strategy × pair × behavioral state`

with states:

> `JPY_CONSENSUS_YOUNG`  
> `JPY_CONSENSUS_MATURING`  
> `JPY_CONSENSUS_MATURE`  
> `JPY_NON_EXTREME`

These should remain analytically separate unless an experiment explicitly models their interaction.

The later consensus-lifecycle work should therefore not be interpreted as replacing the trend-volatility surface. Rather, it introduces another conditioning dimension that may eventually be combined with it.

A future integrated experiment could, in principle, investigate:

> **strategy × pair × trend-volatility regime × behavioral state**

but such an interaction should be treated as a new experiment rather than inferred from either surface independently.

---

## 10. Implications for PhaseAware strategy selection

The historical TF4/MR42 combination remains a useful canonical benchmark because it was empirically motivated by the original 14-pair evaluation.

The trend-volatility evidence changes the interpretation of that benchmark: TF4/MR42 should be regarded as a **baseline/control**, not as an assumption that the same representatives are optimal in every population and regime.

For Reactive-JPY specifically, the trend-volatility surface provides evidence that different strategies have different conditional profiles across HVTF, LVTF, HVR and LVR.

This motivates conditional strategy selection rather than simple global ranking.

However, the historical cell-level results should not be converted directly into a production routing policy. A deployable selector requires out-of-sample, fold-aware evaluation and must account for pair-level consistency, sparse cells and the difference between realized performance and information available at decision time.

---

## 11. Limitations

### Small family size

Reactive-JPY contains only three pairs: EURJPY, GBPJPY and USDJPY. Family-level results should therefore be interpreted as evidence about this observed population rather than as strong evidence about a broad FX population.

### Pair-level dependence

Repeated walk-forward observations from the same currency pair do not make the observations equivalent to independent currency pairs.

### Sparse cells

Some strategy × pair × regime combinations contain relatively few trades. Sparse cells should therefore be interpreted cautiously rather than treated as equally informative as densely sampled cells.

### Historical evidence

The results describe the historical evaluation period. They do not establish that the same strategy/regime relationships will persist in future data.

### Conditional rather than causal relationships

The analysis establishes empirical conditional-performance relationships. It does not establish why a strategy performs differently in a particular regime or family.

### Relative-performance interpretation

The relative-performance matrix is normalized against each strategy's own family-level baseline. It therefore should not be interpreted as a direct ranking of absolute strategy returns across all strategies.

---

## 12. Analytical artifacts

The baseline investigation produced the following strategy-level artifacts:

- `strategy_regime_profile__baseline.csv`
- `strategy_signal_timeline__baseline.csv`
- `strategy_execution_timeline__baseline.csv`
- `strategy_trades__baseline.csv`

The subsequent analysis produced:

- `baseline_conditional_performance_pairfold.csv`
- `baseline_conditional_performance_stability.csv`
- `baseline_family_conditional_pairfold.csv`
- `baseline_family_conditional_stability.csv`
- `baseline_pair_family_regime.csv`
- `baseline_family_regime_consistency.csv`
- `baseline_relative_family_regime.csv`
- `baseline_relative_pair_family_regime.csv`

These artifacts preserve the progression from global performance to family-level and pair-level trend-volatility analysis.

---

## 13. Current status

The Reactive-JPY trend-volatility surface is established as an **earlier baseline research layer** for the population.

The main working conclusions are:

1. Reactive-JPY has a distinct strategy/regime surface.
2. The relative strategy leader changes across HVTF, LVTF, HVR and LVR.
3. The surface differs from the Persistent population.
4. Pair identity and sample size remain important limitations.
5. The results motivate conditional selection but do not constitute a production routing policy.
6. The later consensus-lifecycle surface is a separate conditioning dimension and should not be conflated with these results.

The current Reactive-JPY strategy research can therefore be organized as two parallel documents:

- **Consensus-lifecycle surface:** current behavioral-state and PhaseAware composition research.
- **Trend-volatility surface:** historical regime-conditioned strategy research documented here.

An eventual four-dimensional study can combine them explicitly, but should be documented as a new research stage rather than appended implicitly to either existing surface.

---

## Appendix A — Interpretation of relative-performance values

Unless otherwise stated, the relative-performance tables use:

> `100 × (E[R | strategy, family, regime] − E[R | strategy, family])`

for readability.

The displayed values are therefore **not conventional percentage returns**. They are the underlying relative-return values multiplied by 100.

Positive values indicate that the regime performed better than the strategy's family-level baseline. Negative values indicate underperformance relative to that baseline.

The top-strategy matrix identifies the highest relative-performing strategy within each family/regime cell. It should be interpreted as a descriptive result, not as evidence that the selected strategy is statistically superior in every cell.

---
