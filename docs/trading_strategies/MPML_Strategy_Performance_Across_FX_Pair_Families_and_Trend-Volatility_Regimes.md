# MPML Strategy Performance Across FX Pair Families and Trend-Volatility Regimes

## 1. Purpose and Research Question

### 1.1 Background

Early MPML research established a fixed strategy combination for the PhaseAware strategy-selection framework. At that stage, before BSVE and MSML were introduced, the strategy universe was evaluated across the full set of 14 FX pairs. **TF4 emerged as the strongest overall trend-following strategy, while MR42 emerged as the strongest overall mean-reversion strategy.** These empirical results provided the straightforward basis for using the TF4/MR42 combination in subsequent PhaseAware evaluation.

The choice was therefore not intended to imply that TF4 and MR42 were universally optimal under every market condition. Rather, they represented the best-performing members of their respective strategy classes under the evaluation framework available at the time.

The research landscape has since changed. MPML now evaluates market behaviour through the four standard trend-volatility regimes:

- **HVTF** — High-Volatility Trend
- **LVTF** — Low-Volatility Trend
- **HVR** — High-Volatility Range
- **LVR** — Low-Volatility Range

and, through the integration with MSML, behavioral surfaces provide additional information about distinct populations of FX pairs.

This creates a new question:

> **Is strategy performance sufficiently homogeneous across FX pairs and market regimes for a globally fixed TF4/MR42 combination to remain an appropriate basis for strategy selection?**

The present investigation addresses this question using baseline MPML runs without behavioral-surface information.

### 1.2 Objectives

The investigation has four related objectives:

1. Determine how individual strategies perform across the four trend-volatility regimes.
2. Determine whether these relationships are stable across the different FX pair families.
3. Separate a strategy's **natural operating surface** from its **conditional performance surface**.
4. Establish an empirical baseline against which future behavioral-surface experiments can be evaluated.

The purpose is **not** to redesign the strategies themselves. Questions concerning the technical construction of individual strategies are intentionally deferred to a separate investigation.

------

# 2. Data and Methodology

## 2.1 Strategy and pair universe

The baseline investigation covers 11 MPML strategies:

- TF1
- TF2
- TF3
- TF4
- TF5
- MR1
- MR2
- MR3
- MR32
- MR42
- MR5

The 14 FX pairs are divided into four analytical families:

| Family           | Pairs                                  | Pair count |
| ---------------- | -------------------------------------- | ---------- |
| **Persistent**   | EURUSD, GBPUSD, NZDUSD, EURGBP, EURAUD | 5          |
| **Reactive-JPY** | EURJPY, GBPJPY, USDJPY                 | 3          |
| **Reactive-CHF** | EURCHF, USDCHF                         | 2          |
| **Unclassified** | AUDJPY, AUDUSD, GBPAUD, USDCAD         | 4          |

The first three families reflect the current MPML/MSML research terminology. The **unclassified** group is retained so that the complete 14-pair universe can be analyzed without assigning pairs to behavioral families for which such a classification has not been established.

The small number of pairs in the reactive families is an important limitation. In particular, results for Reactive-CHF, containing only two pairs, should be interpreted as evidence about the observed pair population rather than as strong evidence about a broad FX population.

## 2.2 Walk-forward evaluation

The analysis uses the existing MPML walk-forward evaluation framework. Performance is evaluated out of sample using the individual strategy runs and the newly introduced strategy-level artifacts.

The primary conditional-performance unit is:

> **strategy × pair × walk-forward fold × entry regime**

This preserves the temporal and cross-pair structure of the evaluation rather than treating individual trades as independent observations.

## 2.3 Conditional performance

The primary performance measure is mean trade return (`pnl_pct`), conditioned on the regime in which the trade was entered.

This distinction is deliberate. A strategy may remain active across multiple regimes during the lifetime of a trade; the present analysis asks the simpler and more directly interpretable question:

> **How does a strategy perform when its trades are initiated in a particular regime?**

The analysis therefore does not attempt to assign the entire lifecycle of a trade to a single regime.

## 2.4 Fold-aware uncertainty

Conditional performance was first examined using pooled trade-level results. These were then subjected to a more conservative pair × walk-forward-fold analysis.

For each strategy × regime combination, pair/fold cells were used as bootstrap clusters rather than treating individual trades as independent observations.

This produces:

- pooled mean trade return;
- pair/fold mean and median;
- pair/fold dispersion;
- 95% cluster-bootstrap confidence intervals;
- probability that the bootstrap mean is greater than zero;
- explicit counts of sparse pair/fold cells.

The distinction is important because a large pooled mean does not necessarily imply that the relationship is stable across time and pairs.

## 2.5 Operating surface versus performance surface

A central finding of this investigation is that two different quantities must be kept separate.

### Operating surface

Where does a strategy actually operate?

P(regime∣strategy)

This was measured using the strategy execution artifacts and expressed as relative preference compared with the underlying regime population.

### Performance surface

Where does the strategy perform well?

E[R∣strategy,regime]

These quantities need not agree.

A strategy may naturally generate much of its activity in a particular regime even though another regime produces better conditional performance.

This distinction becomes particularly important for strategy selection.

------

# 3. Global 14-Pair Baseline

The initial analysis pooled all 14 FX pairs.

This produced a useful first overview, but it also demonstrated why a global strategy × regime surface is insufficient.

Several strategies showed clear regime preferences in their operating surfaces. For example, TF4 and MR32 both displayed strong preferences for trend-oriented regimes, particularly LVTF in the case of MR32.

However, the conditional-performance analysis showed that operating preference does not necessarily correspond to performance advantage.

The most notable globally robust conditional relationships were:

- **MR32 × HVTF:** robust positive conditional performance.
- **MR3 × LVTF:** robust negative conditional performance.

Other apparently attractive pooled relationships became less certain once pair × walk-forward-fold variation was taken into account.

This established the need to examine the pair-family structure explicitly.

------

# 4. FX Pair-Family Analysis

## 4.1 Why family structure matters

The global 14-pair analysis implicitly assumes that a strategy/regime relationship is reasonably homogeneous across the FX universe.

The family analysis tests this assumption directly.

The results show substantial differences between the persistent, Reactive-JPY, Reactive-CHF and unclassified populations.

This is one of the main findings of the investigation:

> **The strategy × regime performance surface is not adequately described by a single surface shared across all 14 FX pairs.**

Instead, strategy performance depends materially on the interaction between:

- strategy,
- trend-volatility regime,
- and FX pair family.

------

## 4.2 Top-performing strategy by family and regime

The following matrix identifies the strategy with the highest relative performance within each family/regime combination.

| Family           | HVTF           | LVTF          | HVR           | LVR           |
| ---------------- | -------------- | ------------- | ------------- | ------------- |
| **Persistent**   | **MR42 +29.2** | **MR1 +13.4** | **TF3 +22.8** | **MR3 +37.9** |
| **Reactive-JPY** | **MR32 +44.2** | **TF3 +19.6** | **MR1 +12.7** | **TF3 +28.8** |
| **Reactive-CHF** | **MR32 +31.1** | **TF2 +36.3** | **TF3 +24.2** | **TF2 +10.7** |
| **Unclassified** | **TF3 +24.5**  | **TF2 +30.6** | **TF5 +12.9** | **MR32 +9.4** |

**Table legend:** Values show relative mean trade return compared with the strategy's own family-level baseline. Displayed values have been multiplied by a factor of **100** for readability and comparison. A positive value therefore indicates that the strategy performs above its normal family-level performance in that regime. The table identifies the highest relative-performing strategy in each cell; it does not by itself establish statistical significance.

Several observations stand out immediately.

First, **no single strategy dominates the four families**.

Second, the best strategy changes substantially with the combination of family and regime.

Third, the historical TF4/MR42 combination does not capture the complete strategy landscape revealed by the current analysis.

For example, MR42 is the strongest relative performer in persistent HVTF, while MR32 becomes the strongest strategy in HVTF for both reactive families. TF3 dominates several ranging-regime cells, while TF2 becomes particularly competitive in Reactive-CHF and the unclassified group.

The implication is not that these individual winners should immediately replace the current PhaseAware strategy set. Rather, the matrix demonstrates that **strategy selection is inherently conditional on the population and market state being evaluated**.

---

## 4.3 Visualizing the Operating vs. Performance Mismatch

![fig1_operating_vs_performance](/home/almqvist/Documents/PycharmProjects/market-phase-ml/docs/trading_strategies/appendix_A/fig/fig1_operating_vs_performance.png)

**Figure 4.1: Strategy Selection – Operating Surface vs. Performance Surface.** This scatter plot visualizes the critical distinction between where a strategy naturally operates (Y-axis: Operating Frequency) and where it performs best relative to its baseline (X-axis: Relative Performance). Each point represents a `strategy × regime` combination, colored by volatility regime (Green: LVR, Orange: HVR, Red: HVTF,  Blue: LVTF). The four quadrants define the strategic landscape:

- **Top-Right (Ideal):** High activity and high relative performance.
- **Top-Left (Trap):** High activity but negative relative performance (e.g., strategies over-exposed to unfavorable regimes).
- **Bottom-Right (Under-utilized Advantage):** Low operating frequency but high relative performance. This quadrant identifies the "hidden alpha" where strategies are under-utilized despite historical outperformance (e.g., MR3 in LVR).
- **Bottom-Left (Ignore):** Low activity and low performance. The plot demonstrates that a strategy's natural operating distribution does not align with its optimal deployment environment, reinforcing the need for conditional selection.

---

## 4.4 Family-Specific Performance Surfaces

![fig2_family_radar](/home/almqvist/Documents/PycharmProjects/market-phase-ml/docs/trading_strategies/appendix_A/fig/fig2_family_radar.png)

**Figure 4.2: Strategy Performance Surfaces by FX Family.** These plots illustrate the heterogeneity of strategy performance across the four FX pair families. Each subplot represents a family (Persistent, Reactive-JPY, Reactive-CHF, Unclassified), with axes representing the four trend-volatility regimes (HVTF, LVTF, HVR, LVR). The lines trace the relative performance profile of key strategies (MR42, MR32, TF4, TF3, MR3).

- **Divergence:** The distinct shapes of the lines across families confirm that no single strategy surface is universal. For instance, the "spiky" profile of MR32 in Reactive-JPY contrasts sharply with the flatter profile in the Persistent family.
- **Scaling:** Values are scaled (×10 or auto-scaled) to make relatively small differences visible. The figure is intended to illustrate the shape of the performance surfaces rather than establish statistical significance.
- **Implication:** This visual evidence supports the conclusion that strategy selection must be conditioned on the specific behavioral family of the target pair.

---

## 4.5 Pair-Level Consistency and Sample Size Limitations

![fig3_consistency_heatmap](/home/almqvist/Documents/PycharmProjects/market-phase-ml/docs/trading_strategies/appendix_A/fig/fig3_consistency_heatmap.png)

**Figure 4.3: Pair-Level Consistency of Strategy Performance (Hierarchically Clustered).** This heatmap displays the fraction of pairs within a family that agree on the positive direction of a strategy's performance in a given regime (1.0 = 100% agreement). Rows (strategies) and columns (family/regime combinations) are reordered via hierarchical clustering to reveal coherent blocks of high or low consistency. The clustering is used for visual organization; it does not constitute a statistical clustering result or establish distinct strategy classes.

- **Robustness:** Dark green blocks (e.g., MR3 in Persistent-LVR) indicate robust, family-wide effects shared by all constituent pairs.
- **Limitations:** The discrete nature of the Reactive-CHF column (values restricted to 0.0, 0.5, 1.0) visually highlights the statistical limitation of small sample sizes (N=2). The two-panel design separates the "All Families" view (discrete scale) from the "Stable Families" view (Persistent + Unclassified, continuous scale), allowing for a clearer assessment of patterns without the noise introduced by low-count families.

------

## 4.6 Persistent-family structure

The persistent family provides the strongest evidence because it contains five pairs.

A particularly clear example is MR3.

Its relative performance surface is approximately:

|         | HVTF | LVTF      | HVR  | LVR       |
| ------- | ---- | --------- | ---- | --------- |
| **MR3** | +2.3 | **−22.1** | +1.4 | **+37.9** |

The pair-level consistency analysis shows that the direction is shared across the persistent pairs: **all five persistent pairs have positive mean performance in LVR, while none has positive mean performance in LVTF** under the minimum-observation criterion.

This is much stronger than a pooled observation. It indicates that the MR3/LVTF and MR3/LVR contrast is not simply the result of one unusually influential currency pair.

MR42 provides another important example. Its persistent-family relative performance is strongly positive in HVTF and negative in HVR, with the positive HVTF direction shared by most of the constituent pairs.

TF4 provides an important counterexample. It has historically been one of the strongest overall trend-following strategies and has a strong operating preference for trend regimes, but within the persistent family its relative performance is better in LVTF than in HVTF. The pair-level results show the same directional pattern across the five persistent pairs.

Thus:

> **A strategy's natural tendency to operate in a regime does not establish that the regime is its best deployment environment.**

------

## 4.7 Reactive-JPY structure

The Reactive-JPY family contains three pairs:

EURJPY, GBPJPY and USDJPY.

The family-level performance surface differs materially from the persistent family.

One notable result is TF3 in LVR, which shows positive conditional performance and is supported by the three-pair family structure. However, the smaller number of pairs means that apparent family-level relationships should be treated more cautiously than five-pair persistent-family results.

The important conclusion at this stage is therefore not that a particular strategy has been established as the universal Reactive-JPY winner, but that:

> **Reactive-JPY exhibits a strategy/regime structure that is materially different from the persistent family.**

This is directly relevant to future MSML behavioral-surface experiments.

------

## 4.8 Reactive-CHF structure

Reactive-CHF contains only two pairs, EURCHF and USDCHF.

Several strong-looking strategy/regime relationships appear in this population, including positive results for MR32 in HVTF and MR1 in LVTF, as well as several negative trend-following relationships.

Because only two independent pairs are available, these results should be treated primarily as **hypotheses and observed pair-population patterns**, rather than as broadly established family-level effects.

Nevertheless, the divergence from the persistent and Reactive-JPY surfaces is itself informative.

------

## 4.9 Unclassified pairs

The unclassified group contains AUDJPY, AUDUSD, GBPAUD and USDCAD.

This group is retained to avoid forcing an established behavioral classification onto pairs for which one has not been established.

Its strategy/regime surface is less coherent than that of the persistent family. This provides a useful comparison and reinforces the value of distinguishing established behavioral families from a generic residual group.

The unclassified group should therefore not be interpreted as a fourth behavioral family.

------

# 5. Pair-Level Consistency

Family-level averages can conceal substantial differences between constituent pairs. The pair-level analysis therefore examined whether the direction of each family/regime relationship was shared across individual pairs.

A useful measure is the **pair consistency**:

C=#pairs with sufficient observations#pairs with positive mean return

using 25 trades per pair/regime as the minimum observation threshold.

This produces an important distinction between:

1. **family-level effects shared by constituent pairs**;
2. **family-level effects driven by a subset of pairs**;
3. **pair-specific effects that happen to appear in a family aggregate**.

The persistent-family MR3 result is a particularly strong example of the first category.

Conversely, some apparent effects in the smaller families depend on one or two pairs and should therefore be regarded as considerably less stable.

This analysis demonstrates that **pair identity remains an important source of variation even after introducing pair families**.

------

# 6. Relative Performance Surfaces

Raw conditional returns are difficult to compare between strategies and families because strategies have different overall return characteristics.

The final baseline analysis therefore normalizes each regime against the strategy's own family-level performance:

Δs,f,r=E[R∣s,f,r]−E[R∣s,f]

where:

- s = strategy,
- f = FX pair family,
- r = trend-volatility regime.

This produces a **relative performance surface**.

The interpretation changes from:

> "Is this regime profitable?"

to:

> **"Is this regime better or worse than the strategy's normal performance within this FX family?"**

This is more directly relevant to strategy selection.

It also makes the mismatch between operating and performance surfaces particularly visible.

------

## 6.1 Operating preference versus relative performance

The investigation identifies several cases where a strategy naturally operates in a regime that is not its strongest relative-performance regime.

TF4 is a particularly clear example in the persistent family:

- it naturally operates disproportionately in trend regimes;
- its relative performance is nevertheless better in LVTF than HVTF;
- the pair-level direction is consistent across the persistent pairs.

MR3 provides an even more striking example:

- it under-operates in LVR;
- LVR is nevertheless strongly positive relative to its persistent-family baseline;
- all five persistent pairs agree on the positive direction.

This suggests that **strategy selection should not simply reproduce the strategy's natural historical operating distribution**.

The selector may instead benefit from identifying where a strategy has historically demonstrated a relative advantage, even when that environment is not where the strategy naturally generates the most activity.

------

# 7. Implications for MPML Strategy Selection

## 7.1 The TF4/MR42 benchmark

The historical use of TF4/MR42 remains well motivated.

TF4 was identified as the strongest overall trend-following strategy and MR42 as the strongest overall mean-reversion strategy across the 14-pair universe during the early MPML work. This made the pair a reasonable and empirically grounded PhaseAware benchmark.

The present study does **not** invalidate that historical conclusion.

What has changed is the scope of the research problem.

The introduction of MSML behavioral surfaces means that MPML is no longer simply asking:

> Which two strategies perform best across the complete FX universe?

It is increasingly asking:

> **Which strategy is best suited to this particular FX population, market regime and behavioral state?**

Under that formulation, a globally fixed TF4/MR42 combination becomes a baseline rather than an obvious universal optimum.

------

## 7.2 A more conditional selection framework

The results suggest that future strategy selection should consider at least:

strategy×FX family×trend-volatility regime

and potentially:

strategy×FX family×trend-volatility regime×behavioral state

The latter is the natural direction for integration with MSML.

The purpose of the behavioral-surface experiments can therefore be sharpened:

> **Determine whether behavioral states explain, refine or improve upon the strategy/regime differences already observed in the baseline MPML population.**

This is a substantially stronger experimental question than simply asking whether behavioral information improves a fixed TF4/MR42 benchmark.

------

# 8. Limitations

Several limitations should be kept explicit.

### Small family sizes

The persistent family contains five pairs, Reactive-JPY three and Reactive-CHF only two. Consequently, apparent family-specific effects become progressively less reliable as the number of constituent pairs decreases.

### Pair-level dependence

Walk-forward folds provide repeated observations over time, but they do not make different observations from the same currency pair statistically equivalent to independent currency pairs.

### Sparse cells

Some strategy × pair × regime combinations contain relatively few trades. Sparse cells are retained and flagged rather than silently discarded.

### Historical evidence

The results describe the observed historical evaluation period. They do not establish that the same strategy/regime relationships will persist in future data.

### Conditional, not causal, relationships

The analysis establishes empirical conditional performance relationships. It does not establish why a strategy performs differently in a particular regime or family.

### Strategy construction is outside scope

The results naturally raise questions about whether the technical construction of individual strategies contributes to their observed regime and family preferences. Those questions are important, but they constitute a separate strategy-development investigation and are not addressed here.

------

# 9. Conclusions

The baseline strategy investigation fills an important gap in the MPML research.

The original TF4/MR42 benchmark was empirically justified by the early 14-pair evaluation. However, the current analysis demonstrates that **global strategy rankings conceal substantial heterogeneity across FX pair families and trend-volatility regimes**.

Three conclusions are particularly important.

### 1. Strategy performance is family-dependent

The strategy/regime surface differs materially between persistent, Reactive-JPY, Reactive-CHF and unclassified pairs.

There is no single strategy that consistently dominates all families and regimes.

### 2. Operating preference and performance are different

A strategy's natural operating surface does not necessarily identify its best deployment environment.

The clearest examples include TF4 in the persistent family and MR3 in the persistent family, where operating behaviour and relative performance point in different directions.

### 3. Strategy selection should become conditional

The evidence supports treating a fixed strategy pair as a useful global baseline rather than as an assumed universal optimum.

The next generation of MPML strategy selection should instead investigate the interaction between:

strategy×family×regime×behavioral state

The behavioral-surface experiments can now be interpreted against this richer baseline.

Rather than asking whether behavioral information can improve a fixed TF4/MR42 combination in the abstract, we can ask a more meaningful question:

> **Does behavioral state information provide additional information for choosing among strategies whose suitability already varies across FX families and trend-volatility regimes?**

That is the principal research direction opened by this investigation.

------

## Appendix A — Analytical Artifacts

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

These artifacts preserve the progression from global performance to family-level and pair-level analysis.

------

## Appendix B — Interpretation of Relative-Performance Tables

Unless otherwise stated, relative-performance tables display:

100×(E[R∣strategy,family,regime]−E[R∣strategy,family])

for readability.

Thus, the displayed values are **not percentages in the conventional sense of a percentage return**; they are the underlying relative-return values multiplied by 100.

Positive values indicate that the regime performed better than the strategy's family-level baseline. Negative values indicate underperformance relative to that baseline.

The top-strategy matrix identifies the highest relative-performing strategy within each family/regime cell. It should be interpreted as a descriptive strategy-ranking result, not as evidence that the selected strategy is statistically superior in every cell.