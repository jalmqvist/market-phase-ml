# Archive Notice

**Status:** Historical document (superseded)

This document represents an earlier stage of the MPML family-topology investigation and is preserved for historical reference only.

Subsequent engineering audits, transfer-validation studies, and dense-overlap analyses substantially improved understanding of the mechanisms underlying the observed family effects. While many individual experimental results contained in this document remain valid, several interpretations and working hypotheses have since been revised or replaced.

In particular, this document predates:

- transfer-validated analysis of DL prediction surfaces,
- overlap-conditioned selector influence analysis,
- dense-DL overlap distribution studies,
- updated explanations of topology convergence,
- the current MPML working model.

Readers seeking the current knowledge state should begin with:

**Family Topology Study v2 (Transfer-Validated Baseline)**

and consult this document only when investigating the historical evolution of hypotheses, experiments, or implementation audits.

This archive remains valuable as a record of the research process, but it should not be treated as the authoritative interpretation of current evidence.

---

# Family-Regime Topology Study (Post-Fix Baseline)

## Status

**Version:** v1 (Post-Fix Baseline)

**Purpose:** Establish a clean reference document based only on experiments executed after fixing the DL regime attachment and artifact loading bugs.

This document supersedes earlier family-topology notes generated before the following issues were identified and resolved:

- DL regime inference bug (LVTF default surface applied to non-LVTF artifacts)
- DL feature attachment failures
- Transfer artifact regime mismatch
- Experiment contamination caused by identical DL surfaces across multiple regimes
- Various caching and reproducibility issues discovered during investigation

Earlier documents should be archived and treated as historical references only.

---

# Table of Contents

1. Background
2. Experimental Design
3. Bug Fixes and Validation
4. Family-Level Results
5. Early-Stage Phase Modelling Analysis
6. DL-Active Topology Analysis
7. Transition Analysis
8. Strategy Duration Analysis
9. USD / JPY Bias Decomposition
10. MPML Selector Audit
11. Current Interpretation
12. Open Questions
13. Working Conclusions

---

## Key Findings at a Glance

### Highest-Confidence Findings

1. DL-active periods alter MPML selector topology.
2. Persistent and Reactive families respond in opposite directions.
3. The family effect survives all known DL attachment and regime-loading bug fixes.
4. The family effect survives USD and JPY decomposition analyses.
5. The strongest currently identified mechanism is strategy persistence rather than transition frequency.

### Persistent Family

DL-active periods tend to:

- Increase MeanReversion occupancy.
- Increase MeanReversion duration: 16.70 → 19.02 bars

- Increase overall selector stability.

### Reactive Family

DL-active periods tend to:

- Decrease MeanReversion occupancy.
- Increase PhaseAware occupancy.
- Dramatically reduce MeanReversion duration: 10.31 → 4.46 bars

### Current Working Hypothesis

Persistent and Reactive MSML families appear to encode genuinely different predictive structures that remain visible after integration into MPML. The strongest evidence currently suggests that these differences manifest primarily through changes in strategy persistence during DL-active periods.

------

# 1. Background

The original objective was to understand how Deep Learning (DL) information affects the behavior of the Market Phase ML (MPML) strategy selector.

A second objective emerged during experimentation:

> Do the Persistent and Reactive MSML pair families induce different selector behavior inside MPML?

This question became increasingly important after multiple analyses showed that selector topology differed substantially between the two families.

Initially these findings were treated cautiously because several bugs were later discovered in the DL attachment pipeline.

Following the fixes, the entire experiment matrix was rerun.

The current document summarizes findings from the corrected experiments only.

------

# 2. Experimental Design

## Families

### Persistent Family

Pairs selected by the MSML persistent-pair discovery process.

### Reactive Family

Pairs selected by the MSML reactive-pair discovery process.

------

## Regimes

Experiments were conducted for:

- LVTF
- HVTF
- LVR
- HVR

using regime-specific DL artifacts.

------

## DL Variants

For each family-regime combination:

### Sentiment

MSML trained with sentiment features.

### NoSentiment

MSML trained without sentiment features.

------

## Awareness Variants

### Aware

MPML receives DL features.

### Blind

MPML receives DL features but sentiment-awareness is disabled.

------

## Total Matrix

2 families × 4 regimes × 2 sentiment modes × 2 awareness modes

= 32 independent MPML runs.

Validation checks confirmed:

- unique artifact hashes
- unique MPML outputs
- DL feature count = 5 for all runs
- correct regime attachment

------

# 3. Bug Fixes and Validation

## DL Regime Attachment Bug

### Root Cause

DL surfaces were initialized using:

```python
DL_REGIME = os.getenv("DL_REGIME", "LVTF")
```

which caused LVTF to become the default surface even when HVR/HVTF/LVR artifacts were loaded.

### Effect

Manifests and attachment logic reported:

```text
dl_surface:
    dl_regime = LVTF
```

for non-LVTF artifacts.

### Resolution

Regime now inferred directly from artifact path.

Result:

- LVTF artifacts attach LVTF surfaces
- HVTF artifacts attach HVTF surfaces
- LVR artifacts attach LVR surfaces
- HVR artifacts attach HVR surfaces

------

## DL Feature Validation

All rerun experiments report:

```text
dl_feature_count = 5
```

indicating successful feature attachment.

This is a critical validation point because earlier faulty runs frequently reported:

```text
dl_feature_count = 0
```

for non-LVTF configurations.

------

# 4. Family-Level Results

## Primary Finding

DL-active periods produce opposite selector responses for Persistent and Reactive families.

### Persistent Family

When DL information becomes available:

```text
MeanReversion      increases
PhaseAware         decreases
TrendFollowing     decreases
```

### Reactive Family

When DL information becomes available:

```text
MeanReversion      decreases
PhaseAware         increases
TrendFollowing     varies by subgroup
```

This sign reversal appears consistently across the corrected experiment set.

------

## Interpretation

The selector does not merely become "more confident."

Instead:

- Persistent-family DL information pushes topology toward Mean Reversion.
- Reactive-family DL information pushes topology away from Mean Reversion.

This is the strongest replicated finding currently available.

------

# 5. Early-Stage Phase Modelling Analysis

## Motivation

A key objection to the family-topology hypothesis is that the observed differences might be created by downstream MPML components such as:

- StrategySelector_Dynamic
- Volatility Guard
- USD-role routing
- JPY-specific pip handling
- PhaseAware routing logic

To test this, the earliest selector-free MPML artifact was analyzed:

results_ml__dl_enabled.csv

This artifact is produced before walk-forward evaluation and before dynamic strategy selection.

---

## Method

The following models were compared:

- Baseline (No Phases)
- Phase as Feature
- Separate Phase Models

For each run and pair, the accuracy uplift of:

Separate Phase Models

relative to

Baseline (No Phases)

was computed.

---

## Results

A strong family-dependent response emerged.

### Persistent Family

Average uplift from Separate Phase Models:

+0.00498

### Reactive Family

Average uplift from Separate Phase Models:

-0.00512

---

## Regime Consistency

The sign of the effect remained stable across all four regimes.

Persistent:

- HVR: positive
- HVTF: positive
- LVR: positive
- LVTF: positive

Reactive:

- HVR: negative
- HVTF: negative
- LVR: negative
- LVTF: negative

No regime produced a sign reversal.

---

## Interpretation

This is the earliest currently identified family signal in the MPML pipeline.

Persistent-family markets appear to benefit from explicit phase segmentation.

Reactive-family markets appear to be harmed by explicit phase segmentation.

Importantly, this effect appears before:

- selector topology
- volatility-guard logic
- USD-role overrides
- JPY-specific routing

and therefore cannot be explained solely by downstream MPML heuristics.

---

## Significance

This finding substantially weakens the hypothesis that family-level topology differences are created by the selector itself.

Instead, the evidence suggests that family structure is already present in the predictive learning problem before strategy selection occurs.

---

### 5.1 Pair-Level Phase Uplift Analysis

#### Motivation

The family-level phase-modelling analysis demonstrated a strong split between persistent and reactive families. However, family averages can potentially hide substantial heterogeneity between individual currency pairs.

To determine whether the family effect was broadly distributed or concentrated in a small number of pairs, a pair-level phase uplift analysis was performed.

------

#### Method

For each pair and run, phase uplift was defined as:

```
Separate Phase Models (avg)
-
Baseline (No Phases)
```

using the earliest selector-free artifact:

```
results_ml__dl_enabled.csv
```

This artifact is generated before:

- StrategySelector_Dynamic
- Volatility Guard
- USD-role routing
- JPY-specific pip handling
- PhaseAware routing

and therefore provides the cleanest available measure of the intrinsic usefulness of phase segmentation.

------

#### Results

| Pair   | Mean Phase Uplift |
| ------ | ----------------- |
| EURUSD | +0.0183           |
| GBPUSD | +0.0133           |
| EURAUD | +0.0063           |
| USDJPY | -0.0008           |
| EURGBP | -0.0015           |
| GBPJPY | -0.0020           |
| EURCHF | -0.0058           |
| EURJPY | -0.0075           |
| USDCHF | -0.0095           |
| NZDUSD | -0.0115           |

Several observations immediately stand out.

------

#### EURUSD and GBPUSD

EURUSD and GBPUSD emerged as exceptionally strong beneficiaries of phase segmentation.

EURUSD produced:

- Mean uplift: +0.0183
- Positive uplift in 100% of runs
- Minimal variation across regimes

GBPUSD similarly produced:

- Mean uplift: +0.0133
- Positive uplift in 100% of runs

Both pairs exhibited remarkably stable behavior across:

- HVR
- HVTF
- LVR
- LVTF

suggesting that the effect is intrinsic to the pair rather than being driven by specific market regimes.

------

#### NZDUSD: A Persistent-Family Outlier

The most surprising result was NZDUSD.

Despite belonging to the persistent family, NZDUSD produced:

- Mean uplift: -0.0115
- Positive uplift in 0% of runs

This negative effect was larger in magnitude than several reactive-family pairs.

NZDUSD therefore behaves more like a reactive-family member than a persistent-family member with respect to phase segmentation.

This finding provides the first direct evidence supporting the sub-family hypothesis that emerged during earlier ABM experiments.

------

#### Pair Effects vs Regime Effects

For most pairs, phase uplift remained remarkably stable across all four regimes.

Examples:

EURUSD:

- HVR: +0.0196
- HVTF: +0.0181
- LVR: +0.0183
- LVTF: +0.0172

NZDUSD:

- HVR: -0.0120
- HVTF: -0.0103
- LVR: -0.0114
- LVTF: -0.0121

The consistency of these values suggests that pair-specific behavior is substantially stronger than regime-specific variation.

------

#### Family Contribution Analysis

The family effect was not driven by a single dominant pair.

The strongest positive contributors were:

- EURUSD
- GBPUSD
- EURAUD

while NZDUSD acted in the opposite direction and partially cancelled the persistent-family effect.

This indicates that the family split is broadly distributed rather than being created by a single outlier.

------

#### Sentiment Contribution

A separate comparison between sentiment-enabled and no-sentiment runs showed only very small changes in phase uplift.

Typical sentiment effects were approximately:

```
0.0001 – 0.0017
```

which is roughly an order of magnitude smaller than the observed pair-level phase uplifts.

At this early stage of the pipeline, phase structure appears to be the dominant signal, while sentiment contributes only marginally.

------

#### Interpretation

The pair-level analysis refines the family hypothesis in two important ways.

First, it confirms that the family effect is not uniformly distributed across all members of a family.

Second, it reveals evidence of internal family structure, particularly within the persistent family.

The strongest beneficiaries of phase segmentation are:

- EURUSD
- GBPUSD
- EURAUD

while NZDUSD behaves as a persistent-family outlier.

This suggests that persistent and reactive families may themselves consist of smaller behavioral sub-families, a possibility that warrants further investigation in future transfer and ablation experiments.

---

### 5.2 Early vs Late: Phase Uplift and Selector Topology

#### Motivation

The pair-level phase uplift analysis demonstrated that certain pairs consistently benefited from phase segmentation while others did not.

This naturally raises the question:

> Do pairs that benefit from phase segmentation subsequently become more MeanReversion-oriented during strategy selection?

If so, this would provide a mechanistic link between the earliest stage of the MPML pipeline and the later selector topology.

------

#### Method

The analysis combined two previously generated datasets.

**Early-stage signal**

Phase uplift was measured using:

```text
results_ml__dl_enabled.csv
```

where:

```text
Phase Uplift
=
Separate Phase Models (avg)
−
Baseline (No Phases)
```

A positive value indicates that phase segmentation improved predictive accuracy.

A negative value indicates that phase segmentation reduced predictive accuracy.

This artifact is generated before:

- StrategySelector_Dynamic
- Volatility Guard
- USD-role routing
- JPY-specific selector effects

and therefore represents the cleanest available estimate of the intrinsic usefulness of phase segmentation.

**Late-stage signal**

Selector topology was measured using:

```text
selector_state_timeline__dl_enabled.csv
```

For each pair and regime, the fraction of bars spent in:

- MeanReversion (MR)
- PhaseAware (PA)
- TrendFollowing (TF)

was calculated.

The analysis was performed separately for:

- HVR
- HVTF
- LVR
- LVTF

producing forty observations:

```text
10 pairs × 4 regimes
```

------

#### Correlation Metric

Relationships were quantified using the Pearson correlation coefficient.

Interpretation:

| Correlation | Meaning                       |
| ----------- | ----------------------------- |
| +1.0        | Perfect positive relationship |
| 0.0         | No linear relationship        |
| -1.0        | Perfect negative relationship |

All correlation values reported below are Pearson correlation coefficients.

------

#### Full-Timeline Results

Across all forty pair-regime observations:

| Relationship                 | Correlation |
| ---------------------------- | ----------- |
| Phase Uplift vs MR Occupancy | +0.235      |
| Phase Uplift vs PA Occupancy | -0.036      |
| Phase Uplift vs TF Occupancy | -0.362      |

The expected relationship between phase uplift and MeanReversion occupancy was present but relatively weak.

The strongest observed relationship was instead a negative correlation between phase uplift and TrendFollowing occupancy.

This suggests that pairs benefiting from phase segmentation tend to spend less time in TrendFollowing rather than substantially more time in MeanReversion.

------

#### DL-Active Results

The analysis was repeated using only bars where:

```text
dl_overlap_state == partial
```

indicating periods where DL information was available.

Results were highly similar:

| Relationship                 | Correlation |
| ---------------------------- | ----------- |
| Phase Uplift vs MR Occupancy | +0.271      |
| Phase Uplift vs PA Occupancy | -0.059      |
| Phase Uplift vs TF Occupancy | -0.338      |

The overall pattern remained unchanged.

This suggests that the relationship between early phase usefulness and later selector topology is largely independent of DL availability.

------

#### Regime-Level Analysis

The strongest regime-specific relationship was observed in LVR:

| Regime | Relationship                 | Correlation |
| ------ | ---------------------------- | ----------- |
| LVR    | Phase Uplift vs TF Occupancy | -0.518      |

This represents a moderate negative relationship.

Pairs that benefited from phase segmentation in Low-Volatility Ranging environments tended to spend substantially less time in TrendFollowing later in the pipeline.

Other regimes produced similar but weaker results:

| Regime | Strongest Correlation |
| ------ | --------------------- |
| HVR    | TF Occupancy = -0.258 |
| HVTF   | MR Occupancy = +0.292 |
| LVTF   | TF Occupancy = -0.279 |

No regime produced a strong positive relationship between phase uplift and MeanReversion occupancy.

------

#### The NZDUSD Counterexample

A particularly important result was provided by NZDUSD.

Earlier analysis showed:

```text
Mean Phase Uplift ≈ -0.0115
```

making NZDUSD one of the strongest negative phase-segmentation examples.

However, selector topology showed:

```text
MR Occupancy ≈ 65–78%
```

depending on regime.

This makes NZDUSD one of the most MeanReversion-dominated pairs in the entire dataset despite receiving little or no benefit from phase segmentation.

NZDUSD therefore serves as a direct counterexample to the simple hypothesis:

```text
Phase Usefulness
    →
MeanReversion Occupancy
```

and demonstrates that MeanReversion allocation alone cannot explain the observed family effect.

------

#### Interpretation

The original working hypothesis was:

```text
Phase Usefulness
    →
More MeanReversion
    →
Family Effect
```

The evidence does not strongly support this explanation.

Instead, the data suggests a weaker but more consistent relationship:

```text
Phase Usefulness
    →
Reduced TrendFollowing Occupancy
```

The strongest positive phase-uplift pairs:

- EURUSD
- GBPUSD
- EURAUD

all exhibited relatively low TrendFollowing exposure.

Conversely, several strongly negative phase-uplift pairs exhibited substantially larger TrendFollowing allocations.

The effect is not large enough to be considered a complete explanation of the family phenomenon, but it is currently the strongest mechanistic bridge identified between the early predictive stage of the MPML pipeline and the later selector topology.

------

#### Conclusion

The analysis provides little evidence that phase usefulness is primarily expressed through increased MeanReversion allocation.

Instead, the most consistent relationship observed throughout the study is a reduction in TrendFollowing occupancy.

The effect survives both:

- full-timeline analysis
- DL-active-only analysis

suggesting that it reflects a broader property of pair behaviour rather than a phenomenon driven exclusively by DL information availability.

Further work is required to determine whether this reduction in TrendFollowing exposure is a direct consequence of phase segmentation or merely a correlated downstream effect.

---

# 6. DL-Active Topology Analysis

## Method

Selector State Timeline (SST) files were analyzed.

Topology during:

```text
dl_overlap_state = partial
```

was compared against:

```text
dl_overlap_state = missing
```

for each run.

This isolates periods where DL information is actually present.

------

## Family-Level Results

Topology shifts during DL-active periods:

| Family     |  MR Δ |  PA Δ |  TF Δ |
| ---------- | ----: | ----: | ----: |
| Persistent | +6.23 | -2.85 | -3.38 |
| Reactive   | -7.84 | +6.62 | +1.22 |

---

### Interpretation

When DL information becomes available:

Persistent-family selectors become substantially more MeanReversion-oriented.

Reactive-family selectors become substantially less MeanReversion-oriented and more PhaseAware-oriented.

The effect is approximately symmetric between the two families.

---

## Regime-Level Results

Persistent:

| Regime |  MR Δ |
| ------ | ----: |
| HVR    | +9.46 |
| HVTF   | +2.67 |
| LVR    | +5.16 |
| LVTF   | +7.64 |

Reactive:

| Regime |  MR Δ |
| ------ | ----: |
| HVR    | -6.91 |
| HVTF   | -9.01 |
| LVR    | -7.65 |
| LVTF   | -7.80 |

The sign remains consistent across every regime.

No family-regime combination produced a sign reversal.

---

## Sample Size Validation

DL-active coverage remained substantial:

- HVR: ~4k–4.5k bars
- HVTF: ~8.6k–9.6k bars
- LVR: ~3.8k–3.9k bars
- LVTF: ~8.9k bars

The effect therefore cannot be attributed to isolated DL-active observations.

------

# 7. Transition Analysis

A transition analysis was performed to test whether occupancy changes arise from direct migration between strategies.

Transitions examined:

```text
MR → PA
PA → MR
MR → TF
TF → MR
PA → TF
TF → PA
```

------

## Result

The expected directional migration pattern was not observed.

Instead:

### Persistent

Both:

```text
MR → PA
PA → MR
```

increase during DL-active periods.

### Reactive

Transition shifts are generally small.

------

## Interpretation

Occupancy changes are not explained by transition frequency alone.

This suggests:

- state persistence
- strategy duration
- dwell time effects

may be more important than transition counts.

------

# 8. Strategy Duration Analysis

## Motivation

Transition analysis produced an unexpected result.

The observed occupancy shifts:

Persistent:
- MeanReversion increases

Reactive:
- MeanReversion decreases
- PhaseAware increases

could not be fully explained by directional migration between strategies.

In particular, transition frequencies changed only modestly and often symmetrically, suggesting that another mechanism was responsible for the topology changes.

To investigate this, strategy persistence was analyzed.

---

## Method

For each pair and run, continuous blocks of:

- MeanReversion
- PhaseAware
- TrendFollowing

were identified from the Selector State Timeline (SST).

For each block, the duration in bars was recorded.

Durations were then compared between:

```text
dl_overlap_state = missing
```

and

```
dl_overlap_state = partial
```

to isolate periods where DL information was available.

------

## Results

### Persistent Family

Average strategy duration (bars):

| Strategy       | Missing | Partial | Δ     |
| -------------- | ------- | ------- | ----- |
| MeanReversion  | 16.70   | 19.02   | +2.33 |
| PhaseAware     | 15.27   | 16.52   | +1.25 |
| TrendFollowing | 8.91    | 10.30   | +1.39 |

All strategies become more persistent during DL-active periods.

However, MeanReversion experiences the largest increase in duration.

This is consistent with the previously observed increase in MeanReversion occupancy.

------

### Reactive Family

Average strategy duration (bars):

| Strategy       | Missing | Partial | Δ     |
| -------------- | ------- | ------- | ----- |
| MeanReversion  | 10.31   | 4.46    | -5.84 |
| PhaseAware     | 17.72   | 18.95   | +1.23 |
| TrendFollowing | 10.21   | 10.18   | -0.02 |

The MeanReversion effect is particularly striking.

During DL-active periods, average MeanReversion duration falls from approximately 10.3 bars to 4.5 bars, representing a reduction of more than 50%.

At the same time, PhaseAware duration increases modestly while TrendFollowing remains largely unchanged.

------

## Interpretation

The duration analysis provides a plausible mechanism for the topology shifts observed throughout the study.

### Persistent Family

DL information appears to increase selector stability.

The selector remains committed to strategies for longer periods, with MeanReversion benefiting most strongly.

This provides a natural explanation for the observed increase in MeanReversion occupancy.

### Reactive Family

DL information appears to destabilize MeanReversion.

The selector still enters MeanReversion states, but exits them substantially more quickly.

Simultaneously, PhaseAware states become slightly more persistent.

This explains why large occupancy shifts can occur even when transition frequencies remain relatively stable.

------

## Key Insight

The primary effect of DL information is not necessarily to change where the selector moves.

Instead, DL information appears to change how long the selector remains committed to a selected strategy.

This distinction helps reconcile the transition analysis with the occupancy analysis and currently represents the strongest candidate mechanism underlying the family-specific topology effects.

---

## Duration and Performance

Duration changes were compared against walk-forward performance metrics.

The strongest relationship observed in the entire experiment matrix was between MeanReversion duration and performance.

Global correlations:

| Metric                    | Correlation |
| ------------------------- | ----------: |
| Sharpe Δ vs MR duration Δ |      -0.858 |
| Return Δ vs MR duration Δ |      -0.870 |

Family-level behavior was highly consistent:

### Persistent Family

- MeanReversion duration increased.
- Sharpe decreased.
- Returns decreased.

### Reactive Family

- MeanReversion duration decreased.
- Sharpe increased.
- Returns increased.

This suggests that family-specific changes in MeanReversion persistence are closely associated with the observed performance differences.

Importantly, the relationship appears directional:

- Longer MeanReversion persistence is associated with poorer performance.
- Shorter MeanReversion persistence is associated with improved performance.

Further work is required to determine whether MeanReversion persistence is the direct causal mechanism or a proxy for a deeper family-specific effect.

---

# 9. USD / JPY Bias Decomposition

## Motivation

A selector audit revealed a hard-coded volatility guard.

For USD-quote pairs:

```text
high volatility
    -> force TrendFollowing
```

This raised the possibility that family effects were merely artifacts of USD routing logic.

------

## Result

The family effect survives decomposition.

### Persistent No-USD

DL-active periods still show:

```text
MR up
PA down
TF down
```

with strong consistency.

### Reactive No-USD

DL-active periods still show:

```text
MR down
PA up
```

with strong consistency.

------

## Conclusion

The family effect is not primarily caused by the USD volatility guard.

The guard may amplify the effect, but cannot fully explain it.

------

## JPY Analysis

Reactive-family JPY and non-JPY pairs exhibit noticeably different response vectors.

This indicates that pair-level structure remains important and should be investigated further.

However, the overall family effect survives the decomposition.

------

# 10. MPML Selector Audit

A dedicated audit was performed to identify potential topology biases.

Key findings:

## Confirmed Mechanisms

### Volatility Guard

High-volatility conditions may force TrendFollowing behavior.

### Pair-Specific Selectors

Selectors are trained separately per pair.

### DL Feature Injection

DL features enter selector training and inference directly.

### Confidence Gating

Selector behavior influenced by confidence thresholds, hysteresis, and hold periods.

### PhaseAware Routing

PhaseAware internally routes between MR and TF depending on phase classification.

------

## Interpretation

The audit identified several mechanisms capable of influencing topology.

However, the USD/JPY decomposition suggests these mechanisms are insufficient to explain the family effect by themselves.

A separate audit document should be maintained for implementation details.

------

# 11. Current Interpretation

Current evidence suggests a two-stage mechanism:

Family Structure
    ↓
Different usefulness of phase information
    ↓
Different interpretation of DL signals
    ↓
Different selector attractors
    ↓
Different topology

The strongest evidence for this interpretation comes from:

1. Early-stage phase-modelling analysis.
2. DL-active topology analysis.
3. Transfer-topology convergence.
4. USD/JPY decomposition.

------

# 12. Open Questions

## Strategy Persistence Drivers

We have established that family topology differences are strongly associated with changes in strategy duration.

The remaining question is why DL information increases MeanReversion persistence for Persistent families while reducing MeanReversion persistence for Reactive families.

Potential explanations include:

- differences in DL feature distributions,
- differences in selector confidence,
- differences in regime occupancy,
- differences in pair composition,
- differences in volatility-guard interactions.

------

## Pair Clustering

Do:

- JPY pairs
- commodity pairs
- European pairs

form distinct response clusters?

------

## Regime Sensitivity

Which regimes produce the strongest family divergence?

Questions remain regarding:

- LVTF
- HVTF
- LVR
- HVR

under the corrected pipeline.

------

## Cross-Family Transfer

The most important pending experiment.

### Train Persistent → Evaluate Reactive

### Train Reactive → Evaluate Persistent

Potential outcomes:

- strong degradation
- weak degradation
- asymmetric degradation

would provide direct evidence regarding whether the families encode distinct predictive structures.

------

# 13. Working Conclusions

Current confidence is highest for the following statements:

1. Family effects are visible before strategy selection occurs.
2. Phase modelling helps Persistent families and harms Reactive families.
3. DL-active periods amplify family divergence.
4. Persistent families become more MeanReversion-oriented during DL-active periods.
5. Reactive families become less MeanReversion-oriented during DL-active periods.

---

### Duration-Performance Reassessment

A strong global relationship was observed between MeanReversion duration changes and performance changes (r ≈ -0.86).

However, this relationship disappeared after conditioning on family identity:

- Persistent: r = 0.33 (not significant)
- Reactive: r = 0.14 (not significant)

This indicates that MeanReversion duration is primarily a marker of the family effect rather than the direct driver of performance differences.

The dominant explanatory variable remains pair family:

- Reactive family consistently benefits from DL information.
- Persistent family consistently deteriorates when DL information is introduced.

The selector topology changes remain important because they reveal how the families respond differently to DL information, but they do not appear to directly explain performance variation within each family.

---

## Transfer Topology Convergence (2026-06-05)

Cross-family transfer experiments reveal almost complete convergence toward the evaluation-family topology.

Using an L1 occupancy-distance metric:

- Persistent→Reactive:
  - distance to Reactive baseline = 0.215
  - distance to Persistent baseline = 57.349
  - 266.7× closer to Reactive

- Reactive→Persistent:
  - distance to Persistent baseline = 0.424
  - distance to Reactive baseline = 57.112
  - 134.7× closer to Persistent

This indicates that selector occupancy structure is overwhelmingly determined by the market family being traded rather than the family used during training.

The evaluation universe acts as a strong attractor for selector topology.

---

The current DL-active analysis uses the framework's
`partial` overlap state, which corresponds to a very broad
DL coverage range (5–95%).

Future work should repeat the analysis using explicit
`dl_overlap_pct` thresholds (e.g. ≥50%, ≥80%, ≥95%)
to determine whether high-confidence DL availability
strengthens the observed relationships.
