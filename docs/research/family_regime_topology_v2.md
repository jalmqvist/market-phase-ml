# Family-Regime Topology Study v2

## Status

**Version:** v2 (Transfer-Validated Baseline)

**Purpose:** Establish the current best-supported understanding of how Deep Learning (DL) information influences Market Phase ML (MPML) after all known regime-loading, feature-attachment, artifact-selection, and transfer-validation issues were identified and corrected.

This document supersedes v1.

Unlike earlier studies, the current version incorporates:

- corrected regime-specific DL surfaces,
- verified DL feature attachment,
- cross-family transfer experiments,
- dense-overlap analysis using explicit `dl_overlap_pct` thresholds,
- transfer-influence validation under high-overlap conditions.

The primary objective is no longer simply to determine whether DL information changes selector behavior.

The objective is now:

> Where does DL information influence MPML, how strong is that influence, and what mechanisms explain the observed family-level differences?

------

# Table of Contents

1. Executive Summary
2. Why Transfer Experiments Matter
   - 2.1 Transfer Definition
   - 2.2 Why Transfer Validation Was Necessary
3. Experimental Validation
   - 3.1 Validation Philosophy
   - 3.2 Major Audit Findings
   - 3.3 Current Confidence Level
4. Early Pipeline Evidence
   - 4.1 Why Early-Stage Analysis Matters
   - 4.2 Phase-Modelling Experiments
   - 4.3 Family-Level Results
   - 4.4 Interpretation
5. Family Topology Effects
   - 5.1 Family Decomposition
   - 5.2 Aggregate Occupancy Results
   - 5.3 Duration Analysis
   - 5.4 Pair-Level Analysis
   - 5.5 Working Interpretation
6. Transfer Experiments
   - 6.1 Transfer Definition
   - 6.2 Why Transfer Validation Was Necessary
   - 6.3 Aggregate Topology Convergence
   - 6.4 A New Question
   - 6.5 Local Influence Analysis
   - 6.6 Agreement Results
   - 6.7 Transfer Findings
7. DL Overlap Analysis
   - 7.1 Defining DL Overlap
   - 7.2 Overlap Distribution
   - 7.3 Dense Overlap Is Not Uniformly Distributed
   - 7.4 The Dense-Overlap Population
   - 7.5 Reinterpreting Transfer Results
   - 7.6 Dense-Overlap Agreement Analysis
   - 7.7 Updated Interpretation
   - 7.8 Overlap Prevalence and Regime Concentration
8. Updated Working Model
   - 8.1 The Original Hypothesis
   - 8.2 Evaluation Environment Dominates Topology
   - 8.3 DL Information Still Matters
   - 8.4 The Importance of Overlap
   - 8.5 Why Trend-Following Regimes Matter
   - 8.6 Family Effects Revisited
   - 8.7 Current Best Explanation
9. Open Questions and Future Directions
   - 9.1 Why Do Family Effects Appear So Early?
   - 9.2 What Creates Dense Overlap Episodes?
   - 9.3 Can MPML Exploit Overlap Explicitly?
   - 9.4 Agent-Based Modelling (ABM)
   - 9.5 DL Architecture Development
   - 9.6 MPML Architecture Development
   - 9.7 Final Assessment

---

# 1. Executive Summary

Five findings currently have the highest confidence.

### Finding 1

Persistent and Reactive families produce systematically different *selector topologies* (i.e. different patterns of strategy occupancy,
residence time, and routing behavior).

When DL information becomes available:

- Persistent-family selectors become more MeanReversion-oriented.
- Reactive-family selectors become less MeanReversion-oriented and more PhaseAware-oriented.

This sign reversal survives:

- regime decomposition,
- sentiment decomposition,
- USD/JPY decomposition,
- all currently known bug fixes.

### Finding 2

The family effect appears before strategy selection.

Phase-segmentation experiments performed on the earliest MPML artifact (`results_ml__dl_enabled.csv`) show opposite responses to explicit phase modelling:

- Persistent family: average accuracy uplift ≈ +0.005
- Reactive family: average accuracy uplift ≈ -0.005

This effect appears before:

- selector routing,
- volatility guards,
- pair-specific execution logic,
- walk-forward evaluation.

The evidence therefore suggests that the family effect is already present in the predictive learning problem itself.

### Finding 3

Cross-family transfer experiments reveal strong topology convergence.

When Persistent DL artifacts are transferred into Reactive environments, selector topology converges almost completely toward the Reactive baseline.

When Reactive DL artifacts are transferred into Persistent environments, topology converges almost completely toward the Persistent baseline.

Observed convergence ratios exceed 100×.

This indicates that aggregate selector topology is determined primarily by the evaluation universe rather than by the family used to generate the transferred DL artifact.

### Finding 4

Transfer influence nevertheless remains detectable locally.

Although aggregate topology converges, selector agreement decreases substantially during periods of high DL overlap.

Typical agreement levels fall from approximately:

95%
→
83–85%

under dense-overlap conditions.

This demonstrates that transferred DL information reaches runtime and influences selector decisions even though aggregate topology remains dominated by the evaluation family.

### Finding 5

Dense DL overlap is structurally concentrated in trend-following regimes.

Using explicit overlap thresholds:

- 35–37% of HVTF/LVTF observations exceed 10% overlap.
- 22–25% exceed 20% overlap.
- 13–17% exceed 30% overlap.

In contrast:

- HVR and LVR contain essentially no observations above 20% overlap.

At overlap ≥20%:

HVTF + LVTF account for 100% of the dense-overlap population.

This finding explains why transfer-induced selector divergence is observed almost exclusively in HVTF and LVTF.

------

# 2. Why Transfer Experiments Matter

## Motivation

The original family-topology studies established that Persistent and Reactive families produce different selector behavior.

However, those results alone do not reveal where the family signal originates.

Two competing explanations remained possible.

### Hypothesis A: DL-Driven Family Structure

Persistent and Reactive DL models learn genuinely different predictive structures.

If this is true, transferring DL artifacts between families should alter selector behavior.

### Hypothesis B: Market-Driven Family Structure

Selector topology is determined primarily by the evaluation market itself.

Under this explanation, transferred DL artifacts should have only limited influence.

The transfer experiments were designed specifically to distinguish between these possibilities.

------

## What "Transfer" Means

The transfer experiments are not classical transfer-learning experiments.

No MPML selector model is transferred.

No MPML phase predictor is transferred.

No strategy-selection logic is transferred.

Instead, the transferred object is the DL prediction surface produced by MSML.

The pipeline can be summarized as:

MSML source family
↓
DL predictions
↓
attached as MPML features
↓
phase predictor training
↓
selector training
↓
selector execution

What is transferred:

- DL prediction surface

What is not transferred:

- selector model
- phase predictor model
- strategy mapping
- runtime routing logic

This distinction is critical.

The transfer experiments therefore measure:

> How much influence does a transferred DL prediction surface exert on the final MPML selector?

rather than:

> Can an entire model trained on one family be reused on another?

------

## Why Transfer Validation Was Necessary

Several earlier engineering audits revealed situations where DL artifacts were attached incorrectly or had little practical influence on runtime behavior.

Examples included:

- regime-inference failures,
- feature-attachment failures,
- awareness-mode inconsistencies,
- artifact-selection bugs.

As a result, observing identical topologies between transferred and baseline runs is not sufficient evidence that transfer has no effect.

The central question becomes:

> Does the transferred DL information successfully propagate through the entire MPML pipeline and influence selector decisions?

The transfer analysis therefore examines both:

1. aggregate topology convergence,
2. local selector divergence under high-overlap conditions.

Together these analyses provide evidence regarding both reachability and influence.

------

# 3. Experimental Validation

## Motivation

The family-topology study evolved over multiple rounds of engineering audits and experimental re-analysis.

Several early findings were later shown to be influenced by implementation issues rather than genuine market effects.

As a result, a substantial portion of the project shifted from hypothesis testing to validation.

The purpose of this section is to establish which experimental artifacts can now be considered trustworthy and therefore suitable for scientific interpretation.

------

## 3.1 Validation Philosophy

A recurring challenge throughout the project was distinguishing between:

- genuine family effects,
- data-pipeline artifacts,
- feature-attachment failures,
- transfer-configuration errors.

Several analyses initially appeared to support strong conclusions, only to later reveal that the underlying DL information was either attached incorrectly or not reaching the intended stage of the MPML pipeline.

Consequently, no result is considered valid unless the following chain can be verified:

DL artifact
→
feature attachment
→
phase predictor
→
selector feature matrix
→
selector decisions
→
selector_state_timeline

This end-to-end validation principle became increasingly important during the transfer studies, where topology similarity alone could not be interpreted as evidence of successful transfer.

------

## 3.2 Major Audit Findings

Multiple engineering audits identified issues capable of materially altering experimental interpretation.

Examples included:

- regime-specific DL attachment issues,
- feature-set inference issues,
- awareness-mode inconsistencies,
- transfer artifact selection issues,
- runtime propagation issues.

The practical consequence of these audits was that several early interpretations required re-validation using corrected artifacts.

Although these findings initially slowed experimental progress, they substantially increased confidence in the final results by ensuring that family-level conclusions were based on verified runtime behavior rather than configuration assumptions.

------

## 3.3 Current Confidence Level

The current evidence base is built entirely on post-audit artifacts.

The following claims now have high confidence:

- family topology differences are real,
- phase-modelling effects differ between families,
- transferred DL surfaces reach runtime,
- selector decisions respond to transferred DL information,
- overlap statistics are computed correctly,
- topology convergence results are not artifacts of missing feature propagation.

This validation foundation is critical because all subsequent sections depend on these assumptions being true.

------

# 4. Early Pipeline Evidence

## Motivation

Topology analysis reveals how selectors behave.

However, topology alone cannot determine where family differences originate.

A central question throughout the study has therefore been:

> Does the family effect emerge during selector routing, or is it already present in the predictive learning problem itself?

To answer this question, we move as far upstream as possible and analyze the earliest MPML artifacts.

------

## 4.1 Why Early-Stage Analysis Matters

The earliest MPML outputs are generated before:

- selector routing,
- volatility guards,
- runtime switching,
- execution-layer adaptations.

If family effects are visible at this stage, then the origin of the effect must lie deeper than selector mechanics.

In other words:

Early-stage differences imply that Persistent and Reactive families create genuinely different prediction problems.

------

## 4.2 Phase-Modelling Experiments

The primary early-stage analysis examined explicit phase modelling using:

results_ml__dl_enabled.csv

For each pair, three approaches were compared:

- Baseline (No Phases)
- Phase as Feature
- Separate Phase Models

The objective was to determine whether explicit phase information improves predictive performance.

------

## 4.3 Family-Level Results

The results revealed a striking asymmetry.

Persistent-family experiments generally benefited from explicit phase modelling.

Reactive-family experiments generally did not.

Average effects were approximately:

Persistent:
+0.005 accuracy uplift

Reactive:
-0.005 accuracy change

Although the absolute values are modest, the sign difference is consistent across multiple decompositions.

This finding is important because it appears before selector construction.

The result therefore suggests that family differences are already present in the predictive structure of the learning problem itself.

------

## 4.4 Interpretation

The early-stage evidence shifts the working hypothesis away from:

> Selectors create family behavior.

and toward:

> Selectors inherit family behavior from upstream predictive structure.

The selector remains important, but it increasingly appears to amplify and organize information that already exists rather than generating the family distinction on its own.

This interpretation is consistent with later topology and transfer results.

------

## What "Topology" Means In This Study

The term topology is used throughout this document in a specialized MPML sense.

It does not refer to mathematical topology, graph topology, or model architecture.

Instead, topology describes the statistical structure of selector behavior.

More specifically, topology refers to:

- strategy occupancy,
- strategy residence time,
- switching behavior,
- regime-specific strategy allocation,
- the distribution of selector decisions through time.

For example:

A selector that spends 70% of its time in MeanReversion and 30% in PhaseAware has a different topology from a selector that spends 30% in MeanReversion and 70% in PhaseAware, even if their overall performance is identical.

Similarly, two selectors may exhibit identical occupancy percentages while possessing different topologies if one switches frequently and the other remains in the same state for extended periods.

Throughout this document, family topology therefore refers to the *characteristic structure of selector behavior* induced by a particular family.

The topology analyses seek to answer:

"How does a family organize strategy usage?"

rather than:

"Which family produces the highest returns?"

This distinction is important because topology provides insight into the mechanisms by which information influences MPML behavior, whereas performance alone often does not.

---

# 5. Family Topology Effects

## Motivation

Having established that family differences emerge early in the pipeline, the next question becomes:

> How do these differences manifest in selector behavior?

Topology analysis addresses this question by examining selector occupancy, strategy preference, and regime-specific allocation patterns.

The goal is not merely to measure performance.

The goal is to understand how information flows through the selector architecture.

------

## 5.1 Family Decomposition

The family experiments compare:

Persistent
vs
Reactive

under identical MPML infrastructure.

This isolates the effect of family construction while holding:

- selector logic,
- execution rules,
- volatility guards,
- strategy universe

constant.

The resulting topology differences can therefore be attributed primarily to family structure.

------

## 5.2 Aggregate Occupancy Results

Across the family experiments, a consistent pattern emerges.

Persistent-family selectors exhibit:

- increased MeanReversion occupancy,
- reduced PhaseAware occupancy.

Reactive-family selectors exhibit:

- reduced MeanReversion occupancy,
- increased PhaseAware occupancy.

The direction of the effect remains stable across:

- sentiment-aware runs,
- sentiment-blind runs,
- USD pairs,
- non-USD pairs,
- regime-specific decompositions.

This stability significantly increases confidence that the effect is structural rather than incidental.

------

## 5.3 Duration Analysis

Occupancy alone cannot distinguish between:

- frequent short-lived visits,
- persistent state residence.

Duration analysis therefore examined selector persistence within each strategy state.

The results showed that family differences are not solely explained by switching frequency.

Instead, family structure influences the characteristic residence times of selected strategies.

This indicates that the selector is not merely choosing different strategies; it is organizing strategy usage differently over time.

------

## 5.4 Pair-Level Analysis

Pair-level decomposition was performed to determine whether family effects were driven by a small number of outlier instruments.

The results showed broad participation across the evaluation universe.

While individual pairs vary in magnitude, the overall family signal is not attributable to a single dominant instrument.

This finding strengthens the interpretation that family effects represent a portfolio-level phenomenon rather than isolated pair-specific behavior.

------

## 5.5 Working Interpretation

At the topology level, the evidence supports the following interpretation:

Persistent-family information appears to encourage greater exploitation of mean-reverting opportunities.

Reactive-family information appears to encourage greater reliance on adaptive phase-sensitive behavior.

The effect is visible:

- early in the pipeline,
- at the selector level,
- across multiple decompositions,
- after correction of known implementation issues.

The remaining question is whether these topology differences originate directly from DL information or primarily from the evaluation environment itself.

The transfer experiments address that question.

---

# 6. Transfer Experiments

## Motivation

The family-topology experiments established that Persistent and Reactive families produce different selector behavior.

However, those experiments alone cannot determine whether the observed differences originate from:

- the DL information itself,
- the evaluation market,
- the MPML selector architecture,
- or some combination of the above.

The transfer experiments were designed to isolate these possibilities.

The central question is:

> If DL information generated by one family is injected into another family, does selector behavior follow the transferred DL information or the evaluation environment?

This is one of the most important questions in the entire study because it determines whether family identity is fundamentally a property of the DL signal or a property of the MPML environment.

------

## 6.1 Transfer Definition

The transfer experiments are frequently misunderstood as classical transfer-learning studies.

They are not.

No selector model is transferred.

No phase predictor is transferred.

No MPML runtime logic is transferred.

Instead, the transferred object is the *DL prediction surface* generated by MSML.

The pipeline is:

Source Family
→
MSML DL Predictions
→
MPML Feature Attachment
→
Phase Predictor Training
→
Selector Training
→
Runtime Execution

Therefore:

Transferred:

- DL prediction surface

Not transferred:

- selector model
- phase predictor model
- strategy mappings
- routing logic
- volatility guards

The experiments therefore answer:

> How much influence does a transferred DL prediction surface exert on the final MPML selector?

rather than:

> Can an entire model trained on one family be reused elsewhere?

This distinction is critical when interpreting the results.

------

## 6.2 Why Transfer Validation Was Necessary

Historically, several engineering audits revealed situations where DL information appeared to exist in configuration files but exerted little or no influence on runtime behavior.

Examples included:

- regime attachment issues,
- feature propagation issues,
- awareness-mode inconsistencies,
- artifact selection errors.

As a result, observing topology similarity alone cannot establish whether transfer is working.

The transfer experiments therefore pursue two separate goals:

1. Verify that transferred information reaches runtime.
2. Measure how strongly transferred information influences selector behavior.

Both questions must be answered before scientific interpretation becomes possible.

------

## 6.3 Aggregate Topology Convergence

The first transfer analysis compared selector topology between:

Persistent baseline
vs
Reactive→Persistent transfer

and

Reactive baseline
vs
Persistent→Reactive transfer

The expectation was that if DL information dominates selector behavior, transferred runs should resemble their source family.

This did not occur.

Instead, transferred runs converged strongly toward the evaluation family.

Observed convergence ratios exceeded 100×.

Regardless of transfer direction:

- Persistent environments remained Persistent-like.
- Reactive environments remained Reactive-like.

This result indicates that aggregate selector topology is determined primarily by the evaluation environment rather than by the family that generated the transferred DL artifact.

------

## 6.4 A New Question

The convergence result immediately creates a new problem.

Topology similarity does not imply that transfer information is irrelevant.

Two interpretations remain possible.

### Interpretation A

Transferred DL information has little influence.

Under this explanation, topology convergence occurs because the evaluation family completely dominates the transferred signal.

### Interpretation B

Transferred DL information influences selector decisions locally, but those changes are too small to alter aggregate topology.

The topology analysis alone cannot distinguish between these explanations.

A more sensitive analysis is required.

------

## 6.5 Local Influence Analysis

To measure local effects, selector decisions were compared directly.

For matched timestamps, pairs, and folds:

- baseline selector decisions,
- transferred selector decisions

were compared using strategy agreement metrics.

The objective was to determine whether *transferred DL information changes individual selector decisions* even when aggregate topology remains stable.

------

## 6.6 Agreement Results

At the aggregate level, transferred runs exhibit very high agreement with baseline selectors.

Typical agreement values fall within:

93–96%

depending on family and experiment variant.

At first glance, this appears to support the interpretation that transfer effects are weak.

However, this conclusion changes substantially once overlap information is introduced.

------

## 6.7 Transfer Findings

The transfer experiments ultimately produce two simultaneous conclusions.

### Global Conclusion

Transferred DL surfaces do not substantially alter aggregate selector topology.

Evaluation-family structure dominates at the topology level.

### Local Conclusion

Transferred DL surfaces do influence selector decisions.

The effect is not large enough to change family identity, but it is large enough to alter many individual routing decisions.

This distinction between:

*global topology*

and

*local decision behavior*

became one of the central findings of the study.

The next section explains why these local effects appear only in specific regions of the state space.

------

# 7. DL Overlap Analysis

## Motivation

The transfer experiments established that transferred DL information reaches runtime.

However, the magnitude of the effect varied substantially across observations.

This raised an important question:

> Under what conditions should transferred DL information be expected to matter?

The original overlap analyses could not answer this question because they grouped together all observations classified as:

dl_overlap_state = partial

This category spans a very large range of overlap values and therefore mixes weak and strong overlap conditions.

A more precise analysis was required.

------

## 7.1 Defining DL Overlap

DL overlap measures the degree of agreement between DL-derived information surfaces.

Higher overlap implies stronger consistency and persistence within the DL signal.

Low overlap implies weak or transient agreement.

The overlap metric therefore provides a natural way to estimate when transferred DL information should have the greatest opportunity to influence MPML decisions.

------

## 7.2 Overlap Distribution

The first finding was unexpected.

DL overlap is extremely sparse.

Across all regimes:

approximately 50–65% of observations have:

dl_overlap_pct = 0

Examples:

HVR:
61.73% zero overlap

HVTF:
49.93% zero overlap

LVR:
61.72% zero overlap

LVTF:
54.94% zero overlap

The median overlap is therefore:

0.00

for three of the four regimes.

This result initially appeared surprising but was later verified through direct distribution analysis.

------

## 7.3 Dense Overlap Is Not Uniformly Distributed

Although overlap is sparse overall, it is not distributed evenly across regimes.

Trend-following regimes contain dramatically more dense-overlap observations.

Global regime statistics:

HVR:
7.87% ≥10%
0.00% ≥20%

HVTF:
35.39% ≥10%
22.55% ≥20%

LVR:
3.94% ≥10%
0.00% ≥20%

LVTF:
36.84% ≥10%
24.71% ≥20%

This represents approximately:

4–9× enrichment

of dense-overlap observations within HVTF and LVTF compared with HVR and LVR.

------

## 7.4 The Dense-Overlap Population

The most important overlap result emerges when examining the composition of the dense-overlap population itself.

For overlap ≥20%:

HVTF:
47.72%

LVTF:
52.28%

HVR:
0.00%

LVR:
0.00%

The result remains essentially unchanged at:

≥30%
≥40%

thresholds.

This means:

> Every dense-overlap observation belongs to either HVTF or LVTF.

The result is remarkably clean and immediately explains several earlier findings.

------

## 7.5 Reinterpreting Transfer Results

The transfer analyses showed that selector divergence appears primarily in:

HVTF
LVTF

while HVR and LVR contributed little usable evidence.

Initially this could have been interpreted as a regime-specific transfer effect.

The overlap analysis suggests a different explanation.

The transfer experiments are not primarily discovering:

"Transfer matters in HVTF/LVTF."

They are discovering:

"Transfer matters when dense overlap exists."

Dense overlap itself is concentrated entirely within:

HVTF
LVTF

for overlap levels capable of generating observable selector divergence.

This interpretation provides a coherent explanation for both the transfer results and the overlap distributions.

------

## 7.6 Dense-Overlap Agreement Analysis

Selector agreement was recomputed under increasingly strict overlap thresholds.

Agreement levels decrease systematically as overlap requirements increase.

Typical behavior:

94–95%
↓
83–85%

under dense-overlap conditions.

This represents approximately:

9–12 percentage-point reductions

in selector agreement.

Importantly, these effects occur despite the strong topology convergence observed earlier.

This confirms that transferred DL information reaches runtime and alters selector behavior, even though aggregate family topology remains largely unchanged.

------

## 7.7 Updated Interpretation

The overlap analysis fundamentally changes how transfer experiments should be interpreted.

The evidence now supports the following chain:

DL overlap
→
concentrated in HVTF/LVTF
→
transferred DL influence
→
local selector divergence
→
aggregate topology convergence

In other words:

> - Transferred DL information matters.
>
> - However, its influence is concentrated in relatively rare but highly structured periods, characterized by **strong overlap** and **trend-following market conditions**.
>
> - Outside those regions, **evaluation-family structure** dominates selector behavior.

This interpretation reconciles all major transfer findings without requiring contradictory explanations.

---

## 7.8 Overlap Prevalence and Regime Concentration

A follow-up overlap analysis was performed using the complete collection of selector-state timelines across all family-regime experiments.

The objective was to determine how frequently meaningful overlap conditions occur and whether overlap behaves as a broadly distributed property of the state space or as a regime-specific phenomenon.

The analysis confirms that dense overlap is relatively uncommon at the global level:

- 21.01% of observations exceed 10% overlap,
- 11.81% exceed 20% overlap,
- 7.52% exceed 30% overlap.

However, this aggregate view is misleading because overlap is not distributed uniformly across regimes.

Regime decomposition reveals a remarkably clean separation:

| Regime | ≥10%   | ≥20%   | ≥30%   |
| ------ | ------ | ------ | ------ |
| HVR    | 7.87%  | 0.00%  | 0.00%  |
| HVTF   | 35.39% | 22.55% | 13.23% |
| LVR    | 3.94%  | 0.00%  | 0.00%  |
| LVTF   | 36.84% | 24.71% | 16.86% |

The result independently reproduces earlier dense-overlap analyses and substantially increases confidence in the conclusion that meaningful overlap is fundamentally a trend-following phenomenon.

Most importantly:

> No HVR or LVR observations exceed 20% overlap.

Consequently, every observation capable of producing strong overlap-conditioned transfer effects originates from either HVTF or LVTF.

This finding strengthens the interpretation that overlap is closely linked to market structure rather than representing a generic property of the DL prediction surfaces themselves.

---

# 8. Updated Working Model

## Motivation

The purpose of this section is not to summarize experimental results.

The results have already been presented.

Instead, the objective is to synthesize those results into a coherent working model that can guide future research.

A useful working model should:

- explain the observed family differences,
- explain the transfer results,
- explain the overlap findings,
- remain consistent with all validated experiments.

The model presented here should be viewed as a hypothesis supported by current evidence rather than a proven description of the underlying market.

------

## 8.1 The Original Hypothesis

Early in the project, a natural interpretation was that family identity originates primarily from the DL models themselves.

Under this view:

Persistent DL
→
Persistent selector behavior

Reactive DL
→
Reactive selector behavior

Transfer experiments were expected to provide a strong test of this hypothesis.

If correct, transferred runs should have inherited much of the topology of their source family.

This prediction was not supported by the data.

------

## 8.2 Evaluation Environment Dominates Topology

The strongest transfer result is the observation of topology convergence.

Regardless of transfer direction:

- Persistent environments remain Persistent-like.
- Reactive environments remain Reactive-like.

This finding implies that selector topology is determined primarily by the evaluation environment rather than by the origin of the transferred DL surface.

Consequently, family identity should not be interpreted as a static property embedded within the DL artifact itself.

Instead, family identity appears to emerge through interaction between:

- DL information,
- MPML training,
- market structure,
- selector adaptation.

This represents an important shift in interpretation compared with earlier versions of the study.

------

## 8.3 DL Information Still Matters

Topology convergence does not imply that DL information is irrelevant.

Dense-overlap analyses demonstrate that transferred DL information alters selector decisions in a measurable way.

Under dense-overlap conditions:

selector agreement falls from approximately:

94–95%

to:

83–85%

depending on regime and transfer direction.

This corresponds to roughly:

9–12 percentage-point reductions

in selector agreement.

These effects are too large to dismiss as statistical noise.

The evidence therefore supports the following conclusion:

> DL information influences selector behavior, but does not dominate selector topology.

This distinction is central to the current interpretation.

------

## 8.4 The Importance of Overlap

One of the most significant findings of the project is that DL influence is highly conditional.

The influence of transferred DL information is not distributed uniformly across the state space.

Instead, meaningful effects appear primarily when:

- overlap is high,
- sentiment persistence is high,
- trend structure is present.

This explains why transfer effects become visible only after introducing overlap-aware analyses.

Without overlap conditioning, the majority of observations belong to regions where DL influence is naturally weak.

The overlap metric therefore provides an important bridge between:

DL behavior

and

selector behavior.

------

## 8.5 Why Trend-Following Regimes Matter

Dense-overlap observations are concentrated almost entirely within:

HVTF
LVTF

At overlap levels above 20%:

HVTF + LVTF account for 100% of observations.

A subsequent prevalence analysis reproduced this result using the full selector-state timeline population.

Across more than 700,000 observations:

- only 11.81% exceeded 20% overlap,
- only 7.52% exceeded 30% overlap,

yet every observation above the 20% threshold originated from HVTF or LVTF.

This result is important because it demonstrates that dense overlap is not merely enriched within trend-following regimes; it is effectively absent elsewhere.

The practical implication is that overlap-conditioned DL influence should be viewed as a specialized phenomenon associated with trend persistence rather than a property expected throughout the broader MPML state space.

This finding suggests that overlap is not merely a property of the DL model.

Instead, overlap appears closely linked to market structure.

A plausible interpretation is that sustained directional behavior produces persistent retail positioning, which in turn produces persistent DL agreement.

Under this interpretation:

trend persistence
→
sentiment persistence
→
DL persistence
→
selector influence

The current evidence is consistent with this chain, although additional work would be required to establish causality.

------

## 8.6 Family Effects Revisited

The original family-topology experiments remain valid.

Persistent and Reactive families clearly produce different selector behavior.

However, the mechanism now appears more nuanced than originally assumed.

The evidence suggests:

- family effects emerge early,
- family effects survive selector construction,
- family effects influence topology,
- transfer effects remain local,
- evaluation environments dominate globally.

In practical terms:

Persistent and Reactive families appear to define different information environments rather than simply different prediction models.

This interpretation reconciles:

- early-stage phase modelling,
- topology analysis,
- transfer convergence,
- overlap-conditioned divergence.

No major result currently contradicts this model.

------

## 8.7 Current Best Explanation

The simplest explanation consistent with all validated evidence is:

DL information influences MPML locally.

Market structure influences MPML globally.

Family identity emerges from the interaction between the two.

Under this model:

- DL affects decisions,
- overlap determines when DL matters,
- topology reflects the evaluation environment,
- family effects are real but not absolute.

At the time of writing, this represents the highest-confidence interpretation of the experimental evidence.

------

# 9. Open Questions and Future Directions

## Motivation

The objective of this section is not to generate a large list of possible future experiments.

Instead, the goal is to identify the questions whose answers are most likely to change future development decisions.

In particular:

- ABM development,
- DL architecture changes,
- MPML architecture changes,
- future data collection.

Only questions capable of influencing those decisions are considered high priority.

------

## 9.1 Why Do Family Effects Appear So Early?

One of the most important unresolved findings is the early-stage phase-modelling asymmetry.

Persistent-family experiments generally benefit from phase modelling.

Reactive-family experiments generally do not.

This effect appears before selector construction.

Consequently, the origin of the family effect may lie within the predictive learning problem itself.

Understanding this mechanism remains a high-priority research objective.

If resolved, it could influence both DL architecture design and future feature engineering.

------

## 9.2 What Creates Dense Overlap Episodes?

An attempted overlap-persistence analysis produced an unexpected methodological result.

Because selector-state timelines are generated within regime-specific experimental universes, simple episode-duration statistics do not directly measure chronological persistence in the underlying market.

This observation highlights an important limitation of the currently available artifacts.

Future work investigating overlap persistence, overlap formation, or overlap decay will likely require reconstruction at the original chronological market level rather than within regime-filtered experiment outputs.

Consequently, the next generation of overlap research should focus not only on explaining overlap, but also on establishing the correct observational framework for studying it.

------

## 9.3 Can MPML Exploit Overlap Explicitly?

Current MPML architectures treat overlap largely as an observed property.

An obvious question is whether overlap itself should become an explicit decision variable.

Potential directions include:

- overlap-aware routing,
- overlap-aware confidence estimation,
- overlap-conditioned selector behavior.

The current study demonstrates that overlap identifies regions where DL information matters most.

Future architectures may benefit from exploiting that information directly.

------

## 9.4 Agent-Based Modelling (ABM)

The overlap findings provide an interesting motivation for ABM research.

The observed chain:

trend persistence
→
sentiment persistence
→
DL persistence

suggests that market structure may emerge from interacting behavioral populations.

ABM provides a natural framework for investigating these dynamics.

Unlike purely predictive models, ABM can potentially explain why particular market states arise.

The overlap results therefore strengthen the case for future ABM work.

------

## 9.5 DL Architecture Development

The transfer results imply that current DL surfaces contain meaningful information.

However, the influence appears localized rather than dominant.

Future DL work should therefore focus on increasing:

- signal quality,
- signal persistence,
- signal relevance during high-overlap periods.

Simply increasing model complexity is unlikely to be sufficient.

The more important question is whether the model captures the mechanisms responsible for overlap formation.

------

## 9.6 MPML Architecture Development

The selector currently appears highly effective at adapting to its evaluation environment.

This is a strength.

However, the transfer analyses also suggest that potentially valuable DL information may be diluted by global adaptation effects.

Future MPML research should therefore investigate:

- stronger utilization of DL information,
- confidence-aware routing,
- overlap-aware strategy allocation,
- regime-aware transfer weighting.

The goal is not necessarily to increase DL influence.

The goal is to use DL influence where it is most informative.

------

## 9.7 Final Assessment

At the beginning of the project, the central question was:

> Does DL information matter?

The current evidence allows a more precise answer.

DL information matters.

But it does not matter everywhere.

Its influence is concentrated in specific market states characterized by:

- persistent sentiment,
- dense overlap,
- trend-following behavior.

Understanding those states now appears substantially more important than measuring average effects across the entire dataset.

For future development, the most promising direction is therefore not merely building stronger models.

It is understanding the conditions under which models become influential.

