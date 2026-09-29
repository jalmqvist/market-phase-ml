# MPML Strategy Research --- Research Hierarchy

**Status:** Active research overview\
**Last updated:** 2026-09-29

------------------------------------------------------------------------

## 1. Purpose

This document is the top-level map of the MPML strategy-performance
research.

The research has evolved from a global 14-pair strategy/regime benchmark
into population-specific investigations. The current documentation
deliberately separates the **trend-volatility surface** from the newer
**behavioral-surface investigations**, so that findings from different
conditioning dimensions are not conflated.

The detailed research records should be treated as the authoritative
sources for their respective surfaces. This document provides the
hierarchy, scope boundaries, and current interpretation.

------------------------------------------------------------------------

## 2. Research hierarchy

The current strategy research is organized into three levels.

### Level 1 --- Global / cross-family trend-volatility baseline

**Document:**

`MPML_Strategy_Performance_Across_FX_Pair_Families_and_Trend-Volatility_Regimes.md`

This is the historical baseline for the MPML strategy-selection
research.

It evaluates:

> **strategy × FX pair family × trend-volatility regime**

across the original 14-pair universe.

The main finding is that strategy performance is not homogeneous across
FX populations and trend-volatility regimes. The historical TF4/MR42
combination therefore remains a useful global benchmark, but not an
assumption of universal optimality.

This document should remain a relatively self-contained historical
baseline. It is intentionally not rewritten to incorporate later
Reactive-JPY behavioral experiments.

### Level 2 --- Population-specific trend-volatility surfaces

The first population-specific trend-volatility record is:

`MPML_Reactive-JPY_Trend-Volatility_Strategy_Findings_2026-09-29.md`

It narrows the same trend-volatility question to:

-   EURJPY
-   GBPJPY
-   USDJPY

and documents the Reactive-JPY strategy/regime surface:

-   HVTF
-   LVTF
-   HVR
-   LVR

This document establishes the Reactive-JPY trend-volatility findings
without mixing them with the later consensus-lifecycle research.

Equivalent documents can eventually be created for other behavioral
populations, such as Persistent, when sufficient evidence exists.

### Level 3 --- Population-specific behavioral surfaces

The current Reactive-JPY behavioral record is:

`MPML_Reactive-JPY_Consensus-Lifecycle_Strategy_Findings_2026-09-29.md`

It evaluates:

> **strategy × pair × consensus-lifecycle state**

across:

-   `JPY_CONSENSUS_YOUNG`
-   `JPY_CONSENSUS_MATURING`
-   `JPY_CONSENSUS_MATURE`
-   `JPY_NON_EXTREME`

This is the current and most developed Reactive-JPY strategy-selection
research. It includes the individual-strategy benchmark, PhaseAware
composition experiment, walk-forward evidence, robustness analysis, and
MR32 implementation transition audit.

------------------------------------------------------------------------

## 3. Current Reactive-JPY research position

Reactive-JPY now has two parallel strategy-performance surfaces.

### Trend-volatility surface

> `strategy × pair × trend-volatility regime`

This is the earlier MPML conditional-performance layer.

The historical Reactive-JPY relative-performance leaders were:

  Regime   Relative leader
-------- -----------------
  HVTF     MR32
  LVTF     TF3
  HVR      MR1
  LVR      TF3

These are descriptive relative-performance results, not production
routing decisions.

### Consensus-lifecycle surface

> `strategy × pair × behavioral state`

This is the newer MSML-derived behavioral layer.

The current PhaseAware investigation retains:

-   `PhaseAware_TF4_MR42` --- canonical control
-   `PhaseAware_TF4_MR2` --- leading robust candidate
-   `PhaseAware_TF2_MR2` --- high-return comparator

The consensus-lifecycle document contains the detailed evidence and
qualifications supporting these roles.

The two surfaces should not be treated as interchangeable.

------------------------------------------------------------------------

## 4. Why the separation matters

A trend-volatility regime and a behavioral lifecycle state describe
different dimensions of the market.

For example:

> `LVTF`

does not mean the same thing as:

> `JPY_CONSENSUS_YOUNG`

and a result conditioned on one should not silently be interpreted as
evidence about the other.

The current research therefore avoids statements such as "Reactive-JPY
prefers TF2" unless the conditioning surface is specified.

Instead, findings should be expressed in their proper context:

> strategy performance conditioned on trend-volatility regime

or:

> strategy performance conditioned on consensus-lifecycle state.

This distinction becomes especially important as the research moves
toward richer conditional strategy selection.

------------------------------------------------------------------------

## 5. The eventual integrated research question

The two Reactive-JPY surfaces can eventually be combined explicitly.

The natural higher-dimensional research question is:

> **strategy × pair × trend-volatility regime × behavioral state**

Such an experiment could investigate whether behavioral state provides
information beyond the already-established trend-volatility and pair
structure.

However, this interaction should be treated as a **new research stage**.

It should not be inferred by mentally combining results from the two
existing documents.

In particular, the current consensus-lifecycle PhaseAware results should
not be described as evidence that behavioral states improve the
trend-volatility selector. That has not yet been tested directly.

------------------------------------------------------------------------

## 6. Historical progression

The research can be understood as a sequence.

### Stage 1 --- Global strategy benchmark

The original 14-pair evaluation established TF4 and MR42 as empirically
strong representatives of the TrendFollowing and MeanReversion classes.

This produced the historical:

> `TF4 + MR42`

PhaseAware benchmark.

### Stage 2 --- Family and trend-volatility analysis

The next investigation demonstrated that strategy performance depends
materially on:

> strategy × family × trend-volatility regime

This established that a single global strategy pair cannot fully
represent the conditional strategy landscape.

### Stage 3 --- Reactive-JPY behavioral surface

The Reactive-JPY population was subsequently examined using the
MSML-derived consensus-lifecycle surface.

The individual-strategy benchmark revealed a particularly notable TF/MR
separation in `JPY_CONSENSUS_YOUNG`, while later PhaseAware experiments
tested alternative TF/MR compositions across all four lifecycle states.

### Stage 4 --- PhaseAware composition investigation

The current Stage A work compared six PhaseAware compositions:

               MR42         MR2         MR5
----- ----------- ----------- -----------
  TF4       Control   Candidate   Candidate
  TF2     Candidate   Candidate   Candidate

The principal current interpretation is:

-   TF4/MR2 provides the broader risk/return improvement relative to the
    canonical control.
-   TF2/MR2 produces the larger return effect but has less stable risk
    evidence.
-   The composition effects were similar with and without the
    Reactive-JPY behavioral surface, so the present evidence does not
    show that behavioral conditioning created the composition advantage.

The detailed evidence belongs in the consensus-lifecycle document rather
than here.

------------------------------------------------------------------------

## 7. Validation principles

Across the hierarchy, several methodological principles should remain
consistent.

### Walk-forward evaluation is primary

MPML strategy research preserves temporal ordering through walk-forward
evaluation.

The primary conditional-performance unit is generally:

> **strategy × pair × walk-forward fold × conditioning state**

depending on the experiment.

### Pair identity matters

Repeated observations from the same currency pair are not equivalent to
independent FX populations.

Family-level or population-level results should therefore be accompanied
by pair-level analysis where possible.

### Sparse cells matter

Trade counts differ substantially between regimes and behavioral states.

Sparse cells should be identified and interpreted cautiously rather than
treated as equally informative as densely sampled cells.

### Operating and performance surfaces are distinct

A strategy's natural operating frequency does not establish where it has
the strongest conditional performance.

### Descriptive winners are not automatically deployable policies

A strategy that has the highest historical relative performance in a
cell is not automatically an appropriate routing rule. Out-of-sample
evaluation, fold-level robustness, pair coverage, and decision-time
information must be considered.

------------------------------------------------------------------------

## 8. Documentation rules going forward

To keep future studies from conflating surfaces, new findings should
follow these rules.

1.  **Name the conditioning surface explicitly in the document title.**
2.  **State the FX population explicitly.**
3.  **Keep trend-volatility and behavioral-state results in separate
    documents unless their interaction is the actual research
    question.**
4.  **Distinguish historical baseline results from current candidate
    investigations.**
5.  **Preserve the canonical PhaseAware control separately from
    candidate compositions.**
6.  **Do not promote a cell-level historical winner directly into a
    production strategy without a dedicated out-of-sample evaluation.**
7.  **When combining surfaces, create a new research stage and document
    rather than silently extending an existing surface.**

------------------------------------------------------------------------

## 9. Current document map

The current strategy-performance documents are:

1. **Global / cross-family baseline**  
   `MPML_Strategy_Performance_Across_FX_Pair_Families_and_Trend-Volatility_Regimes.md`  
   **Population:** 14-pair universe  
   **Conditioning surface:** Family × trend-volatility  
   **Role:** Historical/global baseline

2. **Reactive-JPY trend-volatility**  
   `MPML_Reactive-JPY_Trend-Volatility_Strategy_Findings_2026-09-29.md`  
   **Population:** Reactive-JPY  
   **Conditioning surface:** Trend-volatility  
   **Role:** Population-specific baseline

3. **Reactive-JPY consensus lifecycle**  
   `MPML_Reactive-JPY_Consensus-Lifecycle_Strategy_Findings_2026-09-29.md`  
   **Population:** Reactive-JPY  
   **Conditioning surface:** Consensus lifecycle  
   **Role:** Current behavioral research

The older `MPML_Strategy_Findings_Unified.md` should be treated as a historical unified research record rather than as the primary current document. The more focused documents above provide clearer scope boundaries for future work.

## 10. Future extension

The same hierarchy can be extended naturally as new behavioral
populations are investigated.

For example:

``` text
MPML Strategy Research
│
├── Global
│   └── Strategy Performance Across FX Pair Families and Trend-Volatility Regimes
│
├── Reactive-JPY
│   ├── Trend-Volatility Strategy Findings
│   └── Consensus-Lifecycle Strategy Findings
│
├── Persistent
│   ├── Trend-Volatility Strategy Findings
│   └── Behavioral-Surface Strategy Findings
│
└── Future integrated studies
    └── Strategy × Pair × Trend-Volatility × Behavioral State
```

This structure keeps the historical foundation intact while allowing
population-specific research to develop independently.

------------------------------------------------------------------------

## 11. Final interpretation

The MPML strategy research has progressed from asking:

> **Which strategies perform best globally?**

to asking:

> **How does strategy suitability vary across FX populations and market
> states?**

The trend-volatility research established the first conditional layer.

The Reactive-JPY consensus-lifecycle research adds a second, independent
conditioning layer.

The current evidence therefore supports a research architecture in which
strategy selection is increasingly treated as a conditional problem
rather than a fixed global ranking problem.

The next major methodological step is not to merge the existing findings
informally, but to test explicitly whether the two conditioning surfaces
interact.
