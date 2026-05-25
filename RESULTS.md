# MPML Experimental Findings

This document summarizes the primary findings, observations, and research directions explored in the Market-Phase-ML (MPML) framework.

The project focuses on:

- regime-aware strategy routing
- mixture-of-experts architectures
- walk-forward evaluation
- contextual ML systems
- robustness under non-stationary conditions
- optional deep-learning surface integration from MSML

The experiments prioritize:

- realistic out-of-sample evaluation
- reproducibility
- failure analysis
- causal validation
- robustness over in-sample optimization

---

# Executive Summary

## Core Observations

The project found evidence that:

- regime-aware routing can improve robustness relative to static baselines
- dynamic selectors can reduce drawdowns under certain market conditions
- walk-forward evaluation materially changes apparent strategy quality
- several seemingly promising configurations fail under strict causal evaluation
- DL-derived sentiment surfaces provide conditional rather than universal benefit
- transfer behavior varies significantly across market pair families

The overall outcome was:

> modest but credible improvements under realistic evaluation,
> rather than large but fragile in-sample gains.

This distinction became one of the central themes of the project.

---

# Classical MPML Findings

## Trend Following vs Mean Reversion

The framework initially explored two specialized expert families:

| Expert | Behavior |
|---|---|
| TF (Trend Following) | Attempts to capture directional continuation |
| MR (Mean Reversion) | Attempts to exploit short-term reversals |

Key observations:

- TF generally behaved better during sustained directional movement
- MR was often more stable during sideways or low-volatility conditions
- neither expert dominated consistently across all regimes
- performance strongly depended on volatility structure and market state

This supported the core hypothesis:

> different market environments favor different strategy behaviors.

---

## PhaseAware Routing

The PhaseAware baseline introduced rule-based routing between experts.

Compared to static baselines:

- drawdowns were sometimes reduced
- robustness improved in several evaluation windows
- transitions between regimes became more interpretable

However:

- gains were inconsistent
- hand-engineered regime logic had limited adaptability
- some improvements disappeared under stricter walk-forward validation

This motivated the transition toward ML-driven routing.

---

## Dynamic Selector Findings

The StrategySelector_Dynamic experiments used ML gating (primarily XGBoost-based routing) to select between experts.

Observed behavior:

- routing decisions became more adaptive
- some folds showed improved stability relative to static baselines
- dynamic selection sometimes reduced catastrophic expert mismatches
- uplift was generally modest rather than dramatic

Important:

The strongest value of the selector was often:

> robustness improvement

rather than raw return maximization.

---

# Walk-Forward Evaluation Findings

A major focus of the project was:

> realistic out-of-sample evaluation.

The repository repeatedly hardened the evaluation pipeline against:

- lookahead leakage
- fold overlap contamination
- temporal contamination
- schema drift
- feature ordering instability
- experiment attribution corruption

---

## Expanding-Window Validation

Experiments used:

- expanding-window training
- sequential test folds
- strict causal ordering

rather than random train/test splits.

This produced a much more conservative evaluation environment.

Several strategies that appeared strong under looser evaluation degraded significantly under walk-forward testing.

This became one of the most important findings in the project.

---

## Stability vs Peak Performance

Across many experiments:

- smoother equity curves
- lower drawdowns
- more consistent fold behavior

were often more meaningful than isolated peak performance.

The framework increasingly prioritized:

> stability and robustness

over:

> aggressive in-sample optimization.

---

# Failure Analysis

One of the strongest themes of the project was:

> explicit preservation of negative results.

The repository intentionally documents:

- failed hypotheses
- unstable selectors
- transfer failures
- evaluation collapses
- volatility-induced degradation

rather than hiding them.

---

## Volatility Spike Failures

Several experiments degraded sharply during:

- volatility spikes
- sudden directional reversals
- regime-transition instability

Key observations:

- selectors occasionally became unstable during high-volatility periods
- routing confidence sometimes collapsed during abrupt transitions
- some strategies overfit calm-market structure
- some gains disappeared entirely under stress conditions

These findings motivated:

- volatility guards
- stronger walk-forward validation
- schema hardening
- stricter provenance tracking
- transfer-aware analysis

---

## Feature Schema Failures

During DL integration experiments, the project uncovered:

- silent feature-order corruption
- schema mismatch drift
- missing-feature propagation bugs
- hidden selector fallback behavior

These issues led to:

- hard-fail schema validation
- deterministic feature ordering
- explicit imputation-awareness handling
- provenance-aware manifest semantics

This infrastructure work became a major part of the repository.

---

# MSML Integration Findings

## Overview

MPML integrates optional deep-learning prediction surfaces generated by the companion repository:

> Market-Sentiment-ML (MSML)

The integration experiments explored:

- sentiment vs no-sentiment surfaces
- persistent vs reactive pair families
- transfer behavior across pair groups
- imputation-awareness controls
- sparse temporal DL coverage

---

# Persistent vs Reactive Pair Families

The project explored two broad pair families:

| Family | General Behavior |
|---|---|
| Persistent | Longer directional structure |
| Reactive | Faster directional reversal / noisier behavior |

The experiments investigated whether:

- DL surfaces trained on one family generalized to another
- selector behavior changed across families
- routing quality depended on family structure

Key findings:

- transfer behavior was inconsistent
- some surfaces generalized poorly outside their training family
- several promising transfer hypotheses failed under walk-forward validation
- pair-family structure mattered more than initially expected

---

# Sentiment vs No-Sentiment Surfaces

MSML experiments compared:

| Surface | Description |
|---|---|
| Sentiment-aware | Includes sentiment-derived features |
| No-sentiment | Price/volatility-only surfaces |

Findings:

- sentiment signals sometimes improved contextual routing
- uplift was conditional rather than universal
- some experiments showed minimal difference
- some sentiment surfaces failed to generalize robustly

A major outcome was:

> DL integration produced nuanced effects rather than simple performance gains.

---

# Imputation Awareness

The project explored whether selectors should explicitly know when DL features were imputed or unavailable.

Two broad modes were tested:

| Mode | Description |
|---|---|
| Blind | Selector does not observe imputation state |
| Aware | Missing/imputed DL state propagated explicitly |

Findings:

- imputation awareness sometimes stabilized selector behavior
- effects varied significantly across folds and pair families
- missingness handling became a major robustness concern
- schema integrity proved more important than initially expected

---

# Temporal Coverage Constraints

DL surfaces were only available for modern historical windows (~2019+).

This introduced important constraints:

- overlap-window evaluation
- sparse historical DL coverage
- partial fold contamination risks
- reduced training horizon for some experiments

The project increasingly treated:

> temporal coverage integrity

as a first-class research concern.

---

# Infrastructure & Reproducibility Learnings

A major secondary outcome of the project was the development of:

- provenance-aware experiment infrastructure
- immutable experiment archives
- runtime manifest semantics
- ontology-aware analysis pipelines
- deterministic replay support
- semantic hardening against attribution drift

---

## Provenance Hardening

Several analysis-layer bugs were traced to:

- semantic reconstruction
- variant-name inference
- hidden runtime assumptions
- manifest ambiguity

The repository evolved toward:

> canonical provenance propagation.

This eventually produced:

- explicit experiment surfaces
- runtime-emitted provenance
- factor-conditioned analysis
- semantic anti-corruption validation

---

## Determinism & Replayability

The framework introduced:

- canonical run manifests
- deterministic experiment seeding
- immutable run directories
- schema regression testing
- ontology-aware validation

The goal was:

> reproducible research infrastructure

rather than one-off experimentation.

---

# Current Research Directions

The project is currently exploring:

- transfer-learning behavior across pair families
- HTF vs LVTF surface transfer
- overlap-aware DL evaluation windows
- selector calibration
- online adaptation
- alternate routing models
- feature attribution analysis
- robustness under distribution shift

---

# Conclusions

The repository evolved from:

> a regime-aware trading experiment

into:

> a broader contextual ML routing and experimentation framework.

The strongest outcomes were not necessarily:

- raw returns
- dramatic uplift
- or universal DL gains.

Instead, the project produced:

- realistic walk-forward evaluation infrastructure
- robust provenance-aware experimentation
- transfer-learning analysis capability
- failure-aware ML evaluation
- reproducible contextual-routing research tooling

Several hypotheses failed.

Those failures became some of the most valuable findings in the project because they revealed:

- where apparent ML signal disappears under causal evaluation
- where transfer breaks down
- where selector robustness collapses
- and where infrastructure assumptions fail under realistic conditions.

The project therefore prioritizes:

> rigor, reproducibility, and robustness

over:

> aggressive in-sample optimization.

