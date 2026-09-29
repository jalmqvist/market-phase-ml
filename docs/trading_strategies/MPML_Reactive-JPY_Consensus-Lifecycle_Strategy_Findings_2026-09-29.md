# MPML Reactive-JPY Consensus-Lifecycle Strategy Findings

**Status:** Active research record
**Scope:** Reactive-JPY behavioral/consensus-lifecycle surface
**Last updated:** 2026-09-29

---

## 1. Current research position

The current Reactive-JPY strategy investigation has progressed from an
individual-strategy benchmark to a focused PhaseAware composition study.
The latest evidence does **not** justify changing the canonical
PhaseAware policy, but it has established two concrete compositions for
continued investigation:

| Composition | Current role |
|---|---|
| `PhaseAware_TF4_MR42` | Canonical control |
| `PhaseAware_TF4_MR2` | Leading robust candidate |
| `PhaseAware_TF2_MR2` | High-return comparator |

The important distinction is between **return magnitude** and
**robustness**. `TF2/MR2` produces the largest return advantage in the
current Reactive-JPY matrix, while `TF4/MR2` provides the broader and
more stable return/risk profile. The current working position is
therefore to retain `TF4/MR42` as the control, carry `TF4/MR2` forward
as the leading robustness candidate, and retain `TF2/MR2` as a
higher-return comparison candidate.

No default-policy change has been made.

---

## 2. Latest evidence: PhaseAware composition investigation

### 2.1 Candidate matrix

Stage A evaluated six PhaseAware compositions:

|  | MR42 | MR2 | MR5 |
|---|---:|---:|---:|
| **TF4** | ✓ | ✓ | ✓ |
| **TF2** | ✓ | ✓ | ✓ |

The canonical `TF4/MR42` configuration was retained as the control.

The experiment covered:

- three Reactive-JPY pairs: EURJPY, GBPJPY, USDJPY;
- four consensus-lifecycle states:
    -   `JPY_CONSENSUS_YOUNG`
    -   `JPY_CONSENSUS_MATURING`
    -   `JPY_CONSENSUS_MATURE`
    -   `JPY_NON_EXTREME`;
- six compositions;
- 24 MPML runs;
- 72 pair/state ML cells.

The normal PhaseAware, selector and walk-forward machinery was exercised
rather than bypassed with direct strategy execution.

### 2.2 Primary PhaseAware results

The primary state-conditioned comparison used the ML backtest artifact.

| Composition | Mean return | Median | Trade-weighted return | Positive cells | Worst | Best |
|---|---:|---:|---:|---:|---:|---:|
| **TF2/MR2** | **+15.67%** | −0.71% | **+14.06%** | 5/12 | −16.03% | +62.20% |
| TF2/MR42 | +11.15% | −19.28% | +4.40% | 4/12 | −25.41% | +80.15% |
| TF4/MR2 | +3.58% | **+11.42%** | +3.79% | 8/12 | −19.52% | +18.26% |
| TF2/MR5 | +8.94% | −16.32% | +2.72% | 4/12 | −34.87% | +73.38% |
| TF4/MR5 | +0.05% | −2.55% | −3.15% | 4/12 | −39.16% | +41.52% |
| **TF4/MR42 (control)** | −6.05% | −3.32% | −8.23% | 5/12 | −42.73% | +29.56% |

`TF2/MR2` is therefore the strongest raw-return composition in the
current matrix. Return alone, however, is not the selection criterion.

### 2.3 State-conditioned behavior

Trade-weighted returns by consensus-lifecycle state:

| State | TF4/MR42 | TF4/MR2 | TF4/MR5 | TF2/MR42 | TF2/MR2 | TF2/MR5 |
|---|---:|---:|---:|---:|---:|---:|
| **Young** | −6.57% | +4.99% | −2.14% | +6.28% | **+11.94%** | +1.83% |
| **Maturing** | −8.06% | +4.07% | −3.27% | +7.03% | **+17.16%** | +5.52% |
| **Mature** | −9.66% | +2.93% | −3.14% | +1.36% | **+14.00%** | +1.58% |
| **Non-Extreme** | −8.65% | +3.17% | −4.05% | +2.94% | **+13.17%** | +1.97% |

`TF2/MR2` is the strongest return composition in all four states. This
makes a narrowly state-specific explanation less plausible: the observed
advantage is more consistent with a composition effect that is expressed
across the consensus-lifecycle surface.

`TF4/MR2` also maintains a positive composition advantage across all
four states, with a substantially more stable risk profile.

### 2.4 Return and drawdown relative to the canonical control

For the candidate comparison:

- `ΔReturn = candidate return − control return`
- `ΔDD = candidate Max DD − control Max DD`

Because maximum drawdown is negative, a positive `ΔDD` means a less
severe drawdown.

| Composition | Mean ΔReturn | Median ΔReturn | Mean ΔDD | Return improved | DD improved | Both improved |
|---|---:|---:|---:|---:|---:|---:|
| **TF4/MR2** | **+9.63 pp** | **+13.37 pp** | **+9.02 pp** | 8/12 | **11/12** | **8/12** |
| TF4/MR5 | +6.10 pp | +4.85 pp | +1.85 pp | 8/12 | 11/12 | 8/12 |
| TF2/MR42 | +17.21 pp | +19.41 pp | −4.60 pp | 8/12 | 4/12 | 4/12 |
| **TF2/MR2** | **+21.73 pp** | **+25.11 pp** | +1.23 pp | **10/12** | 4/12 | 4/12 |
| TF2/MR5 | +15.00 pp | +10.48 pp | −4.37 pp | 8/12 | 3/12 | 3/12 |

The current evidence therefore separates the candidates into two
profiles:

**TF4/MR2:** broader and more stable return/risk improvement.

**TF2/MR2:** stronger return effect, but less stable risk behavior.

### 2.5 Pair-level robustness

For `TF4/MR2`, mean differences from the canonical control were:

| Pair | Mean ΔReturn | Mean ΔDD | Candidate mean return |
|---|---:|---:|---:|
| EURJPY | +13.19 pp | +12.86 pp | +11.04% |
| GBPJPY | +21.75 pp | +12.22 pp | −17.50% |
| USDJPY | −6.04 pp | +1.97 pp | +17.20% |

The return effect is pair-dependent, but drawdown improvement is
positive for all three pairs.

For `TF2/MR2`:

| Pair | Mean ΔReturn | Mean ΔDD |
|---|---:|---:|
| EURJPY | −0.04 pp | −2.63 pp |
| GBPJPY | +29.02 pp | +11.08 pp |
| USDJPY | +36.20 pp | −4.76 pp |

The much larger return effect is consequently more dependent on pair
identity, while risk behavior varies substantially across the three
pairs.

### 2.6 Leave-one-pair-out robustness

Removing each Reactive-JPY pair in turn gives:

| Pair omitted | TF4/MR2 weighted ΔReturn | TF4/MR2 mean ΔDD | TF2/MR2 weighted ΔReturn | TF2/MR2 mean ΔDD |
|---|---:|---:|---:|---:|
| EURJPY | +7.78 pp | +7.10 pp | +32.41 pp | +3.16 pp |
| GBPJPY | +3.83 pp | +7.41 pp | +16.89 pp | −3.69 pp |
| USDJPY | +17.33 pp | +12.54 pp | +14.24 pp | +4.23 pp |

`TF4/MR2` retains a positive return and mean drawdown effect after
removing every individual pair. `TF2/MR2` retains a strong return
effect, but its mean drawdown advantage disappears when GBPJPY is
removed.

This is a major reason for treating the two candidates differently.

---

## 3. Walk-forward evidence

Walk-forward evaluation remains the primary out-of-sample validation
mechanism. Pair and behavioral-state perturbations are used as
robustness analyses rather than substitutes for temporal validation.

For the 84 pair/fold observations:

| Candidate | Mean fold return Δ | Median fold return Δ | Mean DD Δ | DD improved |
|---|---:|---:|---:|---:|
| **TF4/MR2** | +0.494 pp | +0.290 pp | **+1.834 pp** | **62/84** |
| TF2/MR2 | +0.954 pp | +0.160 pp | +0.806 pp | 54/84 |

`TF4/MR2` therefore shows particularly broad fold-level drawdown
improvement. `TF2/MR2` has the larger mean fold-level return effect, but
the return advantage is less uniformly accompanied by improved risk.

---

## 4. Does the behavioral surface create the composition advantage?

An important Stage A control was an unconditional baseline without the
Reactive-JPY behavioral surface.

The composition effects were similar:

| Candidate | Unconditional mean ΔReturn | Behavioral mean ΔReturn | Unconditional mean ΔDD | Behavioral mean ΔDD |
|---|---:|---:|---:|---:|
| **TF4/MR2** | +10.31 pp | +9.63 pp | +9.69 pp | +9.02 pp |
| **TF2/MR2** | +23.14 pp | +21.73 pp | — | +1.23 pp |

The similarity indicates that the main candidate separation is not being
created by the consensus-lifecycle conditioning itself. The current
evidence instead points toward **strategy composition** as the dominant
source of the observed difference in these experiments.

This does **not** show that behavioral information is uninformative. It
shows only that these experiments do not provide evidence that the
behavioral conditioning is the source of the PhaseAware composition
advantage.

This is an important boundary for future work: the Reactive-JPY
consensus surface should not be credited with an effect that the
unconditional composition control already reproduces.

---

## 5. MR32 implementation transition audit

A separate implementation audit compared the canonical `TF4/MR42`
control using the old and new MR32 implementations on the current
codebase.

The substantive outputs were equivalent at the pair and
walk-forward-fold levels. The transition therefore does not appear to
have altered the PhaseAware control results.

This is treated as an **implementation transition audit**, not as the
historical reproducibility gate for the broader MPML experiment. The
current new-MR32 `TF4/MR42` run remains the canonical control.

---

## 6. Individual-strategy benchmark: before PhaseAware composition

The PhaseAware investigation was preceded by an individual-strategy
benchmark over:

- TF1--TF5;
- MR1, MR2, MR32, MR42 and MR5;
- EURJPY, GBPJPY and USDJPY;
- the same four consensus-lifecycle states.

The purpose was not to identify a universal winner from a single metric.
It was to determine whether strategy classes behave differently across
the lifecycle and to identify candidates worth carrying into PhaseAware
evaluation.

### 6.1 Young-state TF/MR separation

The clearest individual-strategy observation occurs in:

> **`JPY_CONSENSUS_YOUNG`**

The separation is visible across all three pairs:

- EURJPY: 4/5 TF strategies positive versus 1/5 MR strategies.
- GBPJPY: 4/5 TF strategies positive versus 0/5 MR strategies.
- USDJPY: 2/5 TF strategies positive versus 1/5 MR strategies.

The class-level TF-minus-MR difference is positive for all three pairs
under both strategy-equal and trade-count-weighted aggregation.

Pair-level weighted means:

| Pair | TF weighted mean | MR weighted mean | TF − MR |
|---|---:|---:|---:|
| EURJPY | +0.000945 | −0.001144 | +0.002089 |
| GBPJPY | +0.007783 | −0.003896 | +0.011679 |
| USDJPY | −0.000583 | −0.002467 | +0.001884 |

At the individual-strategy level, all five TF strategies have positive
mean return across the three pairs, while all five MR strategies have
negative mean return.

This is stronger evidence for a **strategy-class separation** than for a
single-strategy effect.

### 6.2 Lifecycle dependence

The separation is not equally strong across the lifecycle.

#### `JPY_CONSENSUS_YOUNG`

This is the strongest and most coherent observation. TF is positive in
aggregate for EURJPY and GBPJPY and less negative than MR for USDJPY. MR
strategies are predominantly negative.

#### `JPY_CONSENSUS_MATURING`

Results become heterogeneous and substantially more data-sparse. EURJPY
shows a TF advantage, while GBPJPY and USDJPY do not support a general
TF-over-MR conclusion.

#### `JPY_CONSENSUS_MATURE`

This state is particularly sparse. Several strategy/pair combinations
contain only one or a few eligible trades, while some contain none.
Apparent differences are therefore exploratory.

#### `JPY_NON_EXTREME`

This state is much more densely sampled and provides an important
contrast. TF does not uniformly outperform MR:

- EURJPY: small TF advantage;
- GBPJPY: MR advantage;
- USDJPY: clear TF advantage.

The absence of a universal TF advantage in this better-sampled state
argues against interpreting the Young result as simple global
superiority of trend following.

### 6.3 Figure: individual strategy/state return distribution

![Reactive-JPY individual-strategy
benchmark](figures/strategy_benchmark_rj_return_vs_trade_count_by_state_v2_5.png)

The figure plots mean eligible-trade return against eligible trade count
for each strategy in each consensus-lifecycle state. It is particularly
useful for seeing the pronounced TF/MR separation in
`JPY_CONSENSUS_YOUNG` and the lower trade density in
`JPY_CONSENSUS_MATURING` and `JPY_CONSENSUS_MATURE`.

### 6.4 Preliminary behavioral hypothesis

The original interpretation was:

> **Early-stage JPY consensus may be associated with subsequent
> directional price behavior, making trend-following strategies more
> effective and mean-reversion strategies less effective during the
> `JPY_CONSENSUS_YOUNG` phase.**

This remains a **behavioral hypothesis, not a causal conclusion**.

The evidence does not establish that sentiment causes the observed price
behavior, nor that a particular TF strategy is universally optimal in
the Young state. The more defensible finding is the repeated separation
between multiple TF and MR implementations.

The later PhaseAware results also matter here: the strongest current
composition effects are not confined to the Young state, and the
unconditional composition control reproduces most of the candidate
advantage.

---

## 7. Earlier individual-strategy robustness findings

Before the PhaseAware composition experiment, the individual-strategy
benchmark provided the following useful candidate picture:

| Strategy | States available | Positive states | Mean return | Worst | Best | Trades |
|---|---:|---:|---:|---:|---:|---:|
| **TF4** | 4 | 3 | +0.215% | −0.004% | +0.576% | 149 |
| TF2 | 3 | 2 | +0.048% | −0.564% | +0.685% | 116 |
| MR2 | 4* | 3* | +0.130% | — | — | — |
| MR5 | 4* | — | −0.012% | — | — | — |
| MR32 | 4* | — | −0.510% | — | — | — |
| MR42 | 4* | — | −0.291% | — | — | — |

Availability varied because of sparse state/pair cells.

An oracle-gap diagnostic also showed that TF2, MR2 and MR5 became local
winners in some pair/state cells, motivating the later PhaseAware
candidate matrix. The oracle itself is not deployable evidence because
it uses realized outcomes.

The important historical conclusion was not that TF4 was universally
optimal. Rather, TF4 had the strongest combination of state coverage,
positive-state frequency and relatively stable aggregate performance
among the TF candidates.

---

## 8. Research interpretation

The Reactive-JPY consensus-lifecycle investigation now supports several
distinct findings.

### Finding 1 --- Strategy-class behavior varies across the consensus lifecycle

The strongest preliminary class-level observation is the TF/MR
separation during `JPY_CONSENSUS_YOUNG`. The effect weakens or becomes
heterogeneous later in the lifecycle.

### Finding 2 --- PhaseAware composition materially affects performance

Changing the concrete TF/MR representatives can materially change
PhaseAware results under the same state-conditioned evaluation
framework.

### Finding 3 --- The current composition effect is not obviously created by behavioral conditioning

The unconditional baseline produces similar candidate-vs-control
differences. The current evidence therefore points more strongly to
composition than to the consensus-lifecycle surface as the source of the
observed PhaseAware separation.

### Finding 4 --- TF4/MR2 has the broadest current robustness profile

Its return advantage is not universal at every pair/state cell, but it
is accompanied by unusually broad drawdown improvement, survives
leave-one-pair-out analysis, and shows broad fold-level risk
improvement.

### Finding 5 --- TF2/MR2 remains an important high-return comparator

Its return advantage is larger and persists after removing any single
pair, but its risk behavior is less stable and more pair-dependent.

### Finding 6 --- The current evidence does not justify a policy change

`PhaseAware_TF4_MR42` remains the canonical control. The next work
should emphasize validation, pair-balanced interpretation, fold-level
stability and testing outside the current Reactive-JPY sample rather
than expanding the candidate universe without a specific hypothesis.

---

## 9. Validation framework

Walk-forward evaluation is the primary out-of-sample validation
mechanism because the underlying market and behavioral surfaces are
time-varying.

The hierarchy used in this investigation is:

1. **Walk-forward evaluation** --- primary temporal out-of-sample
    validation.
2.  **Pair and behavioral-state perturbation** --- robustness analysis.
3.  **Additional temporal holdouts** --- supplementary diagnostics where
    useful.
4.  **Randomized splits** --- diagnostic only; not a primary validation
    basis.

Randomized train/test splitting is not treated as inherently superior
because random mixing can disrupt temporal ordering and the dependence
structure of evolving market environments.

The general principle is:

> **Validation should preserve the structure of the problem being
> modeled.**

---

## 10. Scope boundary: consensus lifecycle versus trend-volatility surface

This document is intentionally a **Reactive-JPY consensus-lifecycle
strategy record**.

The conditioning dimension studied here is the MSML-derived Reactive-JPY
behavioral surface with the four lifecycle states:

- `JPY_CONSENSUS_YOUNG`
- `JPY_CONSENSUS_MATURING`
- `JPY_CONSENSUS_MATURE`
- `JPY_NON_EXTREME`

The individual-strategy benchmark uses MLP `price_trend` artifacts, but
this artifact name should not be confused with MPML's separate
**trend-volatility surface**. The experiments documented here are
organized around the consensus-lifecycle states, not around `HVTF`,
`LVTF`, `HVR` and `LVR`.

The separate MPML trend-volatility investigation covers the strategy ×
pair-family × trend-volatility regime surface and should remain a
separate research document. Its Reactive-JPY baseline is useful
historical context, but its regime results are **not part of the
evidence base for the consensus-lifecycle findings recorded here**.

This separation is important for future studies: otherwise, results from
the two conditioning surfaces can be conflated as if they were
observations from the same experimental population.

A future document should therefore be maintained under a title such as:

> **MPML Reactive-JPY Trend-Volatility Strategy Findings**

and should contain the Reactive-JPY `HVTF/LVTF/HVR/LVR` results
independently of this consensus-lifecycle record.

---

## 11. Source lineage and superseded documents

This document consolidates the Reactive-JPY consensus-lifecycle material
from four earlier records, while deliberately separating it from the
trend-volatility material.

### `MPML_PhaseAware_Stage_A_Findings_2026-09-14.md`

**Role:** latest and most consequential source.

Absorbed material includes:

- six-composition PhaseAware candidate matrix;
- state-conditioned ML results;
- candidate-vs-control return and drawdown analysis;
- pair-level analysis;
- leave-one-pair-out robustness;
- walk-forward fold evidence;
- unconditional composition control;
- MR32 implementation transition audit;
- current candidate status.

This material is presented first in this document.

### `MPML_Preliminary_Finding_TF_MR_Reactive_JPY_Young.md`

**Role:** earlier individual-strategy behavioral finding.

Absorbed material includes:

- post-G3.1 individual-strategy benchmark;
- Young-state TF/MR separation;
- pair-level class comparison;
- lifecycle dependence;
- preliminary behavioral hypothesis;
- sparse-state limitations.

This material is presented after the later PhaseAware evidence because
it explains how the candidate investigation was motivated.

### `MPML_Strategy_Findings_Unified.md`

**Role:** broader research record.

Only the Reactive-JPY consensus-lifecycle sections have been
incorporated here, principally the individual-strategy benchmark,
PhaseAware candidate experiment, validation philosophy and current
status. Its global, persistent-family, Reactive-CHF and trend-volatility
sections are intentionally excluded from this document.

### `MPML Strategy Performance Across FX Pair Families and Trend-Volatility Regimes.md`

**Role:** separate trend-volatility baseline.

This document is **not** merged into the present evidence record. Its
Reactive-JPY `HVTF/LVTF/HVR/LVR` results belong to the trend-volatility
research surface and should remain separately documented.

---

## 12. Current status and next steps

### Current status

- `PhaseAware_TF4_MR42` remains the canonical control.
- `PhaseAware_TF4_MR2` is the leading robust candidate.
- `PhaseAware_TF2_MR2` is retained as the high-return comparator.
- The Young-state TF/MR separation remains a useful behavioral
    hypothesis.
- No default PhaseAware policy change has been made.

### Next steps

The next stage should emphasize:

1. independent robustness and fold-level validation;
2.  pair-balanced aggregation;
3.  evaluation outside the current Reactive-JPY sample;
4.  clarification of what, if anything, the consensus-lifecycle surface
    adds beyond the underlying strategy/regime composition;
5.  interpretation of the candidate behavior without treating the
    Young-state hypothesis as causal sentiment alpha.

The candidate universe should not be expanded broadly unless a specific
hypothesis motivates the additional compositions.

---

## 13. Final summary

The Reactive-JPY consensus-lifecycle research has progressed from an
exploratory observation of TF/MR class separation to a concrete
PhaseAware composition investigation.

The current evidence supports the following working picture:

> **The Reactive-JPY consensus lifecycle contains state-dependent
> differences in strategy-class behavior, with the clearest TF/MR
> separation occurring during `JPY_CONSENSUS_YOUNG`.**

> **PhaseAware performance is materially affected by the concrete TF/MR
> representatives used, but the current composition effect is reproduced
> to a similar degree without behavioral conditioning.**

> **`PhaseAware_TF4_MR2` currently has the broadest robustness profile,
> while `PhaseAware_TF2_MR2` produces the stronger raw-return effect but
> less stable risk behavior.**

> **`PhaseAware_TF4_MR42` remains the canonical control and no policy
> change is currently justified.**

The next research question is therefore not simply which composition has
the highest return, but whether the observed composition differences
remain robust under additional temporal and population-level validation,
and whether the Reactive-JPY consensus lifecycle provides information
beyond the strategy/regime structure already present in the underlying
system.
