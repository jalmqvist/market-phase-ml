# Trading-Strategy-Level Interpretation of Persistent vs Reactive Structures

# Relationship To Statistical Findings

This document is primarily:
- interpretative,
- mechanistic,
- and hypothesis-oriented.

Its purpose is to connect:
- MPML selector behavior,
- trading-strategy properties,
- latent topology hypotheses,
- and ABM-style behavioral interpretations.

Quantitative validation, statistical estimates, variance decomposition, selector diagnostics, and temporal persistence analysis are maintained separately in:

```text
mpml_msml_v_5_phase_1_reference_documentation.md
```

This separation is intentional and helps preserve:

- statistical traceability,
- theoretical clarity,
- and infrastructure reproducibility.

---

## Context

During development of MPML, the combinatoric span of trading strategies had to be constrained due to the computational cost of dynamic strategy selection and walkforward backtesting.

The final architecture therefore converged on a deliberately minimal strategy basis:

- TF4 (trend-following)
- MR42 (mean-reversion)

These were selected because they consistently ranked among the strongest performers across both major and minor currency pairs.

Initially, this restriction appeared primarily practical.
However, the V5 Phase-1 investigations suggest that the limited strategy basis may have unintentionally revealed a much deeper structural distinction between the persistent and reactive pair families.

Specifically:

- persistent and reactive structures appear to interact fundamentally differently with the underlying assumptions embedded inside TF4, MR42, PhaseAware routing, and Dynamic Selector adaptation.
- these differences were not explicitly designed into MPML.
- the family assignments themselves originated upstream from MSML investigations (ABM initially, later DL-assisted), not from MPML optimization.

The resulting downstream behavioral separation therefore constitutes an important independent validation of the ontology.

---

# 1. Strategy Definitions and Embedded Assumptions

## TF4 — Trend-Following Pullback Continuation

TF4 is not a pure breakout strategy.

The strategy combines:

- LWMA slope persistence
- stochastic pullback detection
- continuation re-entry after local exhaustion

Core logic:

- enter long when:
  - LWMA is rising
  - stochastic becomes oversold
- enter short when:
  - LWMA is falling
  - stochastic becomes overbought

This creates a very specific implicit assumption set.

## TF4 Implicit Assumptions

| Assumption | Meaning |
|---|---|
| trend persistence exists | directional drift survives locally |
| pullbacks are temporary | reversals are bounded |
| stochastic extremes are exhaustion events | not true regime changes |
| continuation dominates interruption | trend survives local perturbation |
| latent structure persists long enough | re-entry remains exploitable |

Importantly, TF4 assumes:

> pullbacks are interruptions of an underlying directional structure.

This is a much stronger assumption than simple trend-following.

---

## MR42 — Structured Mean Reversion

MR42 is fundamentally a bounded-reversion system.

Although simpler conceptually than TF4, it contains several strong implicit assumptions:

| Assumption | Meaning |
|---|---|
| deviations revert | local equilibrium exists |
| reversals are bounded | instability is limited |
| pullbacks relax rather than explode | volatility eventually contracts |
| latent structure survives shocks | reversion remains meaningful |
| equilibrium persists long enough | exploitation window survives |

MR42 therefore assumes:

> local disequilibrium is temporary rather than structurally transformative.

This becomes extremely important when comparing persistent vs reactive families.

---

## PhaseAware — Defensive Stabilization Layer

PhaseAware introduces:

- volatility suppression
- spike handling
- fallback routing
- defensive occupancy
- heuristic stabilization

Its role is not primarily predictive.

Instead, PhaseAware acts as:

> a damage-limitation and regime-stabilization controller.

This became increasingly clear during V5 Phase-1.

---

## Dynamic Selector — Adaptive Policy Arbitration

The Dynamic Selector introduces:

- adaptive routing
- temporal policy arbitration
- occupancy optimization
- transition handling
- contextual adaptation

Critically, the selector architecture contains:

- hysteresis
- minimum hold periods
- maximum hold periods
- volatility suppression
- spike overrides

Therefore the selector implicitly assumes:

| Assumption | Meaning |
|---|---|
| policy persistence matters | adaptation accumulates value |
| latent structure survives locally | switching decisions remain meaningful |
| transitions are not instantaneous | exploitation continuity exists |
| selector state can stabilize | routing continuity has informational value |

This assumption appears substantially more compatible with persistent structures than reactive structures.

---

# 2. TF4 → MR42 Transition

## Persistent Family

Observed transition:

| Strategy | Mean Return |
|---|---:|
| TF4 | +6.1% |
| MR42 | +45.8% |

Approximate uplift:

```text
≈ +40 percentage points
```

## Interpretation

This strongly suggests:

> persistent pairs are NOT simple trend systems.

Instead, persistent structures appear to support:

- bounded pullback-release cycles
- delayed reversion
- metastable positioning
- oscillatory persistence
- structured local equilibrium

The data therefore suggests that persistent structures contain:

> persistence WITH bounded relaxation.

This is extremely compatible with:

- delayed trader adaptation
- behavioral crowding
- slow information digestion
- metastable collective positioning

which aligns closely with the original ABM/MSML interpretation.

---

## Reactive Family

Observed transition:

| Strategy | Mean Return |
|---|---:|
| TF4 | -21.8% |
| MR42 | -7.6% |

Approximate uplift:

```text
≈ +14 percentage points
```

However:

- the system remains structurally weak.
- MR improves behavior but does not stabilize it.

## Interpretation

Reactive structures appear hostile to BOTH:

- continuation assumptions
- stable mean-reversion assumptions

Possible reasons:

| Reactive Property | Consequence |
|---|---|
| abrupt transitions | continuation invalidated |
| rapid reversals | stochastic re-entry becomes trap-entry |
| fragmented trends | LWMA continuity collapses |
| unstable equilibrium | reversion assumptions fail |
| volatility persistence | pullbacks become regime transitions |

This suggests:

> reactive structures are fundamentally transition-dominated.

This may explain why:

- TF continuation fails,
- while MR stabilization also remains weak.

---

# 3. MR42 → PhaseAware Transition

## Persistent Family

Observed transition:

| Strategy | Mean Return |
|---|---:|
| MR42 | +45.8% |
| PhaseAware | +45.0% |

Approximate delta:

```text
≈ -0.8 percentage points
```

## Interpretation

This is extremely revealing.

Despite introducing:

- volatility suppression
- defensive routing
- fallback stabilization
- spike handling

PhaseAware does NOT materially improve persistent-family performance.

This strongly suggests:

> persistent structures are already naturally exploitable by bounded MR logic.

Meaning:

- persistent systems may already possess sufficient latent stability,
- making heavy defensive stabilization less necessary.

This is one of the strongest pieces of evidence supporting:

- metastable equilibrium,
- bounded release dynamics,
- and persistent latent structure.

---

## Reactive Family

Observed transition:

| Strategy | Mean Return |
|---|---:|
| MR42 | -7.6% |
| PhaseAware | +0.7% |

Approximate uplift:

```text
≈ +8.3 percentage points
```

## Interpretation

This is one of the clearest indicators that reactive structures are:

> instability-sensitive.

Raw MR appears to overcommit into:

- unstable transitions
- volatility bursts
- collapsing equilibrium states
- continuation traps

PhaseAware helps because it:

- suppresses MR during dangerous periods
- introduces defensive occupancy
- reduces catastrophic exposure
- stabilizes routing during volatility expansion

Therefore:

> PhaseAware functions primarily as a damage limiter inside reactive structures.

This is an extremely important systems-level interpretation.

---

# 4. PhaseAware → Dynamic Selector Transition

## Persistent Family

Observed transition:

| Strategy | Mean Return |
|---|---:|
| PhaseAware | +45.0% |
| DynamicSelector | +80.7% |

Approximate uplift:

```text
≈ +35.7 percentage points
```

## Interpretation

This is arguably the most important transition in the entire architecture.

The Dynamic Selector appears to exploit:

- occupancy timing
- policy persistence
- adaptive continuity
- transition arbitration
- contextual routing geometry

Persistent structures appear highly compatible with these assumptions.

Specifically:

| Persistent Property | Selector Advantage |
|---|---|
| metastable structure | routing continuity useful |
| slower transitions | adaptation accumulates value |
| bounded reversion | occupancy optimization meaningful |
| latent persistence | policy continuity exploitable |

This suggests:

> adaptive exploitation is possible because the environment remains coherent long enough for the controller to stabilize.

This is one of the strongest architectural findings observed so far.

---

## Reactive Family

Observed transition:

| Strategy | Mean Return |
|---|---:|
| PhaseAware | +0.7% |
| DynamicSelector | +12.2% |

Approximate uplift:

```text
≈ +11.5 percentage points
```

Still positive — but substantially smaller than persistent structures.

## Interpretation

Reactive structures appear to violate:

> policy persistence assumptions.

The Dynamic Selector requires:

- temporal continuity
- stable arbitration
- persistent exploitation windows
- coherent selector state

Reactive structures instead exhibit:

- abrupt reorganizations
- confidence collapse
- policy interruption cascades
- rapid latent-state mutation

Therefore:

> the environment reorganizes faster than the controller stabilizes.

This likely explains:

- higher selector entropy
- larger fallback occupancy
- fragmented exploitation geometry
- weaker adaptive continuity

---

# 5. Emerging Structural Interpretation

The architecture chain may now be reinterpreted as:

| Transition | Primary Functional Role |
|---|---|
| TF4 → MR42 | exploit bounded reversion structure |
| MR42 → PhaseAware | suppress catastrophic instability |
| PhaseAware → Dynamic | optimize adaptive exploitation continuity |

Importantly, the architecture chain increasingly resembles:

```text
behavioral-control specialization under heterogeneous latent market topology
```

rather than:

- simple strategy replacement,
- or isolated predictive optimization.

This interpretation emerged independently from:

- selector statistics,
- routing geometry,
- temporal persistence analysis,
- and trading-strategy interaction behavior.

---

# Final Session-Level Conceptual Consolidation

The strongest conceptual consolidation emerging from the V5 Phase-1 investigations is likely:

```text
persistent structures support adaptive exploitation continuity,
while reactive structures destabilize policy persistence assumptions
```

Importantly, this interpretation is now supported simultaneously by:

- selector geometry,
- temporal persistence,
- routing occupancy,
- variance decomposition,
- confidence dynamics,
- trading-strategy interactions,
- and adaptive stability behavior.

The project narrative therefore increasingly resembles:

```text
adaptive behavioral topology modeling
```

rather than:

```text
sentiment-driven predictive alpha generation
```

This represents a substantial conceptual maturation of the MPML ↔ MSML research direction, and is substantially richer than:

- “better strategy selection,”
- or “trend vs mean reversion.”

---

# 6. Persistent vs Reactive — Revised Interpretation After Broker Replication

The broker-data replication substantially strengthened confidence in the existence of a genuine structural distinction between persistent and reactive pair families.

Importantly, the strongest replicated findings were not performance metrics.

Instead, replication occurred primarily at the controller-behavior level:

- arbitration duration,
- routing entropy,
- occupancy distributions,
- recovery preferences,
- and transition topology.

This distinction is important because controller-level behavior is considerably less sensitive to individual market regimes than aggregate Sharpe ratios.

The broker replication therefore increases confidence that the persistent/reactive distinction reflects a real latent structural property rather than a performance artifact.

------

## Persistent Structures

Persistent structures appear increasingly compatible with long-horizon adaptive exploitation.

The selector behaves as though:

- latent structure survives uncertainty,
- equilibrium eventually re-emerges,
- waiting has informational value,
- and adaptive state continuity remains useful.

Several broker-replicated findings support this interpretation:

- substantially longer PhaseAware occupancy during overlap,
- strong recovery preference toward MeanReversion,
- increasing recovery conservatism as overlap density rises,
- and moderate entropy increases under uncertainty.

The resulting picture is not one of simple trend-following.

Instead, persistent structures appear to contain:

> persistence with bounded relaxation.

The controller behaves as though uncertainty is temporary and exploitable structure is expected to return.

This makes persistent families appear fundamentally uncertainty-tolerant.

------

## Reactive Structures

Reactive structures exhibit a markedly different response.

The broker replication showed:

- large increases in routing entropy,
- strong growth in PhaseAware occupancy,
- substantial increases in MeanReversion abandonment,
- and continued preference for TrendFollowing recovery after arbitration.

In particular:

MR → PhaseAware transitions increased from approximately 10% to approximately 25% during overlap-active periods.

This suggests that reactive structures do not merely contain more noise.

Instead, the controller behaves as though local equilibrium assumptions become unreliable under uncertainty.

The resulting behavior is highly exploratory:

- exploitation states are abandoned more aggressively,
- arbitration dominates occupancy,
- and routing flexibility increases substantially.

Reactive structures therefore appear less consistent with uncertainty tolerance and more consistent with opportunity-seeking adaptation.

The controller behaves as though exploitable opportunities exist, but their persistence is uncertain and must be continually re-evaluated.

------

# 7. Revised Structural Hypothesis

The strongest current interpretation is no longer:

> persistent structures are easier to model than reactive structures.

Instead, the evidence increasingly supports:

> persistent and reactive structures represent fundamentally different organizational topologies.

Persistent structures appear to support:

- metastability,
- bounded relaxation,
- equilibrium persistence,
- and adaptive continuity.

Reactive structures appear to support:

- rapid reorganization,
- instability-sensitive transitions,
- opportunity-driven exploitation,
- and elevated uncertainty management.

This interpretation survived:

- independent data sources,
- extended historical periods,
- overlap-geometry changes,
- and selector-topology replication.

Not all performance findings survived replication.

However, the controller-level findings largely did.

This significantly increases confidence that the distinction reflects a genuine structural property of the underlying market topology.

------

# 8. Confidence Level After Broker Replication

Current confidence assessment:

| Finding                                          | Confidence |
| ------------------------------------------------ | ---------- |
| Persistent vs Reactive distinction exists        | High       |
| Family effect dominates DL effect                | High       |
| Overlap changes controller behavior              | High       |
| Overlap increases arbitration duration           | High       |
| Overlap increases routing entropy                | High       |
| Reactive structures abandon MR more aggressively | High       |
| Dense overlap improves performance               | Low        |
| Reactive recovery-policy flip                    | Low        |

The strongest surviving interpretation of the V5 Phase-1 investigations is therefore:

> sentiment overlap systematically changes how uncertainty is managed by the adaptive controller.

This finding replicated across both Yahoo and broker datasets.

The primary remaining uncertainty is no longer whether the interpretation is real.

Instead, the primary uncertainty is whether these findings generalize beyond the LVTF regime.

Future investigations of HVTF, LVMR, and HVMR therefore represent the most important remaining validation step.



