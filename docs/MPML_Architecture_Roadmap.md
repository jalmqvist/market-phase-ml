# MPML Architecture Roadmap

**Status:** Draft (July 2026)

---

# 1. Purpose

Market Phase Machine Learning (MPML) is the third stage of the behavioral
research pipeline.

Its purpose is **not** to discover predictive signals.

Its purpose is to determine **how those signals should be traded** under
realistic walk-forward evaluation.

The long-term research pipeline is:

```
BSVE
    ↓
MSML
    ↓
MPML
    ↓
MRML
```

where

- **BSVE** defines behavioral state spaces.
- **MSML** evaluates predictive value.
- **MPML** evaluates trading value.
- **MRML** performs portfolio-level execution and risk management.

---

# 2. Design Philosophy

MPML should become a reusable strategy recommendation engine rather than a
backtesting application.

Future components should communicate through stable interfaces rather than
repository-specific file formats.

Every major abstraction should correspond to a genuine financial concept.

Examples:

Behavioral Surface

Behavioral State

Strategy

Recommendation

Experiment

rather than implementation details.

---

### Metadata-Driven Architecture

As MPML evolves, metadata becomes a primary architectural mechanism rather than merely documentation.

Earlier versions of MPML relied on hardcoded knowledge about behavioral surfaces, strategy families and evaluation workflows. Future versions should instead describe these capabilities through metadata exposed by registries.

Components should communicate by asking *what* an object supports rather than *which implementation* it belongs to.

This allows Behavioral Surfaces, Strategies, Recommendation Engines and future extensions to evolve independently while remaining interoperable through stable interfaces.

```
Registry

↓

Metadata

↓

Capabilities

↓

Generic algorithms
```

instead of

```
Hardcoded classes

↓

if surface == LVTF

↓

if strategy == TF4

↓

special-case logic
```



---

# 3. Responsibilities

## MPML owns

- Walk-forward evaluation
- Strategy evaluation
- Strategy ranking
- Strategy recommendation
- Behavioral state interpretation
- Experiment metadata
- Recommendation generation

## MPML does NOT own

- Broker APIs
- Live execution
- Portfolio allocation
- Position sizing
- Risk profiles
- Drawdown limits
- Exposure management
- User accounts

Those belong to MRML.

---



# 4. Behavioral Surface Abstraction

Historically MPML has operated on a single hardcoded market representation:

```
Trend × Volatility
```

This abstraction is no longer sufficient.

Future versions should treat all market representations uniformly.

```
BehavioralSurface

    TrendVolSurface

    ReactiveJPYSurface

    ReactiveCHFSurface

    PersistentSurface

    ...
```

Each Behavioral Surface exposes:

```
surface_id

surface_version

state_id

state_name

metadata
```

The strategy selector should never care how a Behavioral Surface was
constructed.

---

## Behavioral Surface Contract

Behavioral Surfaces are produced by BSVE/MSML and consumed by MPML.

MPML does not construct Behavioral Surfaces itself.

Instead, MPML depends only on a stable metadata contract describing each
surface.

Every Behavioral Surface should expose, at minimum,

```
surface_id

surface_version

display_name

states[]

metadata
```

Every Behavioral State should expose

```
state_id

display_name

description

metadata
```

MPML should treat these objects as immutable research artifacts.

Their internal construction, calibration and validation remain the
responsibility of BSVE/MSML.

---

### DL Prediction Artifact Contract

Behavioral prediction artifacts produced by MSML expose the following canonical identity:

```
surface_id
surface_version
state_id
```

together with

```
model
target_horizon
feature_set
```

These fields uniquely identify one behavioral prediction surface.

MPML should use these fields directly for artifact selection and runtime propagation.

`dl_regime` remains available only as a deprecated compatibility alias for legacy Trend/Vol artifacts and must not be used as the canonical runtime identifier.

---

# 5. Behavioral Surface Registry

Behavioral Surfaces should be loaded through a registry.

Instead of

```
if LVTF:
```

MPML should perform

```
surface = registry.load(surface_id)
```

Benefits

- No hardcoded surfaces
- Plug-in architecture
- Multiple simultaneous surfaces
- Independent evolution of BSVE and MPML

---

# 6. Behavioral State

Behavioral States become first-class entities.

Every state should contain

```
state_id

surface_id

display_name

description

metadata
```

Behavioral States are dynamical market objects rather than simple labels.

Future extensions may include

- confidence
- maturity
- persistence
- transition probability

without changing downstream APIs.

---

# 7. Strategy Registry

Strategies should become metadata-driven.

Instead of selecting strategies directly, MPML maintains a registry of
available strategies.

Each strategy contains

```
strategy_id

family

entry rule

exit rule

supported surfaces

supported states

supported asset classes

required indicators

dependencies

tags

metadata
```

The Strategy Registry should describe strategy capabilities rather than encode selection logic. Selection algorithms operate entirely on registry metadata, allowing new strategies to participate automatically without requiring changes to the recommendation engine.

---

# 8. Walk-forward Evaluation

Walk-forward remains the canonical evaluation procedure.

Its responsibilities are

```
Historical Data

↓

Behavioral State

↓

Strategy Evaluation

↓

Performance Metrics

↓

Ranking
```

Walk-forward outputs become evidence rather than final decisions.

---

# 9. Strategy Ranking

Strategy rankings are generated offline using walk-forward evaluation.

Unlike earlier MPML implementations, the ranking engine should not contain
hardcoded knowledge of specific strategies or behavioral surfaces.

Instead, rankings emerge by combining metadata from the Behavioral Surface
Registry and the Strategy Registry.

```
Behavioral Surface

↓

Behavioral State

↓

Compatible Strategies

↓

Walk-forward Evidence

↓

Strategy Ranking
```

Ranking criteria may include

- Expected Sharpe
- Expected Return
- Expected Maximum Drawdown
- Stability across folds
- Walk-forward consistency
- Confidence
- Number of supporting observations

The ranking engine should operate entirely on registry metadata rather than
strategy-specific decision trees.

This allows newly introduced Behavioral Surfaces and Strategies to participate
in the recommendation process without modifications to the ranking algorithm.

---

# 10. Recommendation Engine

The Recommendation Engine becomes MPML's primary output.

Instead of

```
TF4
```

MPML returns

```
Behavioral Surface

↓

Behavioral State

↓

Strategy Registry

↓

Compatible Strategies

↓

Walk-forward Ranking

↓

Recommendation
```

Each recommendation contains

```
strategy

rank

behavioral_state

expected return

expected Sharpe

expected drawdown

walk-forward support

confidence

metadata
```

The Recommendation object is produced by combining Behavioral Surface metadata, Behavioral State metadata and Strategy Registry metadata through generic ranking algorithms. **It forms the primary interface between MPML and MRML.**

---

# 11. Recommendation Philosophy

Recommendations represent evidence rather than decisions.

MPML answers

> Which strategies have historically performed best in the current market
> state?

It does **not** answer

> Should a trade be placed?

That decision belongs to MRML.

---

# 12. MRML Interface

MRML should consume Recommendation objects.

Typical workflow

```
Current Market

↓

Behavioral Surface

↓

Behavioral State

↓

MPML Recommendation

↓

Risk Controller

↓

Broker Execution
```

MPML remains completely unaware of

- account balance
- risk profile
- current positions
- portfolio exposure

This separation greatly simplifies both repositories.

---

# 13. Strategy Recommendation Lifecycle

Offline

```
Historical Data

↓

Behavioral Surface

↓

Behavioral State

↓

Strategy Registry

↓

Walk-forward Evaluation

↓

Strategy Ranking

↓

Recommendation Table
```

Online

```
Current Market

↓

Behavioral Surface

↓

Behavioral State

↓

Recommendation Lookup

↓

Ranked Strategies

↓

Recommendation
```

No walk-forward computation occurs during live execution.

Future recommendation engines should remain entirely metadata-driven. Their
responsibility is to compose information provided by registries rather than to
encode knowledge about individual Behavioral Surfaces or Strategies.

---

# 14. Experiment Architecture

Every MPML experiment should produce a manifest containing

```
dataset_version

behavioral_surface

surface_version

git_commit

experiment_type

timestamp

random_seed

parameters
```

This provides reproducibility without requiring external experiment tracking
systems.

---

# 15. Output Artifacts

Future MPML outputs should include

```
strategy_registry.parquet

strategy_rankings.parquet

recommendations.parquet

experiment_manifest.json
```

CSV summaries remain useful for inspection but should no longer be considered
the primary interface.

---

# 16. Repository Interaction

```
BSVE

↓

Behavioral Surface

↓

MSML

↓

Behavioral Prediction Artifacts

↓

MPML
```

Each repository owns exactly one layer.

---

# Behavioral Surface Ownership

Behavioral Surfaces originate entirely within BSVE/MSML.

Examples include

- Trend/Vol Surface
- Reactive JPY Surface
- Reactive CHF Surface
- Persistent Surface

MPML should never duplicate the logic used to generate or calibrate these
surfaces.

Instead, MPML consumes Behavioral Surface metadata together with Behavioral
State labels produced by BSVE/MSML.

Future Behavioral Surfaces should become available to MPML simply by
registering their metadata rather than modifying MPML algorithms.

---

## Behavioral Surface Manifest

Each Behavioral Surface should be accompanied by metadata describing its
identity and provenance.

Suggested fields include

```
surface_id

surface_version

dataset_version

calibration_version

state_spec_version

created

description
```

MPML should preserve this metadata in experiment manifests wherever practical,
allowing downstream analyses to trace recommendations back to the exact
Behavioral Surface used during evaluation.

---

## State Naming

Behavioral State identifiers should be treated as stable external contracts.

Where historical naming inconsistencies exist (for example HVR vs HVMR),
compatibility aliases may be provided internally.

However, Behavioral Surface registries should expose a single canonical
identifier for each state.

Future Behavioral Surfaces should avoid introducing multiple names for the
same market state.

---

## Canonical Identity Rule

Throughout MPML,

```
surface_id
surface_version
state_id
```

define Behavioral Surface identity.

No runtime component should infer Behavioral Surface identity from

```
dl_regime
```

except inside explicit backward-compatibility adapters.

---

## Legacy Compatibility

During the Behavioral Surface migration, MPML may encounter legacy artifacts containing `dl_regime`.

Compatibility adapters may populate

```
surface_id
state_id
```

from

```
dl_regime
```

for historical Trend/Vol artifacts.

Newly produced artifacts should always use the canonical Behavioral Surface identity.

---

## Consumer Expectations

Behavioral prediction artifacts consumed by MPML are expected to provide:

```
pair
entry_time
prediction_available_timestamp

model
surface_id
surface_version
state_id
target_horizon
feature_set
```

MPML must perform all artifact selection using the behavioral identity fields.

`prediction_available_timestamp` is the causal timestamp used for temporal validation.

`prediction_generated_timestamp` and `artifact_created_timestamp` are provenance only.

---

# 17. Planned Evolution

## Phase A (Completed)

Behavioral Surface Registry

Objective

Introduce Behavioral Surfaces as first-class objects while preserving existing
Trend/Vol functionality.

Deliverables

- Behavioral Surface abstraction
- Behavioral State abstraction
- Behavioral Surface Registry
- Trend/Vol implementation
- Reactive-JPY example implementation
- Registry-backed manifests
- Backward compatibility

This phase intentionally preserves existing runtime behaviour.

---

## Phase B

Behavioral Surface Integration

Objective

Replace all runtime assumptions based on `dl_regime` with Behavioral Surface identity (`surface_id`, `surface_version`, `state_id`). Runtime components should consume the canonical artifact identity directly rather than reconstructing Behavioral Surface information from legacy fields.

The Behavioral Surface Registry introduced in Phase A establishes the required
abstractions, but MPML still propagates Trend/Vol concepts (for example
`DL_REGIME`) throughout the runtime.

This phase removes those assumptions and allows the runtime to operate on

```
surface_id

+

state_id
```

rather than

```
LVTF

HVTF

LVR

HVR
```

Deliverables

- Surface-aware runtime
- Behavioral Surface propagated through experiment pipeline
- Behavioral State propagated through selector pipeline
- Surface-aware manifests
- Surface-aware artifact loading
- Surface-aware experiment metadata

No Strategy Registry work is included in this phase.

---

## Phase C

Strategy Registry

Objective

Generalize strategy selection using metadata.

Deliverables

- Strategy metadata
- Strategy Registry
- Supported Behavioral Surfaces
- Supported Behavioral States
- Strategy compatibility metadata

Strategies become metadata-driven objects rather than hardcoded selector
choices.

---

## Phase D

Recommendation Engine

Objective

Generate metadata-driven strategy recommendations.

Deliverables

- Ranked recommendations
- Recommendation object
- Recommendation export
- Generic ranking engine

Recommendations become the primary interface consumed by MRML.

---

## Phase E

Experiment Metadata

Objective

Strengthen experiment provenance.

Deliverables

- Rich manifests
- Dataset traceability
- Behavioral Surface provenance
- Strategy Registry provenance
- Recommendation provenance

---

## Phase F

MRML Integration

Objective

Expose a stable recommendation interface.

Deliverables

- Recommendation API
- Portfolio integration
- Risk-management boundary
- Stable repository interface

---

# 18. Future Extensions

Possible future work

- Hierarchical Behavioral Surfaces
- Multi-surface recommendations
- Confidence calibration
- Bayesian strategy ranking
- Ensemble recommendations
- Reinforcement-learning selectors
- Online adaptation
- Experiment database / MLflow backend

---

# 19. Guiding Principle

Behavioral States describe markets.

Strategies describe trading behavior.

Recommendations connect the two.

MPML should therefore evolve from a collection of backtests into a reusable
behavioral strategy recommendation engine.

MRML can then focus exclusively on the independent problem of portfolio risk
management and live execution.

---

# Appendix A — Current Behavioral Surfaces

TrendVolSurface

States

```
LVTF
HVTF
LVR
HVR
```

ReactiveJPYSurface

States

```
JPY_NON_EXTREME
JPY_CONSENSUS_YOUNG
JPY_CONSENSUS_MATURING
JPY_CONSENSUS_MATURE
```

ReactiveCHFSurface

Reserved

PersistentSurface

Reserved

This appendix documents only the public metadata exposed by each Behavioral
Surface.

Behavioral definitions remain the responsibility of BSVE/MSML.