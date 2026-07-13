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

Predictive Validation

↓

MPML

↓

Strategy Recommendation

↓

MRML

↓

Risk Management

↓

Broker Execution
```

Each repository owns exactly one layer.

---

# 17. Planned Evolution

## Phase A

Behavioral Surface Registry

- Surface abstraction
- Registry
- Trend/Vol implementation
- Reactive-JPY implementation

---

## Phase B

Strategy Registry

- Strategy metadata
- Supported surfaces
- Supported states

---

## Phase C

Recommendation Engine

- Ranked recommendations
- Recommendation object
- Recommendation export

---

## Phase D

Experiment Metadata

- Rich manifests
- Provenance
- Dataset traceability

---

## Phase E

MRML Integration

- Stable Recommendation interface
- Portfolio execution
- Risk management

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