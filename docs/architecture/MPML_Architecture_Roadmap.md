# MPML Architecture Roadmap

**Status:** Living Architecture (July 2026)

This document describes both the long-term architecture and the current implementation state of MPML.

Completed phases represent architectural milestones that have been implemented and experimentally validated. Future phases describe planned evolution while preserving compatibility with the completed architecture.

---

# Current Implementation Status

The foundational architectural migration of MPML is complete.

Completed architectural foundations:

✓ Behavioral Surface abstraction
✓ Behavioral Surface Registry
✓ Canonical Behavioral Identity
✓ Behavioral Prediction Artifact interface
✓ Behavioral Aggregation Layer
✓ Metadata-driven Strategy Registry
✓ Evaluation Policy Registry
✓ StrategyEvaluation representation
✓ Canonical evaluation serialization
✓ Recommendation representation

MPML now exposes three stable research artifacts:

- Behavioral Prediction Artifacts (imported from MSML)
- StrategyEvaluation artifacts (produced by MPML)
- Recommendation artifacts (produced by MPML, public MPML→MRML interface)

StrategyEvaluation remains the canonical evidence layer.
Recommendation is the canonical repository boundary between MPML and MRML.

No further architectural work is currently planned for Behavioral Surface integration, StrategyEvaluation representation, or Recommendation representation.

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

## Stable Repository Boundaries

The boundary between repositories should be defined by canonical research artifacts rather than implementation details.

Each repository owns exactly one stage of the research pipeline and communicates exclusively through stable artifact contracts.

For example,

```
Behavioral Surface

↓

Behavioral Prediction Artifact

↓

Behavioral Features

↓

Strategy Recommendation
```

rather than through repository-specific internal data structures.

Whenever practical:

- producers define canonical metadata
- consumers validate rather than reconstruct metadata
- compatibility adapters remain isolated
- canonical identities flow unchanged across repository boundaries

This principle allows Behavioral Surfaces to evolve independently of MPML while preserving reproducibility and backward compatibility.

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

# Evaluation Policies

Strategy compatibility and experiment scope are independent concepts.

The Strategy Registry describes which strategies are capable of operating under a given Behavioral Surface or Behavioral State.

Evaluation Policies determine which compatible strategies participate in a particular experiment.

For example

```
Behavioral Surface

↓

Strategy Registry

↓

Compatible Strategies

↓

Evaluation Policy

↓

Strategies Evaluated
```

Typical policies may include

```
phaseaware_default

registry_all

trend_following_only

mean_reversion_only

custom
```

Evaluation Policies exist to preserve reproducible research while preventing unnecessary combinatorial growth.

Current implementation status

- `src/strategy_registry.py` defines `StrategyDefinition`,
  `StrategyCapabilities`, `StrategyRegistry`, `EvaluationPolicy` and
  `EvaluationPolicyRegistry`.
- The default registry wraps existing TF and MR implementations rather than
  rewriting them.
- `phaseaware_default` preserves the legacy benchmark by resolving to
  `TF4 + MR42` through policy metadata.
- Runtime code now resolves the PhaseAware benchmark pair through the policy
  registry while keeping walk-forward behavior unchanged.

They are not recommendation engines.

They simply define experimental scope.

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

Recommendation objects are derived from StrategyEvaluation objects through a
ranking process.

MPML therefore exposes two complementary representations:

- StrategyEvaluation — stable historical evidence
- Recommendation — interpreted ranking of that evidence

Only Recommendation objects cross the repository boundary into MRML.
StrategyEvaluation remains the canonical evidence layer within MPML.

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

behavioral_feature_version

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

The canonical MPML research outputs are:

    strategy_evaluations.parquet
    recommendations.parquet
    experiment_manifest.json

Strategy registry metadata remains an internal MPML capability description
rather than a required run artifact.

CSV summaries remain useful for inspection but are not considered the primary
MPML→MRML interface.

---

# 16. Repository Interaction

```
BSVE

↓

Behavioral Surfaces

↓

MSML

↓

Behavioral Prediction Artifacts (H1)

↓

Behavioral Aggregation Layer

↓

Daily Behavioral Features (D1)

↓

MPML
```

Each repository owns exactly one architectural layer.

- **BSVE** defines Behavioral Surfaces and Behavioral States.
- **MSML** generates predictive Behavioral Artifacts at the native modeling resolution.
- **MPML** transforms those artifacts into features suitable for walk-forward trading evaluation.
- **MRML** consumes MPML recommendations for portfolio construction and execution.

The Behavioral Aggregation Layer forms MPML's adaptation boundary between hourly behavioral predictions and daily trading models. It is an internal MPML component and is not part of the external MSML artifact contract.

## Repository Boundary

Behavioral prediction artifacts form the contractual boundary between MSML and MPML.

MSML is responsible for

- constructing Behavioral Surfaces
- generating Behavioral predictions
- exporting canonical artifact metadata

MPML is responsible for

- discovering compatible artifacts
- validating artifact metadata
- consuming predictions during walk-forward evaluation

MPML should never recreate Behavioral Surface identity internally.

Instead, Behavioral identity is imported directly from MSML artifacts.

---

## Behavioral Aggregation Layer

Behavioral prediction artifacts produced by MSML currently operate at **hourly (H1)** resolution, while MPML performs walk-forward evaluation on **daily (D1)** market data.

To preserve causal evaluation while allowing behavioral information to influence D1 models, MPML introduces a Behavioral Aggregation Layer.

Responsibilities:

```
Behavioral Prediction Artifact (H1)

↓

Temporal validation

↓

Leakage-safe aggregation

↓

Daily behavioral feature generation

↓

D1 feature matrix
```

The aggregation layer:

- validates prediction timestamps
- preserves causal ordering
- performs H1→D1 temporal alignment
- generates deterministic daily behavioral summary features
- exposes a stable D1 interface to downstream models

This layer is an internal implementation detail of MPML.

MSML remains unaware of the aggregation process.

Future versions of MPML operating natively on H1 data may bypass this layer entirely while preserving the external repository interface.

---

## Behavioral Feature Contract

Behavioral prediction artifacts remain the canonical interface between MSML and MPML.

Within MPML, these artifacts are transformed into a stable set of daily behavioral features suitable for D1 machine-learning models.

Current features include

```
dl_signal_mean_24h

dl_signal_std_24h

dl_signal_last

dl_signal_abs_mean

dl_signal_flip_count
```

These features summarize the behavior of hourly prediction signals over the previous trading day while preserving causal ordering.

They are intentionally treated as optional model features.

Models remain fully functional when behavioral features are unavailable, allowing historical Trend/Vol experiments and non-behavioral evaluations to execute without modification.

Future behavioral features may be added without changing the external artifact contract.

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

## Phase B (Completed)

### Behavioral Surface Runtime Integration

### Objective

Replace hardcoded runtime assumptions based on `dl_regime` with Behavioral Surface metadata while preserving existing runtime behaviour.

The Behavioral Surface Registry introduced in Phase A now propagates canonical Behavioral Surface metadata throughout the runtime.

Deliverables

- Surface-aware runtime
- Behavioral Surface propagated through experiment pipeline
- Behavioral Surface manifests
- Canonical experiment metadata
- Runtime compatibility bridge
- Backward-compatible Trend/Vol execution

Notes

Runtime behaviour intentionally remains unchanged.

Non-Trend/Vol Behavioral Surfaces may execute without DL predictions while the runtime migration is completed.

------

## Phase C (Completed)

---

### Phase C Completion Note

Phase C establishes the permanent architectural boundary between MSML and MPML.

Completion of this phase represents more than implementation of the Behavioral Prediction Artifact interface. It confirms that independently developed Behavioral Surfaces can be exported by MSML and consumed by MPML through a stable, surface-agnostic contract.

Behavioral Prediction Artifacts now constitute the canonical producer–consumer interface between the repositories.

This interface has been validated through end-to-end walk-forward evaluation using multiple Behavioral Surfaces, demonstrating:

- canonical Behavioral Identity propagation
- deterministic artifact discovery
- leakage-safe H1→D1 behavioral aggregation
- behavioral feature generation
- downstream strategy adaptation
- reproducible changes in trading performance

Future Behavioral Surfaces are expected to integrate through this interface without requiring architectural modifications to MPML.

Subsequent phases therefore focus on extending MPML capabilities rather than continuing Behavioral Surface integration.

---

### Behavioral Prediction Artifact Integration

### Objective

Enable MPML to consume canonical Behavioral prediction artifacts produced by MSML.

Behavioral prediction artifacts are no longer identified by `dl_regime`.

Instead, every artifact is identified by

```
surface_id

surface_version

state_id

model

target_horizon

feature_set
```

MPML should discover, validate and consume these artifacts directly without reconstructing Behavioral identity from legacy metadata.

### Deliverables

- Behavioral Artifact Resolver
- Surface-aware artifact discovery
- Canonical artifact validation
- Runtime prediction loading
- Prediction cache integration
- Behavioral Aggregation Layer
- Leakage-safe H1→D1 aggregation
- Daily behavioral feature generation
- Backward compatibility with legacy Trend/Vol artifacts
- Runtime logging for Behavioral artifact loading

Behavioral prediction artifacts become optional behavioral features within the existing walk-forward machine-learning pipeline.

This phase intentionally preserves existing phase prediction, strategy-selection and recommendation algorithms.

Its purpose is to establish a clean producer–consumer boundary between MSML and MPML while enabling behavioral information to participate in existing model pipelines through stable interfaces.

Behavioral Prediction Artifacts should now be regarded as stable research infrastructure rather than experimental extensions.

Future architectural evolution should build upon this interface rather than replacing it.

### Phase C Validation

Phase C has now been validated through end-to-end integration between MSML and MPML.

Behavioral Prediction Artifacts produced by MSML have been successfully consumed by MPML using the canonical Behavioral Surface identity

```
surface_id
surface_version
state_id
```

without reconstructing Behavioral identity from legacy runtime metadata.

Validation has demonstrated:

- successful artifact discovery
- canonical identity propagation
- leakage-safe H1→D1 behavioral aggregation
- downstream feature generation
- walk-forward strategy adaptation
- measurable trading impact across multiple Behavioral Surfaces

Behavioral Prediction Artifacts therefore constitute the stable producer–consumer interface between MSML and MPML.

Future Behavioral Surfaces should integrate through this interface without requiring architectural changes to MPML.

------

## Phase D

### Strategy Registry

> Phase D is considered architecturally complete.
>
> The Strategy Registry now provides the permanent metadata layer describing trading capabilities independently of runtime implementation. 
>
> Future work may add additional strategy metadata, but no further architectural changes are expected for this phase.
>
> Subsequent phases build upon the Strategy Registry rather than modifying it.

### Objective

Introduce trading behaviors as first-class architectural objects.

Deliverables

- StrategyDefinition abstraction

- Strategy Registry

- Capability metadata

- Capability queries

- Evaluation Policy Registry

- Default PhaseAware policy

- Backward compatibility

Strategies become metadata-driven objects rather than hardcoded selector choices.

Status

- Implemented
- Default strategy metadata lives in `src/strategy_registry.py`
- Capability queries include `all()`, `get()`, `by_family()`,
  `supporting_surface()`, `supporting_state()`, and `supporting_asset()`
- Evaluation policy `phaseaware_default` now supplies the legacy TF4/MR42
  benchmark without hardcoded selector logic

------

## Phase E (Completed)

### Strategy Evaluation Representation

> Phase E is considered architecturally complete.
>
> StrategyEvaluation now forms MPML's canonical evaluation artifact.
>
> Walk-forward evaluation, deterministic identity, serialization and experiment provenance have been unified into a stable evidence layer.
>
> Future work may extend evaluation metadata, but Recommendation semantics should build upon StrategyEvaluation rather than modifying it.

### Objective

Represent the evidence produced by walk-forward evaluation as stable,
repository-independent domain objects.

### Deliverables

✓ StrategyEvaluation abstraction

✓ Deterministic evaluation identity

✓ Canonical evaluation metadata

✓ strategy_evaluations.parquet

✓ Evaluation schema versioning

✓ Experiment manifest integration

✓ Backward-compatible runtime integration

### Validation

Phase E has been validated through end-to-end execution.

Validation confirms:

- byte-identical legacy benchmark outputs
- unchanged walk-forward behaviour
- deterministic evaluation identity
- stable parquet serialization
- backward-compatible experiment manifests

StrategyEvaluation should now be regarded as a stable research artifact rather than an implementation detail.

------

## Phase F

### Experiment Provenance

Objective

Continue strengthening experiment reproducibility and provenance.

This phase extends existing experiment manifests while preserving compatibility with earlier evaluation artifacts.

Typical additions may include:

- registry versions
- Behavioral Surface provenance
- Recommendation provenance
- software version metadata
- execution environment metadata

Phase F intentionally does not alter evaluation or recommendation semantics.

------

## Phase G

### Recommendation Representation

Objective

Recommendation objects SHALL reference StrategyEvaluation objects and MUST NOT duplicate evaluation evidence already represented by StrategyEvaluation.

Recommendation exists to express ranking and interpretation only.

Historical performance metrics, confidence estimates and other evaluation evidence remain the responsibility of StrategyEvaluation.

This separation preserves a single canonical evidence layer while allowing recommendation policies to evolve independently.

### Phase G1 — Recommendation Representation

**Status: Complete (Phase G1)**

Deliverables

- Recommendation abstraction (`src/recommendation.py`)
- Deterministic recommendation identity (`build_recommendation_id`)
- Recommendation serialization (`recommendations.parquet`)
- Recommendation schema versioning (`RECOMMENDATION_SCHEMA_VERSION = "1.0.0"`)
- Recommendation manifest integration (`recommendation_schema_version`, `recommendation_count`)
- Backward-compatible runtime integration

#### Recommendation

`Recommendation` is MPML's canonical recommendation artifact and the public MPML→MRML interface.

```python
@dataclass(frozen=True)
class Recommendation:
    recommendation_id: str   # deterministic SHA-256 derived ID
    evaluation_id: str       # references StrategyEvaluation
    rank: int
    recommendation_policy: str
    metadata: dict[str, Any]
```

`Recommendation` is intentionally lightweight.  It references `StrategyEvaluation` through
`evaluation_id` and does not duplicate evaluation evidence (expected_return, expected_sharpe,
expected_drawdown, confidence, stability, fold statistics).  Those belong exclusively to
`StrategyEvaluation`.

`recommendation_id` is derived deterministically from `RECOMMENDATION_SCHEMA_VERSION`,
`evaluation_id`, `recommendation_policy`, and `rank`.  IDs are stable across runs.

#### recommendations.parquet

One row per `Recommendation`.  Columns mirror the dataclass fields.
`metadata` is JSON-encoded.

`Recommendation` artifacts reference `StrategyEvaluation` through `evaluation_id`.

MPML produces the following artifacts after a completed run:

```
strategy_evaluations.parquet   — canonical historical evidence (StrategyEvaluation)
recommendations.parquet        — public MPML→MRML interface (Recommendation)
experiment_manifest.json       — run manifest with recommendation_schema_version and recommendation_count
```

This phase introduced Recommendation representation only. Recommendation semantics are introduced in G2.

Recommendation policy remains unchanged.

### Phase G2 — Recommendation Policy

**Status: Complete (Phase G2)**

> G2 is considered complete. Recommendation semantics now provide a
> deterministic `sharpe_rank_v1` ranking policy with optional Top-N output,
> while preserving StrategyEvaluation and walk-forward evaluation unchanged.

### Objective

Introduce deterministic recommendation semantics on top of the canonical
StrategyEvaluation evidence layer.

qG2 defines how a collection of StrategyEvaluation objects is interpreted and
converted into an ordered collection of Recommendation objects.

A recommendation policy is defined as a deterministic mapping:

    StrategyEvaluation[]
            ↓
    Recommendation Policy
            ↓
    Recommendation[]

Recommendation policies operate on evaluation evidence. They do not perform
walk-forward evaluation and do not modify StrategyEvaluation objects.

### Deliverables

- Generic recommendation builder
- Explicit recommendation policy abstraction
- Deterministic default ranking policy
- Top-N recommendation support
- Stable recommendation ordering
- Stable MPML recommendation interface

The initial policy should remain deliberately simple and transparent. It should
provide a deterministic ranking over StrategyEvaluation evidence without
introducing sophisticated multi-objective or statistical ranking machinery.

Recommendation identity remains deterministic and continues to be derived from
the evaluation identity, policy and rank.

### Recommendation Confidence

G2 does not introduce a new statistical definition of recommendation confidence.

Although confidence is a potential recommendation attribute, no arbitrary
confidence metric should be invented solely to satisfy the architectural
interface. Until a justified confidence methodology exists, confidence remains
optional metadata rather than a required ranking input.

Confidence calibration is therefore deferred to future research.

### Architectural Constraints

G2 MUST:

- preserve StrategyEvaluation unchanged
- preserve walk-forward evaluation unchanged
- keep recommendation policies independent of individual strategies and
  Behavioral Surfaces
- produce deterministic recommendations from deterministic evaluation input
- avoid hardcoded strategy-specific ranking logic

G2 MUST NOT introduce:

- new Behavioral Surface abstractions
- new evaluation evidence models
- live or portfolio decision logic
- complex ranking frameworks without a demonstrated research requirement

Recommendation semantics remain an interpretation layer over the existing
evidence layer.

### Phase G3 — Evaluation Scope and Strategy Selection

### Objective

Provide explicit user control over which registered strategies participate in an
MPML evaluation.

G3 exposes a small, user-facing override of the existing evaluation scope. It
does not introduce a new strategy-selection framework.

The Strategy Registry remains the authority for strategy identity and
capabilities. Existing Evaluation Policies continue to define the default
experimental scope. An explicit CLI strategy selection allows the user to
narrow that scope for targeted experiments.

The objective is practical experiment control:

    Strategy Registry
            ↓
    Evaluation Scope
            ↓
    Strategies Evaluated
            ↓
    StrategyEvaluation
            ↓
    Recommendation

### Strategy Selection

A single repeatable CLI option should support both individual and multiple
strategy selection:

    --strategy STRATEGY_ID

For example:

    --strategy TF4

or:

    --strategy TF4 --strategy MR42

The option selects the strategies to be evaluated; it does not alter their
implementation or evaluation semantics.

The default invocation, without explicit strategy selection, MUST preserve the
existing benchmark evaluation scope exactly.



### Deliverables

- Repeatable `--strategy STRATEGY_ID` CLI option
- Support for evaluating one or multiple explicitly named strategies
- Validation of requested strategy identifiers against the Strategy Registry
- Clear handling of unknown strategy identifiers
- Clear handling of strategies incompatible with the active Behavioral Surface
- Experiment manifest recording of the resolved evaluation scope
- Backward-compatible default behavior

### Evaluation Scope Semantics

G3 controls which strategy evidence is generated.

It does not control how that evidence is ranked.

The distinction is therefore:

    G3
    Which strategies are evaluated?
            ↓
    StrategyEvaluation
            ↓
    G2
    How are the resulting evaluations ranked?
            ↓
    Recommendation

Explicit strategy selection MUST affect only the set of strategies evaluated.

It MUST NOT alter:

- walk-forward logic
- strategy implementation
- StrategyEvaluation semantics
- recommendation policy
- recommendation ranking
- Behavioral Surface definitions

### Provenance

The experiment manifest should record the resolved evaluation scope, including
the effective strategy identifiers used for the run.

This applies both to explicitly selected strategies and to the default evaluation
scope, so that an experiment is self-describing and reproducible.

The manifest should record the effective scope rather than relying solely on
the original CLI invocation.

### Design Constraints

G3 should reuse the existing Strategy Registry and Evaluation Policy mechanisms.

It MUST NOT introduce:

- a separate strategy-selection abstraction
- strategy-specific CLI branches
- separate strategy configuration files
- duplicated strategy metadata
- changes to strategy implementations
- changes to the Recommendation model
- additional recommendation policies

The CLI should remain intentionally small.

G3 is limited to explicit strategy-level evaluation scope. Additional evaluation
controls or policy selection may be introduced in a future phase if a concrete
research requirement emerges.

### Non-Goals

G3 does not introduce a general-purpose experiment configuration framework.

It does not expose all Evaluation Policy options through the CLI.

It does not redesign the Strategy Registry.

It does not change the default MPML benchmark.

Its purpose is simply to make targeted strategy evaluation possible while
preserving the existing runtime and artifact contracts.

---

### Phase G4 — Stable MPML–MRML Recommendation Interface

**Objective**

Establish `Recommendation` as the stable, validated repository boundary
between MPML and MRML.

G4 defines the guarantees that a downstream MRML consumer may rely upon when
reading MPML recommendation artifacts, without requiring knowledge of MPML
strategy implementations, walk-forward internals or recommendation-generation
logic.

The objective is contract stability rather than additional recommendation
intelligence.

### Deliverables

- Canonical Recommendation serialization contract
- Explicit Recommendation schema/version contract
- Deterministic recommendation identity and ordering guarantees
- Referential integrity between Recommendation and StrategyEvaluation
- Recommendation provenance sufficient to identify the originating experiment
  and evaluation evidence
- Serialization/deserialization round-trip validation
- Contract-level validation tests
- Stable MPML→MRML artifact documentation

### Recommendation Contract

The Recommendation artifact remains intentionally lightweight:

    Recommendation
        ├── recommendation_id
        ├── evaluation_id
        ├── rank
        ├── recommendation_policy
        └── metadata

Recommendation MUST NOT duplicate StrategyEvaluation evidence.

In particular, expected return, expected Sharpe, drawdown, confidence,
stability and fold/trade statistics remain properties of StrategyEvaluation.

The `evaluation_id` provides the explicit reference from a Recommendation to
its supporting evidence.

### MRML Boundary

MRML may consume Recommendation artifacts without knowledge of:

- MPML strategy implementations
- Behavioral Surface construction
- walk-forward implementation
- evaluation internals
- recommendation-generation implementation

MPML MUST NOT expose repository-specific implementation objects across this
boundary.

MRML is responsible for deciding how, or whether, a Recommendation should be
used in portfolio and execution decisions.

### Contract Guarantees

G4 should establish that, for a supported schema version:

- Recommendation serialization is deterministic and lossless
- Recommendation identity is deterministic
- Recommendation ordering is deterministic for identical evidence and policy
- each Recommendation references a valid StrategyEvaluation
- schema versions are explicit
- provenance permits the recommendation to be traced to its originating
  experiment/evaluation
- unsupported or malformed artifacts fail validation clearly

### Non-Goals

G4 does not introduce:

- new recommendation policies
- new ranking criteria
- portfolio logic
- risk management
- execution logic
- live data handling
- MRML implementation
- new strategy-selection mechanisms
- new evaluation evidence models

G4 is complete when the Recommendation artifact can be treated as a stable
external contract rather than an MPML-internal output.


---

BSVE
        │
        ▼
Behavioral Surface

        │
        ▼
MSML
        │
        ▼
Behavioral Prediction Artifact

        │
        ▼
MPML
        │
        ▼
Walk-forward Evaluation

        │
        ▼
StrategyEvaluation

        │
        ▼
Recommendation

        │
        ▼

MRML
        │
        ▼
Portfolio Decision

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
- Native H1 MPML execution (removing the D1 aggregation layer)

---

# Guiding Principle

Behavioral Surfaces describe market structure.

Strategies describe trading behavior.

StrategyEvaluation records canonical historical evidence.

Recommendations interpret that evidence.

MRML decides how (or whether) to act upon those recommendations.

Each architectural layer therefore contributes exactly one new abstraction while preserving the contracts established by the previous layer.

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