# MPML Project Description

Market-Phase-ML (MPML) is a research framework for evaluating trading
strategies under behavioral context and strict walk-forward validation.

The repository combines:

- behavioral market-state interpretation
- metadata-aware trading strategy definitions
- optional deep-learning feature surfaces
- dynamic routing experiments
- reproducible evaluation and analysis

## Trading Ontology

MPML now treats **Strategy** as a first-class architectural object.

`src/strategy_registry.py` defines:

- `StrategyDefinition`
- `StrategyCapabilities`
- `StrategyRegistry`
- `EvaluationPolicy`
- `EvaluationPolicyRegistry`

The Strategy Registry describes what each strategy supports, including:

- Behavioral Surfaces
- Behavioral States
- asset coverage
- required indicators and features
- trade-direction support
- tags and metadata

Evaluation Policies define experiment scope separately from compatibility.

The initial policy, `phaseaware_default`, preserves the legacy PhaseAware
benchmark by resolving to `TF4 + MR42` through registry metadata rather than
hardcoded runtime logic.

This architectural layer does not change recommendation quality, strategy
performance, or walk-forward behavior; it establishes the metadata foundation
for future recommendation engines.
