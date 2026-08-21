# Behavioral Surface Contract

**Repository Interface between BSVE/MSML and MPML**

---

## Purpose

This document defines the public contract between BSVE/MSML and MPML.

Behavioral Surfaces are produced by BSVE/MSML and consumed by MPML.

MPML intentionally does **not** know how Behavioral Surfaces are calibrated,
constructed or validated. Those responsibilities remain within BSVE/MSML.

Instead, MPML depends only upon the public metadata described here.

This separation allows both repositories to evolve independently while
maintaining compatibility.

---

# Ownership

Repository responsibilities are intentionally separated.

| Repository | Responsibility                                               |
| ---------- | ------------------------------------------------------------ |
| BSVE/MSML  | Construct Behavioral Surfaces                                |
| BSVE/MSML  | Calibrate Behavioral States                                  |
| BSVE/MSML  | Validate Behavioral Surface quality                          |
| MPML       | Consume Behavioral Surfaces                                  |
| MPML       | Evaluate strategies conditioned on Behavioral States         |
| MPML       | Produce strategy recommendations                             |
| MRML       | Consume MPML recommendations and perform execution and risk management |

MPML should never duplicate Behavioral Surface construction logic.

---

# Behavioral Surface

Every Behavioral Surface should expose the following public metadata.

| Field           | Description                        |
| --------------- | ---------------------------------- |
| surface_id      | Stable machine-readable identifier |
| surface_version | Semantic version                   |
| display_name    | Human-readable name                |
| states          | Collection of Behavioral States    |
| metadata        | Optional extensible metadata       |

The internal implementation is not part of the public contract.

---

# Behavioral Prediction Artifact

Behavioral Surfaces classify market behaviour.

Predictive models generate forecasts conditioned on those Behavioral States.

These are intentionally separate artifacts.

Behavioral Surfaces answer

> "What market state are we currently observing?"

Prediction artifacts answer

> "Given this Behavioral State, what does the trained model predict?"

MPML consumes both artifacts.

Behavioral Surface metadata determines which prediction artifacts are
applicable, while prediction artifacts provide the model outputs used during
strategy evaluation.

Behavioral prediction artifacts should expose, at minimum,

| Field                | Description                          |
| -------------------- | ------------------------------------ |
| surface_id           | Behavioral Surface identifier        |
| surface_version      | Surface version used during training |
| state_id             | Behavioral State used for training   |
| model                | Model family (MLP, LSTM, …)          |
| target_horizon       | Prediction horizon                   |
| feature_set          | Training feature set                 |
| prediction_timestamp | Artifact creation timestamp          |
| metadata             | Optional extensible metadata         |

Prediction-specific fields (such as probabilities, confidence scores or signal
strength) remain model-dependent and are outside the scope of this contract.

---

# Behavioral State

Every Behavioral State should expose the following public metadata.

| Field        | Description                        |
| ------------ | ---------------------------------- |
| state_id     | Stable machine-readable identifier |
| display_name | Human-readable name                |
| description  | Short description                  |
| metadata     | Optional extensible metadata       |

Behavioral States are immutable value objects.

---

# Current Behavioral Surfaces

The following Behavioral Surfaces currently form part of the public contract.

## Trend × Volatility

Surface ID

```
trend_vol
```

Canonical states

```
LVTF
HVTF
LVR
HVR
```

Legacy aliases accepted for compatibility

```
LVMR → LVR
HVMR → HVR
```

---

## Reactive JPY

Surface ID

```
reactive_jpy
```

Canonical states

```
JPY_NON_EXTREME

JPY_CONSENSUS_YOUNG

JPY_CONSENSUS_MATURING

JPY_CONSENSUS_MATURE
```

---

# Compatibility

Behavioral State identifiers constitute stable external interfaces.

Future releases should preserve canonical identifiers whenever practical.

Where historical names require migration, compatibility aliases may be
provided, but only one canonical identifier should exist for each state.

---

This contract intentionally specifies repository interfaces rather than model
implementations.

Behavioral Surface artifacts and Behavioral Prediction artifacts constitute the
public interface between MSML and MPML.

How those artifacts are produced remains an internal implementation detail of
BSVE/MSML.

---

# Architectural Principle

Behavioral Surfaces should be treated as immutable research artifacts.

MPML consumes Behavioral Surface metadata but should remain agnostic to the
algorithms used to generate those artifacts.

Future Behavioral Surfaces should become available to MPML by registering new
metadata rather than modifying MPML algorithms.

---

# Relationship to the Behavioral Surface Registry

This document defines the repository interface between BSVE/MSML and MPML.

Implementation details of the MPML registry are documented separately in

```
docs/behavioral/behavioral_surface_registry.md
```

The registry is one possible implementation of this contract.

The contract itself is implementation-independent.

---

# Standalone Strategy Evaluation and Behavioral Performance Attribution

## Evaluation Semantics

When an explicit individual strategy (e.g. `--strategy TF1`) is evaluated
alongside a Behavioral Surface and state (e.g. `--behavioral=JPY_CONSENSUS_YOUNG`),
MPML applies **performance-attribution conditioning**, not signal or execution
conditioning.

Specifically:

1. **Signal generation is unconditional.** The strategy computes signals on
   the complete walk-forward test fold.  Rolling indicators, technical signals,
   and stop/target levels are not modified.

2. **Execution is unconditional.** The backtester runs on the complete fold.
   SL/TP hits, trade entries, and trade exits are resolved over the full bar
   sequence, regardless of whether individual bars are state-active.

3. **Conditioning is attribution-only.** After execution, each completed trade
   is annotated with a `behavioral_eligible` flag.  A trade is eligible when
   its **entry observation** falls on a state-active bar.

## State-Active Bar Definition

A bar is state-active if the D1 DL prediction features joined by
`attach_dl_features()` are non-null on that bar:

```
state_active = df_test[D1_FEATURE_COLS].notna().any(axis=1)
```

This is the established per-bar behavioral availability indicator used
throughout the MPML DL pipeline.  No new definition is introduced.

## Trade Lifecycle Preservation

Once a trade is classified as behaviorally eligible (entered on an active bar),
its **complete realized lifecycle** is attributed to the behavioral state:

- The exit may occur on an inactive bar.
- The realized P&L, SL/TP outcome, and exit price are not altered.
- The trade is not force-closed at behavioral state boundaries.

A trade entered on an **inactive** bar receives `behavioral_eligible=False`
and is excluded from conditional performance attribution, even if the trade's
exit occurs on an active bar.

## Auditable Research Artifacts

The `strategy_trades__dl_enabled.csv` artifact exposes:

| Column                  | Description                                          |
| ----------------------- | ---------------------------------------------------- |
| `behavioral_eligible`   | `True` when trade entry is state-active              |
| `behavioral_surface_id` | Behavioral Surface identifier                        |
| `behavioral_state_id`   | Behavioral State identifier                          |

These columns are only present when `_strategy_only_scope=True` and the DL
runtime is enabled.  Baseline runs (no behavioral surface) do not include
these columns, preserving backward compatibility.

## Causal Guarantee

The state-active mask is derived from `D1_FEATURE_COLS` joined by
`attach_dl_features()`, which enforces a minimum one-day merge lag between
the prediction `available_timestamp` and the bar's observation date.  No
future information is introduced by the attribution step.

## Rationale

This design was chosen because standalone strategies use rolling indicators
(`rolling`, `shift`, `ffill`) that are positional with respect to the input
DataFrame index.  Pre-filtering `df_test` to state-active bars before signal
generation would corrupt rolling window computations and cause SL/TP hit
detection to miss breaches during inactive gaps.  Attribution at the
trade-entry level avoids both failure modes while preserving the ability to
compare strategy performance across different behavioral states.