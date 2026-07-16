# Behavioral Surface Registry

**Phase A — Registry and Abstractions | Phase B — Runtime Integration**

> See [`docs/MPML_Architecture_Roadmap.md`](../MPML_Architecture_Roadmap.md) for the
> full design rationale, long-term phases and guiding principles.

---

## Overview

Prior to Phase A, MPML assumed a single hardcoded market representation based
on the Trend × Volatility regime (LVTF, HVTF, LVR, HVR).  This assumption was
embedded throughout the codebase as raw string comparisons, making it difficult
to introduce additional market representations without modifying core logic.

Phase A introduces three abstractions that allow MPML to support multiple
Behavioral Surfaces simultaneously:

| Concept | Role |
|---|---|
| `BehavioralSurface` | A named, versioned market representation that partitions markets into discrete states |
| `BehavioralState` | An immutable value object representing one state within a surface |
| `BehavioralSurfaceRegistry` | A registry that loads and exposes available surfaces |

Phase B makes Behavioral Surfaces the **canonical runtime abstraction**.
Instead of routing on `DL_REGIME` / `LVTF` / `HVTF` internally, the MPML
execution pipeline now carries Behavioral Surface metadata directly.

---

## Package Layout

```
mpml/
    behavioral/
        __init__.py       Public API exports
        base.py           BehavioralSurface ABC + BehavioralState dataclass
        registry.py       BehavioralSurfaceRegistry + default_registry
        trend_vol.py      TrendVolSurface (first built-in surface)
        reactive_jpy.py   ReactiveJPYSurface (second built-in surface)
        compat.py         Compatibility wrappers for existing code
```

---

## Quick Start

```python
from mpml.behavioral import registry

# Load a surface
surface = registry.load("trend_vol")

# Inspect available states
print(surface.state_ids())
# ['LVTF', 'HVTF', 'LVR', 'HVR']

# Resolve a state
state = surface.get_state("LVTF")
print(state.display_name)
# 'Low-Volatility Trend-Following'

# One-liner via registry
state = registry.get_state("trend_vol", "LVTF")

# See all registered surfaces
print(registry.available())
# ['reactive_jpy', 'trend_vol']
```

---

## BehavioralSurface

An abstract base class (`mpml.behavioral.base.BehavioralSurface`) that every
surface implementation must extend.

### Required attributes

| Attribute | Type | Description |
|---|---|---|
| `surface_id` | `str` | Stable machine-readable identifier (e.g. `"trend_vol"`) |
| `surface_version` | `str` | Semantic version string |
| `display_name` | `str` | Human-readable label |

### Required methods

| Method | Returns | Description |
|---|---|---|
| `states()` | `list[BehavioralState]` | All states in canonical order |
| `get_state(state_id)` | `BehavioralState` | Resolve a state by ID or alias |
| `metadata()` | `dict` | Serialisable surface metadata |

### Convenience methods

| Method | Returns | Description |
|---|---|---|
| `state_ids()` | `list[str]` | Canonical state IDs |

---

## BehavioralState

An immutable frozen dataclass (`mpml.behavioral.base.BehavioralState`) that
represents one market state within a surface.

### Fields

| Field | Type | Description |
|---|---|---|
| `state_id` | `str` | Canonical machine-readable identifier |
| `display_name` | `str` | Human-readable label |
| `surface_id` | `str` | Parent surface identifier |
| `description` | `str` | Brief description of the market regime |
| `metadata` | `dict` | Extensible attributes (aliases, tags, …) |

`BehavioralState` objects are hashable and can be used as dict keys or set
members.  `str(state)` returns `state.state_id`.

---

## BehavioralSurfaceRegistry

A dictionary-backed registry (`mpml.behavioral.registry.BehavioralSurfaceRegistry`)
that loads and exposes surfaces by `surface_id`.

### Methods

| Method | Description |
|---|---|
| `register(surface)` | Add a surface to the registry |
| `load(surface_id)` | Return a surface by ID (raises `KeyError` if unknown) |
| `get_state(surface_id, state_id)` | Load a surface and resolve a state in one call |
| `available()` | Sorted list of registered surface IDs |
| `surface_metadata(surface_id)` | Metadata dict for one surface |
| `all_metadata()` | Metadata for all surfaces |

---

## Default Registry

The module-level `default_registry` is pre-populated with all built-in
surfaces and is accessible as `mpml.behavioral.registry`:

```python
from mpml.behavioral import registry

registry.available()
# ['reactive_jpy', 'trend_vol']
```

---

## Built-in Surfaces

### TrendVolSurface (`"trend_vol"`)

The existing Trend × Volatility market representation, now a first-class
Behavioral Surface.  Public behaviour is unchanged.

| State ID | Aliases | Description |
|---|---|---|
| `LVTF` | — | Low-Volatility Trend-Following |
| `HVTF` | — | High-Volatility Trend-Following |
| `LVR` | `LVMR` | Low-Volatility Ranging |
| `HVR` | `HVMR` | High-Volatility Ranging |

> **Compatibility note:** `HVMR` and `LVMR` are legacy MPML names for `HVR`
> and `LVR` respectively.  Both are accepted as aliases by `get_state()`.
> MSML uses `HVR` / `LVR` as canonical names; this surface follows that
> convention.

---

### ReactiveJPYSurface (`"reactive_jpy"`)

A second Behavioral Surface modelling JPY-driven behavioural regimes using
consensus-stage state labels produced by BSVE/MSML.

| State ID | Description |
|---|---|
| `JPY_NON_EXTREME` | No meaningful consensus signal |
| `JPY_CONSENSUS_YOUNG` | Early-stage consensus forming |
| `JPY_CONSENSUS_MATURING` | Consensus strengthening |
| `JPY_CONSENSUS_MATURE` | Established, mature consensus |

> **Note:** This surface does not yet drive walk-forward experiments.
> Its role in Phase A is to validate that MPML can host multiple surfaces
> simultaneously.

---

## Compatibility Wrappers

`mpml.behavioral.compat` provides helpers for existing code that uses
`dl_regime` strings or `src.phases` phase labels.

### `dl_regime_to_state(dl_regime)`

Convert an MSML `dl_regime` string to a `BehavioralState`:

```python
from mpml.behavioral.compat import dl_regime_to_state

state = dl_regime_to_state("LVTF")
# BehavioralState(state_id='LVTF', ...)

state = dl_regime_to_state("HVMR")  # alias accepted
# BehavioralState(state_id='HVR', ...)
```

### `phase_label_to_state(phase_label)`

Convert a `src.phases.MarketPhaseDetector` label to a `BehavioralState`:

```python
from mpml.behavioral.compat import phase_label_to_state

state = phase_label_to_state("HV_Trend")
# BehavioralState(state_id='HVTF', ...)
```

### `resolve_behavioral_state_for_surface(surface_id, dl_regime)` *(Phase B)*

Resolve the canonical behavioral `state_id` for a runtime surface from a
`dl_regime` string.  This is the Phase B bridge between MSML's `dl_regime`
vocabulary and the Behavioral Surface abstraction.

**Rules:**

- `"trend_vol"` surface: returns `dl_regime` unchanged (LVTF, HVTF, HVR, LVR are valid state IDs).
- Any other surface: returns `None` — the DL artifact regime vocabulary has no meaning outside TrendVol.
- `None` or empty `dl_regime`: returns `None`.

```python
from mpml.behavioral.compat import resolve_behavioral_state_for_surface

# TrendVol: dl_regime IS the state_id
resolve_behavioral_state_for_surface("trend_vol", "LVTF")   # → "LVTF"
resolve_behavioral_state_for_surface("trend_vol", None)      # → None

# ReactiveJPY: dl_regime has no meaning
resolve_behavioral_state_for_surface("reactive_jpy", "LVTF") # → None
resolve_behavioral_state_for_surface("reactive_jpy", None)   # → None
```

Use this in `main()` to compute `state_id` before calling
`build_behavioral_surface_manifest_block`, avoiding a `KeyError` when a
non-TrendVol surface is selected.

### `build_behavioral_surface_manifest_block(surface_id, state_id=None)`

Build the `behavioral_surface` block for experiment manifests:

```python
from mpml.behavioral.compat import build_behavioral_surface_manifest_block

block = build_behavioral_surface_manifest_block("trend_vol", "LVTF")
# {
#     "surface_id": "trend_vol",
#     "surface_version": "1.0.0",
#     "display_name": "Trend / Volatility",
#     "behavioral_state": {
#         "state_id": "LVTF",
#         "display_name": "Low-Volatility Trend-Following",
#         "description": "...",
#     },
# }

# ReactiveJPY with no state — does not raise
block = build_behavioral_surface_manifest_block("reactive_jpy")
# {
#     "surface_id": "reactive_jpy",
#     "surface_version": "1.0.0",
#     "display_name": "Reactive JPY",
# }
```

---

## Phase B — Runtime Propagation

Phase B ensures that Behavioral Surface metadata flows continuously from
experiment configuration through to manifest emission without being
converted back into Trend/Vol-specific concepts.

### Canonical runtime identity

A run's Behavioral Surface identity is defined by:

```
surface_id           (e.g. "trend_vol", "reactive_jpy")
surface_version      (e.g. "1.0.0")
state_id             (e.g. "LVTF" for trend_vol; None for reactive_jpy)
```

`dl_regime` is treated as **deprecated compatibility metadata** — it is
preserved in the manifest for backward compatibility but is no longer the
primary routing key.

### Runtime flow in `main()`

```
CLI --behavioral-surface
         │
         ▼
_resolved_behavioral_surface_id  (validated against registry)
_resolved_behavioral_surface     (loaded surface object)
         │
         ▼
resolve_behavioral_state_for_surface(surface_id, dl_regime)
         │
         ├─ trend_vol  → state_id = dl_regime  (e.g. "LVTF")
         └─ other      → state_id = None
         │
         ▼
build_behavioral_surface_manifest_block(surface_id, state_id)
build_runtime_experiment_surface(..., behavioral_surface_id, behavioral_surface_version, behavioral_state_id)
         │
         ▼
run_manifest.json
  behavioral_surface.surface_id
  behavioral_surface.surface_version
  behavioral_surface.behavioral_state.state_id   (trend_vol only)
  experiment_surface.behavioral_surface
  experiment_surface.behavioral_surface_version
  experiment_surface.behavioral_state            (trend_vol only)
```

### DL artifact guard

When `--behavioral-surface reactive_jpy` is used with `DL_SIGNALS_ENABLED=true`,
the runtime emits a descriptive informational message and disables DL features:

```
[INFO] DL artifact loading is not yet implemented for behavioral surface
'reactive_jpy' (only 'trend_vol' is supported).  DL features will not be
attached for this run.  Implement a surface-specific artifact resolver to
enable DL support.
```

This replaces the previous `KeyError` crash.  The run proceeds to completion
using the ReactiveJPY surface without DL signals.

### Manifest fields (Phase B additions)

`experiment_surface` now includes:

| Field | Type | Description |
|---|---|---|
| `behavioral_surface` | `str` | Canonical surface ID (e.g. `"trend_vol"`) |
| `behavioral_surface_version` | `str` | Surface semantic version |
| `behavioral_state` | `str` \| absent | State ID for the run (TrendVol only) |

These fields are absent when the caller does not provide surface identity
(e.g. legacy manifest parsing).  The existing `msml_regime` field is
preserved as deprecated compatibility metadata.

### `msml_regime` deprecation

`msml_regime` in `experiment_surface` is a Trend × Volatility compatibility
field.  From Phase B onwards:

- For `trend_vol` runs: `msml_regime` is still populated from `dl_regime`.
- For other surfaces: `msml_regime` is set to `"unknown"`.

New runtime logic must not use `msml_regime` as a primary selector key.
Use `behavioral_surface` + `behavioral_state` instead.

---

## Experiment Manifests

Every run manifest contains a `behavioral_surface` block and Phase B adds
Behavioral Surface fields to `experiment_surface`:

```json
{
  "behavioral_surface": {
    "surface_id": "trend_vol",
    "surface_version": "1.0.0",
    "display_name": "Trend / Volatility",
    "behavioral_state": {
      "state_id": "LVTF",
      "display_name": "Low-Volatility Trend-Following",
      "description": "..."
    }
  },
  "experiment_surface": {
    "behavioral_surface": "trend_vol",
    "behavioral_surface_version": "1.0.0",
    "behavioral_state": "LVTF",
    "msml_regime": "LVTF",
    "..."
  }
}
```

For `reactive_jpy` runs, `behavioral_state` is absent (no DL regime mapping):

```json
{
  "behavioral_surface": {
    "surface_id": "reactive_jpy",
    "surface_version": "1.0.0",
    "display_name": "Reactive JPY"
  },
  "experiment_surface": {
    "behavioral_surface": "reactive_jpy",
    "behavioral_surface_version": "1.0.0",
    "msml_regime": "unknown",
    "..."
  }
}
```

---

## CLI

The `--behavioral-surface` argument selects the active Behavioral Surface:

```bash
python main.py --behavioral-surface trend_vol      # default — identical to omitting the flag
python main.py --behavioral-surface reactive_jpy   # select ReactiveJPY surface
```

The `BEHAVIORAL_SURFACE` environment variable provides the same control:

```bash
BEHAVIORAL_SURFACE=reactive_jpy python main.py
```

**Precedence:** `--behavioral-surface` > `BEHAVIORAL_SURFACE` env > `"trend_vol"`.

Existing runs that omit the flag will default to `trend_vol` and produce
identical results.

---

## Adding a New Behavioral Surface

1. Create `mpml/behavioral/<your_surface>.py` and implement `BehavioralSurface`.
2. Register it in `mpml/behavioral/registry.py::_build_default_registry()`.
3. Add tests in `tests/test_behavioral_surface.py`.

No other files need to be modified.  The registry pattern ensures future
surfaces become available throughout MPML simply by being registered.

---

## Out of Scope (Phase B)

The following are explicitly deferred to later roadmap phases:

- Strategy Registry
- Recommendation Engine
- MRML integration
- Strategy ranking redesign
- DL artifact support for non-TrendVol surfaces

---

## References

- [`docs/MPML_Architecture_Roadmap.md`](../MPML_Architecture_Roadmap.md)
- [`mpml/behavioral/base.py`](../../mpml/behavioral/base.py)
- [`mpml/behavioral/registry.py`](../../mpml/behavioral/registry.py)
- [`mpml/behavioral/trend_vol.py`](../../mpml/behavioral/trend_vol.py)
- [`mpml/behavioral/reactive_jpy.py`](../../mpml/behavioral/reactive_jpy.py)
- [`mpml/behavioral/compat.py`](../../mpml/behavioral/compat.py)
- [`src/experiment_surface_runtime.py`](../../src/experiment_surface_runtime.py)
- [`tests/test_behavioral_surface.py`](../../tests/test_behavioral_surface.py)
- [`tests/test_runtime_experiment_surface.py`](../../tests/test_runtime_experiment_surface.py)
