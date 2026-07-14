# Behavioral Surface Registry

**Phase A of the MPML Architecture Roadmap**

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
```

---

## Experiment Manifests

Every run manifest now contains a `behavioral_surface` block:

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
  }
}
```

The `behavioral_state` key is present when a `dl_regime` is configured.

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

## Out of Scope (Phase A)

The following are explicitly deferred to later roadmap phases:

- Strategy Registry (Phase B)
- Recommendation Engine (Phase C)
- Rich experiment metadata / provenance (Phase D)
- MRML integration (Phase E)
- Plugin / external surface loading

---

## References

- [`docs/MPML_Architecture_Roadmap.md`](../MPML_Architecture_Roadmap.md)
- [`mpml/behavioral/base.py`](../../mpml/behavioral/base.py)
- [`mpml/behavioral/registry.py`](../../mpml/behavioral/registry.py)
- [`mpml/behavioral/trend_vol.py`](../../mpml/behavioral/trend_vol.py)
- [`mpml/behavioral/reactive_jpy.py`](../../mpml/behavioral/reactive_jpy.py)
- [`mpml/behavioral/compat.py`](../../mpml/behavioral/compat.py)
- [`tests/test_behavioral_surface.py`](../../tests/test_behavioral_surface.py)
