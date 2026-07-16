"""
mpml.behavioral — Behavioral Surface abstractions.

This package implements Phase A of the MPML Architecture Roadmap
(see docs/MPML_Architecture_Roadmap.md).

Instead of hardcoded regime names (LVTF, HVTF, LVR, HVR) scattered
throughout the codebase, downstream components should use:

    from mpml.behavioral import registry, BehavioralSurface, BehavioralState

    surface = registry.load("trend_vol")
    state   = surface.get_state("LVTF")

Public API
----------
BehavioralSurface
    Abstract base class for every Behavioral Surface implementation.
BehavioralState
    Immutable value object representing one state within a surface.
BehavioralSurfaceRegistry
    Registry that loads and exposes available surfaces.
registry
    The default, pre-populated global registry instance containing
    TrendVolSurface and ReactiveJPYSurface.
"""

from mpml.behavioral.base import BehavioralSurface, BehavioralState
from mpml.behavioral.registry import BehavioralSurfaceRegistry
from mpml.behavioral import registry as _registry_module
from mpml.behavioral.compat import (
    dl_regime_to_state,
    phase_label_to_state,
    build_behavioral_surface_manifest_block,
    resolve_behavioral_state_for_surface,
)

# The default global registry — pre-populated with built-in surfaces.
registry: BehavioralSurfaceRegistry = _registry_module.default_registry

__all__ = [
    "BehavioralSurface",
    "BehavioralState",
    "BehavioralSurfaceRegistry",
    "registry",
    "dl_regime_to_state",
    "phase_label_to_state",
    "build_behavioral_surface_manifest_block",
    "resolve_behavioral_state_for_surface",
]
