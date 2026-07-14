"""
mpml.behavioral.registry — Behavioral Surface Registry.

The registry is the single entry point for loading Behavioral Surfaces.
Instead of branching on hardcoded regime names

    if regime == "LVTF":
        ...

downstream code should use

    from mpml.behavioral import registry

    surface = registry.load("trend_vol")
    state   = surface.get_state(state_id)

Benefits
--------
* No hardcoded surface names in selector logic.
* New surfaces can be added by calling :meth:`register` without modifying
  any existing code.
* Supports multiple simultaneous surfaces.
* BSVE/MSML and MPML can evolve independently through this stable interface.

See docs/MPML_Architecture_Roadmap.md §5 for the full design rationale.
"""
from __future__ import annotations

from typing import Any

from mpml.behavioral.base import BehavioralSurface, BehavioralState


class BehavioralSurfaceRegistry:
    """
    Registry that loads and exposes available Behavioral Surfaces.

    In Phase A the registry is backed by a simple dictionary of built-in
    surfaces.  No plugin discovery is required.  Future phases may extend
    this class to support external surface manifests or dynamic loading.

    Usage
    -----
    Use the pre-populated :data:`default_registry` instance rather than
    constructing a new registry unless you need an isolated environment
    (e.g. for testing).

        >>> from mpml.behavioral import registry
        >>> surface = registry.load("trend_vol")
        >>> state   = surface.get_state("LVTF")
        >>> print(state.display_name)
        Low-Volatility Trend-Following
    """

    def __init__(self) -> None:
        self._surfaces: dict[str, BehavioralSurface] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, surface: BehavioralSurface) -> None:
        """Add *surface* to the registry.

        Parameters
        ----------
        surface : BehavioralSurface
            The surface to register.  Its :attr:`~BehavioralSurface.surface_id`
            is used as the lookup key.  Registering a surface with a
            duplicate *surface_id* silently replaces the existing entry;
            this is intentional to allow test overrides.

        Raises
        ------
        TypeError
            If *surface* is not an instance of :class:`BehavioralSurface`.
        """
        if not isinstance(surface, BehavioralSurface):
            raise TypeError(
                f"Expected a BehavioralSurface instance; "
                f"got {type(surface).__name__!r}"
            )
        self._surfaces[surface.surface_id] = surface

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def load(self, surface_id: str) -> BehavioralSurface:
        """Return the :class:`BehavioralSurface` registered under *surface_id*.

        Parameters
        ----------
        surface_id : str
            Identifier used when the surface was registered (e.g.
            ``"trend_vol"``, ``"reactive_jpy"``).

        Returns
        -------
        BehavioralSurface

        Raises
        ------
        KeyError
            If *surface_id* is not registered.
        """
        if surface_id not in self._surfaces:
            raise KeyError(
                f"BehavioralSurfaceRegistry: unknown surface_id {surface_id!r}. "
                f"Available: {self.available()}"
            )
        return self._surfaces[surface_id]

    def get_state(self, surface_id: str, state_id: str) -> BehavioralState:
        """Convenience shortcut — load a surface and resolve a state in one call.

        Parameters
        ----------
        surface_id : str
            Behavioral Surface identifier.
        state_id : str
            State identifier (or alias) within that surface.

        Returns
        -------
        BehavioralState

        Raises
        ------
        KeyError
            If either *surface_id* or *state_id* is unknown.
        """
        return self.load(surface_id).get_state(state_id)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def available(self) -> list[str]:
        """Return a sorted list of registered surface IDs."""
        return sorted(self._surfaces)

    def surface_metadata(self, surface_id: str) -> dict[str, Any]:
        """Return the metadata dict for *surface_id*.

        Convenience wrapper around
        :meth:`BehavioralSurface.metadata`.
        """
        return self.load(surface_id).metadata()

    def all_metadata(self) -> dict[str, dict[str, Any]]:
        """Return metadata for every registered surface, keyed by surface_id."""
        return {sid: s.metadata() for sid, s in self._surfaces.items()}

    def __contains__(self, surface_id: str) -> bool:
        return surface_id in self._surfaces

    def __repr__(self) -> str:
        return (
            f"BehavioralSurfaceRegistry(surfaces={self.available()})"
        )


# ---------------------------------------------------------------------------
# Default registry — pre-populated with all built-in surfaces
# ---------------------------------------------------------------------------

def _build_default_registry() -> BehavioralSurfaceRegistry:
    """Construct and return the pre-populated default registry."""
    # Import here to avoid circular imports at module load time.
    from mpml.behavioral.trend_vol import TrendVolSurface
    from mpml.behavioral.reactive_jpy import ReactiveJPYSurface

    reg = BehavioralSurfaceRegistry()
    reg.register(TrendVolSurface())
    reg.register(ReactiveJPYSurface())
    return reg


#: Pre-populated global registry instance.
#: Imported as ``registry`` by :mod:`mpml.behavioral`.
default_registry: BehavioralSurfaceRegistry = _build_default_registry()
