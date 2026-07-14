"""
mpml.behavioral.base — Core abstractions for Behavioral Surfaces.

Behavioral Surfaces are produced by BSVE/MSML and consumed by MPML.
MPML never constructs or calibrates surfaces; it only consumes their
metadata through this stable contract.

See docs/MPML_Architecture_Roadmap.md §4–6 for the full design rationale.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# BehavioralState
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BehavioralState:
    """
    Immutable value object representing one state within a Behavioral Surface.

    Behavioral States are dynamical market objects rather than simple string
    labels.  Treating them as first-class objects allows future extensions
    (confidence, maturity, persistence, transition probability) to be added
    without changing downstream APIs.

    Attributes
    ----------
    state_id : str
        Canonical machine-readable identifier (e.g. ``"LVTF"``).
    display_name : str
        Human-readable label (e.g. ``"Low-Volatility Trend-Following"``).
    surface_id : str
        Identifier of the parent :class:`BehavioralSurface` that owns this
        state (e.g. ``"trend_vol"``).
    description : str
        One- or two-sentence description of the market regime this state
        represents.  Defaults to empty string.
    metadata : dict
        Extensible bag of additional attributes (aliases, tags, …).
        Downstream components should never rely on the exact keys present
        here; they may vary between surface implementations.
    """

    state_id: str
    display_name: str
    surface_id: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.state_id:
            raise ValueError("BehavioralState.state_id must not be empty")
        if not self.surface_id:
            raise ValueError("BehavioralState.surface_id must not be empty")

    def __str__(self) -> str:
        return self.state_id


# ---------------------------------------------------------------------------
# BehavioralSurface
# ---------------------------------------------------------------------------

class BehavioralSurface(ABC):
    """
    Abstract base class for all Behavioral Surface implementations.

    A Behavioral Surface is a named, versioned market representation that
    partitions observable market behaviour into a discrete set of
    :class:`BehavioralState` objects.

    MPML should never care *how* a surface was built — it depends only on
    the interface below.  All construction, calibration and validation
    logic remains the responsibility of BSVE/MSML.

    Subclass contract
    -----------------
    Implementations must set:

    * ``surface_id``      — stable machine-readable identifier (snake_case)
    * ``surface_version`` — semantic version string (``"1.0.0"``)
    * ``display_name``    — human-readable name

    and implement the abstract methods :meth:`states`, :meth:`get_state`
    and :meth:`metadata`.
    """

    #: Stable machine-readable surface identifier (e.g. ``"trend_vol"``).
    surface_id: str

    #: Semantic version of this surface definition (e.g. ``"1.0.0"``).
    surface_version: str

    #: Human-readable name (e.g. ``"Trend / Volatility"``).
    display_name: str

    @abstractmethod
    def states(self) -> list[BehavioralState]:
        """Return all :class:`BehavioralState` objects defined by this surface.

        Returns
        -------
        list[BehavioralState]
            States in a stable, reproducible order.
        """

    @abstractmethod
    def get_state(self, state_id: str) -> BehavioralState:
        """Return the :class:`BehavioralState` for *state_id*.

        Parameters
        ----------
        state_id : str
            Canonical state identifier.  Implementations may optionally
            accept compatibility aliases (e.g. ``"HVR"`` alongside the
            canonical ``"HVMR"``).

        Returns
        -------
        BehavioralState

        Raises
        ------
        KeyError
            If *state_id* (including any recognised aliases) is unknown.
        """

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return a serialisable metadata dictionary describing this surface.

        The returned dict should contain at minimum the fields mandated by
        the Behavioral Surface Contract (§4 of the roadmap):

        * ``surface_id``
        * ``surface_version``
        * ``display_name``
        * ``description``

        Additional fields are permitted and encouraged.

        Returns
        -------
        dict[str, Any]
        """

    # ------------------------------------------------------------------
    # Convenience helpers (non-abstract, shared by all implementations)
    # ------------------------------------------------------------------

    def state_ids(self) -> list[str]:
        """Return a list of canonical state IDs defined by this surface."""
        return [s.state_id for s in self.states()]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"surface_id={self.surface_id!r}, "
            f"surface_version={self.surface_version!r})"
        )
