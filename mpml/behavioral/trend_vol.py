"""
mpml.behavioral.trend_vol — TrendVol Behavioral Surface.

This module implements the existing Trend × Volatility market representation
as the first registered Behavioral Surface.

States
------
LVTF  — Low-Volatility Trend-Following
HVTF  — High-Volatility Trend-Following
LVR   — Low-Volatility Ranging  (LVMR in older MPML taxonomy; aliases provided)
HVR   — High-Volatility Ranging (HVMR in older MPML taxonomy; aliases provided)

Backward-compatibility note
----------------------------
Older MPML code refers to these states as LVTF / HVTF / LVMR / HVMR.
MSML uses LVTF / HVTF / LVR / HVR as the canonical dl_regime vocabulary.
This surface treats LVR and HVR as the canonical state_ids (matching MSML)
and exposes LVMR / HVMR as accepted aliases via :meth:`get_state`.

See docs/MPML_Architecture_Roadmap.md Appendix A and the note on
"State Naming" in §16 for the full naming rationale.
"""
from __future__ import annotations

from typing import Any

from mpml.behavioral.base import BehavioralState, BehavioralSurface


# ---------------------------------------------------------------------------
# State definitions
# ---------------------------------------------------------------------------

_STATES: list[BehavioralState] = [
    BehavioralState(
        state_id="LVTF",
        display_name="Low-Volatility Trend-Following",
        surface_id="trend_vol",
        description=(
            "Low volatility, strongly trending market. "
            "Trend-following strategies typically perform best."
        ),
        metadata={
            "volatility": "low",
            "trend": "trending",
        },
    ),
    BehavioralState(
        state_id="HVTF",
        display_name="High-Volatility Trend-Following",
        surface_id="trend_vol",
        description=(
            "High volatility, strongly trending market. "
            "Trend-following strategies remain applicable but require "
            "wider stop-losses to accommodate elevated volatility."
        ),
        metadata={
            "volatility": "high",
            "trend": "trending",
        },
    ),
    BehavioralState(
        state_id="LVR",
        display_name="Low-Volatility Ranging",
        surface_id="trend_vol",
        description=(
            "Low volatility, directionless ranging market. "
            "Mean-reversion strategies typically perform best."
        ),
        metadata={
            "volatility": "low",
            "trend": "ranging",
            # LVMR (Low-Volatility Mean-Reversion) is the legacy MPML name
            # for this state; accepted as an alias by get_state().
            "aliases": ["LVMR"],
        },
    ),
    BehavioralState(
        state_id="HVR",
        display_name="High-Volatility Ranging",
        surface_id="trend_vol",
        description=(
            "High volatility, directionless ranging market. "
            "Mean-reversion approaches are applicable but position sizing "
            "must account for elevated volatility."
        ),
        metadata={
            "volatility": "high",
            "trend": "ranging",
            # HVMR (High-Volatility Mean-Reversion) is the legacy MPML name
            # for this state; accepted as an alias by get_state().
            "aliases": ["HVMR"],
        },
    ),
]

# Alias → canonical state_id map.
# Populated automatically from each state's metadata["aliases"].
_ALIAS_MAP: dict[str, str] = {}
for _state in _STATES:
    for _alias in _state.metadata.get("aliases", []):
        _ALIAS_MAP[_alias] = _state.state_id

# Fast lookup by canonical state_id.
_STATE_MAP: dict[str, BehavioralState] = {s.state_id: s for s in _STATES}


# ---------------------------------------------------------------------------
# TrendVolSurface
# ---------------------------------------------------------------------------

class TrendVolSurface(BehavioralSurface):
    """
    Behavioral Surface for the Trend × Volatility market representation.

    This is the first and default Behavioral Surface in MPML.  It wraps
    the four regime states that were previously hardcoded throughout the
    codebase (LVTF, HVTF, LVR, HVR) into the :class:`BehavioralSurface`
    contract so that downstream code can treat them uniformly alongside
    future surfaces such as ReactiveJPY.

    Compatibility aliases
    ---------------------
    ``get_state("HVMR")`` and ``get_state("LVMR")`` resolve to the canonical
    ``HVR`` and ``LVR`` states respectively, preserving backward compatibility
    with older MPML analysis code that used the HVMR/LVMR naming convention.
    """

    surface_id: str = "trend_vol"
    surface_version: str = "1.0.0"
    display_name: str = "Trend / Volatility"

    def states(self) -> list[BehavioralState]:
        """Return the four Trend × Volatility states in canonical order."""
        return list(_STATES)

    def get_state(self, state_id: str) -> BehavioralState:
        """Return the :class:`BehavioralState` for *state_id* or a known alias.

        Parameters
        ----------
        state_id : str
            Canonical ID (``"LVTF"``, ``"HVTF"``, ``"LVR"``, ``"HVR"``) or
            a recognised alias (``"LVMR"``, ``"HVMR"``).

        Raises
        ------
        KeyError
            If *state_id* is not recognised.
        """
        canonical = _ALIAS_MAP.get(state_id, state_id)
        if canonical not in _STATE_MAP:
            raise KeyError(
                f"TrendVolSurface: unknown state_id {state_id!r}. "
                f"Valid IDs: {sorted(_STATE_MAP)} "
                f"Aliases: {sorted(_ALIAS_MAP)}"
            )
        return _STATE_MAP[canonical]

    def metadata(self) -> dict[str, Any]:
        """Return metadata describing this surface."""
        return {
            "surface_id": self.surface_id,
            "surface_version": self.surface_version,
            "display_name": self.display_name,
            "description": (
                "Two-dimensional market representation that classifies bars "
                "by volatility level (high / low, split by rolling median ATR%) "
                "and trend strength (trending / ranging, determined by ADX threshold). "
                "Produces four mutually exclusive states: LVTF, HVTF, LVR, HVR."
            ),
            "state_ids": self.state_ids(),
            "aliases": dict(_ALIAS_MAP),
            "source": "BSVE/MSML",
        }
