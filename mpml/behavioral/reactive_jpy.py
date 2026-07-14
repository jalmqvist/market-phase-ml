"""
mpml.behavioral.reactive_jpy — ReactiveJPY Behavioral Surface.

This module provides a second Behavioral Surface as a concrete example of
MPML's multi-surface capability.  Its purpose is to validate that MPML can
support surfaces beyond the original Trend × Volatility representation.

States
------
JPY_NON_EXTREME          — No consensus signal, non-extreme positioning
JPY_CONSENSUS_YOUNG      — Early-stage consensus forming
JPY_CONSENSUS_MATURING   — Consensus strengthening
JPY_CONSENSUS_MATURE     — Established, mature consensus

Important note
--------------
This surface does **not** yet drive walk-forward experiments.
State generation, calibration and validation remain the exclusive
responsibility of BSVE/MSML.  MPML consumes only the state labels and
metadata defined here.

See docs/MPML_Architecture_Roadmap.md Appendix A.
"""
from __future__ import annotations

from typing import Any

from mpml.behavioral.base import BehavioralState, BehavioralSurface


# ---------------------------------------------------------------------------
# State definitions
# ---------------------------------------------------------------------------

_STATES: list[BehavioralState] = [
    BehavioralState(
        state_id="JPY_NON_EXTREME",
        display_name="JPY Non-Extreme",
        surface_id="reactive_jpy",
        description=(
            "No meaningful consensus signal; JPY positioning is within "
            "normal bounds.  No directional behavioural bias."
        ),
        metadata={
            "consensus_stage": "none",
            "aliases": [],
        },
    ),
    BehavioralState(
        state_id="JPY_CONSENSUS_YOUNG",
        display_name="JPY Consensus — Young",
        surface_id="reactive_jpy",
        description=(
            "Early-stage JPY consensus forming.  Positioning has begun to "
            "move directionally but the signal is immature and may reverse."
        ),
        metadata={
            "consensus_stage": "young",
            "aliases": [],
        },
    ),
    BehavioralState(
        state_id="JPY_CONSENSUS_MATURING",
        display_name="JPY Consensus — Maturing",
        surface_id="reactive_jpy",
        description=(
            "Strengthening JPY consensus.  Directional bias is becoming "
            "established; the signal has persisted long enough to be "
            "considered reliable in historical analysis."
        ),
        metadata={
            "consensus_stage": "maturing",
            "aliases": [],
        },
    ),
    BehavioralState(
        state_id="JPY_CONSENSUS_MATURE",
        display_name="JPY Consensus — Mature",
        surface_id="reactive_jpy",
        description=(
            "Established, mature JPY consensus.  Directional signal is "
            "strong and persistent.  Historically associated with "
            "sustained behavioural trends in JPY pairs."
        ),
        metadata={
            "consensus_stage": "mature",
            "aliases": [],
        },
    ),
]

# Fast lookup by canonical state_id.
_STATE_MAP: dict[str, BehavioralState] = {s.state_id: s for s in _STATES}


# ---------------------------------------------------------------------------
# ReactiveJPYSurface
# ---------------------------------------------------------------------------

class ReactiveJPYSurface(BehavioralSurface):
    """
    Behavioral Surface for the Reactive-JPY market representation.

    This surface models JPY-driven behavioural regimes using consensus-based
    state labels produced by BSVE/MSML.  It is included here as the second
    registered Behavioral Surface to demonstrate that MPML can host multiple
    surfaces simultaneously.

    This surface does not yet drive walk-forward experiments; its role in
    Phase A is architectural validation only.
    """

    surface_id: str = "reactive_jpy"
    surface_version: str = "1.0.0"
    display_name: str = "Reactive JPY"

    def states(self) -> list[BehavioralState]:
        """Return the four Reactive-JPY states in canonical order."""
        return list(_STATES)

    def get_state(self, state_id: str) -> BehavioralState:
        """Return the :class:`BehavioralState` for *state_id*.

        Parameters
        ----------
        state_id : str
            One of: ``"JPY_NON_EXTREME"``, ``"JPY_CONSENSUS_YOUNG"``,
            ``"JPY_CONSENSUS_MATURING"``, ``"JPY_CONSENSUS_MATURE"``.

        Raises
        ------
        KeyError
            If *state_id* is not recognised.
        """
        if state_id not in _STATE_MAP:
            raise KeyError(
                f"ReactiveJPYSurface: unknown state_id {state_id!r}. "
                f"Valid IDs: {sorted(_STATE_MAP)}"
            )
        return _STATE_MAP[state_id]

    def metadata(self) -> dict[str, Any]:
        """Return metadata describing this surface."""
        return {
            "surface_id": self.surface_id,
            "surface_version": self.surface_version,
            "display_name": self.display_name,
            "description": (
                "Consensus-based Behavioral Surface for JPY-driven market "
                "regimes.  States are derived from BSVE/MSML analysis of "
                "speculative JPY positioning and classify the market into "
                "four consensus stages: non-extreme, young, maturing, mature."
            ),
            "state_ids": self.state_ids(),
            "aliases": {},
            "source": "BSVE/MSML",
            "note": (
                "This surface does not yet drive walk-forward experiments. "
                "It is registered to validate multi-surface capability."
            ),
        }
