"""
mpml.behavioral.persistent — Persistent Behavioral Surface.

This module defines the runtime representation of the Persistent behavioral
surface produced by BSVE/MSML.

The Persistent surface represents a Commitment Level × Commitment Trajectory
state space with nine canonical states. MPML consumes these state identifiers
and metadata; it does not construct, calibrate, or validate the surface.

Surface definition
------------------
surface_id:
    persistent

surface_version:
    0.1.0

Behavioral dimensions:
    Commitment Level     — LOW / MID / HIGH
    Commitment Trajectory — LOW / MID / HIGH

Canonical state IDs:
    PERSISTENT_LL ... PERSISTENT_HH

The nine states are a representation of the Persistent behavioral surface,
not independently validated predictive mechanisms.

See the Persistent Behavioral Surface roadmap for the scientific definition
and calibration protocol.
"""

from __future__ import annotations

from typing import Any

from mpml.behavioral.base import BehavioralState, BehavioralSurface


# ---------------------------------------------------------------------------
# State definitions
# ---------------------------------------------------------------------------

_STATES: list[BehavioralState] = [
    BehavioralState(
        state_id="PERSISTENT_LL",
        display_name="Persistent — Low / Low",
        surface_id="persistent",
        description=(
            "Low commitment level and low commitment trajectory."
        ),
    ),
    BehavioralState(
        state_id="PERSISTENT_LM",
        display_name="Persistent — Low / Mid",
        surface_id="persistent",
        description=(
            "Low commitment level and medium commitment trajectory."
        ),
    ),
    BehavioralState(
        state_id="PERSISTENT_LH",
        display_name="Persistent — Low / High",
        surface_id="persistent",
        description=(
            "Low commitment level and high commitment trajectory."
        ),
    ),
    BehavioralState(
        state_id="PERSISTENT_ML",
        display_name="Persistent — Mid / Low",
        surface_id="persistent",
        description=(
            "Medium commitment level and low commitment trajectory."
        ),
    ),
    BehavioralState(
        state_id="PERSISTENT_MM",
        display_name="Persistent — Mid / Mid",
        surface_id="persistent",
        description=(
            "Medium commitment level and medium commitment trajectory."
        ),
    ),
    BehavioralState(
        state_id="PERSISTENT_MH",
        display_name="Persistent — Mid / High",
        surface_id="persistent",
        description=(
            "Medium commitment level and high commitment trajectory."
        ),
    ),
    BehavioralState(
        state_id="PERSISTENT_HL",
        display_name="Persistent — High / Low",
        surface_id="persistent",
        description=(
            "High commitment level and low commitment trajectory."
        ),
    ),
    BehavioralState(
        state_id="PERSISTENT_HM",
        display_name="Persistent — High / Mid",
        surface_id="persistent",
        description=(
            "High commitment level and medium commitment trajectory."
        ),
    ),
    BehavioralState(
        state_id="PERSISTENT_HH",
        display_name="Persistent — High / High",
        surface_id="persistent",
        description=(
            "High commitment level and high commitment trajectory."
        ),
    ),
]


# Fast lookup by canonical state_id.
_STATE_MAP: dict[str, BehavioralState] = {
    state.state_id: state for state in _STATES
}


# ---------------------------------------------------------------------------
# PersistentSurface
# ---------------------------------------------------------------------------

class PersistentSurface(BehavioralSurface):
    """
    Behavioral Surface for the Persistent commitment lifecycle representation.

    The surface contains nine canonical states formed by the combination of
    Commitment Level and Commitment Trajectory. State generation and
    calibration are the responsibility of BSVE/MSML; MPML only consumes the
    resulting state identifiers and metadata.
    """

    surface_id: str = "persistent"
    surface_version: str = "0.1.0"
    display_name: str = "Persistent Commitment Lifecycle"

    def states(self) -> list[BehavioralState]:
        """Return the nine Persistent states in canonical order."""
        return list(_STATES)

    def get_state(self, state_id: str) -> BehavioralState:
        """Return the Persistent state for *state_id*.

        Parameters
        ----------
        state_id : str
            One of the nine canonical ``PERSISTENT_*`` identifiers.

        Raises
        ------
        KeyError
            If *state_id* is not recognised.
        """
        if state_id not in _STATE_MAP:
            raise KeyError(
                f"PersistentSurface: unknown state_id {state_id!r}. "
                f"Valid IDs: {sorted(_STATE_MAP)}"
            )

        return _STATE_MAP[state_id]

    def metadata(self) -> dict[str, Any]:
        """Return metadata describing the Persistent surface."""
        return {
            "surface_id": self.surface_id,
            "surface_version": self.surface_version,
            "display_name": self.display_name,
            "description": (
                "Behavioral Surface representing the Persistent commitment "
                "lifecycle through a Commitment Level × Commitment Trajectory "
                "state space with nine canonical states."
            ),
            "state_ids": self.state_ids(),
            "aliases": {},
            "source": "BSVE/MSML",
        }
