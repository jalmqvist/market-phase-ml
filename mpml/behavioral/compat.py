"""
mpml.behavioral.compat — Compatibility helpers for the Behavioral Surface migration.

These helpers allow existing MPML code that uses ``dl_regime`` strings
(``"LVTF"``, ``"HVTF"``, ``"HVR"``, ``"LVR"``), or the internal
``MarketPhaseDetector`` phase labels (``"HV_Trend"``, ``"LV_Trend"``,
``"HV_Ranging"``, ``"LV_Ranging"``), to resolve
:class:`~mpml.behavioral.base.BehavioralState` objects through the registry
without requiring callers to be rewritten.

The functions here are intentionally thin wrappers.  They accept the existing
vocabulary and delegate to the registry, acting as a **seam** between old and
new code.

Migration guidance
------------------
Existing callers should remain unchanged in the short term.  Over time,
direct uses of ``dl_regime`` strings can be replaced with

    surface = registry.load(surface_id)
    state   = surface.get_state(state_id)

as described in docs/MPML_Architecture_Roadmap.md §5.
"""
from __future__ import annotations

from typing import Any

from mpml.behavioral.base import BehavioralState
from mpml.behavioral.registry import default_registry


# ---------------------------------------------------------------------------
# dl_regime → BehavioralState
# ---------------------------------------------------------------------------

#: MSML dl_regime values recognised by the TrendVol surface.
#: HVR and LVR are canonical in MSML; HVMR / LVMR are MPML-internal aliases.
_DL_REGIME_SURFACE: str = "trend_vol"


def dl_regime_to_state(dl_regime: str) -> BehavioralState:
    """Resolve an MSML *dl_regime* string to a :class:`BehavioralState`.

    This is a convenience wrapper for legacy code that holds ``dl_regime``
    values (e.g. ``"LVTF"``, ``"HVTF"``, ``"HVR"``, ``"LVR"``, ``"HVMR"``,
    ``"LVMR"``) and needs a :class:`BehavioralState` object.

    Parameters
    ----------
    dl_regime : str
        A value from MSML's regime vocabulary, or one of the MPML-internal
        aliases (``"HVMR"``, ``"LVMR"``).

    Returns
    -------
    BehavioralState
        The corresponding state within :class:`~mpml.behavioral.trend_vol.TrendVolSurface`.

    Raises
    ------
    KeyError
        If *dl_regime* is not recognised by the TrendVol surface.

    Examples
    --------
    >>> from mpml.behavioral.compat import dl_regime_to_state
    >>> state = dl_regime_to_state("LVTF")
    >>> state.display_name
    'Low-Volatility Trend-Following'
    >>> dl_regime_to_state("HVMR").state_id   # alias accepted
    'HVR'
    """
    return default_registry.get_state(_DL_REGIME_SURFACE, dl_regime)


# ---------------------------------------------------------------------------
# phases.py phase label → BehavioralState
# ---------------------------------------------------------------------------

# Maps src/phases.py internal labels to TrendVol state_ids.
_PHASE_LABEL_TO_STATE_ID: dict[str, str] = {
    "HV_Trend":   "HVTF",
    "LV_Trend":   "LVTF",
    "HV_Ranging": "HVR",
    "LV_Ranging": "LVR",
}


def phase_label_to_state(phase_label: str) -> BehavioralState:
    """Resolve a :mod:`src.phases` phase label to a :class:`BehavioralState`.

    :mod:`src.phases` uses labels such as ``"HV_Trend"`` internally.
    This helper maps those labels to the corresponding TrendVol
    :class:`BehavioralState` object.

    Parameters
    ----------
    phase_label : str
        One of ``"HV_Trend"``, ``"LV_Trend"``, ``"HV_Ranging"``,
        ``"LV_Ranging"`` (as produced by
        :meth:`~src.phases.MarketPhaseDetector.detect_phases`).

    Returns
    -------
    BehavioralState

    Raises
    ------
    KeyError
        If *phase_label* is not a recognised label.

    Examples
    --------
    >>> from mpml.behavioral.compat import phase_label_to_state
    >>> state = phase_label_to_state("HV_Trend")
    >>> state.state_id
    'HVTF'
    """
    state_id = _PHASE_LABEL_TO_STATE_ID.get(phase_label)
    if state_id is None:
        raise KeyError(
            f"phase_label_to_state: unknown phase label {phase_label!r}. "
            f"Recognised labels: {sorted(_PHASE_LABEL_TO_STATE_ID)}"
        )
    return default_registry.get_state(_DL_REGIME_SURFACE, state_id)


# ---------------------------------------------------------------------------
# Runtime state resolution
# ---------------------------------------------------------------------------

#: Surface IDs whose states are addressable by DL artifact dl_regime values.
#: Only the TrendVol surface maps directly from MSML dl_regime tokens.
_DL_REGIME_CAPABLE_SURFACES: frozenset[str] = frozenset({_DL_REGIME_SURFACE})


def resolve_behavioral_state_for_surface(
    surface_id: str,
    dl_regime: str | None,
) -> str | None:
    """Resolve the canonical behavioral state_id for a runtime surface.

    This is the Phase B bridge between ``dl_regime`` (a Trend/Vol artifact
    concept) and :class:`~mpml.behavioral.base.BehavioralState`.

    For surfaces that are based on the Trend × Volatility representation
    (``"trend_vol"``), the *dl_regime* value is a valid state identifier and
    is returned directly.

    For all other surfaces, the DL artifact's ``dl_regime`` has no meaning;
    ``None`` is returned so that callers can omit the state from the manifest
    rather than raising a ``KeyError``.

    Parameters
    ----------
    surface_id : str
        Registered Behavioral Surface identifier (e.g. ``"trend_vol"``,
        ``"reactive_jpy"``).
    dl_regime : str | None
        MSML dl_regime value from the runtime DL surface configuration, or
        ``None`` when no artifact is configured.

    Returns
    -------
    str | None
        The canonical ``state_id`` for this surface, or ``None`` when no
        state mapping is available (non-TrendVol surfaces, or empty regime).

    Examples
    --------
    >>> resolve_behavioral_state_for_surface("trend_vol", "LVTF")
    'LVTF'
    >>> resolve_behavioral_state_for_surface("reactive_jpy", "LVTF") is None
    True
    >>> resolve_behavioral_state_for_surface("trend_vol", None) is None
    True
    """
    if surface_id not in _DL_REGIME_CAPABLE_SURFACES:
        return None
    if not dl_regime:
        return None
    return dl_regime


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def build_behavioral_surface_manifest_block(
    surface_id: str,
    state_id: str | None = None,
) -> dict[str, Any]:
    """Build a serialisable manifest block for a Behavioral Surface.

    This helper produces the ``behavioral_surface`` key for experiment
    manifests (see docs/MPML_Architecture_Roadmap.md §14).

    Parameters
    ----------
    surface_id : str
        Identifier of the Behavioral Surface to describe
        (e.g. ``"trend_vol"``).
    state_id : str | None
        Optional state identifier to include in the manifest.
        Pass a ``dl_regime`` or phase label; aliases are resolved
        to canonical IDs automatically.  If *None*, the
        ``behavioral_state`` key is omitted from the block.

    Returns
    -------
    dict[str, Any]
        A dict suitable for embedding directly in a run manifest under
        the key ``"behavioral_surface"``.

    Examples
    --------
    >>> block = build_behavioral_surface_manifest_block("trend_vol", "LVTF")
    >>> block["surface_id"]
    'trend_vol'
    >>> block["behavioral_state"]["state_id"]
    'LVTF'
    """
    surface = default_registry.load(surface_id)
    meta = surface.metadata()
    block: dict[str, Any] = {
        "surface_id": meta["surface_id"],
        "surface_version": meta["surface_version"],
        "display_name": meta["display_name"],
    }
    if state_id is not None:
        state = surface.get_state(state_id)
        block["behavioral_state"] = {
            "state_id": state.state_id,
            "display_name": state.display_name,
            "description": state.description,
        }
    return block
