from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import pandas as pd

from mpml.behavioral.compat import dl_regime_to_state
from src.dl_config import (
    infer_dl_regime_from_artifact_path,
    resolve_dl_prediction_artifact_path,
)
from src.dl_daily_features import compute_d1_features, empty_d1_df
from src.dl_surface_loader import load_dl_surface

# Legacy default retained to preserve historical TrendVol runtime behaviour
# when no explicit state/regime is provided by configuration or artifact naming.
_LEGACY_DEFAULT_TREND_VOL_REGIME = "LVTF"


@dataclass(frozen=True)
class BehavioralArtifactRuntime:
    enabled: bool
    artifact_path: Path | None
    surface_selector: dict[str, Any]
    state_id: str | None
    h1_predictions: pd.DataFrame
    d1_predictions: pd.DataFrame
    diagnostics: list[str] = field(default_factory=list)


def resolve_behavioral_artifact_runtime(
    *,
    dl_runtime_enabled: bool,
    behavioral_surface_id: str,
    behavioral_surface_version: str,
    dl_surface: dict[str, Any],
    behavioral_state_id: str | None = None,
    explicit_artifact_path: str | None = None,
) -> BehavioralArtifactRuntime:
    selector = dict(dl_surface)
    selector["surface_id"] = behavioral_surface_id
    selector["surface_version"] = behavioral_surface_version

    if not dl_runtime_enabled:
        return BehavioralArtifactRuntime(
            enabled=False,
            artifact_path=None,
            surface_selector=selector,
            state_id=behavioral_state_id,
            h1_predictions=pd.DataFrame(),
            d1_predictions=empty_d1_df(),
            diagnostics=[
                "Behavioral prediction runtime disabled.",
                f"  surface_id : {behavioral_surface_id}",
                f"  state_id   : {behavioral_state_id or 'not resolved (runtime disabled)'}",
            ],
        )

    artifact_path = _resolve_artifact_path(explicit_artifact_path)
    if artifact_path is None:
        _discovery_mode = "explicit path" if explicit_artifact_path else "config discovery"
        return BehavioralArtifactRuntime(
            enabled=False,
            artifact_path=None,
            surface_selector=selector,
            state_id=behavioral_state_id,
            h1_predictions=pd.DataFrame(),
            d1_predictions=empty_d1_df(),
            diagnostics=[
                "Behavioral prediction runtime disabled: no prediction artifact was resolved.",
                f"  surface_id      : {behavioral_surface_id}",
                f"  discovery       : {_discovery_mode}",
            ],
        )

    resolved_state_id, resolved_regime = _resolve_state_id(
        behavioral_surface_id=behavioral_surface_id,
        explicit_state_id=behavioral_state_id,
        selector=selector,
        artifact_path=artifact_path,
    )
    if resolved_state_id is None:
        raise ValueError(
            "Behavioral artifact resolution failed: missing state_id for runtime selection. "
            "Provide canonical state_id (or legacy dl_regime for TrendVol compatibility)."
        )

    selector["state_id"] = resolved_state_id
    selector["target_horizon"] = int(selector["target_horizon"])
    if resolved_regime is not None:
        selector["dl_regime"] = resolved_regime

    h1_predictions = load_dl_surface(artifact_path, selector, strict=True)
    if h1_predictions.empty:
        raise ValueError(
            "Behavioral artifact resolution failed: no compatible predictions found "
            f"for selector={selector} in artifact={artifact_path}."
        )
    d1_predictions = compute_d1_features(h1_predictions)

    pairs = sorted(h1_predictions["pair"].dropna().astype(str).unique())
    _discovery_mode = "explicit path" if explicit_artifact_path else "config discovery"
    _compat_note = (
        f"legacy TrendVol dl_regime={resolved_regime!r}"
        if resolved_regime is not None
        else "canonical state_id"
    )
    diagnostics = [
        "--- Behavioral Prediction Artifact ---",
        f"  surface_id      : {behavioral_surface_id}",
        f"  surface_version : {behavioral_surface_version}",
        f"  state_id        : {resolved_state_id}",
        f"  model           : {selector.get('model')}",
        f"  target_horizon  : {selector.get('target_horizon')}",
        f"  feature_set     : {selector.get('feature_set')}",
        "--- Artifact Discovery ---",
        f"  path            : {artifact_path}",
        f"  discovery       : {_discovery_mode}",
        f"  state_resolution: {_compat_note}",
        "--- Prediction Coverage ---",
        f"  h1_rows         : {len(h1_predictions)}",
        f"  d1_rows         : {len(d1_predictions)}",
        f"  pairs           : {' '.join(p.upper() for p in pairs) if pairs else 'none'}",
        f"  pair_overlap    : {_pair_overlap_pct(pairs):.1f}%",
        "Behavioral prediction runtime ready.",
    ]

    return BehavioralArtifactRuntime(
        enabled=True,
        artifact_path=artifact_path,
        surface_selector=selector,
        state_id=resolved_state_id,
        h1_predictions=h1_predictions,
        d1_predictions=d1_predictions,
        diagnostics=diagnostics,
    )


def _resolve_artifact_path(explicit_artifact_path: str | None) -> Path | None:
    if explicit_artifact_path:
        explicit = Path(explicit_artifact_path).expanduser()
        return explicit if explicit.exists() else None
    resolved = resolve_dl_prediction_artifact_path()
    return Path(resolved) if resolved is not None else None


def _resolve_state_id(
    *,
    behavioral_surface_id: str,
    explicit_state_id: str | None,
    selector: dict[str, Any],
    artifact_path: Path,
) -> tuple[str | None, str | None]:
    if explicit_state_id:
        state_id = str(explicit_state_id).strip()
        if state_id:
            return state_id, None

    selector_state = str(selector.get("state_id", "")).strip()
    if selector_state:
        return selector_state, None

    if behavioral_surface_id == "trend_vol":
        regime = str(selector.get("dl_regime", "")).strip().upper()
        if not regime:
            regime = infer_dl_regime_from_artifact_path(artifact_path) or ""
        if not regime:
            regime = _LEGACY_DEFAULT_TREND_VOL_REGIME
        if regime:
            try:
                state_id = dl_regime_to_state(regime).state_id
            except KeyError as exc:
                raise ValueError(
                    "Legacy TrendVol compatibility adapter failed while resolving "
                    f"dl_regime={regime!r} from artifact={artifact_path}. "
                    "Ensure dl_regime is a valid TrendVol state token or provide "
                    "an explicit BEHAVIORAL_STATE_ID/state_id."
                ) from exc
            return state_id, regime

    return None, None


def _pair_overlap_pct(pairs: list[str]) -> float:
    artifact_pairs = sorted(set(p.upper() for p in pairs))
    if not artifact_pairs:
        return 0.0
    raw_active = os.getenv("ACTIVE_PAIRS", "")
    requested_pairs = []
    for raw_pair in raw_active.split(","):
        normalized = raw_pair.strip().upper()
        if normalized:
            requested_pairs.append(normalized)
    if not requested_pairs:
        return 100.0
    requested_set = set(requested_pairs)
    overlap = len(requested_set & set(artifact_pairs))
    return 100.0 * float(overlap) / float(len(requested_set))
