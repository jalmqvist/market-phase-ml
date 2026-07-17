from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import pandas as pd

from mpml.behavioral.registry import default_registry
from src.dl_config import (
    infer_dl_regime_from_artifact_path,
    resolve_dl_prediction_artifact_path,
)
from src.dl_daily_features import compute_d1_features, empty_d1_df
from src.dl_surface_loader import load_dl_surface


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
            diagnostics=["Behavioral prediction runtime disabled."],
        )

    artifact_path = _resolve_artifact_path(explicit_artifact_path)
    if artifact_path is None:
        return BehavioralArtifactRuntime(
            enabled=False,
            artifact_path=None,
            surface_selector=selector,
            state_id=behavioral_state_id,
            h1_predictions=pd.DataFrame(),
            d1_predictions=empty_d1_df(),
            diagnostics=[
                "Behavioral prediction runtime disabled: no prediction artifact was resolved.",
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

    pairs = sorted(h1_predictions["pair"].dropna().astype(str).unique().tolist())
    diagnostics = [
        f"Behavioral Surface: {behavioral_surface_id}",
        f"Behavioral Surface version: {behavioral_surface_version}",
        "Searching Behavioral artifacts...",
        "Found compatible artifact.",
        f"model: {selector.get('model')}",
        f"state: {resolved_state_id}",
        f"rows: {len(h1_predictions)}",
        f"pairs: {' '.join(p.upper() for p in pairs) if pairs else 'none'}",
        f"prediction overlap: {_pair_overlap_pct(pairs):.1f}%",
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
            if behavioral_surface_id in default_registry:
                state_id = default_registry.get_state(behavioral_surface_id, state_id).state_id
            return state_id, None

    selector_state = str(selector.get("state_id", "")).strip()
    if selector_state:
        if behavioral_surface_id in default_registry:
            selector_state = default_registry.get_state(behavioral_surface_id, selector_state).state_id
        return selector_state, None

    if behavioral_surface_id == "trend_vol":
        regime = str(selector.get("dl_regime", "")).strip().upper()
        if not regime:
            regime = infer_dl_regime_from_artifact_path(artifact_path) or ""
        if regime:
            state_id = default_registry.get_state("trend_vol", regime).state_id
            return state_id, regime

    return None, None


def _pair_overlap_pct(pairs: list[str]) -> float:
    artifact_pairs = sorted(set(p.upper() for p in pairs))
    if not artifact_pairs:
        return 0.0
    raw_active = os.getenv("ACTIVE_PAIRS", "")
    requested_pairs = [p.strip().upper() for p in raw_active.split(",") if p.strip()]
    if not requested_pairs:
        return 100.0
    requested_set = set(requested_pairs)
    overlap = len(requested_set & set(artifact_pairs))
    return 100.0 * float(overlap) / float(len(requested_set))
