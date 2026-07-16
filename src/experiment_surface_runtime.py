from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from experiment_semantics import EXPERIMENT_SURFACE_VERSION

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_PERSISTENT_PAIRS = frozenset({"EURUSD", "GBPUSD", "NZDUSD", "EURGBP", "EURAUD"})
_REACTIVE_PAIRS = frozenset({"USDJPY", "EURJPY", "GBPJPY", "EURCHF", "USDCHF"})
_FEATURE_TO_SENTIMENT_MAP = {
    "price_trend": "sentiment",
    "trend_vol_only": "no_sentiment",
}
# (artifact_pattern, training_pair_family, evaluation_pair_family)
_TRANSFER_ARTIFACT_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("persistent_to_reactive", "persistent", "reactive"),
    ("persistent-to-reactive", "persistent", "reactive"),
    ("reactive_to_persistent", "reactive", "persistent"),
    ("reactive-to-persistent", "reactive", "persistent"),
)


def _decode_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
    return None


def _decode_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.isdigit():
            return int(normalized)
    return None


def _decode_str(value: Any) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    if value is None:
        return None
    return str(value)


def _flatten_metadata(raw: Any, *, prefix: str = "") -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    flattened: dict[str, Any] = {}
    for key, value in raw.items():
        key_str = str(key)
        full_key = f"{prefix}.{key_str}" if prefix else key_str
        flattened[full_key] = value
        flattened[key_str] = value
        if isinstance(value, dict):
            flattened.update(_flatten_metadata(value, prefix=full_key))
    return flattened


def _read_parquet_kv_metadata(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except Exception:
        return {}
    try:
        raw_metadata = pq.read_metadata(path).metadata
    except Exception:
        return {}
    if not raw_metadata:
        return {}
    decoded: dict[str, Any] = {}
    for key, value in raw_metadata.items():
        try:
            decoded[key.decode("utf-8")] = value.decode("utf-8")
        except Exception:
            continue
    return decoded


def _read_sidecar_metadata(path: Path) -> dict[str, Any]:
    sidecar_candidates = [
        path.with_suffix(".manifest.json"),
        path.with_suffix(path.suffix + ".manifest.json"),
    ]
    for candidate in sidecar_candidates:
        if not candidate.exists():
            continue
        try:
            parsed = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        return _flatten_metadata(parsed)
    return {}


def _lookup(raw: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in raw and raw[key] not in (None, ""):
            return raw[key]
    return None


def _env_bool(name: str) -> bool | None:
    return _decode_bool(os.getenv(name))


def _env_str(name: str) -> str | None:
    return _decode_str(os.getenv(name))


def _normalize_sentiment_surface(value: Any) -> str | None:
    """Normalize legacy/canonical sentiment surface values; return None when invalid."""
    if isinstance(value, bool):
        return "sentiment" if value else "no_sentiment"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"sentiment", "no_sentiment", "none"}:
            return normalized
    return None


def _infer_pair_family_from_text(value: Any) -> str | None:
    text = _decode_str(value)
    if not text:
        return None
    normalized = text.lower()
    has_persistent = "persistent" in normalized
    has_reactive = "reactive" in normalized
    if has_persistent and not has_reactive:
        return "persistent"
    if has_reactive and not has_persistent:
        return "reactive"
    return None


def _infer_pair_families_from_artifact_text(value: Any) -> tuple[str | None, str | None]:
    """
    Infer training/evaluation pair families from artifact-like text.

    Supported transfer patterns:
      persistent_to_reactive_*  -> (persistent, reactive)
      reactive_to_persistent_*  -> (reactive, persistent)

    For non-transfer names, preserves legacy behavior by inferring only a
    single training family (persistent/reactive) and leaving evaluation unset.
    """
    text = _decode_str(value)
    if not text:
        return None, None
    normalized = text.lower()
    for pattern, training_family, evaluation_family in _TRANSFER_ARTIFACT_PATTERNS:
        if pattern in normalized:
            return training_family, evaluation_family
    return _infer_pair_family_from_text(text), None


def _infer_eval_family_from_active_pairs() -> str | None:
    """Infer evaluation family only when ACTIVE_PAIRS is a pure persistent/reactive subset."""
    raw = os.getenv("ACTIVE_PAIRS", "")
    if not raw.strip():
        return None
    active_pairs = {p.strip().upper() for p in raw.split(",") if p.strip()}
    if not active_pairs:
        return None
    if active_pairs.issubset(_PERSISTENT_PAIRS):
        return "persistent"
    if active_pairs.issubset(_REACTIVE_PAIRS):
        return "reactive"
    return None


def _resolve_imputation_awareness(experiment_factors: Mapping[str, Any]) -> str:
    awareness = _decode_bool(experiment_factors.get("missing_indicators_enabled"))
    if awareness is True:
        return "aware"
    return "blind"


def build_runtime_experiment_surface(
    *,
    dl_runtime_enabled: bool,
    dl_surface: dict[str, Any],
    dl_artifact_path: Path | None,
    experiment_factors: dict[str, Any],
    artifact_metadata: Mapping[str, Any] | None = None,
    behavioral_surface_id: str | None = None,
    behavioral_surface_version: str | None = None,
    behavioral_state_id: str | None = None,
) -> dict[str, Any]:
    """
    Build canonical runtime ``experiment_surface`` for manifest emission.

    Field precedence:
      1) Explicit artifact metadata (caller-provided, parquet kv metadata, sidecar)
      2) Canonical runtime configuration (surface dict, experiment factors, env)

    Parameters
    ----------
    behavioral_surface_id : str | None
        Canonical Behavioral Surface identifier (e.g. ``"trend_vol"``,
        ``"reactive_jpy"``).  When provided, emitted in the returned dict
        as ``"behavioral_surface"``.
    behavioral_surface_version : str | None
        Semantic version string for the Behavioral Surface (e.g. ``"1.0.0"``).
        When provided, emitted as ``"behavioral_surface_version"``.
    behavioral_state_id : str | None
        Canonical state identifier within the Behavioral Surface for this run
        (e.g. ``"LVTF"`` for TrendVol).  When provided, emitted as
        ``"behavioral_state"``.  ``None`` for surfaces that do not map from
        DL artifact regimes (e.g. ``"reactive_jpy"``).
    """
    merged_metadata: dict[str, Any] = {}
    merged_metadata.update(_flatten_metadata(dict(artifact_metadata or {})))
    if dl_artifact_path is not None:
        merged_metadata.update(_flatten_metadata(_read_parquet_kv_metadata(dl_artifact_path)))
        merged_metadata.update(_flatten_metadata(_read_sidecar_metadata(dl_artifact_path)))

    explicit_sentiment_surface = _normalize_sentiment_surface(
        _lookup(
            merged_metadata,
            "sentiment_surface",
            "surface.sentiment_surface",
            "msml.sentiment_surface",
            "mpml.sentiment_surface",
        )
    )
    if explicit_sentiment_surface is None:
        explicit_sentiment_surface = _normalize_sentiment_surface(
            _env_str("MPML_SENTIMENT_SURFACE")
        )

    explicit_training_pair_family = _decode_str(
        _lookup(
            merged_metadata,
            "training_pair_family",
            "surface.training_pair_family",
            "msml.training_pair_family",
            "mpml.training_pair_family",
            "pair_family.train",
        )
    ) or _decode_str(
        _lookup(
            merged_metadata,
            "pair_family.training",
        )
    ) or _env_str("MPML_TRAINING_PAIR_FAMILY")

    explicit_evaluation_pair_family = _decode_str(
        _lookup(
            merged_metadata,
            "evaluation_pair_family",
            "surface.evaluation_pair_family",
            "msml.evaluation_pair_family",
            "mpml.evaluation_pair_family",
            "pair_family.eval",
            "pair_family.evaluation",
        )
    ) or _env_str("MPML_EVALUATION_PAIR_FAMILY")

    feature_surface = _decode_str(
        _lookup(
            merged_metadata,
            "feature_surface",
            "surface.feature_surface",
            "msml.feature_surface",
            "mpml.feature_surface",
            "feature_set",
            "surface.feature_set",
        )
    ) or _env_str("MPML_FEATURE_SURFACE") or _decode_str(dl_surface.get("feature_set")) or "unknown"

    artifact_source = _decode_str(
        _lookup(
            merged_metadata,
            "artifact_source",
            "surface.artifact_source",
            "msml.artifact_source",
            "artifact.path",
            "artifact_path",
            "source_path",
        )
    ) or _env_str("MPML_ARTIFACT_SOURCE") or (_decode_str(str(dl_artifact_path)) if dl_artifact_path else "unknown")

    surface_semantics_version = _decode_int(
        _lookup(
            merged_metadata,
            "surface_semantics_version",
            "surface.surface_semantics_version",
            "msml.surface_semantics_version",
            "mpml.surface_semantics_version",
        )
    )
    if surface_semantics_version is None:
        surface_semantics_version = int(EXPERIMENT_SURFACE_VERSION)

    target_horizon = _decode_int(
        _lookup(
            merged_metadata,
            "target_horizon",
            "surface.target_horizon",
            "msml.target_horizon",
        )
    )
    if target_horizon is None:
        target_horizon = _decode_int(dl_surface.get("target_horizon"))

    artifact_model = _decode_str(
        _lookup(
            merged_metadata,
            "artifact_model",
            "model",
            "surface.model",
            "msml.model",
        )
    ) or _decode_str(dl_surface.get("model")) or "unknown"

    msml_regime = _decode_str(
        _lookup(
            merged_metadata,
            "msml_regime",
            "dl_regime",
            "surface.dl_regime",
            "surface.msml_regime",
            "msml.dl_regime",
        )
    ) or _decode_str(experiment_factors.get("msml_regime")) or _decode_str(dl_surface.get("dl_regime")) or "unknown"

    dl_enabled = bool(dl_runtime_enabled)
    sentiment_surface = explicit_sentiment_surface
    if not dl_enabled:
        sentiment_surface = "none"
    elif sentiment_surface is None:
        sentiment_surface = _FEATURE_TO_SENTIMENT_MAP.get(feature_surface)
    if sentiment_surface is None:
        sentiment_surface = "none"

    training_family_from_metadata, evaluation_family_from_metadata = _infer_pair_families_from_artifact_text(
        _lookup(
            merged_metadata,
            "artifact_source",
            "surface.artifact_source",
            "artifact.path",
            "artifact_path",
            "source_path",
        )
    )
    training_family_from_path, evaluation_family_from_path = _infer_pair_families_from_artifact_text(
        dl_artifact_path
    )
    training_pair_family = (
        explicit_training_pair_family
        or training_family_from_metadata
        or training_family_from_path
        or "unknown"
    )

    evaluation_pair_family = (
        explicit_evaluation_pair_family
        or evaluation_family_from_metadata
        or evaluation_family_from_path
        or _infer_eval_family_from_active_pairs()
        or "unknown"
    )

    imputation_awareness = _resolve_imputation_awareness(experiment_factors)

    result: dict[str, Any] = {
        "surface_semantics_version": surface_semantics_version,
        "surface_source": "artifact_introspection",
        "training_pair_family": training_pair_family,
        "evaluation_pair_family": evaluation_pair_family,
        "sentiment_surface": sentiment_surface,
        "feature_surface": feature_surface,
        "imputation_awareness": imputation_awareness,
        "artifact_source": artifact_source,
        "dl_enabled": dl_enabled,
        "selector_enabled": bool(experiment_factors.get("selector_enabled")),
        "overlap_only": bool(experiment_factors.get("overlap_only")),
        "msml_regime": msml_regime,
        "target_horizon": target_horizon,
        "artifact_model": artifact_model,
    }

    # Behavioral Surface provenance — emitted when caller provides surface identity.
    # These fields carry the canonical surface representation through the manifest
    # without converting back to Trend/Vol-specific regime strings.
    if behavioral_surface_id is not None:
        result["behavioral_surface"] = behavioral_surface_id
    if behavioral_surface_version is not None:
        result["behavioral_surface_version"] = behavioral_surface_version
    if behavioral_state_id is not None:
        result["behavioral_state"] = behavioral_state_id

    return result
