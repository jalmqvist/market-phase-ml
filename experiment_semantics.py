"""
Canonical MPML experiment semantics (factor-first).

Architecture note
-----------------
As of the v5 factorial matrix, experiment semantics are sourced from an explicit
``experiment_surface`` manifest block — NOT inferred from variant letters, generation
labels, DL flags, or directory names.

* ``experiment_surface`` is the canonical source of truth for analysis attribution.
* Variant letters are now only runtime architecture selectors (missing-indicator ON/OFF,
  selector enabled, DL-capable runtime).  They do NOT imply parquet-level properties
  such as sentiment surface or training-pair family.
* Legacy manifests (pre-v5) retain old variant-based semantics and are flagged with
  ``legacy_semantics=True``; they emit a warning banner in reports.
"""

from __future__ import annotations

from typing import Any

CURRENT_EXPERIMENT_SEMANTICS_VERSION = 3
# Surface-level ontology version (independent of runtime semantics_version).
EXPERIMENT_SURFACE_VERSION = 5
LEGACY_VARIANT = "U"
VALID_EXPERIMENT_VARIANTS = {"A", "B", "C", "D", "E", "F"}
EXPERIMENT_RUN_FAMILY = "factorial_v1"

# Runtime factor keys — describe runtime architecture configuration.
EXPERIMENT_FACTOR_KEYS: tuple[str, ...] = (
    "dl_enabled",
    "sentiment_enabled",
    "missing_indicators_enabled",
    "msml_regime",
    "overlap_only",
    "selector_enabled",
)

# Surface factor keys — describe the parquet / artifact / training configuration.
# These are the canonical v5 factor dimensions for analysis attribution.
EXPERIMENT_SURFACE_FACTOR_KEYS: tuple[str, ...] = (
    "surface_source",
    "sentiment_surface",
    "training_pair_family",
    "evaluation_pair_family",
    "feature_surface",
    "artifact_source",
    "surface_semantics_version",
)
CANONICAL_SENTIMENT_SURFACES: frozenset[str] = frozenset(
    {"sentiment", "no_sentiment", "none"}
)

DEFAULT_EXPERIMENT_FACTORS: dict[str, Any] = {
    "dl_enabled": True,
    "sentiment_enabled": True,
    "missing_indicators_enabled": False,
    "msml_regime": "LVTF",
    "overlap_only": False,
    "selector_enabled": True,
}

EXPERIMENT_VARIANTS: dict[str, dict[str, Any]] = {
    "A": {
        "generation": "gen1",
        "sentiment_enabled": True,
        "missing_indicators_enabled": False,
        "dl_enabled": True,
        "msml_regime": "LVTF",
        "overlap_only": False,
        "selector_enabled": True,
        "semantic_label": "Gen1_A",
        "run_meaning": "sentiment ON + missing indicator OFF (Gen1)",
    },
    "B": {
        "generation": "gen1",
        "sentiment_enabled": False,
        "missing_indicators_enabled": False,
        "dl_enabled": False,
        "msml_regime": "LVTF",
        "overlap_only": False,
        "selector_enabled": True,
        "semantic_label": "Gen1_B",
        "run_meaning": "sentiment OFF + missing indicator OFF (Gen1 baseline)",
    },
    "C": {
        "generation": "gen2",
        "sentiment_enabled": True,
        "missing_indicators_enabled": True,
        "dl_enabled": True,
        "msml_regime": "LVTF",
        "overlap_only": False,
        "selector_enabled": True,
        "semantic_label": "Gen2_C",
        "run_meaning": "sentiment ON + missing indicator ON (Gen2)",
    },
    "D": {
        "generation": "gen2",
        "sentiment_enabled": False,
        "missing_indicators_enabled": True,
        "dl_enabled": False,
        "msml_regime": "LVTF",
        "overlap_only": False,
        "selector_enabled": True,
        "semantic_label": "Gen2_D",
        "run_meaning": "sentiment OFF + missing indicator ON (Gen2 baseline)",
    },
    "E": {
        "generation": "gen1",
        "sentiment_enabled": True,
        "missing_indicators_enabled": True,
        "dl_enabled": True,
        "msml_regime": "LVTF",
        "overlap_only": False,
        "selector_enabled": True,
        "semantic_label": "Gen1_E",
        "run_meaning": "sentiment ON + missing indicator ON (Gen1)",
    },

    "F": {
        "generation": "gen2",
        "sentiment_enabled": True,
        "missing_indicators_enabled": False,
        "dl_enabled": True,
        "msml_regime": "LVTF",
        "overlap_only": False,
        "selector_enabled": True,
        "semantic_label": "Gen2_F",
        "run_meaning": "sentiment ON + missing indicator OFF (Gen2)",
    },
}

LEGACY_RUN_MEANING = "legacy or unknown experiment semantics"


def normalize_variant(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    if normalized in VALID_EXPERIMENT_VARIANTS:
        return normalized
    return None


def variant_semantics(variant: str | None) -> dict[str, Any] | None:
    normalized = normalize_variant(variant)
    if normalized is None:
        return None
    return dict(EXPERIMENT_VARIANTS[normalized])


def _as_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _as_msml_regime(value: Any, *, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip().upper()
    return default


def normalize_experiment_factors(
    raw_factors: dict[str, Any] | None = None,
    *,
    fallback_sentiment_enabled: Any = None,
    fallback_missing_indicators_enabled: Any = None,
    fallback_dl_enabled: Any = None,
    fallback_msml_regime: Any = None,
) -> dict[str, Any]:
    raw_factors = raw_factors if isinstance(raw_factors, dict) else {}
    normalized: dict[str, Any] = {}

    normalized["dl_enabled"] = _as_bool(
        raw_factors.get("dl_enabled", fallback_dl_enabled),
        default=bool(DEFAULT_EXPERIMENT_FACTORS["dl_enabled"]),
    )
    normalized["sentiment_enabled"] = _as_bool(
        raw_factors.get("sentiment_enabled", fallback_sentiment_enabled),
        default=bool(DEFAULT_EXPERIMENT_FACTORS["sentiment_enabled"]),
    )
    normalized["missing_indicators_enabled"] = _as_bool(
        raw_factors.get("missing_indicators_enabled", fallback_missing_indicators_enabled),
        default=bool(DEFAULT_EXPERIMENT_FACTORS["missing_indicators_enabled"]),
    )
    normalized["msml_regime"] = _as_msml_regime(
        raw_factors.get("msml_regime", fallback_msml_regime),
        default=str(DEFAULT_EXPERIMENT_FACTORS["msml_regime"]),
    )
    normalized["overlap_only"] = _as_bool(
        raw_factors.get("overlap_only"),
        default=bool(DEFAULT_EXPERIMENT_FACTORS["overlap_only"]),
    )
    normalized["selector_enabled"] = _as_bool(
        raw_factors.get("selector_enabled"),
        default=bool(DEFAULT_EXPERIMENT_FACTORS["selector_enabled"]),
    )
    return {key: normalized[key] for key in EXPERIMENT_FACTOR_KEYS}


def build_experiment_metadata_from_variant(
    variant: str,
    *,
    factor_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_variant(variant)
    if normalized is None:
        raise ValueError(
            f"Invalid experiment variant: {variant!r}. "
            f"Allowed values={sorted(VALID_EXPERIMENT_VARIANTS)}"
        )
    semantics = EXPERIMENT_VARIANTS[normalized]
    factors = normalize_experiment_factors(
        {
            "dl_enabled": semantics["dl_enabled"],
            "sentiment_enabled": semantics["sentiment_enabled"],
            "missing_indicators_enabled": semantics["missing_indicators_enabled"],
            "msml_regime": semantics["msml_regime"],
            "overlap_only": semantics["overlap_only"],
            "selector_enabled": semantics["selector_enabled"],
            **(factor_overrides or {}),
        }
    )
    return {
        "run_family": EXPERIMENT_RUN_FAMILY,
        "generation": semantics["generation"],
        "variant": normalized,
        "sentiment_enabled": bool(factors["sentiment_enabled"]),
        "missing_indicators_enabled": bool(factors["missing_indicators_enabled"]),
        "msml_regime": factors["msml_regime"],
        "dl_enabled": bool(factors["dl_enabled"]),
        "factors": factors,
        "semantic_label": semantics["semantic_label"],
        "legacy_semantics": False,
        "semantics_version": CURRENT_EXPERIMENT_SEMANTICS_VERSION,
    }


# ---------------------------------------------------------------------------
# v5 experiment surface — canonical source of truth for analysis attribution
# ---------------------------------------------------------------------------


def normalize_experiment_surface(
    raw_surface: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Canonicalize an ``experiment_surface`` manifest block.

    This is the ONLY source of truth for parquet/artifact/training-level
    semantic attribution in the v5 factorial framework.  It is NOT derived
    from variant letters, generation labels, DL flags, or directory names.

    Parameters
    ----------
    raw_surface:
        The raw ``experiment_surface`` dict from a manifest.  ``None`` or
        non-dict values are treated as absent (returns an empty dict with
        ``surface_semantics_version=None``).

    Returns
    -------
    dict with keys from ``EXPERIMENT_SURFACE_FACTOR_KEYS``.
    All string fields are stripped; absent fields are ``None``.
    ``sentiment_surface`` preserves canonical v5 string ontology:
    ``"sentiment"`` | ``"no_sentiment"`` | ``"none"``.
    Legacy bool values are accepted for backward compatibility and mapped to
    ``"sentiment"`` (True) / ``"no_sentiment"`` (False).
    ``surface_semantics_version`` is coerced to ``int | None``.
    """
    if not isinstance(raw_surface, dict):
        return {key: None for key in EXPERIMENT_SURFACE_FACTOR_KEYS}

    def _str_or_none(val: Any) -> str | None:
        if isinstance(val, str) and val.strip():
            return val.strip()
        return None

    def _sentiment_surface_or_none(val: Any) -> str | None:
        if isinstance(val, str):
            normalized = val.strip().lower()
            if normalized in CANONICAL_SENTIMENT_SURFACES:
                return normalized
            return None
        if isinstance(val, bool):
            return "sentiment" if val else "no_sentiment"
        return None

    def _int_or_none(val: Any) -> int | None:
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.strip().isdigit():
            return int(val.strip())
        return None

    return {
        "surface_source": _str_or_none(raw_surface.get("surface_source")),
        "sentiment_surface": _sentiment_surface_or_none(raw_surface.get("sentiment_surface")),
        "training_pair_family": _str_or_none(raw_surface.get("training_pair_family")),
        "evaluation_pair_family": _str_or_none(raw_surface.get("evaluation_pair_family")),
        "feature_surface": _str_or_none(raw_surface.get("feature_surface")),
        "artifact_source": _str_or_none(raw_surface.get("artifact_source")),
        "surface_semantics_version": _int_or_none(raw_surface.get("surface_semantics_version")),
    }


def is_v5_surface(surface: dict[str, Any] | None) -> bool:
    """Return True when *surface* is a well-formed v5 experiment surface block."""
    if not isinstance(surface, dict):
        return False
    # Must have at least the version marker and one substantive field.
    has_version = surface.get("surface_semantics_version") is not None
    sentiment_surface = surface.get("sentiment_surface")
    if isinstance(sentiment_surface, str):
        has_sentiment = sentiment_surface.strip().lower() in CANONICAL_SENTIMENT_SURFACES
    elif isinstance(sentiment_surface, bool):
        # Backward compatibility for transitional manifests.
        has_sentiment = True
    else:
        has_sentiment = False
    return has_version and has_sentiment
