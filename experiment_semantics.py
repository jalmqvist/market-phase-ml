"""
Canonical MPML experiment semantics (variant-first).
"""

from __future__ import annotations

from typing import Any

CURRENT_EXPERIMENT_SEMANTICS_VERSION = 2
LEGACY_VARIANT = "U"
VALID_EXPERIMENT_VARIANTS = {"A", "B", "C", "D"}

EXPERIMENT_VARIANTS: dict[str, dict[str, Any]] = {
    "A": {
        "generation": "gen1",
        "sentiment_enabled": True,
        "missing_indicators_enabled": False,
        "semantic_label": "Gen1_A",
        "run_meaning": "sentiment ON + missing indicator OFF (Gen1)",
    },
    "B": {
        "generation": "gen1",
        "sentiment_enabled": False,
        "missing_indicators_enabled": False,
        "semantic_label": "Gen1_B",
        "run_meaning": "sentiment OFF + missing indicator OFF (Gen1 baseline)",
    },
    "C": {
        "generation": "gen2",
        "sentiment_enabled": True,
        "missing_indicators_enabled": True,
        "semantic_label": "Gen2_C",
        "run_meaning": "sentiment ON + missing indicator ON (Gen2)",
    },
    "D": {
        "generation": "gen2",
        "sentiment_enabled": False,
        "missing_indicators_enabled": True,
        "semantic_label": "Gen2_D",
        "run_meaning": "sentiment OFF + missing indicator ON (Gen2 baseline)",
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


def build_experiment_metadata_from_variant(variant: str) -> dict[str, Any]:
    normalized = normalize_variant(variant)
    if normalized is None:
        raise ValueError(
            f"Invalid experiment variant: {variant!r}. "
            f"Allowed values={sorted(VALID_EXPERIMENT_VARIANTS)}"
        )
    semantics = EXPERIMENT_VARIANTS[normalized]
    return {
        "generation": semantics["generation"],
        "variant": normalized,
        "sentiment_enabled": bool(semantics["sentiment_enabled"]),
        "missing_indicators_enabled": bool(semantics["missing_indicators_enabled"]),
        "semantic_label": semantics["semantic_label"],
        "legacy_semantics": False,
        "semantics_version": CURRENT_EXPERIMENT_SEMANTICS_VERSION,
    }
