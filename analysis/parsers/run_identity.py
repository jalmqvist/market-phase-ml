"""
analysis/parsers/run_identity.py
=================================
Canonical run identity and experiment-semantics helpers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from experiment_semantics import (
    LEGACY_RUN_MEANING,
    LEGACY_VARIANT,
    VALID_EXPERIMENT_VARIANTS,
    normalize_experiment_factors,
    is_v5_surface,
)

_TS_RE = re.compile(r"(\d{8}T\d{6}Z)")

def _extract_timestamp(text: str | None) -> str | None:
    if not text:
        return None
    match = _TS_RE.search(text)
    return match.group(1) if match else None


def _slugify_path(path: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9/_-]+", "_", path)
    return cleaned.strip("/").replace("/", "__") or "root"


def _normalize_gen(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered == "gen1":
        return "gen1"
    if lowered == "gen2":
        return "gen2"
    return None


def _normalize_variant(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    if normalized in VALID_EXPERIMENT_VARIANTS:
        return normalized
    return None


# Matches archive directory names like fp_gen1_A, fp_gen2_D, and suffixed reruns
# such as fp_gen1_B_retry or fp_gen2_C-copy.
_ARCHIVE_SENTINEL_RE = re.compile(r"fp_(gen[12])_([A-F])(?:$|[_-])", re.IGNORECASE)


def _build_run_meaning(
    generation: str | None,
    sentiment_enabled: bool | None,
    missing_indicators_enabled: bool | None,
) -> str:
    if (
        generation not in {"gen1", "gen2"}
        or not isinstance(sentiment_enabled, bool)
        or not isinstance(missing_indicators_enabled, bool)
    ):
        return LEGACY_RUN_MEANING
    sentiment = "sentiment ON" if sentiment_enabled else "sentiment OFF"
    missing = (
        "missing indicator ON"
        if missing_indicators_enabled
        else "missing indicator OFF"
    )
    return f"{sentiment} + {missing} ({generation})"


def _build_surface_run_meaning(surface: dict[str, Any]) -> str:
    """
    Build a human-readable run meaning from v5 ``experiment_surface`` metadata.

    This avoids cross-referencing variant letters or generation labels — the
    description is self-contained and directly readable.

    Example output:
        "persistent training family + sentiment surface + blind runtime (DL)"
    """
    parts: list[str] = []

    training_family = surface.get("training_pair_family")
    eval_family = surface.get("evaluation_pair_family")
    if training_family:
        family_str = f"{training_family} training family"
        if eval_family and eval_family != training_family:
            family_str += f" → {eval_family} evaluation"
        parts.append(family_str)

    sentiment = surface.get("sentiment_surface")
    if sentiment is True:
        parts.append("sentiment surface")
    elif sentiment is False:
        parts.append("no-sentiment surface")

    feature_surface = surface.get("feature_surface")
    if feature_surface:
        parts.append(f"{feature_surface} features")

    if not parts:
        return LEGACY_RUN_MEANING
    return " + ".join(parts)


def _validate_archive_sentinel(
    archive_dir_name: str,
    generation: str | None,
    variant: str | None,
) -> None:
    match = _ARCHIVE_SENTINEL_RE.search(archive_dir_name)
    if not match:
        return
    expected_generation = match.group(1).lower()
    expected_variant = match.group(2).upper()
    if generation != expected_generation or variant != expected_variant:
        raise RuntimeError(
            "Analysis attribution corruption detected: "
            f"archive_dir={archive_dir_name!r} encodes {expected_generation}/{expected_variant} "
            f"but manifest.experiment resolved to {generation!r}/{variant!r}."
        )


def infer_run_identity(
    *,
    archive_root: Path,
    run_dir: Path,
    experiment_gen: str,
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Build canonical identity and semantic labels for a discovered run.
    """
    run_section = (manifest or {}).get("run") or {}
    manifest_run_id = (manifest or {}).get("run_id") or run_section.get("run_id")
    manifest_timestamp = (manifest or {}).get("timestamp_utc") or run_section.get("timestamp_utc")
    timestamp = (
        manifest_timestamp
        or _extract_timestamp(manifest_run_id)
        or "unknown_ts"
    )

    identity_warnings: list[str] = []
    if manifest_timestamp is None:
        identity_warnings.append(
            "Manifest timestamp missing; using run_id timestamp fallback or unknown_ts."
        )

    experiment_block = (manifest or {}).get("experiment") or {}
    experiment_surface = (manifest or {}).get("experiment_surface") or {}
    surface_source = (manifest or {}).get("surface_source", "legacy_variant_fallback")

    explicit_gen = _normalize_gen(experiment_block.get("generation"))
    explicit_variant = _normalize_variant(experiment_block.get("variant"))
    factors = normalize_experiment_factors(
        experiment_block.get("factors"),
        fallback_sentiment_enabled=experiment_block.get("sentiment_enabled"),
        fallback_missing_indicators_enabled=experiment_block.get("missing_indicators_enabled"),
        fallback_dl_enabled=experiment_block.get("dl_enabled", (manifest or {}).get("dl_enabled")),
        fallback_msml_regime=experiment_block.get("msml_regime"),
    )
    sentiment_enabled = factors.get("sentiment_enabled") if isinstance(factors.get("sentiment_enabled"), bool) else None
    missing_indicators_enabled = (
        factors.get("missing_indicators_enabled")
        if isinstance(factors.get("missing_indicators_enabled"), bool)
        else None
    )
    semantic_label = experiment_block.get("semantic_label")
    semantic_label = (
        semantic_label.strip()
        if isinstance(semantic_label, str) and semantic_label.strip()
        else None
    )

    if manifest and not experiment_block:
        identity_warnings.append(
            "Manifest experiment block missing; using legacy unknown semantics (variant='U')."
        )

    variant = explicit_variant or LEGACY_VARIANT
    final_gen = explicit_gen or "unknown"
    legacy_semantics = not (
        explicit_gen in {"gen1", "gen2"}
        and explicit_variant in VALID_EXPERIMENT_VARIANTS
        and isinstance(factors.get("sentiment_enabled"), bool)
        and isinstance(factors.get("missing_indicators_enabled"), bool)
        and semantic_label is not None
    )

    if explicit_gen is None and experiment_block:
        identity_warnings.append("Experiment generation invalid or missing in manifest experiment block.")
    if explicit_variant is None and experiment_block:
        identity_warnings.append("Experiment variant invalid or missing in manifest experiment block.")
    if experiment_block and "factors" not in experiment_block:
        identity_warnings.append("Experiment factors block missing; using compatibility fallbacks from legacy fields.")
    if sentiment_enabled is None and experiment_block:
        identity_warnings.append("Experiment sentiment_enabled missing or non-boolean in manifest experiment block.")
    if missing_indicators_enabled is None and experiment_block:
        identity_warnings.append(
            "Experiment missing_indicators_enabled missing or non-boolean in manifest experiment block."
        )
    if semantic_label is None and experiment_block:
        identity_warnings.append("Experiment semantic_label missing or empty in manifest experiment block.")

    if explicit_variant is None and experiment_block.get("variant") is not None:
        identity_warnings.append(
            f"Experiment variant invalid: {experiment_block.get('variant')!r} "
            f"(expected one of {sorted(VALID_EXPERIMENT_VARIANTS)})."
        )

    if final_gen == "unknown" or variant == LEGACY_VARIANT:
        identity_warnings.append(
            "Variant unresolved from canonical manifest experiment metadata; assigned variant='U'."
        )

    # Warn when v5 experiment_surface is absent from an otherwise valid manifest.
    if manifest and not is_v5_surface(experiment_surface):
        if surface_source == "missing_experiment_surface":
            identity_warnings.append(
                "experiment_surface block missing in a modern manifest; semantic attribution is invalid "
                "until an explicit v5 experiment_surface block is emitted."
            )
        else:
            identity_warnings.append(
                "experiment_surface block absent or incomplete; run will use legacy_variant_fallback semantics. "
                "Add experiment_surface to the manifest for factor-first attribution."
            )

    archive_root = archive_root.resolve()
    run_dir = run_dir.resolve()
    try:
        archive_relpath = str(run_dir.relative_to(archive_root))
    except ValueError:
        archive_relpath = run_dir.name

    archive_slug = _slugify_path(archive_relpath)
    sentinel_gen = final_gen if final_gen != "unknown" else None
    sentinel_variant = variant if variant != LEGACY_VARIANT else None
    _validate_archive_sentinel(run_dir.name, sentinel_gen, sentinel_variant)
    semantic_name = f"{final_gen}_{variant}"
    semantic_run_id = f"{semantic_name}__{timestamp}"
    canonical_run_id = f"{semantic_run_id}__{archive_slug}"

    # Build human-readable run meaning:
    # For v5 manifests, derive from experiment_surface (self-describing).
    # For legacy manifests, fall back to old variant-based description.
    if is_v5_surface(experiment_surface):
        run_meaning = _build_surface_run_meaning(experiment_surface)
    elif surface_source == "legacy_variant_fallback":
        run_meaning = _build_run_meaning(
            final_gen if final_gen != "unknown" else None,
            sentiment_enabled,
            missing_indicators_enabled,
        )
    else:
        run_meaning = LEGACY_RUN_MEANING

    return {
        "run_id": canonical_run_id,
        "semantic_run_name": semantic_name,
        "semantic_run_id": semantic_run_id,
        "experiment_gen": final_gen,
        "run_variant": variant,
        "sentiment_enabled": sentiment_enabled,
        "missing_indicators_enabled": missing_indicators_enabled,
        "factors": factors,
        "experiment_surface": experiment_surface,
        "surface_source": surface_source,
        "run_family": experiment_block.get("run_family"),
        "semantic_label": semantic_label,
        "legacy_semantics": legacy_semantics,
        "run_meaning": run_meaning,
        "archive_relpath": archive_relpath,
        "archive_slug": archive_slug,
        "timestamp_utc": timestamp,
        "identity_warnings": identity_warnings,
        "manifest_run_id": manifest_run_id,
    }
