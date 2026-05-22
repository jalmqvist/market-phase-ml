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
    CURRENT_EXPERIMENT_SEMANTICS_VERSION,
    LEGACY_RUN_MEANING,
    LEGACY_VARIANT,
    VALID_EXPERIMENT_VARIANTS,
    variant_semantics,
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
    explicit_gen = _normalize_gen(experiment_block.get("generation"))
    explicit_variant = _normalize_variant(experiment_block.get("variant"))
    sentiment_enabled_raw = experiment_block.get("sentiment_enabled")
    sentiment_enabled = sentiment_enabled_raw if isinstance(sentiment_enabled_raw, bool) else None
    missing_indicators_raw = experiment_block.get("missing_indicators_enabled")
    missing_indicators_enabled = (
        missing_indicators_raw if isinstance(missing_indicators_raw, bool) else None
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

    variant = LEGACY_VARIANT
    final_gen = "unknown"
    legacy_semantics = True
    semantics_version = experiment_block.get("semantics_version")
    has_current_semantics_version = semantics_version == CURRENT_EXPERIMENT_SEMANTICS_VERSION

    if explicit_gen is None and experiment_block:
        identity_warnings.append("Experiment generation invalid or missing in manifest experiment block.")
    if explicit_variant is None and experiment_block:
        identity_warnings.append("Experiment variant invalid or missing in manifest experiment block.")
    if sentiment_enabled is None and experiment_block:
        identity_warnings.append("Experiment sentiment_enabled missing or non-boolean in manifest experiment block.")
    if missing_indicators_enabled is None and experiment_block:
        identity_warnings.append(
            "Experiment missing_indicators_enabled missing or non-boolean in manifest experiment block."
        )
    if semantic_label is None and experiment_block:
        identity_warnings.append("Experiment semantic_label missing or empty in manifest experiment block.")

    if experiment_block and not has_current_semantics_version:
        identity_warnings.append(
            "Manifest experiment semantics version missing or stale; "
            "treating manifest as legacy semantics (variant='U')."
        )

    if explicit_variant and has_current_semantics_version:
        canonical_semantics = variant_semantics(explicit_variant)
        if canonical_semantics is not None:
            variant = explicit_variant
            final_gen = canonical_semantics["generation"]
            legacy_semantics = False
            if explicit_gen and explicit_gen != canonical_semantics["generation"]:
                identity_warnings.append(
                    f"Variant conflict: explicit generation='{explicit_gen}' "
                    f"does not match canonical generation='{canonical_semantics['generation']}'."
                )
            if sentiment_enabled is not None and sentiment_enabled != canonical_semantics["sentiment_enabled"]:
                identity_warnings.append(
                    f"Variant conflict: explicit sentiment_enabled={sentiment_enabled} "
                    f"does not match canonical sentiment_enabled={canonical_semantics['sentiment_enabled']}."
                )
            if (
                missing_indicators_enabled is not None
                and missing_indicators_enabled != canonical_semantics["missing_indicators_enabled"]
            ):
                identity_warnings.append(
                    "Variant conflict: explicit missing_indicators_enabled="
                    f"{missing_indicators_enabled} does not match canonical "
                    f"missing_indicators_enabled={canonical_semantics['missing_indicators_enabled']}."
                )
            if semantic_label and semantic_label != canonical_semantics["semantic_label"]:
                identity_warnings.append(
                    f"Variant conflict: explicit semantic_label='{semantic_label}' "
                    f"does not match canonical semantic_label='{canonical_semantics['semantic_label']}'."
                )
        else:
            identity_warnings.append(
                f"Experiment variant invalid: {explicit_variant!r} "
                f"(expected one of {sorted(VALID_EXPERIMENT_VARIANTS)})."
            )

    if final_gen == "unknown" or variant == LEGACY_VARIANT:
        identity_warnings.append(
            "Variant unresolved from canonical manifest experiment metadata; assigned variant='U'."
        )

    archive_root = archive_root.resolve()
    run_dir = run_dir.resolve()
    try:
        archive_relpath = str(run_dir.relative_to(archive_root))
    except ValueError:
        archive_relpath = run_dir.name

    archive_slug = _slugify_path(archive_relpath)
    semantic_name = f"{final_gen}_{variant}"
    semantic_run_id = f"{semantic_name}__{timestamp}"
    canonical_run_id = f"{semantic_run_id}__{archive_slug}"

    return {
        "run_id": canonical_run_id,
        "semantic_run_name": semantic_name,
        "semantic_run_id": semantic_run_id,
        "experiment_gen": final_gen,
        "run_variant": variant,
        "legacy_semantics": legacy_semantics,
        "run_meaning": (
            canonical_semantics["run_meaning"]
            if not legacy_semantics and (canonical_semantics := variant_semantics(variant))
            else LEGACY_RUN_MEANING
        ),
        "archive_relpath": archive_relpath,
        "archive_slug": archive_slug,
        "timestamp_utc": timestamp,
        "identity_warnings": identity_warnings,
        "manifest_run_id": manifest_run_id,
    }
