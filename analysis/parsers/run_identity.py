"""
analysis/parsers/run_identity.py
=================================
Canonical run identity and experiment-semantics helpers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_TS_RE = re.compile(r"(\d{8}T\d{6}Z)")

_RUN_MEANINGS = {
    "A": "sentiment ON + missing indicator OFF (Gen1)",
    "B": "sentiment OFF + missing indicator OFF (Gen1 baseline)",
    "C": "sentiment ON + missing indicator ON (Gen2)",
    "D": "sentiment OFF + missing indicator ON (Gen2 baseline)",
    "U": "unknown experiment semantics",
}


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
    if lowered in {"gen1", "g1"}:
        return "gen1"
    if lowered in {"gen2", "g2"}:
        return "gen2"
    return None


def _manifest_experiment_gen(manifest: dict[str, Any] | None) -> str | None:
    payload = manifest or {}
    candidates = [
        payload.get("experiment_gen"),
        (payload.get("run") or {}).get("experiment_gen"),
        (payload.get("experiment") or {}).get("gen"),
        (payload.get("flags") or {}).get("EXPERIMENT_GEN"),
    ]
    for candidate in candidates:
        normalized = _normalize_gen(candidate)
        if normalized:
            return normalized
    return None


def _manifest_variant(manifest: dict[str, Any] | None) -> str | None:
    payload = manifest or {}
    candidates = [
        payload.get("run_variant"),
        (payload.get("run") or {}).get("run_variant"),
        (payload.get("experiment") or {}).get("variant"),
    ]
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        normalized = candidate.strip().upper()
        if normalized in {"A", "B", "C", "D"}:
            return normalized
    return None


def _variant_from_gen_and_dl(experiment_gen: str, dl_enabled: bool | None) -> str | None:
    if dl_enabled is None:
        return None
    if experiment_gen == "gen1":
        return "A" if dl_enabled else "B"
    if experiment_gen == "gen2":
        return "C" if dl_enabled else "D"
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

    explicit_gen = _manifest_experiment_gen(manifest)
    inferred_gen = _normalize_gen(experiment_gen)
    final_gen = explicit_gen or inferred_gen or "unknown"
    if explicit_gen and inferred_gen and explicit_gen != inferred_gen:
        identity_warnings.append(
            f"Experiment generation conflict: discovery='{inferred_gen}' manifest='{explicit_gen}'."
        )

    dl_enabled_raw = (manifest or {}).get("dl_enabled")
    dl_enabled: bool | None = None if dl_enabled_raw is None else bool(dl_enabled_raw)
    explicit_variant = _manifest_variant(manifest)
    derived_variant = _variant_from_gen_and_dl(final_gen, dl_enabled)
    variant = explicit_variant or derived_variant or "U"

    if explicit_variant and derived_variant and explicit_variant != derived_variant:
        identity_warnings.append(
            f"Variant conflict: explicit='{explicit_variant}' derived='{derived_variant}'."
        )
    if variant == "U":
        identity_warnings.append(
            "Variant could not be inferred safely from manifest/config; assigned variant='U'."
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
        "run_meaning": _RUN_MEANINGS.get(variant, _RUN_MEANINGS["U"]),
        "archive_relpath": archive_relpath,
        "archive_slug": archive_slug,
        "timestamp_utc": timestamp,
        "identity_warnings": identity_warnings,
        "manifest_run_id": manifest_run_id,
    }
