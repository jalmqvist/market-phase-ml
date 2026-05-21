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
_VARIANT_RE = re.compile(r"(?:^|[_\-/])(gen[12])[_\-/]?([ABCD])(?:$|[_\-/])", re.IGNORECASE)
_VARIANT_LETTER_RE = re.compile(r"(?:^|[_\-/])([ABCD])(?:$|[_\-/])", re.IGNORECASE)

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


def _variant_from_gen_and_dl(experiment_gen: str, dl_enabled: bool | None) -> str | None:
    if dl_enabled is None:
        return None
    if experiment_gen == "gen1":
        return "A" if dl_enabled else "B"
    if experiment_gen == "gen2":
        return "C" if dl_enabled else "D"
    return None


def _variant_from_name(name: str, experiment_gen: str) -> str | None:
    match = _VARIANT_RE.search(name)
    if match:
        gen = match.group(1).lower()
        variant = match.group(2).upper()
        if gen == experiment_gen:
            return variant

    loose = _VARIANT_LETTER_RE.search(name)
    if not loose:
        return None
    variant = loose.group(1).upper()
    if experiment_gen == "gen1" and variant in {"A", "B"}:
        return variant
    if experiment_gen == "gen2" and variant in {"C", "D"}:
        return variant
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
    run_section = (manifest or {}).get("primary", {}).get("run") or {}
    manifest_run_id = (manifest or {}).get("run_id") or run_section.get("run_id")
    timestamp = (
        (manifest or {}).get("timestamp_utc")
        or run_section.get("timestamp_utc")
        or _extract_timestamp(manifest_run_id)
        or _extract_timestamp(run_dir.name)
        or "unknown_ts"
    )

    dl_enabled = (manifest or {}).get("dl_enabled")
    variant_from_manifest = _variant_from_gen_and_dl(experiment_gen, dl_enabled)
    name_context = f"{run_dir.parent.name}/{run_dir.name}"
    variant_from_name = _variant_from_name(name_context, experiment_gen)
    variant = variant_from_name or variant_from_manifest or "U"

    identity_warnings: list[str] = []
    if variant_from_name and variant_from_manifest and variant_from_name != variant_from_manifest:
        identity_warnings.append(
            "Variant mismatch: directory naming implies "
            f"{variant_from_name} but manifest implies {variant_from_manifest}."
        )

    archive_root = archive_root.resolve()
    run_dir = run_dir.resolve()
    try:
        archive_relpath = str(run_dir.relative_to(archive_root))
    except ValueError:
        archive_relpath = run_dir.name

    archive_slug = _slugify_path(archive_relpath)
    semantic_name = f"{experiment_gen}_{variant}"
    semantic_run_id = f"{semantic_name}__{timestamp}"
    canonical_run_id = f"{semantic_run_id}__{archive_slug}"

    return {
        "run_id": canonical_run_id,
        "semantic_run_name": semantic_name,
        "semantic_run_id": semantic_run_id,
        "run_variant": variant,
        "run_meaning": _RUN_MEANINGS.get(variant, _RUN_MEANINGS["U"]),
        "archive_relpath": archive_relpath,
        "archive_slug": archive_slug,
        "timestamp_utc": timestamp,
        "identity_warnings": identity_warnings,
        "manifest_run_id": manifest_run_id,
    }

