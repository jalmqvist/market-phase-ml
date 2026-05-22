"""
analysis/parsers/manifest_parser.py
=====================================
Parse MPML canonical manifest files (``run_manifest.json`` or legacy ``run_manifest_*.json``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _manifest_files(run_dir: Path) -> list[Path]:
    manifests: list[Path] = []
    canonical = run_dir / "run_manifest.json"
    if canonical.exists():
        manifests.append(canonical)
    manifests.extend(sorted(run_dir.glob("run_manifest_*.json")))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for m in manifests:
        resolved = m.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(m)
    return deduped


def _parse_experiment_block(data: dict[str, Any]) -> dict[str, Any]:
    experiment = data.get("experiment") or {}
    if not isinstance(experiment, dict):
        return {}
    generation = experiment.get("generation")
    variant = experiment.get("variant")
    sentiment_enabled = experiment.get("sentiment_enabled")
    missing_indicators_enabled = experiment.get("missing_indicators_enabled")
    semantic_label = experiment.get("semantic_label")
    return {
        "generation": generation,
        "variant": variant,
        "sentiment_enabled": sentiment_enabled,
        "missing_indicators_enabled": missing_indicators_enabled,
        "semantic_label": semantic_label,
    }


def parse_manifest(run_dir: Path) -> dict[str, Any] | None:
    """
    Parse the canonical manifest in *run_dir*.

    Integrity rules:
    * 0 manifests  -> return ``None`` (legacy fallback path)
    * 1 manifest   -> return parsed canonical manifest metadata
    * >1 manifests -> raise ``ValueError`` (ambiguous provenance)
    """
    manifest_files = _manifest_files(run_dir)
    if not manifest_files:
        return None
    if len(manifest_files) > 1:
        raise ValueError(
            f"Manifest integrity failure: expected exactly one manifest in {run_dir}, "
            f"found {len(manifest_files)}."
        )

    manifest_path = manifest_files[0]
    try:
        data = json.loads(manifest_path.read_text(errors="ignore"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Manifest parse failure in {manifest_path}: {exc}") from exc

    dl_section = data.get("dl") or {}
    run_section = data.get("run") or {}
    wf_section = data.get("walkforward") or {}
    flags_section = data.get("flags") or {}
    dl_enabled = dl_section.get("dl_enabled")

    return {
        "manifest_count": 1,
        "manifest_path": str(manifest_path),
        "run_id": run_section.get("run_id"),
        "dl_enabled": None if dl_enabled is None else bool(dl_enabled),
        "dl_surface": dl_section.get("dl_surface"),
        "dl_surface_string": dl_section.get("dl_surface_string"),
        "dl_artifact_path": dl_section.get("dl_artifact_path"),
        "dl_mode_tag": dl_section.get("dl_mode_tag"),
        "walkforward": wf_section,
        "flags": flags_section,
        "git_sha": run_section.get("git_sha"),
        "timestamp_utc": run_section.get("timestamp_utc"),
        "python_version": run_section.get("python_version"),
        "run": run_section,
        "experiment": _parse_experiment_block(data),
        "raw": data,
    }
