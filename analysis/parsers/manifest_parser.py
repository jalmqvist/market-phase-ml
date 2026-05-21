"""
analysis/parsers/manifest_parser.py
=====================================
Parse MPML ``run_manifest_*.json`` files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_manifest(run_dir: Path) -> dict[str, Any] | None:
    """
    Parse the canonical manifest in *run_dir*.

    Integrity rules:
    * 0 manifests  -> return ``None`` (legacy fallback path)
    * 1 manifest   -> return parsed canonical manifest metadata
    * >1 manifests -> raise ``ValueError`` (ambiguous provenance)
    """
    manifest_files = sorted(run_dir.glob("run_manifest_*.json"))
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

    return {
        "manifest_count": 1,
        "manifest_path": str(manifest_path),
        "run_id": run_section.get("run_id"),
        "dl_enabled": None if "dl_enabled" not in dl_section else bool(dl_section.get("dl_enabled")),
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
        "experiment": data.get("experiment") or {},
        "raw": data,
    }
