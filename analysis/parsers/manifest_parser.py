"""
analysis/parsers/manifest_parser.py
=====================================
Parse MPML ``run_manifest_*.json`` files.

Manifest files record the full run configuration: costs, DL surface
settings, walkforward parameters, flags, and software versions.  They
are the authoritative source for:

* whether DL signals were enabled (``dl.dl_enabled``)
* the DL surface specification (model, regime, horizon)
* whether sentiment features were active (inferred from ``dl_surface``)
* walkforward / ablation flags

Sentiment ON/OFF semantics
---------------------------
"Sentiment" in the MPML context means the DL prediction surface provides
a *sentiment* proxy signal (e.g. price-direction probability from a
LSTM/MLP trained on market-sentiment-ml outputs).

* Sentiment **ON**  — ``dl.dl_enabled = true`` and a valid
  ``dl_artifact_path`` is present.
* Sentiment **OFF** — ``dl.dl_enabled = false`` (baseline mode).

Experiment variants are labelled A–D in comparative studies:

* **A** — sentinel ON,  Gen1 missing-indicator semantics
* **B** — sentinel OFF, Gen1 (baseline comparison for A)
* **C** — sentinel ON,  Gen2 missing-indicator semantics
* **D** — sentinel OFF, Gen2 (baseline comparison for C)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_manifest(run_dir: Path) -> dict[str, Any] | None:
    """
    Find and parse all ``run_manifest_*.json`` files in *run_dir*.

    Multiple manifests may exist when a directory holds both a
    ``__baseline`` and a ``__dl_enabled`` run.  All are merged into a
    list under ``manifests``; the *primary* manifest is selected by
    preferring ``__dl_enabled`` > ``__baseline`` > first-found.

    Returns
    -------
    dict or None
        Parsed manifest dict with the following top-level keys:
        ``manifests``, ``primary``, ``dl_enabled``, ``run_id``,
        ``dl_surface``, ``walkforward``, ``flags``.
        Returns None if no manifest files are found.
    """
    manifest_files = sorted(run_dir.glob("run_manifest_*.json"))
    if not manifest_files:
        return None

    manifests: list[dict[str, Any]] = []
    for mf in manifest_files:
        try:
            data = json.loads(mf.read_text(errors="ignore"))
            data["_source_file"] = str(mf.name)
            manifests.append(data)
        except Exception as exc:  # noqa: BLE001
            manifests.append({"_source_file": str(mf.name), "_parse_error": str(exc)})

    # Select the primary manifest: prefer DL-enabled over baseline.
    primary = manifests[0]
    for m in manifests:
        mode_tag = (m.get("dl") or {}).get("dl_mode_tag", "")
        if "__dl_enabled" in mode_tag:
            primary = m
            break

    dl_section = primary.get("dl") or {}
    run_section = primary.get("run") or {}
    wf_section = primary.get("walkforward") or {}
    flags_section = primary.get("flags") or {}

    return {
        "manifests": manifests,
        "primary": primary,
        "run_id": run_section.get("run_id"),
        "dl_enabled": bool(dl_section.get("dl_enabled", False)),
        "dl_surface": dl_section.get("dl_surface"),
        "dl_surface_string": dl_section.get("dl_surface_string"),
        "dl_artifact_path": dl_section.get("dl_artifact_path"),
        "dl_mode_tag": dl_section.get("dl_mode_tag"),
        "walkforward": wf_section,
        "flags": flags_section,
        "git_sha": run_section.get("git_sha"),
        "timestamp_utc": run_section.get("timestamp_utc"),
        "python_version": run_section.get("python_version"),
    }
