"""
analysis/parsers/run_discovery.py
==================================
Discover MPML run directories inside an archive root.

Integrity hardening rules:
* A canonical run root must contain exactly one ``run_manifest_*.json``.
* Legacy CSV-only fallback is allowed only when explicitly marked via
  ``.mpml_legacy_run_root`` in that directory.
* Discovery is non-recursive beyond a discovered run root so nested
  exports/copies cannot be re-discovered as separate runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Generator

_LEGACY_RUN_MARKER_FILE = ".mpml_legacy_run_root"
_LEGACY_CSV_MARKERS = [
    "results_ml*.csv",
    "walkforward_results*.csv",
    "results_summary*.csv",
]


def _manifest_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("run_manifest_*.json"))


def _is_legacy_run_root(directory: Path) -> bool:
    if not (directory / _LEGACY_RUN_MARKER_FILE).exists():
        return False
    for pattern in _LEGACY_CSV_MARKERS:
        if any(directory.glob(pattern)):
            return True
    return False


def _classify_run_root(directory: Path) -> str | None:
    manifests = _manifest_files(directory)
    if len(manifests) == 1:
        return "manifest"
    if len(manifests) > 1:
        raise ValueError(
            f"Run discovery integrity failure: multiple manifests found in run root {directory}"
        )
    if _is_legacy_run_root(directory):
        return "legacy"
    return None


def _extract_experiment_gen_from_manifest(directory: Path) -> str:
    manifest_file = _manifest_files(directory)[0]
    try:
        payload = json.loads(manifest_file.read_text(errors="ignore"))
    except Exception:  # noqa: BLE001
        return "unknown"

    candidates = [
        payload.get("experiment_gen"),
        (payload.get("run") or {}).get("experiment_gen"),
        (payload.get("experiment") or {}).get("gen"),
        (payload.get("flags") or {}).get("EXPERIMENT_GEN"),
    ]
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        lowered = candidate.strip().lower()
        if lowered in {"gen1", "g1"}:
            return "gen1"
        if lowered in {"gen2", "g2"}:
            return "gen2"
    return "unknown"


def discover_runs(
    root: Path,
    *,
    recursive: bool = True,
) -> Generator[tuple[Path, str], None, None]:
    """
    Yield ``(run_dir, experiment_gen)`` pairs found under *root*.

    ``experiment_gen`` is inferred only from explicit manifest metadata
    (or ``"unknown"`` when absent/legacy).
    """
    root = Path(root).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Archive root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Archive root is not a directory: {root}")

    yielded: set[Path] = set()

    def _yield_once(path: Path, *, mode: str) -> Generator[tuple[Path, str], None, None]:
        resolved = path.resolve()
        if resolved in yielded:
            return
        yielded.add(resolved)
        if mode == "manifest":
            yield resolved, _extract_experiment_gen_from_manifest(resolved)
            return
        yield resolved, "unknown"

    root_mode = _classify_run_root(root)
    if root_mode:
        # Provenance guarantee: once we identify a run root we do not recurse
        # under it, avoiding nested pseudo-run contamination.
        yield from _yield_once(root, mode=root_mode)
        return

    if not recursive:
        return

    for current_root, dirs, _files in os.walk(root, topdown=True):
        dirs.sort()
        current_path = Path(current_root)
        if current_path == root:
            continue
        mode = _classify_run_root(current_path)
        if not mode:
            continue
        yield from _yield_once(current_path, mode=mode)
        dirs[:] = []
