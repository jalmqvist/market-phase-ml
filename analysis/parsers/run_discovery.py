"""
analysis/parsers/run_discovery.py
==================================
Discover MPML run directories inside an archive root.

A "run directory" is any directory that contains at least one
``run_manifest_*.json`` file or at least one ``results_ml*.csv`` /
``walkforward_results*.csv`` file.  This makes discovery robust to
partial runs and to the mix of old (v1) and new (v2) output shapes.

Gen1 vs Gen2 semantics
-----------------------
The experiment generation is inferred from the ``run_manifest`` flags
or from the directory / run-id name:

* **Gen1** — missing-indicator semantics OFF.  DL features that are
  absent for a bar are simply dropped; the PhaseAware baseline fires.
* **Gen2** — missing-indicator semantics ON.  A synthetic boolean
  ``dl_missing_indicator`` column signals absence explicitly, so the
  XGBoost gating model can learn a "no-DL" regime policy.

The generation label is advisory: it is written into the summary JSON
under ``meta.experiment_gen`` and is used by the comparison layer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Generator

# Any directory containing these glob patterns is treated as a run root.
_RUN_MARKERS = [
    "run_manifest_*.json",
    "results_ml*.csv",
    "walkforward_results*.csv",
    "results_summary*.csv",
]

# Regex to infer experiment generation from directory / run-id names.
_GEN2_RE = re.compile(r"gen2|g2|missing.ind|missing_ind", re.IGNORECASE)


def _has_run_files(directory: Path) -> bool:
    """Return True if *directory* contains at least one recognised MPML output."""
    for pattern in _RUN_MARKERS:
        if any(directory.glob(pattern)):
            return True
    return False


def _infer_gen(directory: Path) -> str:
    """
    Heuristic: inspect directory name and run_manifest filenames for
    generation hints.  Falls back to ``"gen1"`` (safe default).
    """
    candidate_text = directory.name
    for manifest in directory.glob("run_manifest_*.json"):
        candidate_text += " " + manifest.stem
    if _GEN2_RE.search(candidate_text):
        return "gen2"
    return "gen1"


def discover_runs(
    root: Path,
    *,
    recursive: bool = True,
) -> Generator[tuple[Path, str], None, None]:
    """
    Yield ``(run_dir, experiment_gen)`` pairs found under *root*.

    Parameters
    ----------
    root:
        Top-level archive directory (e.g. ``results_archive/`` or a
        specific run dir such as ``results_archive/fp_gen1_A/``).
    recursive:
        When True (default) descend into sub-directories.  Set to
        False if *root* is itself a single run directory.

    Yields
    ------
    tuple[Path, str]
        ``(run_directory_path, experiment_generation)`` where
        ``experiment_generation`` is ``"gen1"`` or ``"gen2"``.
    """
    root = Path(root).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Archive root does not exist: {root}")

    yielded: set[Path] = set()

    def _yield_once(path: Path) -> Generator[tuple[Path, str], None, None]:
        resolved = path.resolve()
        if resolved in yielded:
            return
        yielded.add(resolved)
        yield resolved, _infer_gen(resolved)

    # If root itself is a run directory, yield it directly.
    if _has_run_files(root):
        yield from _yield_once(root)
        # Also descend in case sub-directories are individual run flavours.

    if not recursive:
        return

    # Walk depth-first; yield any child directory that looks like a run.
    for child in sorted(root.rglob("*")):
        if child.is_dir() and child != root and _has_run_files(child):
            yield from _yield_once(child)
