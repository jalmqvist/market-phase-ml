#!/usr/bin/env python3
"""
analysis/pipeline.py
=====================
Single-command MPML analysis orchestrator (framework v2).

Usage
-----
::

    # Analyse all runs in an archive directory:
    python analysis/pipeline.py results_archive/

    # Analyse a specific run directory:
    python analysis/pipeline.py results_archive/fp_gen1_A/

    # Write outputs to a custom directory:
    python analysis/pipeline.py results_archive/ --output-dir my_reports/

What it does
-------------
1. **Discover** all run directories under the archive root.
2. **Parse** CSV outputs, run manifests, and (as fallback) log files.
3. **Build** a normalised summary JSON per run.
4. **Generate** comparisons: sentiment ON/OFF, Gen1 vs Gen2, selector uplift.
5. **Render** a unified markdown report.
6. **Write** outputs to the ``--output-dir`` directory.

Output files
------------
- ``<output_dir>/summaries/<run_id>.summary.json``  — per-run summary
- ``<output_dir>/comparisons.json``                  — cross-run comparison
- ``<output_dir>/report.md``                         — human-readable report

Architecture
-------------
::

    pipeline.py
        ↓ discover_runs()        (parsers/run_discovery.py)
        ↓ parse_run_csvs()       (parsers/csv_parsers.py)
        ↓ parse_manifest()       (parsers/manifest_parser.py)
        ↓ parse_log()            (parsers/log_parser.py  ← fallback)
        ↓ build_run_summary()    (this file)
        ↓ compare_*()            (comparisons/*.py)
        ↓ render_markdown_report() (reports/markdown_report.py)

Experiment semantics
---------------------
Gen1 vs Gen2:
    Refers to missing-indicator semantics for DL features.
    Gen1 = no indicator; Gen2 = explicit ``dl_missing_indicator`` column.

Sentiment ON/OFF:
    Refers to whether ``DL_SIGNALS_ENABLED=true`` was set for the run.
    Sentiment ON = DL prediction surface attached to each bar.
    OFF = pure regime/feature baseline.

Experiment variants (comparative studies):
    A — Sentiment ON,  Gen1
    B — Sentiment OFF, Gen1  (baseline for A)
    C — Sentiment ON,  Gen2
    D — Sentiment OFF, Gen2  (baseline for C)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Internal imports — all relative to the project root.
# Add the project root to sys.path so this script works when run directly
# from any working directory.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.parsers.run_discovery import discover_runs
from analysis.parsers.csv_parsers import parse_run_csvs
from analysis.parsers.manifest_parser import parse_manifest
from analysis.parsers.run_identity import infer_run_identity
from analysis.parsers.log_parser import parse_log
from analysis.comparisons.sentiment import compare_sentiment_variants
from analysis.comparisons.selector import compare_selector_uplift
from analysis.comparisons.gen_comparison import compare_gen1_gen2
from analysis.reports.markdown_report import render_markdown_report
from analysis.validation import validate_summaries, sort_summaries_deterministically


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------


def build_run_summary(
    run_dir: Path,
    archive_root: Path,
    experiment_gen: str = "gen1",
) -> dict[str, Any]:
    """
    Build a normalised summary dict for a single run directory.

    This is the central data structure consumed by all comparison and
    report modules.  It is also serialised as JSON for archival.

    Parameters
    ----------
    run_dir:
        Path to a single run directory.
    experiment_gen:
        ``"gen1"`` or ``"gen2"`` — inferred by ``discover_runs``.

    Returns
    -------
    dict with the following top-level keys:

    ``run_id``
        Canonical run identifier (from manifest or directory name).
    ``meta``
        Run metadata: dl_enabled, experiment_gen, git_sha, etc.
    ``csvs``
        Parsed CSV sections (may contain None values for absent files).
    ``log``
        Legacy log-parsed data (None if no log file found).
    ``coverage``
        DL coverage summary extracted from available data.
    ``warnings``
        List of non-fatal issues encountered during parsing.
    """
    warnings: list[str] = []

    # --- manifests --------------------------------------------------------
    manifest = parse_manifest(run_dir)
    if not manifest:
        warnings.append(f"No run_manifest_*.json found in {run_dir.name}.")

    identity = infer_run_identity(
        archive_root=archive_root,
        run_dir=run_dir,
        experiment_gen=experiment_gen,
        manifest=manifest,
    )
    run_id = identity["run_id"]
    experiment = (manifest or {}).get("experiment") or {}

    # --- CSVs -------------------------------------------------------------
    csvs = parse_run_csvs(run_dir)
    for err in csvs.pop("_errors", []):
        warnings.append(f"CSV parse error [{err['file']}]: {err['error']}")

    # Report which expected sections are missing
    expected_sections = [
        "ml_accuracy", "backtest", "walkforward_summary", "walkforward_per_pair",
        "walkforward_per_fold", "selector_comparison", "ablation_aggregate",
        "ablation_per_pair", "vol_guard_summary", "vol_guard_per_fold",
        "results_summary", "results_per_pair",
    ]
    missing_sections = [s for s in expected_sections if not csvs.get(s)]
    if missing_sections:
        warnings.append(
            f"Missing CSV sections (partial run or older format): "
            + ", ".join(missing_sections)
        )

    # --- log (fallback) ---------------------------------------------------
    log = parse_log(run_dir)
    if not log and not csvs.get("_files_found"):
        warnings.append(
            "No CSV files and no log file found — directory may be empty or corrupt."
        )

    warnings.extend(identity.get("identity_warnings", []))

    # --- coverage summary -------------------------------------------------
    coverage = _build_coverage_summary(csvs, log, manifest)

    # --- meta -------------------------------------------------------------
    meta = {
        "experiment_gen": experiment.get("generation") or identity.get("experiment_gen") or experiment_gen,
        "run_variant": experiment.get("variant") or identity.get("run_variant"),
        "sentiment_enabled": experiment.get("sentiment_enabled"),
        "missing_indicators_enabled": experiment.get("missing_indicators_enabled"),
        "semantic_label": experiment.get("semantic_label"),
        "experiment": experiment,
        "dl_enabled": (manifest or {}).get("dl_enabled"),
        "dl_surface": (manifest or {}).get("dl_surface"),
        "dl_surface_string": (manifest or {}).get("dl_surface_string"),
        "dl_artifact_path": (manifest or {}).get("dl_artifact_path"),
        "walkforward_params": (manifest or {}).get("walkforward"),
        "flags": (manifest or {}).get("flags"),
        "reproducibility": (manifest or {}).get("reproducibility") or {},
        "feature_ordering": (manifest or {}).get("feature_ordering") or {},
        "git_sha": (manifest or {}).get("git_sha"),
        "timestamp_utc": identity.get("timestamp_utc"),
        "python_version": (manifest or {}).get("python_version"),
        "run_dir": str(run_dir),
        "files_found": csvs.pop("_files_found", []),
        "manifest_present": bool(manifest),
        "legacy_mode": not bool(manifest),
        "manifest_diagnostics": {
            "manifest_count": (manifest or {}).get("manifest_count", 0),
            "manifest_path": (manifest or {}).get("manifest_path"),
            "manifest_timestamp": (manifest or {}).get("timestamp_utc"),
            "manifest_run_id": (manifest or {}).get("run_id"),
            "dl_mode_tag": (manifest or {}).get("dl_mode_tag"),
        },
        "semantic_run_name": identity.get("semantic_run_name"),
        "semantic_run_id": identity.get("semantic_run_id"),
        "run_meaning": identity.get("run_meaning"),
        "archive_relpath": identity.get("archive_relpath"),
        "archive_slug": identity.get("archive_slug"),
        "manifest_run_id": identity.get("manifest_run_id"),
        "identity_warnings": identity.get("identity_warnings"),
    }

    return {
        "run_id": run_id,
        "meta": meta,
        "csvs": csvs,
        "log": log,
        "coverage": coverage,
        "warnings": warnings,
    }


def _build_coverage_summary(
    csvs: dict[str, Any],
    log: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Derive a DL coverage summary from available data sources.

    Priority: vol_guard_summary CSV → log dl_coverage → manifest flags.
    """
    # Try CSV-based vol guard (newer runs)
    vg_rows = csvs.get("vol_guard_summary") or []
    overlap_diag = _build_overlap_window_diagnostics(csvs, log)
    if vg_rows:
        pairs = [r.get("Pair") or r.get("pair") for r in vg_rows if r.get("Pair") or r.get("pair")]
        return {
            "source": "vol_guard_summary_csv",
            "pairs_with_data": pairs,
            "overlap_window": overlap_diag,
        }

    # Fall back to log-parsed coverage
    if log and log.get("dl_coverage"):
        dl_cov = log["dl_coverage"]
        avg = sum(dl_cov.values()) / len(dl_cov) if dl_cov else 0.0
        return {
            "source": "log",
            "per_pair": dl_cov,
            "avg_coverage_pct": round(avg, 2),
            "n_pairs": len(dl_cov),
            "overlap_window": overlap_diag,
        }

    # Try manifest
    if manifest and manifest.get("dl_enabled"):
        return {"source": "manifest", "dl_enabled": True, "per_pair": {}, "overlap_window": overlap_diag}

    return {"source": "none", "dl_enabled": False, "overlap_window": overlap_diag}


def _build_overlap_window_diagnostics(
    csvs: dict[str, Any],
    log: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Foundational overlap-window diagnostics (non-invasive extension points).
    """
    fold_rows = csvs.get("walkforward_per_fold") or []
    dl_cov = (log or {}).get("dl_coverage") or {}
    diagnostics: dict[str, Any] = {
        "fold_rows_present": len(fold_rows),
        "dl_pairs_with_coverage": sorted(dl_cov.keys()),
        "dl_active_years": [],
        "per_year_fold_counts": {},
        "attachment_persistence_note": "Scaffold only; add true per-fold attachment persistence from future exports.",
    }
    years: dict[int, int] = {}
    for row in fold_rows:
        for k, v in row.items():
            if not isinstance(v, str):
                continue
            if "year" in k.lower() or "date" in k.lower() or "time" in k.lower():
                if len(v) >= 4 and v[:4].isdigit():
                    y = int(v[:4])
                    years[y] = years.get(y, 0) + 1
    diagnostics["per_year_fold_counts"] = {str(y): years[y] for y in sorted(years)}
    diagnostics["dl_active_years"] = [str(y) for y in sorted(years) if y >= 2019]
    total = sum(years.values())
    overlap = sum(c for y, c in years.items() if y >= 2019)
    diagnostics["overlap_fold_coverage_pct"] = round((100.0 * overlap / total), 2) if total else None
    return diagnostics


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    archive_root: Path,
    output_dir: Path,
    *,
    verbose: bool = False,
) -> None:
    """
    Full analysis pipeline.

    Parameters
    ----------
    archive_root:
        Root directory containing one or more run directories.
    output_dir:
        Directory where outputs are written.
    verbose:
        Print progress messages.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(exist_ok=True)

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # 1. Discover runs
    _log(f"🔍 Discovering runs under: {archive_root}")
    discovered = list(discover_runs(archive_root))

    if not discovered:
        print(
            f"⚠  No run directories found under '{archive_root}'.\n"
            "   Ensure run directories contain exactly one run_manifest_*.json "
            "or explicit legacy marker (.mpml_legacy_run_root)."
        )
        return

    _log(f"   Found {len(discovered)} run(s).")
    seen_run_dirs: set[Path] = set()
    dedupe_warnings: list[str] = []

    # 2. Parse + build summaries
    summaries: list[dict[str, Any]] = []
    for run_dir, experiment_gen in discovered:
        if run_dir in seen_run_dirs:
            dedupe_warnings.append(f"Duplicate discovered run directory skipped: {run_dir}")
            continue
        seen_run_dirs.add(run_dir)
        _log(f"   Parsing: {run_dir.name}  [gen={experiment_gen}]")
        summary = build_run_summary(run_dir, archive_root, experiment_gen)

        if summary["warnings"]:
            for w in summary["warnings"]:
                _log(f"   ⚠  {w}")

        summaries.append(summary)

    if dedupe_warnings:
        for s in summaries:
            s.setdefault("warnings", []).extend(dedupe_warnings)
            s.setdefault("meta", {}).setdefault("pipeline_diagnostics", {})["dedupe_warnings"] = dedupe_warnings

    summaries = sort_summaries_deterministically(summaries)
    validation = validate_summaries(summaries)
    if validation["errors"]:
        error_text = "\n".join(f"- {e}" for e in validation["errors"])
        raise RuntimeError(f"Structural validation failed:\n{error_text}")

    # Write per-run summary JSON (after successful validation and deterministic ordering)
    for summary in summaries:
        run_id = summary["run_id"]
        summary_path = summaries_dir / f"{run_id}.summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str, sort_keys=True)
        _log(f"   ✓ wrote {summary_path.relative_to(output_dir.parent)}")

    # 3. Generate comparisons
    _log("\n📊 Generating comparisons …")
    comparisons: dict[str, Any] = {}

    comparisons["sentiment"] = compare_sentiment_variants(summaries)
    _log(f"   sentiment: {len(comparisons['sentiment'].get('delta_table', []))} delta rows")

    comparisons["selector"] = compare_selector_uplift(summaries)
    _log(f"   selector: {len(comparisons['selector'].get('aggregate', {}))} pairs")

    comparisons["gen"] = compare_gen1_gen2(summaries)
    _log(f"   gen1vs2: {len(comparisons['gen'].get('delta_table', []))} delta rows")
    comparisons["validation"] = validation

    comparisons_path = output_dir / "comparisons.json"
    with open(comparisons_path, "w") as f:
        json.dump(comparisons, f, indent=2, default=str, sort_keys=True)
    _log(f"   ✓ wrote {comparisons_path.relative_to(output_dir.parent)}")

    # 4. Render markdown report
    _log("\n📝 Rendering report …")
    report_md = render_markdown_report(summaries, comparisons, validation=validation)
    report_path = output_dir / "report.md"
    report_path.write_text(report_md)
    _log(f"   ✓ wrote {report_path.relative_to(output_dir.parent)}")

    # 5. Summary
    print(
        f"\n✅ Analysis complete.\n"
        f"   Runs processed : {len(summaries)}\n"
        f"   Output dir     : {output_dir}\n"
        f"   Report         : {report_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MPML analysis pipeline v2 — single-command experiment analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analysis/pipeline.py results_archive/
  python analysis/pipeline.py results_archive/fp_gen1_A/ --output-dir reports/gen1_A/
  python analysis/pipeline.py results/ --output-dir analysis/output/ --verbose
""",
    )
    parser.add_argument(
        "archive",
        type=Path,
        help="Archive root directory (or specific run directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/output"),
        help="Output directory for reports and summaries (default: analysis/output/).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress messages.",
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            archive_root=args.archive,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
    except FileNotFoundError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
