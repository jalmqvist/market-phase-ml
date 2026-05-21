"""
analysis/validation.py
=======================
Integrity validation for analysis framework v2.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def sort_summaries_deterministically(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        summaries,
        key=lambda s: (
            (s.get("meta") or {}).get("experiment_gen") or "zzz",
            (s.get("meta") or {}).get("run_variant") or "Z",
            (s.get("meta") or {}).get("timestamp_utc") or "zzz",
            (s.get("meta") or {}).get("archive_relpath") or "zzz",
            s.get("run_id") or "zzz",
        ),
    )


def validate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    provenance_errors: list[str] = []
    provenance_warnings: list[str] = []
    semantic_errors: list[str] = []
    semantic_warnings: list[str] = []
    manifest_errors: list[str] = []
    manifest_warnings: list[str] = []

    run_ids = [s.get("run_id") for s in summaries if s.get("run_id")]
    for rid, count in Counter(run_ids).items():
        if count > 1:
            provenance_errors.append(f"Duplicate canonical run_id detected: {rid} (count={count})")

    semantic_ids = [
        (s.get("meta") or {}).get("semantic_run_id")
        for s in summaries
        if (s.get("meta") or {}).get("semantic_run_id")
    ]
    for sid, count in Counter(semantic_ids).items():
        if count > 1:
            provenance_errors.append(
                f"Duplicate semantic_run_id detected: {sid} (count={count}) "
                "(semantic identity must be unique per archive run root)."
            )

    for summary in summaries:
        run_id = summary.get("run_id", "unknown")
        meta = summary.get("meta") or {}
        csvs = summary.get("csvs") or {}
        manifest_diag = meta.get("manifest_diagnostics") or {}

        manifest_count = manifest_diag.get("manifest_count") or 0
        if manifest_count > 1:
            manifest_errors.append(
                f"{run_id}: multiple manifests discovered in one run directory (count={manifest_count})."
            )
        if manifest_count == 0:
            manifest_warnings.append(
                f"{run_id}: missing manifest (legacy mode); semantics may be incomplete."
            )

        manifest_timestamp = manifest_diag.get("manifest_timestamp")
        identity_timestamp = meta.get("timestamp_utc")
        if manifest_timestamp and identity_timestamp and manifest_timestamp != identity_timestamp:
            manifest_errors.append(
                f"{run_id}: conflicting manifest timestamp ({manifest_timestamp}) vs identity timestamp ({identity_timestamp})."
            )

        files_found = meta.get("files_found") or []
        if not meta.get("manifest_present") and not files_found and not summary.get("log"):
            provenance_errors.append(f"{run_id}: malformed archive (no manifest, no CSV, no log).")

        variant = meta.get("run_variant") or "U"
        gen = meta.get("experiment_gen") or "unknown"
        if gen == "gen1" and variant in {"C", "D"}:
            semantic_errors.append(f"{run_id}: semantic conflict (gen1 run cannot use variant {variant}).")
        if gen == "gen2" and variant in {"A", "B"}:
            semantic_errors.append(f"{run_id}: semantic conflict (gen2 run cannot use variant {variant}).")
        if variant == "U":
            semantic_warnings.append(f"{run_id}: variant unresolved (U); semantic comparisons may be skipped.")

        for warning in meta.get("identity_warnings") or []:
            if "conflict" in warning.lower():
                semantic_errors.append(f"{run_id}: {warning}")
            else:
                provenance_warnings.append(f"{run_id}: {warning}")

        expected_markers = {"walkforward_summary", "walkforward_per_fold"}
        missing = sorted(marker for marker in expected_markers if not csvs.get(marker))
        if missing:
            provenance_warnings.append(
                f"{run_id}: missing optional CSV sections: {', '.join(missing)}."
            )
        if not summary.get("log"):
            provenance_warnings.append(f"{run_id}: log file absent.")

    if not summaries:
        provenance_errors.append("No summaries available after discovery/parsing.")

    errors = provenance_errors + semantic_errors + manifest_errors
    warnings = provenance_warnings + semantic_warnings + manifest_warnings
    diagnostics = [
        f"runs_total={len(summaries)}",
        f"errors={len(errors)}",
        f"warnings={len(warnings)}",
    ]

    return {
        "errors": errors,
        "warnings": warnings,
        "diagnostics": diagnostics,
        "sections": {
            "provenance_integrity": {
                "errors": provenance_errors,
                "warnings": provenance_warnings,
            },
            "semantic_integrity": {
                "errors": semantic_errors,
                "warnings": semantic_warnings,
            },
            "manifest_integrity": {
                "errors": manifest_errors,
                "warnings": manifest_warnings,
            },
        },
    }
