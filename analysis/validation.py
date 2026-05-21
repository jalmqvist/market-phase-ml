"""
analysis/validation.py
=======================
Integrity validation for analysis framework v2.
"""

from __future__ import annotations

from collections import Counter, defaultdict
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
    errors: list[str] = []
    warnings: list[str] = []
    diagnostics: list[str] = []

    run_ids = [s.get("run_id") for s in summaries if s.get("run_id")]
    semantic_ids = [(s.get("meta") or {}).get("semantic_run_id") for s in summaries if (s.get("meta") or {}).get("semantic_run_id")]
    timestamp_ids = [
        (
            (s.get("meta") or {}).get("timestamp_utc"),
            (s.get("meta") or {}).get("archive_relpath"),
            s.get("run_id"),
        )
        for s in summaries
    ]

    for rid, count in Counter(run_ids).items():
        if count > 1:
            errors.append(f"Duplicate canonical run_id detected: {rid} (count={count})")

    for sid, count in Counter(semantic_ids).items():
        if count > 1:
            warnings.append(
                f"Duplicate semantic_run_id detected across archive directories: {sid} (count={count}). "
                "Canonical IDs remain unique via archive path suffix."
            )

    by_ts: dict[str, list[tuple[str | None, str | None]]] = defaultdict(list)
    for ts, relpath, rid in timestamp_ids:
        if ts:
            by_ts[ts].append((relpath, rid))
    for ts, entries in by_ts.items():
        if len(entries) > 1:
            unique_dirs = sorted({e[0] for e in entries if e[0]})
            if len(unique_dirs) > 1:
                warnings.append(
                    f"Timestamp reused across multiple run directories ({ts}): "
                    + ", ".join(unique_dirs)
                )

    # Per-run structural checks
    gen_variant_map: dict[str, set[str]] = defaultdict(set)
    gen_dl_map: dict[str, set[bool]] = defaultdict(set)
    for s in summaries:
        meta = s.get("meta") or {}
        run_id = s.get("run_id", "unknown")
        variant = meta.get("run_variant") or "U"
        gen = meta.get("experiment_gen") or "unknown"
        gen_variant_map[gen].add(variant)
        if meta.get("dl_enabled") is not None:
            gen_dl_map[gen].add(bool(meta.get("dl_enabled")))

        for w in meta.get("identity_warnings") or []:
            warnings.append(f"{run_id}: {w}")

        manifest_diag = meta.get("manifest_diagnostics") or {}
        for pe in manifest_diag.get("parse_errors") or []:
            errors.append(f"{run_id}: malformed manifest ({pe})")

        manifest_timestamps = manifest_diag.get("timestamps") or []
        if len(set(manifest_timestamps)) > 1:
            warnings.append(
                f"{run_id}: conflicting manifest timestamps detected: "
                + ", ".join(sorted(set(manifest_timestamps)))
            )

        files_found = set(meta.get("files_found") or [])
        if not files_found:
            warnings.append(f"{run_id}: no recognised CSV files discovered.")

        expected_markers = {
            "walkforward_summary",
            "walkforward_per_fold",
        }
        csvs = s.get("csvs") or {}
        missing = sorted(k for k in expected_markers if not csvs.get(k))
        if missing:
            warnings.append(f"{run_id}: missing key CSV sections: {', '.join(missing)}")

        if not meta.get("manifest_present") and not files_found and not s.get("log"):
            errors.append(f"{run_id}: malformed archive (no manifest, no CSV, no log).")

    # Cross-run structure checks for A/B/C/D assumptions
    if "gen1" in gen_variant_map:
        if "A" not in gen_variant_map["gen1"] or "B" not in gen_variant_map["gen1"]:
            warnings.append("Gen1 comparison incomplete: expected both A and B variants.")
    if "gen2" in gen_variant_map:
        if "C" not in gen_variant_map["gen2"] or "D" not in gen_variant_map["gen2"]:
            warnings.append("Gen2 comparison incomplete: expected both C and D variants.")

    # Gen label consistency sanity checks
    for gen, variants in gen_variant_map.items():
        if gen == "gen1" and any(v in {"C", "D"} for v in variants):
            warnings.append("Inconsistent gen labelling: gen1 group contains C/D variants.")
        if gen == "gen2" and any(v in {"A", "B"} for v in variants):
            warnings.append("Inconsistent gen labelling: gen2 group contains A/B variants.")

    if not summaries:
        errors.append("No summaries available after discovery/parsing.")

    diagnostics.append(f"runs_total={len(summaries)}")
    diagnostics.append(f"errors={len(errors)}")
    diagnostics.append(f"warnings={len(warnings)}")
    return {"errors": errors, "warnings": warnings, "diagnostics": diagnostics}

