"""
analysis/comparisons/dl_conditional.py
=======================================
DL-conditioned metric aggregation for V5 dataset analysis.

Purpose
-------
Enable analysis of selector behaviour conditioned on DL overlap state —
without rerunning or altering the existing experiment matrix.

Conditions analysed
-------------------
``full``
    All folds, all pairs (full OOS period).
``dl_active``
    Folds estimated to fall within the DL overlap window.
``dl_missing``
    Folds estimated to fall outside the DL overlap window.
``transition``
    Folds at the DL active/missing boundary (enter + exit transitions).

Temporal heuristic
------------------
V5 fold CSV rows do not carry explicit timestamps, so DL state assignment
uses a positional heuristic anchored to the ``overlap_fold_coverage_pct``
field from the coverage block.  The last ``K`` folds (by index) per pair are
classified as DL_ACTIVE, where ``K = round(N * overlap_fraction)``.  This
reflects the walk-forward temporal ordering assumption: later folds correspond
to more recent calendar periods.

When ``selector_state_timeline.csv`` is present, richer per-bar metrics become
available (switch density, true occupancy entropy, confidence collapse).  The
pipeline automatically uses timeline data when available.

Backwards compatibility
-----------------------
All functions accept ``None`` or empty inputs and return scaffold dicts with
``data_available=False``.  Existing pipeline outputs are unaffected when
``--conditional-analysis`` is not passed.

Design goal
-----------
The purpose is understanding WHEN and HOW DL changes selector behaviour:

- Does DL reduce selector entropy?
- Does DL stabilise routing?
- Are DL effects concentrated into sparse conditional windows?
- Are reactive failures transition-dominated?
- Does selector geometry change materially during DL-active periods?
"""

from __future__ import annotations

import math
from typing import Any

from analysis.diagnostics.selector_diagnostics import (
    compute_selector_entropy,
    compute_selector_entropy_per_pair,
    compute_switch_density,
    compute_switch_density_conditioned,
    compute_confidence_collapse_metrics,
)
from analysis.diagnostics.transition_windows import (
    classify_folds_dl_state,
    classify_timeline_dl_state,
    extract_transition_windows,
    summarize_transition_windows,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_dl_conditional_analysis(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build DL-conditioned analysis across all provided run summaries.

    Parameters
    ----------
    summaries:
        List of normalised summary dicts from ``analysis.pipeline.build_run_summary``.

    Returns
    -------
    dict with keys:
        ``per_run``          : list of per-run conditional analysis dicts
        ``aggregate_table``  : list of row dicts suitable for report rendering
        ``warnings``         : list of warning strings
        ``data_available``   : bool — True when at least one run has fold data
        ``metadata``         : provenance metadata, including ``dl_state_assignment_method``
                               (``"heuristic_fold_position"``, ``"timeline_exact"``, or
                               ``"unknown"``).  When runs differ, the most conservative
                               method (``"heuristic_fold_position"``) is reported.
    """
    warnings: list[str] = []
    per_run: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []

    if not summaries:
        warnings.append("No summaries provided; DL conditional analysis is empty.")
        return {
            "per_run": [],
            "aggregate_table": [],
            "warnings": warnings,
            "data_available": False,
            "metadata": {"dl_state_assignment_method": "unknown"},
        }

    any_data = False
    run_methods: list[str] = []

    for summary in summaries:
        run_id = summary.get("run_id", "unknown")
        result = _analyse_run(summary)
        if result.get("data_available"):
            any_data = True
        if result.get("warnings"):
            for w in result["warnings"]:
                warnings.append(f"[{run_id}] {w}")

        run_methods.append(result.get("dl_state_assignment_method", "unknown"))
        per_run.append({"run_id": run_id, **result})

        # Build aggregate table rows.
        for window_label, cond in result.get("conditional_metrics", {}).items():
            row: dict[str, Any] = {
                "run_id": run_id,
                "window": window_label,
                "sharpe_dynamic": cond.get("sharpe_dynamic"),
                "sharpe_delta": cond.get("sharpe_delta"),
                "return_delta": cond.get("return_delta"),
                "maxdd_delta": cond.get("maxdd_delta"),
                "n_folds": cond.get("n_folds"),
                "selector_entropy": cond.get("selector_entropy"),
                "normalized_entropy": cond.get("normalized_entropy"),
                "switches_per_1000_bars": cond.get("switches_per_1000_bars"),
                "data_available": cond.get("data_available", False),
            }
            aggregate_rows.append(row)

    # Derive the aggregate assignment method.  Priority order (highest first):
    # timeline_exact > per_fold_timestamp_overlap > heuristic_fold_position > unknown.
    # Runs with no data ("unknown") are excluded from promotion to a better method.
    if not run_methods:
        aggregate_method = "unknown"
    elif any(m == "heuristic_fold_position" for m in run_methods):
        aggregate_method = "heuristic_fold_position"
    elif any(m == "per_fold_timestamp_overlap" for m in run_methods):
        aggregate_method = "per_fold_timestamp_overlap"
    elif any(m == "timeline_exact" for m in run_methods):
        aggregate_method = "timeline_exact"
    else:
        aggregate_method = "unknown"

    return {
        "per_run": per_run,
        "aggregate_table": aggregate_rows,
        "warnings": warnings,
        "data_available": any_data,
        "metadata": {"dl_state_assignment_method": aggregate_method},
    }


# ---------------------------------------------------------------------------
# Per-run analysis
# ---------------------------------------------------------------------------


def _analyse_run(summary: dict[str, Any]) -> dict[str, Any]:
    """Build DL conditional analysis for a single run summary."""
    warnings: list[str] = []
    run_id = summary.get("run_id", "unknown")
    csvs = summary.get("csvs") or {}
    coverage = summary.get("coverage") or {}

    fold_rows = csvs.get("walkforward_per_fold") or []
    timeline_rows = csvs.get("selector_state_timeline") or []
    overlap_window = coverage.get("overlap_window") or {}

    has_fold_data = bool(fold_rows)
    has_timeline_data = bool(timeline_rows)

    if not has_fold_data and not has_timeline_data:
        warnings.append("No walkforward_per_fold or selector_state_timeline data.")
        return {
            "data_available": False,
            "has_fold_data": False,
            "has_timeline_data": False,
            "overlap_fold_coverage_pct": None,
            "dl_state_assignment_method": "unknown",
            "conditional_metrics": {},
            "selector_entropy": _empty_entropy_result(),
            "switch_density": _empty_switch_density(),
            "confidence_collapse": _empty_confidence_collapse(),
            "transition_summary": _empty_transition_summary(),
            "warnings": warnings,
        }

    # --- Derive aggregate overlap_pct ---
    # Prefer per-fold dl_overlap_pct column (exact, from timestamps); fall
    # back to the pipeline-computed aggregate stored in coverage.overlap_window.
    has_per_fold_overlap = has_fold_data and any(
        r.get("dl_overlap_pct") is not None for r in fold_rows
    )
    if has_per_fold_overlap:
        _active = sum(1 for r in fold_rows if r.get("dl_overlap_state") in ("active", "partial"))
        _total = len(fold_rows)
        overlap_pct: float | None = round(100.0 * _active / _total, 2) if _total else None
    else:
        overlap_pct = overlap_window.get("overlap_fold_coverage_pct")

    # --- Fold-level DL state classification ---
    classified_folds = classify_folds_dl_state(
        fold_rows,
        overlap_fold_coverage_pct=overlap_pct,
    )

    if overlap_pct is None and not has_per_fold_overlap:
        warnings.append(
            "overlap_fold_coverage_pct not available; folds labelled dl_state_unknown. "
            "DL conditional slicing will be unavailable for this run."
        )

    # --- Per-condition fold subsets ---
    all_folds = classified_folds
    active_folds = [r for r in classified_folds if r.get("dl_active")]
    missing_folds = [r for r in classified_folds if not r.get("dl_active")
                     and r.get("dl_state") != "dl_state_unknown"]
    transition_folds = extract_transition_windows(classified_folds, n_before=1, n_after=1)

    # --- Timeline-level DL state classification (if available) ---
    classified_timeline: list[dict[str, Any]] = []
    if has_timeline_data:
        classified_timeline = classify_timeline_dl_state(timeline_rows)
    timeline_active = [r for r in classified_timeline if r.get("dl_state") in ("dl_active", "dl_transition_enter")]
    timeline_missing = [r for r in classified_timeline if r.get("dl_state") in ("dl_missing",)]

    # --- Conditional metrics ---
    conditional_metrics: dict[str, dict[str, Any]] = {
        "full": _aggregate_fold_metrics(all_folds),
        "dl_active": _aggregate_fold_metrics(active_folds),
        "dl_missing": _aggregate_fold_metrics(missing_folds),
        "transition": summarize_transition_windows(transition_folds),
    }

    # Enrich with entropy.
    for window_label, folds_for_window in [
        ("full", all_folds),
        ("dl_active", active_folds),
        ("dl_missing", missing_folds),
        ("transition", transition_folds),
    ]:
        entropy_result = compute_selector_entropy(folds_for_window)
        conditional_metrics[window_label]["selector_entropy"] = entropy_result.get("selector_entropy")
        conditional_metrics[window_label]["normalized_entropy"] = entropy_result.get("normalized_entropy")
        conditional_metrics[window_label]["occupancy_concentration"] = entropy_result.get("occupancy_concentration")
        conditional_metrics[window_label]["fold_sharpe_std"] = entropy_result.get("fold_sharpe_std")

    # Enrich with timeline-level switch density (if timeline available).
    if has_timeline_data:
        switch_cond = compute_switch_density_conditioned(classified_timeline)
        for window_label, switch_key in [
            ("full", "full"),
            ("dl_active", "dl_active"),
            ("dl_missing", "dl_missing"),
        ]:
            sw = switch_cond.get(switch_key, {})
            conditional_metrics[window_label]["switches_per_1000_bars"] = sw.get("switches_per_1000_bars")
            conditional_metrics[window_label]["total_switches"] = sw.get("total_switches")

        # Transition window switch density.
        transition_timeline = [
            r for r in classified_timeline
            if r.get("dl_state") in ("dl_transition_enter", "dl_transition_exit")
        ]
        transition_sw = compute_switch_density(transition_timeline)
        conditional_metrics["transition"]["switches_per_1000_bars"] = transition_sw.get("switches_per_1000_bars")
    else:
        for window_label in ("full", "dl_active", "dl_missing", "transition"):
            conditional_metrics[window_label]["switches_per_1000_bars"] = None

    # --- Full diagnostics ---
    entropy_full = compute_selector_entropy(all_folds)
    entropy_per_pair = compute_selector_entropy_per_pair(all_folds)
    switch_density_full = compute_switch_density(classified_timeline) if has_timeline_data else _empty_switch_density()
    confidence_collapse = compute_confidence_collapse_metrics(classified_timeline) if has_timeline_data else _empty_confidence_collapse()
    transition_summary = summarize_transition_windows(transition_folds)

    return {
        "data_available": True,
        "has_fold_data": has_fold_data,
        "has_timeline_data": has_timeline_data,
        "overlap_fold_coverage_pct": overlap_pct,
        "dl_state_assignment_method": (
            "timeline_exact" if has_timeline_data
            else "per_fold_timestamp_overlap" if has_per_fold_overlap
            else "heuristic_fold_position"
        ),
        "n_folds_total": len(all_folds),
        "n_folds_dl_active": len(active_folds),
        "n_folds_dl_missing": len(missing_folds),
        "n_transition_rows": len(transition_folds),
        "conditional_metrics": conditional_metrics,
        "selector_entropy": entropy_full,
        "selector_entropy_per_pair": entropy_per_pair,
        "switch_density": switch_density_full,
        "confidence_collapse": confidence_collapse,
        "transition_summary": transition_summary,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Metric aggregation helpers
# ---------------------------------------------------------------------------


def _aggregate_fold_metrics(
    fold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute mean Sharpe/Return/MaxDD metrics for a set of fold rows.

    Parameters
    ----------
    fold_rows:
        Classified fold rows (from ``classify_folds_dl_state``).

    Returns
    -------
    dict with per-metric means and supporting counts.
    """
    if not fold_rows:
        return {
            "sharpe_dynamic": None,
            "sharpe_baseline": None,
            "sharpe_delta": None,
            "return_dynamic": None,
            "return_baseline": None,
            "return_delta": None,
            "maxdd_dynamic": None,
            "maxdd_baseline": None,
            "maxdd_delta": None,
            "n_folds": 0,
            "data_available": False,
        }

    accum: dict[str, list[float]] = {}
    csv_to_key = {
        "Sharpe_Dynamic": "sharpe_dynamic",
        "Sharpe_Baseline": "sharpe_baseline",
        "Sharpe_Delta": "sharpe_delta",
        "Return_Dynamic": "return_dynamic",
        "Return_Baseline": "return_baseline",
        "Return_Delta": "return_delta",
        "MaxDD_Dynamic": "maxdd_dynamic",
        "MaxDD_Baseline": "maxdd_baseline",
        "MaxDD_Delta": "maxdd_delta",
    }
    for row in fold_rows:
        for csv_col, key in csv_to_key.items():
            val = row.get(csv_col)
            if val is None:
                continue
            try:
                accum.setdefault(key, []).append(float(val))
            except (TypeError, ValueError):
                continue

    result: dict[str, Any] = {
        key: (round(sum(vals) / len(vals), 4) if vals else None)
        for key, vals in accum.items()
    }
    # Fill any missing keys with None.
    for key in csv_to_key.values():
        result.setdefault(key, None)

    result["n_folds"] = len(fold_rows)
    result["data_available"] = bool(accum)
    return result


# ---------------------------------------------------------------------------
# Empty scaffold dicts (used when data is unavailable)
# ---------------------------------------------------------------------------


def _empty_entropy_result() -> dict[str, Any]:
    return {
        "selector_entropy": None,
        "normalized_entropy": None,
        "occupancy_concentration": None,
        "fold_sharpe_std": None,
        "n_folds": 0,
        "outcome_counts": {},
        "data_available": False,
        "requires_timeline": False,
    }


def _empty_switch_density() -> dict[str, Any]:
    return {
        "switches_per_1000_bars": None,
        "mean_hold_duration": None,
        "median_hold_duration": None,
        "total_bars": 0,
        "total_switches": 0,
        "data_available": False,
        "requires_timeline": True,
    }


def _empty_confidence_collapse() -> dict[str, Any]:
    return {
        "confidence_collapse_count": None,
        "mean_confidence_recovery_time": None,
        "fallback_entry_rate": None,
        "fallback_exit_rate": None,
        "total_bars": 0,
        "data_available": False,
        "requires_timeline": True,
    }


def _empty_transition_summary() -> dict[str, Any]:
    return {
        "n_transition_rows": 0,
        "n_windows": 0,
        "metrics": {},
        "data_available": False,
    }
