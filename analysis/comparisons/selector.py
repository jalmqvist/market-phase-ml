"""
analysis/comparisons/selector.py
==================================
Selector uplift comparison: baseline (PhaseAware rule-based) vs dynamic
(XGBoost-gated StrategySelector_Dynamic) on OOS walk-forward folds.

Selector uplift semantics
--------------------------
``baseline_vs_dynamic_comparison__*.csv`` captures the uplift of the
XGBoost-gated *StrategySelector_Dynamic* over the rule-based *PhaseAware*
baseline:

* Positive Sharpe_Delta → ML routing outperforms hand-coded regime policy.
* Negative Sharpe_Delta → ML routing underperforms; baseline preferred.

The comparison is evaluated *per pair* and in *aggregate* (mean over pairs).
Per-fold breakdown (from ``walkforward_results_per_fold``) can reveal
whether uplift is consistent across time or concentrated in specific periods.
"""

from __future__ import annotations

from typing import Any


def compare_selector_uplift(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate selector uplift across all provided run summaries.

    Parameters
    ----------
    summaries:
        List of normalised summary dicts.

    Returns
    -------
    dict with keys:
        ``per_run``  — list of {run_id, rows} where rows come from
                       ``selector_comparison`` CSV section
        ``aggregate``— {pair → {metric → mean_delta}} across all runs
        ``warnings`` — list of warning strings
    """
    warnings: list[str] = []
    per_run: list[dict[str, Any]] = []
    pair_accum: dict[str, dict[str, list[float]]] = {}

    if not summaries:
        warnings.append("No run summaries provided; selector comparison is empty.")

    for summary in summaries:
        run_id = summary.get("run_id", "unknown")
        rows = (summary.get("csvs") or {}).get("selector_comparison") or []
        if not rows:
            # Fall back to walkforward_summary delta columns if available
            rows = _extract_from_walkforward(summary)
        if not rows:
            warnings.append(f"Run {run_id}: no selector comparison data found.")

        per_run.append({"run_id": run_id, "rows": rows})

        for row in rows:
            pair = row.get("Pair") or row.get("pair") or "unknown"
            for metric in ("Sharpe_Delta", "Return_Delta", "MaxDD_Delta"):
                val = row.get(metric)
                if val is None:
                    continue
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                pair_accum.setdefault(pair, {}).setdefault(metric, []).append(fval)

    aggregate = {
        pair: {metric: sum(vals) / len(vals) for metric, vals in metrics.items()}
        for pair, metrics in pair_accum.items()
    }

    return {
        "per_run": per_run,
        "aggregate": aggregate,
        "warnings": warnings,
    }


def _extract_from_walkforward(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract delta columns from walkforward_summary as a fallback when no
    dedicated selector comparison CSV is available.
    """
    rows = (summary.get("csvs") or {}).get("walkforward_summary") or []
    out = []
    for row in rows:
        delta_row: dict[str, Any] = {}
        pair = row.get("Pair") or row.get("pair")
        if pair:
            delta_row["pair"] = pair
        for col in ("Sharpe_Delta", "Return_Delta", "MaxDD_Delta"):
            if row.get(col) is not None:
                delta_row[col] = row[col]
        if len(delta_row) > 1:  # more than just 'pair'
            out.append(delta_row)
    return out
