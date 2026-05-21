"""
analysis/comparisons/sentiment.py
===================================
Sentiment ON/OFF comparison logic.

Sentiment ON/OFF experiment design
------------------------------------
The MPML pipeline can be run in two modes per experiment generation:

* **Sentiment ON** (variants A, C) — ``DL_SIGNALS_ENABLED=true``.
  DL prediction features from the market-sentiment-ml surface are
  attached to each bar.  The XGBoost selector uses them as additional
  input dimensions.
* **Sentiment OFF** (variants B, D) — ``DL_SIGNALS_ENABLED=false``
  (baseline).  Equivalent to a pure regime/feature-based selector with
  no external signal.

Experiment variants:

========= =========== ============================
Variant   Sentiment   Missing-indicator semantics
========= =========== ============================
A         ON          Gen1 (indicator OFF)
B         OFF         Gen1 (indicator OFF)
C         ON          Gen2 (indicator ON)
D         OFF         Gen2 (indicator ON)
========= =========== ============================

This module compares A vs B (Gen1 sentiment effect) and C vs D (Gen2
sentiment effect) on any walkforward / backtest metric present in the
summaries.
"""

from __future__ import annotations

from typing import Any


def compare_sentiment_variants(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compare Sentiment-ON vs Sentiment-OFF runs across summaries.

    A summary is labelled "sentiment_on" if its manifest reports
    ``dl_enabled = True``; otherwise "sentiment_off".

    Parameters
    ----------
    summaries:
        List of normalised summary dicts (as produced by
        ``analysis.pipeline.build_run_summary``).

    Returns
    -------
    dict with keys:
        ``sentiment_on``  — subset with DL enabled
        ``sentiment_off`` — subset without DL
        ``delta_table``   — list of {pair, metric, on_val, off_val, delta}
                           for each pair × metric combination found in
                           both subsets
        ``warnings``      — list of warning strings
    """
    on_runs = [s for s in summaries if s.get("meta", {}).get("dl_enabled")]
    off_runs = [s for s in summaries if not s.get("meta", {}).get("dl_enabled")]

    warnings: list[str] = []
    if not on_runs:
        warnings.append("No sentiment-ON runs found; delta table will be empty.")
    if not off_runs:
        warnings.append("No sentiment-OFF runs found; delta table will be empty.")

    delta_table = _build_delta_table(on_runs, off_runs)

    return {
        "sentiment_on": [s["run_id"] for s in on_runs],
        "sentiment_off": [s["run_id"] for s in off_runs],
        "delta_table": delta_table,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WF_METRICS = [
    ("walkforward_summary", "Sharpe_Dynamic", "sharpe_dynamic"),
    ("walkforward_summary", "Sharpe_Delta",   "sharpe_delta"),
    ("walkforward_summary", "Return_Delta",   "return_delta"),
    ("walkforward_summary", "MaxDD_Delta",    "maxdd_delta"),
]


def _extract_pair_metrics(
    summaries: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """
    Return ``{pair → {metric → avg_value}}`` across all summaries.

    Values are averaged over runs when multiple runs share a pair.
    """
    accum: dict[str, dict[str, list[float]]] = {}

    for summary in summaries:
        csvs = summary.get("csvs") or {}
        for section_key, csv_col, metric_name in _WF_METRICS:
            rows = csvs.get(section_key) or []
            for row in rows:
                pair = row.get("Pair") or row.get("pair") or "unknown"
                val = row.get(csv_col)
                if val is None:
                    continue
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                accum.setdefault(pair, {}).setdefault(metric_name, []).append(fval)

    return {
        pair: {metric: sum(vals) / len(vals) for metric, vals in metrics.items()}
        for pair, metrics in accum.items()
    }


def _build_delta_table(
    on_runs: list[dict[str, Any]],
    off_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build per-pair, per-metric delta rows (ON − OFF)."""
    on_metrics = _extract_pair_metrics(on_runs)
    off_metrics = _extract_pair_metrics(off_runs)

    pairs = sorted(set(on_metrics) | set(off_metrics))
    rows = []
    for pair in pairs:
        on_pair = on_metrics.get(pair, {})
        off_pair = off_metrics.get(pair, {})
        all_metrics = sorted(set(on_pair) | set(off_pair))
        for metric in all_metrics:
            on_val = on_pair.get(metric)
            off_val = off_pair.get(metric)
            delta = (on_val - off_val) if (on_val is not None and off_val is not None) else None
            rows.append({
                "pair": pair,
                "metric": metric,
                "sentiment_on": on_val,
                "sentiment_off": off_val,
                "delta_on_minus_off": delta,
            })
    return rows
