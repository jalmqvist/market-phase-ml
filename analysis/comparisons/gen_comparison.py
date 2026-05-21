"""
analysis/comparisons/gen_comparison.py
========================================
Gen1 vs Gen2 comparison: missing-indicator semantics.

Gen1 vs Gen2 semantics
------------------------
The "generation" of an experiment refers to how absent DL features are
handled when the DL prediction surface does not cover a given bar:

* **Gen1** — missing DL features are simply absent.  The XGBoost
  StrategySelector receives ``NaN`` values and falls back to the
  PhaseAware policy via the existing NaN guard.  The model has no
  explicit signal that DL data is unavailable.

* **Gen2** — a synthetic boolean column ``dl_missing_indicator`` is
  added.  When DL features are absent, the indicator is 1; otherwise 0.
  This gives the gating model an explicit "no-DL" regime signal,
  potentially enabling a third learned policy: "act conservatively when
  DL data is unavailable."

The DL prediction surface covers roughly the modern era (~2019+), so
Gen1 and Gen2 may diverge most on pairs or folds where DL coverage is
partial.

Research question: does the missing-indicator improve OOS Sharpe on
pairs where DL coverage is low?  This is the primary Gen1 vs Gen2
investigation.
"""

from __future__ import annotations

from typing import Any


def compare_gen1_gen2(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compare Gen1 (no missing indicator) vs Gen2 (with missing indicator).

    Parameters
    ----------
    summaries:
        List of normalised summary dicts.

    Returns
    -------
    dict with keys:
        ``gen1`` — list of run_ids labelled gen1
        ``gen2`` — list of run_ids labelled gen2
        ``delta_table`` — per-pair delta table (gen2 − gen1)
        ``coverage_comparison`` — per-pair DL coverage from each gen
        ``warnings``
    """
    gen1_runs = [s for s in summaries if s.get("meta", {}).get("experiment_gen") == "gen1"]
    gen2_runs = [s for s in summaries if s.get("meta", {}).get("experiment_gen") == "gen2"]

    warnings: list[str] = []
    if not gen1_runs:
        warnings.append("No Gen1 runs found; Gen1 vs Gen2 delta table will be empty.")
    if not gen2_runs:
        warnings.append("No Gen2 runs found; Gen1 vs Gen2 delta table will be empty.")

    delta_table = _build_gen_delta_table(gen1_runs, gen2_runs)
    coverage_comparison = _build_coverage_comparison(gen1_runs, gen2_runs)

    return {
        "gen1": [s["run_id"] for s in gen1_runs],
        "gen2": [s["run_id"] for s in gen2_runs],
        "delta_table": delta_table,
        "coverage_comparison": coverage_comparison,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_METRICS = [
    ("walkforward_summary", "Sharpe_Delta", "sharpe_delta"),
    ("walkforward_summary", "Return_Delta",  "return_delta"),
    ("walkforward_summary", "MaxDD_Delta",   "maxdd_delta"),
]


def _extract_pair_metrics(
    summaries: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    accum: dict[str, dict[str, list[float]]] = {}
    for summary in summaries:
        csvs = summary.get("csvs") or {}
        for section_key, csv_col, metric_name in _METRICS:
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


def _build_gen_delta_table(
    gen1_runs: list[dict[str, Any]],
    gen2_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    g1 = _extract_pair_metrics(gen1_runs)
    g2 = _extract_pair_metrics(gen2_runs)
    pairs = sorted(set(g1) | set(g2))
    rows = []
    for pair in pairs:
        g1_pair = g1.get(pair, {})
        g2_pair = g2.get(pair, {})
        all_metrics = sorted(set(g1_pair) | set(g2_pair))
        for metric in all_metrics:
            g1_val = g1_pair.get(metric)
            g2_val = g2_pair.get(metric)
            delta = (g2_val - g1_val) if (g1_val is not None and g2_val is not None) else None
            rows.append({
                "pair": pair,
                "metric": metric,
                "gen1": g1_val,
                "gen2": g2_val,
                "delta_gen2_minus_gen1": delta,
            })
    return rows


def _build_coverage_comparison(
    gen1_runs: list[dict[str, Any]],
    gen2_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare average DL coverage per pair across generations."""

    def avg_coverage(runs: list[dict[str, Any]]) -> dict[str, float]:
        accum: dict[str, list[float]] = {}
        for summary in runs:
            log = summary.get("log") or {}
            dl_cov = log.get("dl_coverage") or {}
            for pair, cov in dl_cov.items():
                accum.setdefault(pair, []).append(float(cov))
        return {pair: sum(vals) / len(vals) for pair, vals in accum.items()}

    g1_cov = avg_coverage(gen1_runs)
    g2_cov = avg_coverage(gen2_runs)
    pairs = sorted(set(g1_cov) | set(g2_cov))
    return [
        {
            "pair": pair,
            "dl_coverage_gen1": g1_cov.get(pair),
            "dl_coverage_gen2": g2_cov.get(pair),
        }
        for pair in pairs
    ]
