"""
analysis/comparisons/gen_comparison.py
========================================
Gen1 vs Gen2 comparison: missing-indicator semantics.
"""

from __future__ import annotations

from typing import Any


def compare_gen1_gen2(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compare Gen1 vs Gen2 with strict variant cohorts:
    * sentiment_on  -> A vs C
    * sentiment_off -> B vs D
    """
    variants: dict[str, list[dict[str, Any]]] = {v: [] for v in ("A", "B", "C", "D")}
    unknown: list[str] = []
    for summary in summaries:
        experiment = ((summary.get("meta") or {}).get("experiment") or {})
        variant = (experiment.get("variant") or "U").upper()
        if variant in variants:
            variants[variant].append(summary)
        else:
            unknown.append(summary.get("run_id", "unknown"))

    warnings: list[str] = []
    if unknown:
        warnings.append(
            "Skipped runs with unresolved semantics (variant U): " + ", ".join(sorted(unknown))
        )

    delta_table: list[dict[str, Any]] = []
    valid_comparisons: list[str] = []
    incomplete_comparisons: list[str] = []
    invalid_comparisons: list[str] = []

    if variants["A"] and variants["C"]:
        delta_table.extend(_build_gen_delta_table(variants["A"], variants["C"], cohort="sentiment_on"))
        valid_comparisons.append("sentiment_on:A_vs_C")
    else:
        incomplete_comparisons.append("sentiment_on:A_vs_C")
        missing = [v for v in ("A", "C") if not variants[v]]
        warnings.append(
            "Gen comparison invalid for sentiment ON: missing variant(s) "
            + ", ".join(missing)
            + " (need A and C)."
        )

    if variants["B"] and variants["D"]:
        delta_table.extend(_build_gen_delta_table(variants["B"], variants["D"], cohort="sentiment_off"))
        valid_comparisons.append("sentiment_off:B_vs_D")
    else:
        incomplete_comparisons.append("sentiment_off:B_vs_D")
        missing = [v for v in ("B", "D") if not variants[v]]
        warnings.append(
            "Gen comparison invalid for sentiment OFF: missing variant(s) "
            + ", ".join(missing)
            + " (need B and D)."
        )

    if not valid_comparisons:
        invalid_comparisons.append("gen_matrix")

    gen1_runs = [s["run_id"] for s in variants["A"] + variants["B"]]
    gen2_runs = [s["run_id"] for s in variants["C"] + variants["D"]]
    coverage_comparison = _build_coverage_comparison(
        variants["A"] + variants["B"],
        variants["C"] + variants["D"],
    )

    return {
        "gen1": gen1_runs,
        "gen2": gen2_runs,
        "delta_table": delta_table,
        "coverage_comparison": coverage_comparison,
        "warnings": warnings,
        "cohorts": {
            "sentiment_on": {"gen1": [s["run_id"] for s in variants["A"]], "gen2": [s["run_id"] for s in variants["C"]]},
            "sentiment_off": {"gen1": [s["run_id"] for s in variants["B"]], "gen2": [s["run_id"] for s in variants["D"]]},
        },
        "matrix": {
            "expected_variants": ["A", "B", "C", "D"],
            "present_variants": [v for v in ("A", "B", "C", "D") if variants[v]],
        },
        "valid_comparisons": valid_comparisons,
        "incomplete_comparisons": incomplete_comparisons,
        "invalid_comparisons": invalid_comparisons,
    }


_METRICS = [
    ("walkforward_summary", "Sharpe_Delta", "sharpe_delta"),
    ("walkforward_summary", "Return_Delta", "return_delta"),
    ("walkforward_summary", "MaxDD_Delta", "maxdd_delta"),
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
        pair: {
            metric: sum(metrics[metric]) / len(metrics[metric])
            for metric in sorted(metrics)
        }
        for pair, metrics in sorted(accum.items())
    }


def _build_gen_delta_table(
    gen1_runs: list[dict[str, Any]],
    gen2_runs: list[dict[str, Any]],
    *,
    cohort: str,
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
                "cohort": cohort,
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
    def avg_coverage(runs: list[dict[str, Any]]) -> dict[str, float]:
        accum: dict[str, list[float]] = {}
        for summary in runs:
            log = summary.get("log") or {}
            dl_cov = log.get("dl_coverage") or {}
            for pair, cov in dl_cov.items():
                accum.setdefault(pair, []).append(float(cov))
        return {
            pair: sum(accum[pair]) / len(accum[pair])
            for pair in sorted(accum)
        }

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
