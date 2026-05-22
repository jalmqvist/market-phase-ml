"""
analysis/comparisons/sentiment.py
===================================
Sentiment ON/OFF comparison logic.
"""

from __future__ import annotations

from typing import Any


def compare_sentiment_variants(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compare sentiment cohorts using canonical A/B/C/D variant semantics.
    """
    warnings: list[str] = []
    variants: dict[str, list[dict[str, Any]]] = {v: [] for v in ("A", "B", "C", "D")}
    unknown: list[str] = []

    for summary in summaries:
        variant = ((summary.get("meta") or {}).get("run_variant") or "U").upper()
        if variant in variants:
            variants[variant].append(summary)
        else:
            unknown.append(summary.get("run_id", "unknown"))

    if unknown:
        warnings.append(
            "Skipped runs with unresolved semantics (variant U): " + ", ".join(sorted(unknown))
        )

    matrix = {
        "expected_variants": ["A", "B", "C", "D"],
        "present_variants": [v for v in ("A", "B", "C", "D") if variants[v]],
    }

    delta_table: list[dict[str, Any]] = []
    valid_comparisons: list[str] = []
    incomplete_comparisons: list[str] = []
    invalid_comparisons: list[str] = []

    if variants["A"] and variants["B"]:
        delta_table.extend(_build_delta_table(variants["A"], variants["B"], generation="gen1"))
        valid_comparisons.append("gen1:A_vs_B")
    else:
        incomplete_comparisons.append("gen1:A_vs_B")
        missing = [v for v in ("A", "B") if not variants[v]]
        warnings.append(
            "Gen1 sentiment comparison invalid: missing variant(s) "
            + ", ".join(missing)
            + " (A=ON, B=OFF baseline)."
        )

    if variants["C"] and variants["D"]:
        delta_table.extend(_build_delta_table(variants["C"], variants["D"], generation="gen2"))
        valid_comparisons.append("gen2:C_vs_D")
    else:
        incomplete_comparisons.append("gen2:C_vs_D")
        missing = [v for v in ("C", "D") if not variants[v]]
        warnings.append(
            "Gen2 sentiment comparison invalid: missing variant(s) "
            + ", ".join(missing)
            + " (C=ON, D=OFF baseline)."
        )

    if not valid_comparisons:
        invalid_comparisons.append("sentiment_matrix")

    sentiment_on = [s["run_id"] for s in variants["A"] + variants["C"]]
    sentiment_off = [s["run_id"] for s in variants["B"] + variants["D"]]

    return {
        "sentiment_on": sentiment_on,
        "sentiment_off": sentiment_off,
        "delta_table": delta_table,
        "warnings": warnings,
        "grouped": {
            "gen1": {"on": [s["run_id"] for s in variants["A"]], "off": [s["run_id"] for s in variants["B"]]},
            "gen2": {"on": [s["run_id"] for s in variants["C"]], "off": [s["run_id"] for s in variants["D"]]},
        },
        "matrix": matrix,
        "valid_comparisons": valid_comparisons,
        "incomplete_comparisons": incomplete_comparisons,
        "invalid_comparisons": invalid_comparisons,
    }


_WF_METRICS = [
    ("walkforward_summary", "Sharpe_Dynamic", "sharpe_dynamic"),
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
        pair: {
            metric: sum(metrics[metric]) / len(metrics[metric])
            for metric in sorted(metrics)
        }
        for pair, metrics in sorted(accum.items())
    }


def _build_delta_table(
    on_runs: list[dict[str, Any]],
    off_runs: list[dict[str, Any]],
    *,
    generation: str,
) -> list[dict[str, Any]]:
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
                "generation": generation,
                "pair": pair,
                "metric": metric,
                "sentiment_on": on_val,
                "sentiment_off": off_val,
                "delta_on_minus_off": delta,
            })
    return rows
