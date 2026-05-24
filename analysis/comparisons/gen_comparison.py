"""
analysis/comparisons/gen_comparison.py
========================================
Gen1 vs Gen2 comparison: missing-indicator semantics.
"""

from __future__ import annotations

from typing import Any

from analysis.comparisons.factors import (
    build_gen_delta_table,
    factor_crosstab,
    filter_summaries,
    run_ids,
    summary_factors,
)

def compare_gen1_gen2(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compare generation cohorts using generalized factor metadata:
    * sentiment_on  -> gen1 vs gen2
    * sentiment_off -> gen1 vs gen2
    """
    warnings: list[str] = []
    unresolved: list[str] = []
    for summary in summaries:
        sentiment = summary_factors(summary).get("sentiment_enabled")
        if not isinstance(sentiment, bool):
            unresolved.append(summary.get("run_id", "unknown"))
    if unresolved:
        warnings.append("Skipped runs with unresolved factor semantics: " + ", ".join(sorted(unresolved)))

    delta_table: list[dict[str, Any]] = []
    valid_comparisons: list[str] = []
    incomplete_comparisons: list[str] = []
    invalid_comparisons: list[str] = []

    g1_on = filter_summaries(summaries, generation="gen1", factors={"sentiment_enabled": True})
    g2_on = filter_summaries(summaries, generation="gen2", factors={"sentiment_enabled": True})
    if g1_on and g2_on:
        delta_table.extend(build_gen_delta_table(g1_on, g2_on, cohort="sentiment_on"))
        valid_comparisons.append("sentiment_on:A_vs_C")
    else:
        incomplete_comparisons.append("sentiment_on:A_vs_C")
        missing_parts: list[str] = []
        if not g1_on:
            missing_parts.append("gen1/sentiment_enabled=True")
        if not g2_on:
            missing_parts.append("gen2/sentiment_enabled=True")
        warnings.append("Gen comparison invalid for sentiment ON: missing cohort(s) " + ", ".join(missing_parts) + ".")

    g1_off = filter_summaries(summaries, generation="gen1", factors={"sentiment_enabled": False})
    g2_off = filter_summaries(summaries, generation="gen2", factors={"sentiment_enabled": False})
    if g1_off and g2_off:
        delta_table.extend(build_gen_delta_table(g1_off, g2_off, cohort="sentiment_off"))
        valid_comparisons.append("sentiment_off:B_vs_D")
    else:
        incomplete_comparisons.append("sentiment_off:B_vs_D")
        missing_parts = []
        if not g1_off:
            missing_parts.append("gen1/sentiment_enabled=False")
        if not g2_off:
            missing_parts.append("gen2/sentiment_enabled=False")
        warnings.append("Gen comparison invalid for sentiment OFF: missing cohort(s) " + ", ".join(missing_parts) + ".")

    if not valid_comparisons:
        invalid_comparisons.append("gen_matrix")

    gen1_group = filter_summaries(summaries, generation="gen1")
    gen2_group = filter_summaries(summaries, generation="gen2")
    gen1_runs = run_ids(gen1_group)
    gen2_runs = run_ids(gen2_group)
    if not summaries:
        warnings.append("No run summaries provided for generation comparison.")
    coverage_comparison = _build_coverage_comparison(
        gen1_group,
        gen2_group,
    )

    return {
        "gen1": gen1_runs,
        "gen2": gen2_runs,
        "delta_table": delta_table,
        "coverage_comparison": coverage_comparison,
        "warnings": warnings,
        "cohorts": {
            "sentiment_on": {"gen1": run_ids(g1_on), "gen2": run_ids(g2_on)},
            "sentiment_off": {"gen1": run_ids(g1_off), "gen2": run_ids(g2_off)},
        },
        "matrix": {
            "expected_variants": ["A", "B", "C", "D"],
            "present_variants": sorted(
                {
                    (s.get("meta") or {}).get("run_variant")
                    for s in summaries
                    if isinstance((s.get("meta") or {}).get("run_variant"), str)
                }
            ),
            "factor_dimensions": factor_crosstab(summaries),
        },
        "valid_comparisons": valid_comparisons,
        "incomplete_comparisons": incomplete_comparisons,
        "invalid_comparisons": invalid_comparisons,
    }


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
