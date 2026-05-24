"""
analysis/comparisons/gen_comparison.py
========================================
Gen1 vs Gen2 comparison: missing-indicator semantics.
"""

from __future__ import annotations

from typing import Any

from experiment_semantics import EXPERIMENT_VARIANTS

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

    sentiment_values = sorted(
        {
            summary_factors(s).get("sentiment_enabled")
            for s in summaries
            if isinstance(summary_factors(s).get("sentiment_enabled"), bool)
        }
    )
    missing_values = sorted(
        {
            summary_factors(s).get("missing_indicators_enabled")
            for s in summaries
            if isinstance(summary_factors(s).get("missing_indicators_enabled"), bool)
        }
    )
    cohorts: dict[str, dict[str, list[str]]] = {}
    for sentiment_enabled in sentiment_values:
        for missing_enabled in missing_values:
            g1_runs = filter_summaries(
                summaries,
                generation="gen1",
                factors={
                    "sentiment_enabled": sentiment_enabled,
                    "missing_indicators_enabled": missing_enabled,
                },
            )
            g2_runs = filter_summaries(
                summaries,
                generation="gen2",
                factors={
                    "sentiment_enabled": sentiment_enabled,
                    "missing_indicators_enabled": missing_enabled,
                },
            )
            cohort_key = (
                f"sentiment_enabled={str(sentiment_enabled).lower()}/"
                f"missing_indicators_enabled={str(missing_enabled).lower()}"
            )
            cohorts[cohort_key] = {"gen1": run_ids(g1_runs), "gen2": run_ids(g2_runs)}
            if g1_runs and g2_runs:
                delta_table.extend(build_gen_delta_table(g1_runs, g2_runs, cohort=cohort_key))
                valid_comparisons.append(cohort_key)
            else:
                incomplete_comparisons.append(cohort_key)
                missing_parts: list[str] = []
                if not g1_runs:
                    missing_parts.append("gen1")
                if not g2_runs:
                    missing_parts.append("gen2")
                warnings.append(
                    "Gen comparison invalid for "
                    + cohort_key
                    + ": missing cohort(s) "
                    + ", ".join(missing_parts)
                    + "."
                )

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
            **cohorts,
        },
        "matrix": {
            "expected_variants": sorted(EXPERIMENT_VARIANTS),
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
