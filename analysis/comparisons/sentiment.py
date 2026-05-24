"""
analysis/comparisons/sentiment.py
===================================
Sentiment ON/OFF comparison logic.
"""

from __future__ import annotations

from typing import Any

from analysis.comparisons.factors import (
    build_on_off_delta_table,
    factor_crosstab,
    filter_summaries,
    run_ids,
    summary_generation,
    summary_factors,
)

def compare_sentiment_variants(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compare sentiment cohorts using generalized factor metadata.
    """
    warnings: list[str] = []
    unresolved: list[str] = []
    for summary in summaries:
        sentiment = summary_factors(summary).get("sentiment_enabled")
        generation = summary_generation(summary)
        if not isinstance(sentiment, bool) or not isinstance(generation, str):
            unresolved.append(summary.get("run_id", "unknown"))
    if unresolved:
        warnings.append("Skipped runs with unresolved factor semantics: " + ", ".join(sorted(unresolved)))

    matrix = {
        "expected_variants": ["A", "B", "C", "D"],
        "present_variants": sorted(
            {
                (s.get("meta") or {}).get("run_variant")
                for s in summaries
                if isinstance((s.get("meta") or {}).get("run_variant"), str)
            }
        ),
        "factor_dimensions": factor_crosstab(summaries),
    }

    delta_table: list[dict[str, Any]] = []
    valid_comparisons: list[str] = []
    incomplete_comparisons: list[str] = []
    invalid_comparisons: list[str] = []
    grouped: dict[str, dict[str, list[str]]] = {}

    generations = sorted({g for g in (summary_generation(s) for s in summaries) if isinstance(g, str)})
    for gen in generations:
        on_runs = filter_summaries(summaries, generation=gen, factors={"sentiment_enabled": True})
        off_runs = filter_summaries(summaries, generation=gen, factors={"sentiment_enabled": False})
        grouped[gen] = {"on": run_ids(on_runs), "off": run_ids(off_runs)}
        legacy_label = {"gen1": "A_vs_B", "gen2": "C_vs_D"}.get(gen, "on_vs_off")
        comparison_key = f"{gen}:{legacy_label}"
        if on_runs and off_runs:
            delta_table.extend(build_on_off_delta_table(on_runs, off_runs, generation=gen))
            valid_comparisons.append(comparison_key)
        else:
            incomplete_comparisons.append(comparison_key)
            missing_parts: list[str] = []
            if not on_runs:
                missing_parts.append("sentiment_enabled=True")
            if not off_runs:
                missing_parts.append("sentiment_enabled=False")
            warnings.append(
                f"Sentiment comparison invalid for generation={gen}: missing cohort(s) "
                + ", ".join(missing_parts)
                + "."
            )

    if not valid_comparisons:
        invalid_comparisons.append("sentiment_matrix")

    sentiment_on = run_ids(filter_summaries(summaries, factors={"sentiment_enabled": True}))
    sentiment_off = run_ids(filter_summaries(summaries, factors={"sentiment_enabled": False}))
    if not summaries:
        warnings.append("No run summaries provided for sentiment comparison.")

    return {
        "sentiment_on": sentiment_on,
        "sentiment_off": sentiment_off,
        "delta_table": delta_table,
        "warnings": warnings,
        "grouped": grouped,
        "matrix": matrix,
        "valid_comparisons": valid_comparisons,
        "incomplete_comparisons": incomplete_comparisons,
        "invalid_comparisons": invalid_comparisons,
    }
