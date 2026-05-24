"""
analysis/comparisons/factor_comparison.py
==========================================
Generalized comparison outputs based on experiment factors.
"""

from __future__ import annotations

from typing import Any

from analysis.comparisons.factors import (
    build_on_off_delta_table,
    factor_crosstab,
    filter_summaries,
    run_ids,
    summary_generation,
)


def compare_binary_factor(
    summaries: list[dict[str, Any]],
    *,
    factor: str,
    conditioned_on: dict[str, Any] | None = None,
) -> dict[str, Any]:
    conditioned_on = conditioned_on or {}
    gen_condition = conditioned_on.get("generation")
    other_factor_conditions = {
        k: v for k, v in conditioned_on.items()
        if k != "generation"
    }
    on_runs = filter_summaries(
        summaries,
        generation=gen_condition,
        factors={factor: True, **other_factor_conditions},
    )
    off_runs = filter_summaries(
        summaries,
        generation=gen_condition,
        factors={factor: False, **other_factor_conditions},
    )
    generation_label = str(gen_condition) if gen_condition is not None else "all_generations"
    warnings: list[str] = []
    if not on_runs:
        warnings.append(f"Missing cohort: {factor}=True")
    if not off_runs:
        warnings.append(f"Missing cohort: {factor}=False")
    return {
        "factor": factor,
        "conditioned_on": conditioned_on,
        "on": run_ids(on_runs),
        "off": run_ids(off_runs),
        "delta_table": build_on_off_delta_table(on_runs, off_runs, generation=generation_label) if on_runs and off_runs else [],
        "valid": bool(on_runs and off_runs),
        "warnings": warnings,
    }


def build_factor_comparisons(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    generations = sorted({g for g in (summary_generation(s) for s in summaries) if isinstance(g, str)})
    conditioned_sentiment = [
        compare_binary_factor(
            summaries,
            factor="sentiment_enabled",
            conditioned_on={"generation": generation},
        )
        for generation in generations
    ]
    conditioned_missing = [
        compare_binary_factor(
            summaries,
            factor="missing_indicators_enabled",
            conditioned_on={"generation": generation},
        )
        for generation in generations
    ]
    return {
        "factor_crosstab": factor_crosstab(summaries),
        "slices": {
            "dl_enabled_true": run_ids(filter_summaries(summaries, factors={"dl_enabled": True})),
            "dl_enabled_false": run_ids(filter_summaries(summaries, factors={"dl_enabled": False})),
            "msml_regime": factor_crosstab(summaries, factor_keys=["msml_regime"]).get("msml_regime", {}),
            "selector_enabled_true": run_ids(filter_summaries(summaries, factors={"selector_enabled": True})),
            "selector_enabled_false": run_ids(filter_summaries(summaries, factors={"selector_enabled": False})),
        },
        "comparisons": {
            "sentiment_enabled": compare_binary_factor(summaries, factor="sentiment_enabled"),
            "missing_indicators_enabled": compare_binary_factor(summaries, factor="missing_indicators_enabled"),
            "dl_enabled": compare_binary_factor(summaries, factor="dl_enabled"),
            "overlap_only": compare_binary_factor(summaries, factor="overlap_only"),
            "sentiment_enabled_by_generation": conditioned_sentiment,
            "missing_indicators_by_generation": conditioned_missing,
        },
    }

