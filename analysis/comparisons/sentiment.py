"""
analysis/comparisons/sentiment.py
===================================
Sentiment surface ON/OFF comparison logic.

For v5 manifests (``surface_source="manifest"``), sentiment attribution comes from
``experiment_surface.sentiment_surface`` — the parquet-level property.

For legacy manifests, the old variant-based ``factors.sentiment_enabled`` is used.

The comparison is conditioned on:
  * same training_pair_family (surface)
  * same imputation_awareness / missing_indicators_enabled (runtime factor)
  * same dl_enabled (runtime factor)
  * same msml_regime (runtime factor)

Only ``sentiment_surface`` (or ``sentiment_enabled`` for legacy) is varied.
"""

from __future__ import annotations

from typing import Any

from experiment_semantics import EXPERIMENT_VARIANTS

from analysis.comparisons.factors import (
    build_on_off_delta_table,
    factor_crosstab,
    filter_summaries,
    is_invalid_modern_surface_summary,
    is_legacy_summary,
    is_v5_summary,
    run_ids,
    summary_factors,
    summary_generation,
    summary_surface,
)

def compare_sentiment_variants(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compare sentiment cohorts using generalized factor metadata.

    For v5 summaries, uses ``experiment_surface.sentiment_surface`` as the
    distinguishing dimension (parquet-level property).  For legacy summaries,
    falls back to ``factors.sentiment_enabled`` (variant-based).
    """
    warnings: list[str] = []
    unresolved: list[str] = []

    v5_summaries = [s for s in summaries if is_v5_summary(s)]
    legacy_summaries = [s for s in summaries if is_legacy_summary(s)]
    invalid_modern_summaries = [s for s in summaries if is_invalid_modern_surface_summary(s)]

    if invalid_modern_summaries:
        raise RuntimeError(
            "Semantic integrity violation: modern manifest summaries missing valid experiment_surface "
            "(surface_source='missing_experiment_surface'): "
            + ", ".join(sorted(run_ids(invalid_modern_summaries)))
        )

    if legacy_summaries:
        warnings.append(
            f"{len(legacy_summaries)} legacy run(s) use variant-based sentiment attribution "
            "(no experiment_surface): " + ", ".join(
                s.get("run_id", "unknown") for s in legacy_summaries
            )
        )

    # For legacy summaries: validate that sentiment_enabled is resolved.
    for summary in legacy_summaries:
        sentiment = summary_factors(summary).get("sentiment_enabled")
        generation = summary_generation(summary)
        if not isinstance(sentiment, bool) or not isinstance(generation, str):
            unresolved.append(summary.get("run_id", "unknown"))

    # For v5 summaries: validate that sentiment_surface is present.
    for summary in v5_summaries:
        sentiment = summary_surface(summary).get("sentiment_surface")
        if not isinstance(sentiment, bool):
            unresolved.append(summary.get("run_id", "unknown"))

    if unresolved:
        warnings.append("Skipped runs with unresolved sentiment semantics: " + ", ".join(sorted(unresolved)))

    matrix = {
        "expected_variants": sorted(EXPERIMENT_VARIANTS),
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
    grouped: dict[str, dict[str, dict[str, list[str]]]] = {}

    # -------------------------------------------------------------------
    # v5 path: condition on training_pair_family + awareness + dl + regime
    # -------------------------------------------------------------------
    training_families = sorted(
        {
            summary_surface(s).get("training_pair_family")
            for s in v5_summaries
            if summary_surface(s).get("training_pair_family") is not None
        }
    )
    for family in training_families:
        grouped.setdefault(f"family:{family}", {})
        awareness_values = sorted(
            {
                summary_factors(s).get("missing_indicators_enabled")
                for s in v5_summaries
                if summary_surface(s).get("training_pair_family") == family
                and isinstance(summary_factors(s).get("missing_indicators_enabled"), bool)
            }
        )
        for imputation_aware in awareness_values:
            on_runs = filter_summaries(
                v5_summaries,
                factors={"missing_indicators_enabled": imputation_aware},
                surface={"sentiment_surface": True, "training_pair_family": family},
            )
            off_runs = filter_summaries(
                v5_summaries,
                factors={"missing_indicators_enabled": imputation_aware},
                surface={"sentiment_surface": False, "training_pair_family": family},
            )
            cohort_key = f"imputation_awareness={str(imputation_aware).lower()}"
            comparison_key = (
                f"training_family={family}/imputation_awareness={str(imputation_aware).lower()}"
            )
            grouped[f"family:{family}"][cohort_key] = {
                "on": run_ids(on_runs),
                "off": run_ids(off_runs),
            }
            if on_runs and off_runs:
                delta_table.extend(
                    build_on_off_delta_table(
                        on_runs,
                        off_runs,
                        generation=f"{family}:{cohort_key}",
                    )
                )
                valid_comparisons.append(comparison_key)
            else:
                incomplete_comparisons.append(comparison_key)
                missing_parts: list[str] = []
                if not on_runs:
                    missing_parts.append("sentiment_surface=True")
                if not off_runs:
                    missing_parts.append("sentiment_surface=False")
                warnings.append(
                    f"Sentiment comparison incomplete for training_family={family!r}, "
                    f"imputation_awareness={str(imputation_aware).lower()}: missing cohort(s) "
                    + ", ".join(missing_parts) + "."
                )

    # -------------------------------------------------------------------
    # Legacy path: group by generation + imputation_awareness (old logic)
    # -------------------------------------------------------------------
    if legacy_summaries:
        generations = sorted({
            g for g in (summary_generation(s) for s in legacy_summaries)
            if isinstance(g, str)
        })
        for gen in generations:
            grouped.setdefault(f"legacy:{gen}", {})
            missing_values = sorted(
                {
                    summary_factors(s).get("missing_indicators_enabled")
                    for s in legacy_summaries
                    if summary_generation(s) == gen
                    and isinstance(summary_factors(s).get("missing_indicators_enabled"), bool)
                }
            )
            for missing_enabled in missing_values:
                on_runs = filter_summaries(
                    legacy_summaries,
                    generation=gen,
                    factors={
                        "sentiment_enabled": True,
                        "missing_indicators_enabled": missing_enabled,
                    },
                )
                off_runs = filter_summaries(
                    legacy_summaries,
                    generation=gen,
                    factors={
                        "sentiment_enabled": False,
                        "missing_indicators_enabled": missing_enabled,
                    },
                )
                cohort_key = f"imputation_awareness={str(missing_enabled).lower()}"
                comparison_key = (
                    f"legacy:generation={gen}/imputation_awareness={str(missing_enabled).lower()}"
                )
                grouped[f"legacy:{gen}"][cohort_key] = {
                    "on": run_ids(on_runs),
                    "off": run_ids(off_runs),
                }
                if on_runs and off_runs:
                    delta_table.extend(
                        build_on_off_delta_table(
                            on_runs,
                            off_runs,
                            generation=f"legacy:{gen}:{cohort_key}",
                        )
                    )
                    valid_comparisons.append(comparison_key)
                else:
                    incomplete_comparisons.append(comparison_key)

    if not valid_comparisons:
        invalid_comparisons.append("sentiment_matrix")
        warnings.append(
            "No valid sentiment comparisons found (all cohorts incomplete or invalid). "
            "Ensure matching sentiment_surface=True and sentiment_surface=False runs exist "
            "within the same training_family/imputation_awareness cohort."
        )

    # Summary-level on/off slices (union across all).
    sentiment_on = run_ids(
        filter_summaries(v5_summaries, surface={"sentiment_surface": True})
        + filter_summaries(legacy_summaries, factors={"sentiment_enabled": True})
    )
    sentiment_off = run_ids(
        filter_summaries(v5_summaries, surface={"sentiment_surface": False})
        + filter_summaries(legacy_summaries, factors={"sentiment_enabled": False})
    )
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
