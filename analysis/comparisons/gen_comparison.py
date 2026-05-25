"""
analysis/comparisons/gen_comparison.py
========================================
Training-family transfer effect comparison (v5 factor-first).

For v5 manifests, compares runs by ``experiment_surface.training_pair_family``
(e.g. "persistent" vs "reactive"), conditioned on same imputation awareness and
sentiment surface.

For legacy manifests (pre-v5), falls back to the old Gen1 vs Gen2 semantics.

Imputation Awareness terminology
---------------------------------
"Imputation Awareness" (``missing_indicators_enabled``) is used throughout this
module.  The model is not merely "aware of missingness"; it is aware that a value
was *imputed* rather than genuinely observed.  This wording reflects the actual
architectural mechanism.
"""

from __future__ import annotations

from typing import Any

from experiment_semantics import EXPERIMENT_VARIANTS

from analysis.comparisons.factors import (
    build_family_delta_table,
    build_gen_delta_table,
    factor_crosstab,
    filter_summaries,
    is_v5_summary,
    run_ids,
    summary_factors,
    summary_surface,
)


# LEGACY NAME:
# compare_gen1_gen2 now performs training-family comparisons when
# experiment_surface metadata is available (v5 manifests).
# For legacy manifests it falls back to the old Gen1 vs Gen2 semantics.
# The function name is preserved to avoid import breakage in callers.
def compare_gen1_gen2(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compare generation/training-family cohorts using generalized factor metadata.

    v5 path (``surface_source="manifest"``):
        * Compares ``training_pair_family`` cohorts (e.g. persistent vs reactive).
        * Conditioned on same sentiment_surface + same imputation_awareness.

    Legacy path (``surface_source="legacy_variant_fallback"``):
        * Compares Gen1 vs Gen2 using ``factors.sentiment_enabled`` +
          ``factors.missing_indicators_enabled``.
    """
    warnings: list[str] = []

    v5_summaries = [s for s in summaries if is_v5_summary(s)]
    legacy_summaries = [s for s in summaries if not is_v5_summary(s)]

    if legacy_summaries:
        warnings.append(
            f"{len(legacy_summaries)} legacy run(s) use variant-based Gen1/Gen2 semantics: "
            + ", ".join(s.get("run_id", "unknown") for s in legacy_summaries)
        )

    delta_table: list[dict[str, Any]] = []
    valid_comparisons: list[str] = []
    incomplete_comparisons: list[str] = []
    invalid_comparisons: list[str] = []
    cohorts: dict[str, dict[str, list[str]]] = {}

    # -------------------------------------------------------------------
    # v5 path: training-family transfer effect
    # -------------------------------------------------------------------
    # Identify all training_pair_family values present in v5 runs.
    families = sorted(
        {
            summary_surface(s).get("training_pair_family")
            for s in v5_summaries
            if summary_surface(s).get("training_pair_family") is not None
        }
    )

    # Build cross-product: for each pair of families, compare across sentiment × awareness.
    family_pairs = [
        (a, b) for i, a in enumerate(families) for b in families[i + 1:]
    ]
    for family_a, family_b in family_pairs:
        sentiment_values = sorted(
            {
                summary_surface(s).get("sentiment_surface")
                for s in v5_summaries
                if isinstance(summary_surface(s).get("sentiment_surface"), bool)
            }
        )
        awareness_values = sorted(
            {
                summary_factors(s).get("missing_indicators_enabled")
                for s in v5_summaries
                if isinstance(summary_factors(s).get("missing_indicators_enabled"), bool)
            }
        )
        for sentiment in sentiment_values:
            for awareness in awareness_values:
                a_runs = filter_summaries(
                    v5_summaries,
                    factors={"missing_indicators_enabled": awareness},
                    surface={"sentiment_surface": sentiment, "training_pair_family": family_a},
                )
                b_runs = filter_summaries(
                    v5_summaries,
                    factors={"missing_indicators_enabled": awareness},
                    surface={"sentiment_surface": sentiment, "training_pair_family": family_b},
                )
                cohort_key = (
                    f"{family_a}_vs_{family_b}/"
                    f"sentiment_surface={str(sentiment).lower()}/"
                    f"imputation_awareness={str(awareness).lower()}"
                )
                cohorts[cohort_key] = {
                    family_a: run_ids(a_runs),
                    family_b: run_ids(b_runs),
                }
                if a_runs and b_runs:
                    delta_table.extend(
                        build_family_delta_table(
                            a_runs,
                            b_runs,
                            cohort=cohort_key,
                            family_a_label=family_a,
                            family_b_label=family_b,
                        )
                    )
                    valid_comparisons.append(cohort_key)
                else:
                    incomplete_comparisons.append(cohort_key)
                    missing_parts = []
                    if not a_runs:
                        missing_parts.append(family_a)
                    if not b_runs:
                        missing_parts.append(family_b)
                    warnings.append(
                        f"Training-family comparison incomplete for {cohort_key}: "
                        "missing cohort(s) " + ", ".join(missing_parts) + "."
                    )

    # -------------------------------------------------------------------
    # Legacy path: Gen1 vs Gen2 (old variant-based semantics)
    # -------------------------------------------------------------------
    if legacy_summaries:
        _compare_gen1_gen2_legacy(
            legacy_summaries,
            delta_table=delta_table,
            valid_comparisons=valid_comparisons,
            incomplete_comparisons=incomplete_comparisons,
            cohorts=cohorts,
            warnings=warnings,
        )

    if not valid_comparisons:
        invalid_comparisons.append("training_family_matrix")
        warnings.append(
            "No valid training-family or Gen1/Gen2 comparisons found (all cohorts incomplete or invalid). "
            "Ensure matching training_pair_family / generation cohorts exist."
        )

    # Populate flat family/gen slices for report rendering.
    family_slices: dict[str, list[str]] = {}
    for family in families:
        family_slices[family] = run_ids(
            filter_summaries(v5_summaries, surface={"training_pair_family": family})
        )
    # Legacy gen slices (for backward-compat keys in output).
    from analysis.comparisons.factors import filter_summaries as _fs, summary_generation as _sg
    gen1_group = [s for s in legacy_summaries if _sg(s) == "gen1"]
    gen2_group = [s for s in legacy_summaries if _sg(s) == "gen2"]
    gen1_runs = run_ids(gen1_group)
    gen2_runs = run_ids(gen2_group)

    coverage_comparison = _build_coverage_comparison(
        gen1_group,
        gen2_group,
    )

    if not summaries:
        warnings.append("No run summaries provided for generation/training-family comparison.")

    return {
        # Legacy compat keys.
        "gen1": gen1_runs,
        "gen2": gen2_runs,
        # v5 factor-first keys.
        "training_families": family_slices,
        "delta_table": delta_table,
        "coverage_comparison": coverage_comparison,
        "warnings": warnings,
        "cohorts": cohorts,
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


def _compare_gen1_gen2_legacy(
    legacy_summaries: list[dict[str, Any]],
    *,
    delta_table: list[dict[str, Any]],
    valid_comparisons: list[str],
    incomplete_comparisons: list[str],
    cohorts: dict[str, dict[str, list[str]]],
    warnings: list[str],
) -> None:
    """
    Legacy Gen1 vs Gen2 comparison using variant-based ``sentiment_enabled`` /
    ``missing_indicators_enabled`` factors.

    This function is called only for runs whose ``surface_source`` is
    ``"legacy_variant_fallback"``.  It is NOT the canonical analysis path for
    v5 manifests.
    """
    from analysis.comparisons.factors import summary_generation as _sg

    unresolved: list[str] = []
    for summary in legacy_summaries:
        sentiment = summary_factors(summary).get("sentiment_enabled")
        if not isinstance(sentiment, bool):
            unresolved.append(summary.get("run_id", "unknown"))
    if unresolved:
        warnings.append(
            "Skipped legacy runs with unresolved factor semantics: "
            + ", ".join(sorted(unresolved))
        )

    sentiment_values = sorted(
        {
            summary_factors(s).get("sentiment_enabled")
            for s in legacy_summaries
            if isinstance(summary_factors(s).get("sentiment_enabled"), bool)
        }
    )
    missing_values = sorted(
        {
            summary_factors(s).get("missing_indicators_enabled")
            for s in legacy_summaries
            if isinstance(summary_factors(s).get("missing_indicators_enabled"), bool)
        }
    )
    for sentiment_enabled in sentiment_values:
        for missing_enabled in missing_values:
            g1_runs = filter_summaries(
                legacy_summaries,
                generation="gen1",
                factors={
                    "sentiment_enabled": sentiment_enabled,
                    "missing_indicators_enabled": missing_enabled,
                },
            )
            g2_runs = filter_summaries(
                legacy_summaries,
                generation="gen2",
                factors={
                    "sentiment_enabled": sentiment_enabled,
                    "missing_indicators_enabled": missing_enabled,
                },
            )
            cohort_key = (
                f"legacy:sentiment_enabled={str(sentiment_enabled).lower()}/"
                f"imputation_awareness={str(missing_enabled).lower()}"
            )
            cohorts[cohort_key] = {"gen1": run_ids(g1_runs), "gen2": run_ids(g2_runs)}
            if g1_runs and g2_runs:
                delta_table.extend(build_gen_delta_table(g1_runs, g2_runs, cohort=cohort_key))
                valid_comparisons.append(cohort_key)
            else:
                incomplete_comparisons.append(cohort_key)
                missing_parts = []
                if not g1_runs:
                    missing_parts.append("gen1")
                if not g2_runs:
                    missing_parts.append("gen2")
                warnings.append(
                    "Legacy gen comparison incomplete for " + cohort_key
                    + ": missing cohort(s) " + ", ".join(missing_parts) + "."
                )


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
