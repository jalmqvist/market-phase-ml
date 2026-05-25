"""
analysis/comparisons/factor_comparison.py
==========================================
Generalized comparison outputs based on experiment factors.

v5 additions
-------------
* **Imputation Awareness effect**: conditions on same parquet (training_pair_family +
  sentiment_surface); varies ``missing_indicators_enabled``.
* **Training-family effect**: conditions on same awareness + sentiment_surface;
  varies ``training_pair_family``.

"Imputation Awareness" is the canonical term for what was previously called
"missing awareness".  It reflects that the model receives an explicit indicator
that a value was *imputed* rather than genuinely observed.
"""

from __future__ import annotations

from typing import Any

from analysis.comparisons.factors import (
    build_family_delta_table,
    build_on_off_delta_table,
    factor_crosstab,
    filter_summaries,
    is_v5_summary,
    run_ids,
    summary_factors,
    summary_generation,
    summary_surface,
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


def compare_imputation_awareness_effect(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Imputation Awareness effect (v5 factor-first).

    Conditions on: same training_pair_family + same sentiment_surface.
    Varies: missing_indicators_enabled (imputation awareness ON vs OFF).

    Only v5 summaries are included.  Legacy summaries are reported separately.
    """
    v5_summaries = [s for s in summaries if is_v5_summary(s)]
    warnings: list[str] = []
    comparisons: list[dict[str, Any]] = []

    families = sorted(
        {
            summary_surface(s).get("training_pair_family")
            for s in v5_summaries
            if summary_surface(s).get("training_pair_family") is not None
        }
    )
    sentiment_values = sorted(
        {
            summary_surface(s).get("sentiment_surface")
            for s in v5_summaries
            if isinstance(summary_surface(s).get("sentiment_surface"), bool)
        }
    )

    for family in families:
        for sentiment in sentiment_values:
            aware_runs = filter_summaries(
                v5_summaries,
                factors={"missing_indicators_enabled": True},
                surface={"training_pair_family": family, "sentiment_surface": sentiment},
            )
            blind_runs = filter_summaries(
                v5_summaries,
                factors={"missing_indicators_enabled": False},
                surface={"training_pair_family": family, "sentiment_surface": sentiment},
            )
            cohort_key = (
                f"training_family={family}/sentiment_surface={str(sentiment).lower()}"
            )
            valid = bool(aware_runs and blind_runs)
            if not aware_runs:
                warnings.append(
                    f"Imputation Awareness effect: missing imputation_aware=True cohort "
                    f"for {cohort_key}."
                )
            if not blind_runs:
                warnings.append(
                    f"Imputation Awareness effect: missing imputation_aware=False cohort "
                    f"for {cohort_key}."
                )
            comparisons.append(
                {
                    "cohort": cohort_key,
                    "imputation_aware": run_ids(aware_runs),
                    "imputation_blind": run_ids(blind_runs),
                    "delta_table": build_on_off_delta_table(
                        aware_runs, blind_runs, generation=cohort_key
                    ) if valid else [],
                    "valid": valid,
                }
            )

    # Legacy fallback using factors only.
    legacy_summaries = [s for s in summaries if not is_v5_summary(s)]
    if legacy_summaries:
        aware_runs = filter_summaries(legacy_summaries, factors={"missing_indicators_enabled": True})
        blind_runs = filter_summaries(legacy_summaries, factors={"missing_indicators_enabled": False})
        comparisons.append(
            {
                "cohort": "legacy:all",
                "imputation_aware": run_ids(aware_runs),
                "imputation_blind": run_ids(blind_runs),
                "delta_table": build_on_off_delta_table(
                    aware_runs, blind_runs, generation="legacy"
                ) if (aware_runs and blind_runs) else [],
                "valid": bool(aware_runs and blind_runs),
            }
        )

    return {
        "comparisons": comparisons,
        "warnings": warnings,
    }


def compare_training_family_effect(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Training-family transfer effect (v5 factor-first).

    Conditions on: same sentiment_surface + same imputation_awareness.
    Varies: training_pair_family (e.g. persistent vs reactive).

    Only v5 summaries are included.
    """
    v5_summaries = [s for s in summaries if is_v5_summary(s)]
    warnings: list[str] = []
    comparisons: list[dict[str, Any]] = []

    families = sorted(
        {
            summary_surface(s).get("training_pair_family")
            for s in v5_summaries
            if summary_surface(s).get("training_pair_family") is not None
        }
    )
    family_pairs = [
        (a, b) for i, a in enumerate(families) for b in families[i + 1:]
    ]
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

    for family_a, family_b in family_pairs:
        for sentiment in sentiment_values:
            for awareness in awareness_values:
                a_runs = filter_summaries(
                    v5_summaries,
                    factors={"missing_indicators_enabled": awareness},
                    surface={"training_pair_family": family_a, "sentiment_surface": sentiment},
                )
                b_runs = filter_summaries(
                    v5_summaries,
                    factors={"missing_indicators_enabled": awareness},
                    surface={"training_pair_family": family_b, "sentiment_surface": sentiment},
                )
                cohort_key = (
                    f"{family_a}_vs_{family_b}/"
                    f"sentiment_surface={str(sentiment).lower()}/"
                    f"imputation_awareness={str(awareness).lower()}"
                )
                valid = bool(a_runs and b_runs)
                if not a_runs:
                    warnings.append(
                        f"Training-family effect: missing {family_a!r} cohort for {cohort_key}."
                    )
                if not b_runs:
                    warnings.append(
                        f"Training-family effect: missing {family_b!r} cohort for {cohort_key}."
                    )
                comparisons.append(
                    {
                        "cohort": cohort_key,
                        family_a: run_ids(a_runs),
                        family_b: run_ids(b_runs),
                        "delta_table": build_family_delta_table(
                            a_runs, b_runs, cohort=cohort_key,
                            family_a_label=family_a, family_b_label=family_b,
                        ) if valid else [],
                        "valid": valid,
                    }
                )

    return {
        "comparisons": comparisons,
        "warnings": warnings,
    }


def build_factor_comparisons(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    generations = sorted({g for g in (summary_generation(s) for s in summaries) if isinstance(g, str)})
    # "Imputation Awareness" conditioned by generation (legacy label preserved for backward compat).
    conditioned_sentiment = [
        compare_binary_factor(
            summaries,
            factor="sentiment_enabled",
            conditioned_on={"generation": generation},
        )
        for generation in generations
    ]
    # "Imputation Awareness" = missing_indicators_enabled (canonical term).
    conditioned_imputation_awareness = [
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
            # Canonical term: Imputation Awareness (was "missing_indicators_enabled").
            "imputation_awareness": compare_binary_factor(summaries, factor="missing_indicators_enabled"),
            "dl_enabled": compare_binary_factor(summaries, factor="dl_enabled"),
            "overlap_only": compare_binary_factor(summaries, factor="overlap_only"),
            "sentiment_enabled_by_generation": conditioned_sentiment,
            # Canonical term: imputation_awareness_by_generation (was "missing_indicators_by_generation").
            "imputation_awareness_by_generation": conditioned_imputation_awareness,
        },
        # v5 factor-first comparisons.
        "awareness_effect": compare_imputation_awareness_effect(summaries),
        "training_family_effect": compare_training_family_effect(summaries),
    }

