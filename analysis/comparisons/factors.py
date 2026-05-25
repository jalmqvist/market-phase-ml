"""
analysis/comparisons/factors.py
================================
Generalized factor-based cohort filtering and comparison helpers.

Terminology note
-----------------
"Imputation Awareness" (``missing_indicators_enabled``) is the precise term for the
runtime architecture in which the model receives an explicit indicator that a feature
value was *imputed* rather than genuinely observed.  This is more accurate than the
informal "missing awareness" label used in earlier documentation.
"""

from __future__ import annotations

from typing import Any

from experiment_semantics import normalize_experiment_factors, is_v5_surface


def summary_experiment(summary: dict[str, Any]) -> dict[str, Any]:
    return ((summary.get("meta") or {}).get("experiment") or {})


def summary_generation(summary: dict[str, Any]) -> str | None:
    experiment = summary_experiment(summary)
    generation = experiment.get("generation")
    if isinstance(generation, str) and generation.strip():
        return generation.strip().lower()
    fallback = (summary.get("meta") or {}).get("experiment_gen")
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip().lower()
    return None


def summary_factors(summary: dict[str, Any]) -> dict[str, Any]:
    experiment = summary_experiment(summary)
    return normalize_experiment_factors(
        experiment.get("factors"),
        fallback_sentiment_enabled=experiment.get("sentiment_enabled"),
        fallback_missing_indicators_enabled=experiment.get("missing_indicators_enabled"),
        fallback_dl_enabled=(summary.get("meta") or {}).get("dl_enabled"),
        fallback_msml_regime=experiment.get("msml_regime"),
    )


def summary_surface(summary: dict[str, Any]) -> dict[str, Any]:
    """
    Return the canonical ``experiment_surface`` dict for *summary*.

    For v5 manifests this reads the explicit surface block written by the manifest
    parser.  For legacy manifests an all-None dict is returned and callers should
    fall back to ``summary_factors()`` for sentiment/awareness attribution.
    """
    return (summary.get("meta") or {}).get("experiment_surface") or {}


def summary_surface_source(summary: dict[str, Any]) -> str:
    """Return the surface_source tag: ``"manifest"`` or ``"legacy_variant_fallback"``."""
    return (summary.get("meta") or {}).get("surface_source", "legacy_variant_fallback")


def is_v5_summary(summary: dict[str, Any]) -> bool:
    """Return True when *summary* carries a valid v5 experiment_surface block."""
    return is_v5_surface(summary_surface(summary))


def _value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (set, list, tuple, frozenset)):
        return actual in expected
    return actual == expected


def filter_summaries(
    summaries: list[dict[str, Any]],
    *,
    generation: Any = None,
    factors: dict[str, Any] | None = None,
    surface: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Filter summaries by generation, runtime factors, and/or surface factors.

    Parameters
    ----------
    generation:
        If provided, only summaries whose generation matches are returned.
    factors:
        Dict of runtime factor conditions (e.g. ``{"dl_enabled": True}``).
    surface:
        Dict of ``experiment_surface`` conditions (e.g.
        ``{"sentiment_surface": True, "training_pair_family": "persistent"}``).
        Applied only to v5 summaries; legacy summaries are excluded from surface-
        conditioned queries.
    """
    out: list[dict[str, Any]] = []
    factors = factors or {}
    surface = surface or {}
    for summary in summaries:
        if generation is not None and not _value_matches(summary_generation(summary), generation):
            continue
        sf = summary_factors(summary)
        if not all(_value_matches(sf.get(k), v) for k, v in factors.items()):
            continue
        if surface:
            # Surface conditions are only meaningful for v5 summaries.
            if not is_v5_summary(summary):
                continue
            sv = summary_surface(summary)
            if not all(_value_matches(sv.get(k), v) for k, v in surface.items()):
                continue
        out.append(summary)
    return out


def run_ids(summaries: list[dict[str, Any]]) -> list[str]:
    return [s.get("run_id", "unknown") for s in summaries]


_WF_METRICS = [
    ("walkforward_summary", "Sharpe_Dynamic", "sharpe_dynamic"),
    ("walkforward_summary", "Sharpe_Delta", "sharpe_delta"),
    ("walkforward_summary", "Return_Delta", "return_delta"),
    ("walkforward_summary", "MaxDD_Delta", "maxdd_delta"),
]

_GEN_METRICS = [
    ("walkforward_summary", "Sharpe_Delta", "sharpe_delta"),
    ("walkforward_summary", "Return_Delta", "return_delta"),
    ("walkforward_summary", "MaxDD_Delta", "maxdd_delta"),
]


def extract_pair_metrics(
    summaries: list[dict[str, Any]],
    *,
    metrics: list[tuple[str, str, str]],
) -> dict[str, dict[str, float]]:
    accum: dict[str, dict[str, list[float]]] = {}
    for summary in summaries:
        csvs = summary.get("csvs") or {}
        for section_key, csv_col, metric_name in metrics:
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
            metric: sum(vals) / len(vals)
            for metric, vals in sorted(metric_map.items())
        }
        for pair, metric_map in sorted(accum.items())
    }


def build_on_off_delta_table(
    on_runs: list[dict[str, Any]],
    off_runs: list[dict[str, Any]],
    *,
    generation: str,
) -> list[dict[str, Any]]:
    on_metrics = extract_pair_metrics(on_runs, metrics=_WF_METRICS)
    off_metrics = extract_pair_metrics(off_runs, metrics=_WF_METRICS)
    pairs = sorted(set(on_metrics) | set(off_metrics))
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        on_pair = on_metrics.get(pair, {})
        off_pair = off_metrics.get(pair, {})
        for metric in sorted(set(on_pair) | set(off_pair)):
            on_val = on_pair.get(metric)
            off_val = off_pair.get(metric)
            rows.append(
                {
                    "generation": generation,
                    "pair": pair,
                    "metric": metric,
                    "sentiment_on": on_val,
                    "sentiment_off": off_val,
                    "delta_on_minus_off": (on_val - off_val) if (on_val is not None and off_val is not None) else None,
                }
            )
    return rows


def build_gen_delta_table(
    gen1_runs: list[dict[str, Any]],
    gen2_runs: list[dict[str, Any]],
    *,
    cohort: str,
) -> list[dict[str, Any]]:
    g1 = extract_pair_metrics(gen1_runs, metrics=_GEN_METRICS)
    g2 = extract_pair_metrics(gen2_runs, metrics=_GEN_METRICS)
    rows: list[dict[str, Any]] = []
    for pair in sorted(set(g1) | set(g2)):
        g1_pair = g1.get(pair, {})
        g2_pair = g2.get(pair, {})
        for metric in sorted(set(g1_pair) | set(g2_pair)):
            g1_val = g1_pair.get(metric)
            g2_val = g2_pair.get(metric)
            rows.append(
                {
                    "cohort": cohort,
                    "pair": pair,
                    "metric": metric,
                    "gen1": g1_val,
                    "gen2": g2_val,
                    "delta_gen2_minus_gen1": (g2_val - g1_val) if (g1_val is not None and g2_val is not None) else None,
                }
            )
    return rows


def build_family_delta_table(
    family_a_runs: list[dict[str, Any]],
    family_b_runs: list[dict[str, Any]],
    *,
    cohort: str,
    family_a_label: str = "family_a",
    family_b_label: str = "family_b",
) -> list[dict[str, Any]]:
    """
    Build a delta table comparing two training-pair-family cohorts.

    Used for training-family transfer effect comparisons (e.g. persistent vs reactive).
    """
    a = extract_pair_metrics(family_a_runs, metrics=_GEN_METRICS)
    b = extract_pair_metrics(family_b_runs, metrics=_GEN_METRICS)
    rows: list[dict[str, Any]] = []
    for pair in sorted(set(a) | set(b)):
        a_pair = a.get(pair, {})
        b_pair = b.get(pair, {})
        for metric in sorted(set(a_pair) | set(b_pair)):
            a_val = a_pair.get(metric)
            b_val = b_pair.get(metric)
            rows.append(
                {
                    "cohort": cohort,
                    "pair": pair,
                    "metric": metric,
                    family_a_label: a_val,
                    family_b_label: b_val,
                    f"delta_{family_b_label}_minus_{family_a_label}": (
                        (b_val - a_val)
                        if (a_val is not None and b_val is not None)
                        else None
                    ),
                }
            )
    return rows


def factor_crosstab(
    summaries: list[dict[str, Any]],
    *,
    factor_keys: list[str] | None = None,
    surface_keys: list[str] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """
    Build a factor cross-tabulation across all *summaries*.

    Runtime factors (from ``experiment.factors``) are always included.
    Surface factors (from ``experiment_surface``) are included for v5 summaries
    when *surface_keys* is provided or when the default set is used.

    "Imputation Awareness" is the canonical label for ``missing_indicators_enabled``.
    """
    factor_keys = factor_keys or [
        "dl_enabled",
        "sentiment_enabled",
        "missing_indicators_enabled",  # imputation awareness
        "msml_regime",
        "overlap_only",
        "selector_enabled",
    ]
    surface_keys = surface_keys if surface_keys is not None else [
        "sentiment_surface",
        "training_pair_family",
        "evaluation_pair_family",
        "feature_surface",
    ]
    table: dict[str, dict[str, list[str]]] = {}
    for factor in factor_keys:
        grouped: dict[str, list[str]] = {}
        for summary in summaries:
            value = summary_factors(summary).get(factor)
            key = str(value)
            grouped.setdefault(key, []).append(summary.get("run_id", "unknown"))
        table[factor] = {k: sorted(v) for k, v in sorted(grouped.items())}
    for sf in surface_keys:
        grouped = {}
        for summary in summaries:
            if not is_v5_summary(summary):
                grouped.setdefault("legacy", []).append(summary.get("run_id", "unknown"))
                continue
            value = summary_surface(summary).get(sf)
            key = str(value)
            grouped.setdefault(key, []).append(summary.get("run_id", "unknown"))
        if any(k != "legacy" or v for k, v in grouped.items()):
            table[sf] = {k: sorted(v) for k, v in sorted(grouped.items())}
    return table

