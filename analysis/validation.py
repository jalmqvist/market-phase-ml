"""
analysis/validation.py
=======================
Integrity validation for analysis framework v2.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from experiment_semantics import (
    LEGACY_VARIANT,
    VALID_EXPERIMENT_VARIANTS,
    normalize_experiment_factors,
)

_REQUIRED_REPRODUCIBILITY_KEYS = (
    "experiment_seed",
    "numpy_seed",
    "python_random_seed",
    "torch_seed",
)
def sort_summaries_deterministically(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        summaries,
        key=lambda s: (
            (s.get("meta") or {}).get("experiment_gen") or "zzz",
            (s.get("meta") or {}).get("run_variant") or "Z",
            (s.get("meta") or {}).get("timestamp_utc") or "zzz",
            (s.get("meta") or {}).get("archive_relpath") or "zzz",
            s.get("run_id") or "zzz",
        ),
    )


def validate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    provenance_errors: list[str] = []
    provenance_warnings: list[str] = []
    semantic_errors: list[str] = []
    semantic_warnings: list[str] = []
    manifest_errors: list[str] = []
    manifest_warnings: list[str] = []
    reproducibility_errors: list[str] = []
    reproducibility_warnings: list[str] = []

    run_ids = [s.get("run_id") for s in summaries if s.get("run_id")]
    for rid, count in Counter(run_ids).items():
        if count > 1:
            provenance_errors.append(f"Duplicate canonical run_id detected: {rid} (count={count})")

    semantic_ids = [
        (s.get("meta") or {}).get("semantic_run_id")
        for s in summaries
        if (s.get("meta") or {}).get("semantic_run_id")
    ]
    for sid, count in Counter(semantic_ids).items():
        if count > 1:
            provenance_errors.append(
                f"Duplicate semantic_run_id detected: {sid} (count={count}) "
                "(semantic identity must be unique per archive run root)."
            )

    manifest_run_ids = [
        (s.get("meta") or {}).get("manifest_run_id")
        for s in summaries
        if (s.get("meta") or {}).get("manifest_run_id")
    ]
    for manifest_run_id, count in Counter(manifest_run_ids).items():
        if count > 1:
            reproducibility_warnings.append(
                f"Repeated manifest_run_id detected across run roots: {manifest_run_id} (count={count})."
            )

    for summary in summaries:
        run_id = summary.get("run_id", "unknown")
        meta = summary.get("meta") or {}
        csvs = summary.get("csvs") or {}
        manifest_diag = meta.get("manifest_diagnostics") or {}
        experiment = meta.get("experiment") or {}
        factors = normalize_experiment_factors(
            experiment.get("factors"),
            fallback_sentiment_enabled=experiment.get("sentiment_enabled"),
            fallback_missing_indicators_enabled=experiment.get("missing_indicators_enabled"),
            fallback_dl_enabled=experiment.get("dl_enabled", meta.get("dl_enabled")),
            fallback_msml_regime=experiment.get("msml_regime"),
        )

        manifest_count = manifest_diag.get("manifest_count") or 0
        if manifest_count > 1:
            manifest_errors.append(
                f"{run_id}: multiple manifests discovered in one run directory (count={manifest_count})."
            )
        if manifest_count == 0:
            manifest_warnings.append(
                f"{run_id}: missing manifest (legacy mode); semantics may be incomplete."
            )

        manifest_timestamp = manifest_diag.get("manifest_timestamp")
        identity_timestamp = meta.get("timestamp_utc")
        if manifest_timestamp and identity_timestamp and manifest_timestamp != identity_timestamp:
            manifest_errors.append(
                f"{run_id}: conflicting manifest timestamp ({manifest_timestamp}) vs identity timestamp ({identity_timestamp})."
            )

        files_found = meta.get("files_found") or []
        if not meta.get("manifest_present") and not files_found and not summary.get("log"):
            provenance_errors.append(f"{run_id}: malformed archive (no manifest, no CSV, no log).")

        variant = meta.get("run_variant") or LEGACY_VARIANT
        gen = meta.get("experiment_gen") or "unknown"
        manifest_present = bool(meta.get("manifest_present"))
        legacy_semantics = bool(meta.get("legacy_semantics"))
        if manifest_present:
            missing_required: list[str] = []
            if experiment.get("generation") is None:
                missing_required.append("generation")
            if experiment.get("variant") is None:
                missing_required.append("variant")
            if experiment.get("factors") is None:
                missing_required.append("factors")
            semantic_label = experiment.get("semantic_label")
            if not isinstance(semantic_label, str) or not semantic_label.strip():
                missing_required.append("semantic_label")
            if missing_required:
                semantic_warnings.append(
                    f"{run_id}: manifest experiment block incomplete (legacy tolerance): missing {', '.join(missing_required)}."
                )
            if legacy_semantics:
                semantic_warnings.append(
                    f"{run_id}: legacy_semantics=True; run marked as variant '{LEGACY_VARIANT}' and excluded from canonical semantic cohorts."
                )

        exp_generation = experiment.get("generation")
        exp_variant = experiment.get("variant")
        exp_sentiment = factors.get("sentiment_enabled")
        exp_missing = factors.get("missing_indicators_enabled")
        exp_semantic_label = experiment.get("semantic_label")
        if exp_generation is not None and exp_generation not in {"gen1", "gen2"}:
            semantic_errors.append(
                f"{run_id}: invalid experiment generation {exp_generation!r} (expected gen1|gen2)."
            )
        if exp_variant is not None and exp_variant not in VALID_EXPERIMENT_VARIANTS:
            semantic_warnings.append(
                f"{run_id}: non-canonical experiment variant {exp_variant!r} "
                f"(allowed canonical variants: {sorted(VALID_EXPERIMENT_VARIANTS)})."
            )
        if manifest_present:
            if exp_generation in {"gen1", "gen2"} and gen != exp_generation:
                semantic_errors.append(
                    f"{run_id}: identity corruption (meta experiment_gen={gen!r} does not match manifest.experiment.generation={exp_generation!r})."
                )
            if exp_variant in VALID_EXPERIMENT_VARIANTS and variant != exp_variant:
                semantic_errors.append(
                    f"{run_id}: identity corruption (meta run_variant={variant!r} does not match manifest.experiment.variant={exp_variant!r})."
                )
            if isinstance(exp_sentiment, bool) and meta.get("sentiment_enabled") != exp_sentiment:
                semantic_errors.append(
                    f"{run_id}: identity corruption (meta sentiment_enabled={meta.get('sentiment_enabled')!r} does not match manifest.experiment.factors.sentiment_enabled={exp_sentiment!r})."
                )
            if isinstance(exp_missing, bool) and meta.get("missing_indicators_enabled") != exp_missing:
                semantic_errors.append(
                    f"{run_id}: identity corruption (meta missing_indicators_enabled={meta.get('missing_indicators_enabled')!r} does not match manifest.experiment.factors.missing_indicators_enabled={exp_missing!r})."
                )
            if isinstance(exp_semantic_label, str) and exp_semantic_label.strip():
                if meta.get("semantic_label") != exp_semantic_label:
                    semantic_errors.append(
                        f"{run_id}: identity corruption (meta semantic_label={meta.get('semantic_label')!r} does not match manifest.experiment.semantic_label={exp_semantic_label!r})."
                    )
            if exp_generation in {"gen1", "gen2"} and exp_variant in VALID_EXPERIMENT_VARIANTS:
                expected_run_id_prefix = f"{exp_generation}_{exp_variant}__"
                if not run_id.startswith(expected_run_id_prefix):
                    semantic_errors.append(
                        f"{run_id}: identity corruption (canonical run_id must start with {expected_run_id_prefix!r})."
                    )
        if variant == LEGACY_VARIANT:
            semantic_warnings.append(
                f"{run_id}: variant unresolved ({LEGACY_VARIANT}); semantic comparisons may be skipped."
            )

        for warning in meta.get("identity_warnings") or []:
            if "conflict" in warning.lower():
                semantic_errors.append(f"{run_id}: {warning}")
            else:
                provenance_warnings.append(f"{run_id}: {warning}")

        expected_markers = {"walkforward_summary", "walkforward_per_fold"}
        missing = sorted(marker for marker in expected_markers if not csvs.get(marker))
        if missing:
            provenance_warnings.append(
                f"{run_id}: missing optional CSV sections: {', '.join(missing)}."
            )
        if not summary.get("log"):
            provenance_warnings.append(f"{run_id}: log file absent.")

        reproducibility = meta.get("reproducibility") or {}
        feature_ordering = meta.get("feature_ordering") or {}
        if manifest_present:
            missing_seed_keys = [
                key for key in _REQUIRED_REPRODUCIBILITY_KEYS
                if key not in reproducibility
            ]
            if missing_seed_keys:
                reproducibility_warnings.append(
                    f"{run_id}: missing reproducibility metadata key(s): {', '.join(missing_seed_keys)}."
                )

            phase_predictor_order = feature_ordering.get("phase_predictor_by_pair") or {}
            selector_order = feature_ordering.get("strategy_selector_by_pair") or {}
            if not phase_predictor_order:
                reproducibility_warnings.append(
                    f"{run_id}: missing phase predictor feature ordering metadata."
                )
            if not selector_order and meta.get("dl_enabled") is not None:
                reproducibility_warnings.append(
                    f"{run_id}: missing strategy selector feature ordering metadata."
                )

    # Check for duplicate semantic variant within same generation cohort
    semantic_variant_groups: dict[str, list[str]] = {}
    for summary in summaries:
        run_id = summary.get("run_id", "unknown")
        meta = summary.get("meta") or {}
        gen = meta.get("experiment_gen") or "unknown"
        variant = meta.get("run_variant") or LEGACY_VARIANT
        if variant == LEGACY_VARIANT or gen == "unknown":
            continue
        cohort_key = f"{gen}_{variant}"
        semantic_variant_groups.setdefault(cohort_key, []).append(run_id)

    for cohort_key, run_ids in sorted(semantic_variant_groups.items()):
        if len(run_ids) > 1:
            semantic_warnings.append(
                f"Duplicate semantic variant detected within cohort '{cohort_key}': "
                + ", ".join(sorted(run_ids))
                + " — all duplicate runs will be included in comparisons, "
                "which may produce misleading or non-comparable results. "
                "Verify this is intentional (re-run vs distinct experiment)."
            )

    if not summaries:
        provenance_errors.append("No summaries available after discovery/parsing.")

    reproducibility_groups: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        meta = summary.get("meta") or {}
        reproducibility = meta.get("reproducibility") or {}
        experiment_seed = reproducibility.get("experiment_seed")
        if experiment_seed is None:
            continue
        reproducibility_groups.setdefault(str(experiment_seed), []).append(summary)

    for seed, grouped in reproducibility_groups.items():
        if len(grouped) < 2:
            continue
        fingerprints = {
            summary.get("run_id", "unknown"): {
                key: (summary.get("meta") or {}).get("reproducibility", {}).get(key)
                for key in _REQUIRED_REPRODUCIBILITY_KEYS
            }
            for summary in grouped
        }
        if len({tuple(sorted(fp.items())) for fp in fingerprints.values()}) > 1:
            reproducibility_warnings.append(
                f"Runs sharing experiment_seed={seed} have differing reproducibility metadata: "
                + ", ".join(sorted(fingerprints.keys()))
            )

        pair_feature_orders: dict[tuple[str, str], dict[str, tuple[str, ...]]] = {}
        for summary in grouped:
            run_id = summary.get("run_id", "unknown")
            feature_ordering = ((summary.get("meta") or {}).get("feature_ordering") or {})
            for section_name in ("phase_predictor_by_pair", "strategy_selector_by_pair"):
                by_pair = feature_ordering.get(section_name) or {}
                for pair_name, columns in by_pair.items():
                    pair_feature_orders.setdefault((section_name, pair_name), {})[run_id] = tuple(columns or [])

        for (section_name, pair_name), orders in sorted(pair_feature_orders.items()):
            if len(set(orders.values())) > 1:
                reproducibility_warnings.append(
                    f"Runs sharing experiment_seed={seed} have differing feature column order "
                    f"for {section_name}:{pair_name}."
                )

    errors = provenance_errors + semantic_errors + manifest_errors + reproducibility_errors
    warnings = (
        provenance_warnings
        + semantic_warnings
        + manifest_warnings
        + reproducibility_warnings
    )
    diagnostics = [
        f"runs_total={len(summaries)}",
        f"errors={len(errors)}",
        f"warnings={len(warnings)}",
    ]

    return {
        "errors": errors,
        "warnings": warnings,
        "diagnostics": diagnostics,
        "sections": {
            "provenance_integrity": {
                "errors": provenance_errors,
                "warnings": provenance_warnings,
            },
            "semantic_integrity": {
                "errors": semantic_errors,
                "warnings": semantic_warnings,
            },
            "manifest_integrity": {
                "errors": manifest_errors,
                "warnings": manifest_warnings,
            },
            "reproducibility_integrity": {
                "errors": reproducibility_errors,
                "warnings": reproducibility_warnings,
            },
        },
    }
