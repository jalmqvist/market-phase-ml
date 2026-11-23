"""Fast launch validation; this does not resolve artifacts or start an experiment.

Run with ``python -m mpml.preflight`` and the same arguments as ``main.py``.
Argparse reports only its first syntax error. Existence checks are advisory and
race-prone; the run still performs its atomic output-directory creation.
Prediction-artifact loading and compatibility are not checked.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
from typing import Any

from src.launch_env import (
    dl_surface_target_horizon,
    min_dl_train_coverage_pct,
)

ISSUE_CODE_SUMMARIES = {
    "ACTIVE_PAIRS_UNKNOWN": "Configured pair names are not in the supported pair universe.",
    "ARGPARSE": "Arguments are not accepted by the MPML command-line parser.",
    "DL_ALIAS_SUPPRESSED": "The primary prediction artifact setting suppresses its legacy alias.",
    "DL_ARTIFACT_PATH_MISSING": "The explicitly configured prediction artifact does not exist.",
    "ENV_BOOL_PARSED_FALSE": "A DL boolean value looks truthy but is parsed as false.",
    "ENV_INVALID_VALUE": "An environment value cannot be parsed or is unsupported.",
    "GENERATION_VARIANT_MISMATCH": "The requested generation conflicts with the selected variant.",
    "OUTPUT_DIR_EXISTS": "The selected run output directory already exists.",
    "OUTPUT_NAME_INFERS_AWARENESS": "The output name supplies an imputation-awareness hint.",
    "PHASEAWARE_REPEATED": "A PhaseAware representative option was specified more than once.",
    "PHASEAWARE_ROLE_INVALID": "A PhaseAware representative does not match its required role.",
    "STRATEGY_UNKNOWN": "A requested strategy ID is not in the strategy registry.",
    "SURFACE_UNKNOWN": "The requested behavioral surface is not registered.",
    "TIMELINE_NEEDS_DYNAMIC_SELECTOR": "Timeline export requires an active dynamic selector.",
    "TOP_N_NOT_POSITIVE": "The recommendation limit must be a positive integer.",
}


class _ArgumentError(Exception):
    pass


class _RaisingParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _ArgumentError(message)


def _issue(
    code: str,
    source: str,
    message: str,
    severity: str = "error",
) -> dict[str, str]:
    return {
        "code": code,
        "severity": severity,
        "source": source,
        "message": message,
    }


def _validate(argv: list[str] | None = None) -> tuple[list[dict[str, str]], bool]:
    issues: list[dict[str, str]] = []
    malformed_import_values = False
    for name, converter in (
        ("MIN_DL_TRAIN_COVERAGE_PCT", min_dl_train_coverage_pct),
        ("DL_SURFACE_TARGET_HORIZON", dl_surface_target_horizon),
    ):
        try:
            converter()
        except (TypeError, ValueError) as exc:
            issues.append(_issue("ENV_INVALID_VALUE", name, f"Invalid {name} value: {exc}"))
            malformed_import_values = True

    if malformed_import_values:
        return issues, False

    from mpml.cli import build_parser, validate_phaseaware_repetitions
    from src.recommendation import validate_recommendation_top_n
    from src.repro import resolve_experiment_seed

    parser = build_parser(parser_class=_RaisingParser)
    try:
        args = parser.parse_args(argv)
    except _ArgumentError as exc:
        return [_issue("ARGPARSE", "argv", str(exc))], True

    try:
        validate_phaseaware_repetitions(args)
    except ValueError as exc:
        issues.append(_issue("PHASEAWARE_REPEATED", "phaseaware flags", str(exc)))

    try:
        validate_recommendation_top_n(args.recommendation_top_n)
    except ValueError as exc:
        issues.append(_issue("TOP_N_NOT_POSITIVE", "--recommendation-top-n", str(exc)))

    try:
        resolve_experiment_seed(cli_seed=args.experiment_seed)
    except ValueError as exc:
        issues.append(_issue("ENV_INVALID_VALUE", "EXPERIMENT_SEED", str(exc)))

    # Importing main exposes the same launch checks used at run time, but no
    # function below constructs the data pipeline or resolves prediction data.
    with redirect_stdout(io.StringIO()):
        import main

    try:
        main.resolve_market_data_source()
    except ValueError as exc:
        issues.append(_issue("ENV_INVALID_VALUE", "MPML_DATA_SOURCE", str(exc)))

    if (
        os.environ.get("DL_SIGNALS_ENABLED", "false").lower() in {"1", "yes", "on"}
    ):
        issues.append(
            _issue(
                "ENV_BOOL_PARSED_FALSE",
                "DL_SIGNALS_ENABLED",
                "DL_SIGNALS_ENABLED is parsed as false; use 'true' to enable DL signals.",
                "warning",
            )
        )

    active_pairs = {
        pair.strip().upper()
        for pair in os.environ.get("ACTIVE_PAIRS", "").split(",")
        if pair.strip()
    }
    unknown_pairs = sorted(active_pairs - set(main.PAIR_NAMES.values()))
    if unknown_pairs:
        issues.append(
            _issue(
                "ACTIVE_PAIRS_UNKNOWN",
                "ACTIVE_PAIRS",
                f"Unknown active pair name(s) {unknown_pairs}; these names are silently dropped.",
                "warning",
            )
        )

    if (
        "DL_PREDICTION_ARTIFACT_PATH" in os.environ
        and "DL_SIGNALS_CUBE_PATH" in os.environ
    ):
        issues.append(
            _issue(
                "DL_ALIAS_SUPPRESSED",
                "DL_PREDICTION_ARTIFACT_PATH",
                "DL_PREDICTION_ARTIFACT_PATH is set, so DL_SIGNALS_CUBE_PATH is suppressed.",
                "warning",
            )
        )

    explicit_artifact = os.environ.get("DL_PREDICTION_ARTIFACT_PATH", "").strip()
    if (
        os.environ.get("DL_SIGNALS_ENABLED", "false").lower() == "true"
        and explicit_artifact
        and not Path(explicit_artifact).expanduser().exists()
    ):
        issues.append(
            _issue(
                "DL_ARTIFACT_PATH_MISSING",
                "DL_PREDICTION_ARTIFACT_PATH",
                f"Explicit DL prediction artifact path does not exist: {explicit_artifact!r}.",
                "warning",
            )
        )

    try:
        surface_id, _surface = main._resolve_behavioral_surface(
            args.behavioral_surface
        )
    except ValueError as exc:
        issues.append(_issue("SURFACE_UNKNOWN", "behavioral surface", str(exc)))
        surface_id = None

    phaseaware_configuration = None
    try:
        phaseaware_configuration = main.resolve_phaseaware_configuration(
            main.DEFAULT_PHASEAWARE_POLICY_ID,
            phaseaware_tf_strategy_id=(
                args.phaseaware_tf[0] if args.phaseaware_tf else None
            ),
            phaseaware_mr_strategy_id=(
                args.phaseaware_mr[0] if args.phaseaware_mr else None
            ),
        )
    except ValueError as exc:
        issues.append(
            _issue("PHASEAWARE_ROLE_INVALID", "PhaseAware configuration", str(exc))
        )

    scope = None
    if surface_id is not None:
        try:
            scope = main.resolve_evaluation_scope(
                requested_strategy_ids=args.strategy,
                registry=main.get_default_strategy_registry(),
                policy_registry=main.get_default_policy_registry(),
                surface_id=surface_id,
            )
        except ValueError as exc:
            issues.append(_issue("STRATEGY_UNKNOWN", "--strategy", str(exc)))

    selected_variant = (
        args.experiment_variant
        if args.experiment_variant is not None
        else os.getenv("EXPERIMENT_VARIANT", "A")
    )
    try:
        experiment_meta = main._build_experiment_metadata(variant=selected_variant)
    except ValueError as exc:
        issues.append(
            _issue("GENERATION_VARIANT_MISMATCH", "EXPERIMENT_VARIANT", str(exc))
        )
        experiment_meta = None

    generation_hint = (
        args.experiment_generation
        if args.experiment_generation is not None
        else os.getenv("EXPERIMENT_GENERATION")
    )
    if generation_hint is not None and experiment_meta is not None:
        try:
            main._validate_generation_for_variant(generation_hint, experiment_meta)
        except ValueError as exc:
            issues.append(
                _issue(
                    "GENERATION_VARIANT_MISMATCH",
                    "EXPERIMENT_GENERATION",
                    str(exc),
                )
            )

    if scope is not None and phaseaware_configuration is not None:
        plan = main._build_walkforward_execution_plan(
            scope,
            phaseaware_configuration=phaseaware_configuration,
        )
        if (
            args.export_dl_conditional_portfolio_timeline
            and not plan["run_dynamic_selector"]
        ):
            issues.append(
                _issue(
                    "TIMELINE_NEEDS_DYNAMIC_SELECTOR",
                    "--export-dl-conditional-portfolio-timeline",
                    "The dynamic selector is inactive for this evaluation scope.",
                )
            )

    explicit_output = args.output_dir
    if explicit_output is None:
        env_output = os.getenv("MPML_OUTPUT_DIR")
        explicit_output = Path(env_output) if env_output else None
    output_source = (
        "--output-dir" if args.output_dir is not None else "MPML_OUTPUT_DIR"
    )
    if explicit_output is not None and explicit_output.exists():
        issues.append(
            _issue(
                "OUTPUT_DIR_EXISTS",
                output_source,
                f"Output directory already exists: {explicit_output}.",
            )
        )

    output_hint = str(
        args.output_dir or os.getenv("MPML_OUTPUT_DIR") or ""
    ).strip()
    awareness = main.infer_imputation_awareness_from_name(
        Path(output_hint).name if output_hint else None
    )
    if awareness is not None:
        issues.append(
            _issue(
                "OUTPUT_NAME_INFERS_AWARENESS",
                output_source,
                "The output name supplies an imputation-awareness hint that overrides "
                "the selected variant's default.",
                "warning",
            )
        )

    return issues, False


def run_preflight(argv: list[str] | None = None) -> tuple[int, dict[str, Any]]:
    issues, argparse_failed = _validate(argv)
    has_errors = any(issue["severity"] == "error" for issue in issues)
    result = {"schema_version": 1, "issues": issues}
    if argparse_failed:
        return 2, result
    return (1 if has_errors else 0), result


def main(argv: list[str] | None = None) -> int:
    exit_code, result = run_preflight(argv)
    print(json.dumps(result))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
