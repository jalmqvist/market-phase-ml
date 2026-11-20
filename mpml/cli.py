"""Command-line parser for the MPML experiment runner."""

import argparse
from pathlib import Path
from typing import Type

from experiment_semantics import (
    VALID_EXPERIMENT_GENERATIONS,
    VALID_EXPERIMENT_VARIANTS,
)
from mpml.behavioral import registry as behavioral_registry
from src.repro import DEFAULT_EXPERIMENT_SEED
from src.strategy_registry import get_default_strategy_registry


def build_parser(
    parser_class: Type[argparse.ArgumentParser] = argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Build the MPML command-line parser with current registry choices."""
    parser = parser_class(
        description="Run MPML pipeline with run-owned output directories."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Explicit run output directory (writes all run artifacts here). "
            "If omitted, MPML creates results_archive/<gen>_<variant>__<timestamp>/."
        ),
    )
    parser.add_argument(
        "--experiment-generation",
        type=str,
        default=None,
        choices=sorted(VALID_EXPERIMENT_GENERATIONS),
        help="Optional generation consistency hint (gen1|gen2). Must match selected variant semantics if provided.",
    )
    parser.add_argument(
        "--experiment-variant",
        type=str,
        default=None,
        choices=sorted(VALID_EXPERIMENT_VARIANTS),
        help=(
            "Canonical experiment variant (A|B|C|D|E|F). "
            "Precedence: --experiment-variant > EXPERIMENT_VARIANT env > default A."
        ),
    )
    parser.add_argument(
        "--experiment-seed",
        type=int,
        default=None,
        help=(
            "Experiment RNG seed. Precedence: --experiment-seed > EXPERIMENT_SEED env "
            f"> default ({DEFAULT_EXPERIMENT_SEED})."
        ),
    )
    parser.add_argument(
        "--behavioral-surface",
        type=str,
        default=None,
        choices=behavioral_registry.available(),
        help=(
            "Behavioral Surface to use for this run. "
            "Precedence: --behavioral-surface > BEHAVIORAL_SURFACE env > 'trend_vol'. "
            f"Available: {behavioral_registry.available()}. "
            "Trend/Vol (trend_vol) is the default and preserves existing run behaviour."
        ),
    )
    parser.add_argument(
        "--recommendations-per-pair",
        action="store_true",
        default=False,
        help="Generate recommendations_per_pair.parquet from pair-scoped walk-forward evidence.",
    )
    parser.add_argument(
        "--recommendation-top-n",
        type=int,
        default=None,
        dest="recommendation_top_n",
        help=(
            "Return only the top N recommendations ordered by the default recommendation policy "
            "(sharpe_rank_v1). If omitted, all recommendations are returned. "
            "N must be a positive integer."
        ),
    )
    parser.add_argument(
        "--strategy",
        action="append",
        dest="strategy",
        default=None,
        metavar="STRATEGY_ID",
        help=(
            "Evaluate only this strategy ID. Repeatable: --strategy TF4 --strategy MR42. "
            "When omitted, the default evaluation scope from the active EvaluationPolicy "
            "is used (backward-compatible). "
            "IDs are validated against the Strategy Registry. "
            f"Available: {get_default_strategy_registry().available()}"
        ),
    )
    parser.add_argument(
        "--phaseaware-tf",
        action="append",
        dest="phaseaware_tf",
        default=None,
        metavar="STRATEGY_ID",
        help=(
            "Override the TrendFollowing representative used by PhaseAware for this run. "
            "Must be a Strategy Registry TrendFollowing ID, e.g. TF2. "
            "May be provided at most once."
        ),
    )
    parser.add_argument(
        "--phaseaware-mr",
        action="append",
        dest="phaseaware_mr",
        default=None,
        metavar="STRATEGY_ID",
        help=(
            "Override the MeanReversion representative used by PhaseAware for this run. "
            "Must be a Strategy Registry MeanReversion ID, e.g. MR2. "
            "May be provided at most once."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        dest="debug",
        default=False,
        help=(
            "Enable verbose debug output: [AWARENESS], [TRAINING DIAGNOSTICS], "
            "[DL FEATURE USAGE], and other detailed per-fold diagnostics that are "
            "suppressed by default for concise normal output."
        ),
    )
    parser.add_argument(
        "--export-dl-conditional-portfolio-timeline",
        action="store_true",
        dest="export_dl_conditional_portfolio_timeline",
        default=False,
        help=(
            "Opt-in: write dl_conditional_portfolio_timeline.csv (per pair/fold/test-bar "
            "realized dynamic-selector portfolio outcomes with dl_available) to the run "
            "output directory. Disabled by default; does not change strategy behavior."
        ),
    )
    return parser
