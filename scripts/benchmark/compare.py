"""
benchmark.compare

Scientific comparison routines.

No printing. No pandas. No file I/O.

Consumes Benchmark objects and produces comparison objects
used by report.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median

from .model import (
    Benchmark,
    ExperimentResult,
    PairResult,
    WalkforwardPairResult,
)


# ---------------------------------------------------------------------
# Comparison dataclasses
# ---------------------------------------------------------------------

@dataclass(slots=True)
class PairComparison:
    """
    Walk-forward OOS comparison for one FX pair in one experiment.

    baseline        : aggregate PhaseAware result (no DL)
    wf_result       : walk-forward OOS result (with DL)

    The uplift values come directly from the walk-forward file —
    they are the actual OOS deltas, not a subtraction of aggregate
    in-sample values.
    """
    pair: str

    baseline:  PairResult
    wf_result: WalkforwardPairResult

    return_uplift:   float
    sharpe_uplift:   float
    drawdown_uplift: float


@dataclass(slots=True)
class ExperimentComparison:

    experiment: ExperimentResult

    target_pairs:  list[PairComparison]
    control_pairs: list[PairComparison]

    selector_metrics:    dict
    walkforward_metrics: dict


# ---------------------------------------------------------------------
# Pair comparison
# ---------------------------------------------------------------------

def compare_pair(
    baseline:         PairResult,
    wf_result:        WalkforwardPairResult,
    sensitivity_mode: bool = False,
) -> PairComparison:
    """
    Build a PairComparison.

    Normal mode
    -----------
    Uses pre-baked deltas from walkforward_results_per_pair__dl_enabled.csv.
    These were computed by MPML against the baseline that was active
    at experiment run time.

    Sensitivity mode
    ----------------
    Recomputes deltas on-the-fly using the currently loaded baseline.
    This allows swapping the baseline folder without re-running MPML.
    Requires absolute experiment values to be present on wf_result
    (populated by load_absolute_experiment_values in loader.py).

    CAUTION: In sensitivity mode, the computed deltas compare:
        - Experiment OOS values (from dynamic_selector_results_per_pair)
        - Baseline in-sample aggregate values (from results_per_pair__baseline)

    These are NOT from the same evaluation window. Sensitivity-mode deltas
    are therefore not directly comparable to normal-mode pre-baked deltas.
    Use sensitivity mode only when you explicitly need to test how results
    would change against a different baseline, accepting this semantic
    limitation.
    """
    if sensitivity_mode:
        if wf_result.experiment_return == 0.0 \
                and wf_result.experiment_sharpe == 0.0 \
                and wf_result.experiment_drawdown == 0.0:
            raise RuntimeError(
                f"\nSensitivity mode requested but absolute experiment values "
                f"are missing for {wf_result.pair}.\n\n"
                f"Check that dynamic_selector_results_per_pair__dl_enabled.csv "
                f"exists and was loaded correctly."
            )
        return_uplift   = wf_result.experiment_return   - baseline.total_return
        sharpe_uplift   = wf_result.experiment_sharpe   - baseline.sharpe
        drawdown_uplift = wf_result.experiment_drawdown - baseline.max_drawdown
    else:
        return_uplift   = wf_result.return_delta
        sharpe_uplift   = wf_result.sharpe_delta
        drawdown_uplift = wf_result.drawdown_delta

    return PairComparison(
        pair            = baseline.pair,
        baseline        = baseline,
        wf_result       = wf_result,
        return_uplift   = return_uplift,
        sharpe_uplift   = sharpe_uplift,
        drawdown_uplift = drawdown_uplift,
    )


# ---------------------------------------------------------------------
# Whole benchmark
# ---------------------------------------------------------------------

def compare_to_baseline(
    benchmark:        Benchmark,
    sensitivity_mode: bool = False,
) -> list[ExperimentComparison]:

    comparisons = []
    baseline    = benchmark.baseline

    for experiment in benchmark.experiments:

        target   = []
        controls = []

        population = set(experiment.evaluation_population)

        for pair_name in baseline.pair_names:

            if pair_name not in experiment.wf_pairs:
                continue

            comparison = compare_pair(
                baseline         = baseline.pair(pair_name),
                wf_result        = experiment.wf_pair(pair_name),
                sensitivity_mode = sensitivity_mode,
            )

            if pair_name in population:
                target.append(comparison)
            else:
                controls.append(comparison)

        comparisons.append(
            ExperimentComparison(
                experiment          = experiment,
                target_pairs        = target,
                control_pairs       = controls,
                selector_metrics    = experiment.selector_metrics,
                walkforward_metrics = experiment.walkforward_metrics,
            )
        )

    return comparisons


# ---------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------

def _mean(values) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def _median(values) -> float:
    values = list(values)
    return median(values) if values else 0.0


# ---------------------------------------------------------------------
# Target statistics
# ---------------------------------------------------------------------

def target_return(comparison: ExperimentComparison) -> float:
    return _mean(p.return_uplift for p in comparison.target_pairs)

def target_sharpe(comparison: ExperimentComparison) -> float:
    return _mean(p.sharpe_uplift for p in comparison.target_pairs)

def target_drawdown(comparison: ExperimentComparison) -> float:
    return _mean(p.drawdown_uplift for p in comparison.target_pairs)

def target_return_median(comparison: ExperimentComparison) -> float:
    return _median(p.return_uplift for p in comparison.target_pairs)

def target_sharpe_median(comparison: ExperimentComparison) -> float:
    return _median(p.sharpe_uplift for p in comparison.target_pairs)


# ---------------------------------------------------------------------
# Control statistics
# ---------------------------------------------------------------------

def control_return(comparison: ExperimentComparison) -> float:
    return _mean(p.return_uplift for p in comparison.control_pairs)

def control_sharpe(comparison: ExperimentComparison) -> float:
    return _mean(p.sharpe_uplift for p in comparison.control_pairs)

def control_drawdown(comparison: ExperimentComparison) -> float:
    return _mean(p.drawdown_uplift for p in comparison.control_pairs)


# ---------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------

def positive_target_pairs(comparison: ExperimentComparison) -> int:
    return sum(p.sharpe_uplift > 0 for p in comparison.target_pairs)

def positive_control_pairs(comparison: ExperimentComparison) -> int:
    return sum(p.sharpe_uplift > 0 for p in comparison.control_pairs)


# ---------------------------------------------------------------------
# Specialization
# ---------------------------------------------------------------------

def sharpe_specialization(comparison: ExperimentComparison) -> float:
    return target_sharpe(comparison) - control_sharpe(comparison)

def return_specialization(comparison: ExperimentComparison) -> float:
    return target_return(comparison) - control_return(comparison)


# ---------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------

def benchmark_scorecard(comparison: ExperimentComparison) -> dict:
    """
    Compact experiment summary.
    """
    return {
        "Target Return":         target_return(comparison),
        "Target Return Median":  target_return_median(comparison),
        "Target Sharpe":         target_sharpe(comparison),
        "Target Sharpe Median":  target_sharpe_median(comparison),
        "Target Drawdown":       target_drawdown(comparison),
        "Control Return":        control_return(comparison),
        "Control Sharpe":        control_sharpe(comparison),
        "Control Drawdown":      control_drawdown(comparison),
        "Positive Target":       positive_target_pairs(comparison),
        "Positive Controls":     positive_control_pairs(comparison),
        "Sharpe Separation":     sharpe_specialization(comparison),
        "Return Separation":     return_specialization(comparison),
    }


# ---------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------

def rank_by_sharpe(comparisons: list) -> list:
    return sorted(
        comparisons,
        key=lambda c: target_sharpe(c),
        reverse=True,
    )

def rank_by_specialization(comparisons: list) -> list:
    return sorted(
        comparisons,
        key=lambda c: sharpe_specialization(c),
        reverse=True,
    )