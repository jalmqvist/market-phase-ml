"""
benchmark.model

Domain model for MPML benchmark analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


# ---------------------------------------------------------------------
# PairResult
# ---------------------------------------------------------------------

@dataclass(slots=True)
class PairResult:
    """
    Aggregate PhaseAware result for one FX pair.
    Used for the baseline (in-sample aggregate).
    """
    pair: str

    total_return:  float
    sharpe:        float
    max_drawdown:  float

    win_rate:      float
    profit_factor: float
    trades:        int


# ---------------------------------------------------------------------
# WalkforwardPairResult
# ---------------------------------------------------------------------

@dataclass(slots=True)
class WalkforwardPairResult:
    """
    Walk-forward OOS result for one FX pair in one experiment.

    These are the values that actually vary across runs and are
    the correct basis for architectural uplift comparisons.

    Loaded from walkforward_results_per_pair__dl_enabled.csv.
    """
    pair:          str

    return_delta:  float   # OOS return uplift vs no-DL baseline
    sharpe_delta:  float   # OOS Sharpe uplift
    drawdown_delta: float  # OOS drawdown change (negative = better)

    folds:         int

    # absolute experiment values (for sensitivity mode)
    experiment_return: float = 0.0
    experiment_sharpe: float = 0.0
    experiment_drawdown: float = 0.0


# ---------------------------------------------------------------------
# DynamicResult
# ---------------------------------------------------------------------

@dataclass(slots=True)
class DynamicResult:
    """
    Final adaptive MPML trading result.
    Loaded from dynamic_selector_results_per_pair.
    """
    pair:         str
    strategy:     str

    total_return: float
    sharpe:       float
    max_drawdown: float
    trades:       int


# ---------------------------------------------------------------------
# DynamicComparison
# ---------------------------------------------------------------------

@dataclass(slots=True)
class DynamicComparison:
    """
    Baseline vs Dynamic selector comparison.
    Loaded from baseline_vs_dynamic_comparison.
    """
    pair: str

    baseline_return:  float
    dynamic_return:   float
    return_delta:     float

    baseline_sharpe:  float
    dynamic_sharpe:   float
    sharpe_delta:     float

    baseline_drawdown: float
    dynamic_drawdown:  float
    drawdown_delta:    float


# ---------------------------------------------------------------------
# WalkforwardResult  (fold-level summary, not per-pair OOS)
# ---------------------------------------------------------------------

@dataclass(slots=True)
class WalkforwardResult:
    """
    Fold-averaged adaptive uplift.
    Loaded from walkforward_results_per_pair.
    """
    pair:           str

    return_delta:   float
    sharpe_delta:   float
    drawdown_delta: float

    folds:          int


# ---------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------

@dataclass(slots=True)
class ExperimentResult:
    """
    One benchmark experiment.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    name:           str
    architecture:   str
    pair_family:    str
    representation: str
    state:          str
    feature_set:    str
    sentiment_surface: str = "unknown"

    evaluation_population: list[str] = field(default_factory=list)
    folder: str = ""

    # ------------------------------------------------------------------
    # Benchmark results
    # ------------------------------------------------------------------

    #
    # Aggregate baseline PhaseAware results (in-sample).
    # Populated for the baseline experiment only.
    # Experiments use wf_pairs instead.
    #
    pairs: Dict[str, PairResult] = field(default_factory=dict)

    #
    # Walk-forward OOS results per pair.
    # Populated for DL experiments.
    # These are the values that vary across runs.
    #
    wf_pairs: Dict[str, WalkforwardPairResult] = field(
        default_factory=dict
    )

    #
    # Final adaptive selector results
    #
    dynamic_results: Dict[str, DynamicResult] = field(
        default_factory=dict
    )

    #
    # Baseline vs Dynamic comparison
    #
    dynamic_comparison: Dict[str, DynamicComparison] = field(
        default_factory=dict
    )

    #
    # Walk-forward benchmark (fold summary — kept for diagnostics)
    #
    walkforward_results: Dict[str, WalkforwardResult] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    selector_metrics:    dict = field(default_factory=dict)
    walkforward_metrics: dict = field(default_factory=dict)
    metadata:            dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def pair(self, pair: str) -> PairResult:
        return self.pairs[pair]

    def wf_pair(self, pair: str) -> WalkforwardPairResult:
        return self.wf_pairs[pair]

    @property
    def pair_names(self) -> list[str]:
        #
        # Use wf_pairs when available (experiments),
        # fall back to pairs (baseline).
        #
        if self.wf_pairs:
            return sorted(self.wf_pairs.keys())
        return sorted(self.pairs.keys())

    @property
    def n_pairs(self) -> int:
        return len(self.pair_names)

    @property
    def target_pairs(self) -> list[str]:
        return sorted(self.evaluation_population)

    @property
    def control_pairs(self) -> list[str]:
        return sorted(
            set(self.pair_names)
            -
            set(self.evaluation_population)
        )


# ---------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------

@dataclass(slots=True)
class Benchmark:
    """
    Complete benchmark archive.
    """
    baseline:    ExperimentResult
    experiments: list[ExperimentResult]

    @property
    def architectures(self) -> list[str]:
        return sorted(
            {e.architecture for e in self.experiments}
        )

    @property
    def pair_families(self) -> list[str]:
        return sorted(
            {e.pair_family for e in self.experiments}
        )

    @property
    def representations(self) -> list[str]:
        return sorted(
            {e.representation for e in self.experiments}
        )