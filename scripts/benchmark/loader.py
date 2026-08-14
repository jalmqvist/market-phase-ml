"""
benchmark.loader

Loads MPML benchmark archives into the benchmark domain model.

This is intentionally the ONLY module that understands
- MPML manifests
- CSV filenames / suffixes
- Behavioral Surface identifiers
- PhaseAware strategy extraction

Every other module operates purely on domain model objects.
"""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import pandas as pd

from .model import (
    Benchmark,
    ExperimentResult,
    PairResult,
    WalkforwardPairResult,
    DynamicResult,
    DynamicComparison,
    WalkforwardResult,
)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

BASELINE_FOLDER   = "baseline"
MANIFEST_FILE     = "run_manifest.json"
PHASEAWARE_PREFIX = "PhaseAware"

#
# Evaluation populations
#
# Both 'reactive_jpy' and 'trend_vol' target the same three JPY pairs.
# The distinction is the Behavioral Surface representation, not the
# pair universe.
#

EVALUATION_POPULATIONS: dict[str, list[str]] = {
    "reactive_jpy": ["EURJPY", "GBPJPY", "USDJPY"],
    "trend_vol":    ["EURJPY", "GBPJPY", "USDJPY"],
}


# ---------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------

def detect_suffix(folder: Path) -> str:
    """
    Detect the CSV filename suffix for this experiment folder.

    Priority order:
        __dl_enabled  →  DL experiment run
        __baseline    →  No-DL baseline run
        (empty)       →  Legacy / no suffix

    Raises FileNotFoundError if no recognized pattern is found.
    Does NOT fall back to glob — prevents silently loading wrong files.
    """
    patterns = [
        ("results_per_pair__dl_enabled.csv",  "__dl_enabled"),
        ("results_per_pair__baseline.csv",    "__baseline"),
        ("results_per_pair.csv",              ""),
    ]
    for filename, suffix in patterns:
        if (folder / filename).exists():
            # If this is the baseline folder, warn if it's returning __dl_enabled
            if folder.name == BASELINE_FOLDER and suffix == "__dl_enabled":
                warnings.warn(
                    f"Baseline folder {folder.name} contains __dl_enabled file — "
                    "this may cause data contamination."
                )
            return suffix

    available = sorted(f.name for f in folder.glob("results_per_pair*.csv"))
    raise FileNotFoundError(
        f"No recognized results_per_pair CSV in:\n\n{folder}\n\n"
        f"Expected one of: {[p[0] for p in patterns]}\n"
        f"Found: {available or 'none'}"
    )


def load_manifest(folder: Path) -> dict:
    manifest = folder / MANIFEST_FILE
    if not manifest.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n\n{manifest}"
        )
    with open(manifest) as f:
        return json.load(f)


def load_csv(
    folder: Path,
    prefix: str,
    suffix: str,
) -> pd.DataFrame:
    """
    Load  folder / (prefix + suffix + '.csv').

    Strict: the exact file must exist. No glob fallback.
    Raises FileNotFoundError with a helpful message on failure.
    """
    path = folder / f"{prefix}{suffix}.csv"
    if not path.exists():
        available = sorted(
            f.name for f in folder.glob(f"{prefix}*.csv")
        )
        raise FileNotFoundError(
            f"Required CSV not found:\n\n{path}\n\n"
            f"Available: {available or 'none'}"
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------
# Strategy extraction
# ---------------------------------------------------------------------

def extract_phaseaware(
    results: pd.DataFrame,
    folder: Path,
) -> pd.DataFrame:
    """
    Extract the final PhaseAware strategy rows from an aggregate
    results CSV.

    One row per pair is expected. Raises RuntimeError if missing
    or duplicated.
    """
    phaseaware = results[
        results["Strategy"]
        .astype(str)
        .str.startswith(PHASEAWARE_PREFIX)
    ].copy()

    if len(phaseaware) == 0:
        available = results["Strategy"].unique().tolist()
        raise RuntimeError(
            f"No PhaseAware strategy found in:\n\n{folder}\n\n"
            f"Available strategies: {available}"
        )

    duplicated = phaseaware["Pair"].duplicated()
    if duplicated.any():
        raise RuntimeError(
            f"Duplicate PhaseAware rows in:\n\n{folder}\n\n"
            f"{phaseaware[duplicated]}"
        )

    return phaseaware


# ---------------------------------------------------------------------
# Representation mapping
# ---------------------------------------------------------------------

def evaluation_population(representation: str) -> list[str]:
    """
    Convert a Behavioral Surface representation string into the
    FX pair evaluation population.
    """
    key = representation.strip().lower()
    try:
        return EVALUATION_POPULATIONS[key]
    except KeyError:
        known = list(EVALUATION_POPULATIONS.keys())
        raise RuntimeError(
            f"Unknown Behavioral Surface: '{representation}'\n\n"
            f"Known surfaces: {known}"
        )


# ---------------------------------------------------------------------
# Baseline loader
# ---------------------------------------------------------------------

def load_baseline(folder: Path) -> ExperimentResult:
    """
    Load the no-DL baseline experiment.

    The baseline uses:
        results_per_pair__baseline.csv  →  aggregate PhaseAware results
                                           stored in experiment.pairs

    The baseline has no walk-forward DL results (no wf_pairs).
    """
    manifest = load_manifest(folder)
    surface  = manifest["experiment_surface"]

    # FOR BASELINE: always use __baseline suffix — no fallback
    suffix = "__baseline"

    architecture   = surface["artifact_model"]
    representation = surface["behavioral_surface"]
    pair_family    = surface.get("training_pair_family", "unknown")

    state = (
        surface.get("behavioral_state")
        or surface.get("msml_regime")
        or "unknown"
    )
    feature_set       = surface.get("feature_surface", "unknown")
    sentiment_surface = surface.get("sentiment_surface", "unknown")

    population = evaluation_population(representation)

    # ------------------------------------------------------------------
    # Aggregate PhaseAware results
    # ------------------------------------------------------------------

    results = load_csv(folder, "results_per_pair", suffix)
    results = extract_phaseaware(results, folder)

    experiment = ExperimentResult(
        name                  = folder.name,
        architecture          = architecture,
        pair_family           = pair_family,
        representation        = representation,
        state                 = state,
        feature_set           = feature_set,
        sentiment_surface     = sentiment_surface,
        evaluation_population = population,
        folder                = str(folder),
    )

    for _, row in results.iterrows():
        pair = PairResult(
            pair          = row["Pair"],
            total_return  = float(row["Total Return (%)"]),
            sharpe        = float(row["Sharpe Ratio"]),
            max_drawdown  = float(row["Max Drawdown (%)"]),
            win_rate      = float(row["Win Rate (%)"]),
            profit_factor = float(row["Profit Factor"]),
            trades        = int(row["N Trades"]),
        )
        experiment.pairs[pair.pair] = pair

    # ------------------------------------------------------------------
    # Baseline also has dynamic selector and walkforward diagnostics
    # ------------------------------------------------------------------

    _load_optional_diagnostics(experiment, folder, suffix)

    experiment.metadata = manifest

    # Sanity: all evaluation pairs present
    _check_population(experiment, population, folder)

    return experiment


# ---------------------------------------------------------------------
# Experiment loader
# ---------------------------------------------------------------------
def load_absolute_experiment_values(folder: Path, suffix: str) -> dict:
    """
    Load absolute experiment values from dynamic_selector_results_per_pair.
    These are the actual OOS values for the experiment.
    """
    try:
        df = load_csv(folder, "dynamic_selector_results_per_pair", suffix)
        return {
            row["Pair"]: {
                "return": float(row["Total Return (%)"]),
                "sharpe": float(row["Sharpe"]),
                "drawdown": float(row["Max DD (%)"]),
            }
            for _, row in df.iterrows()
        }
    except FileNotFoundError:
        return {}

def load_experiment(folder: Path) -> ExperimentResult:
    """
    Load a single DL benchmark experiment.

    Uses:
        walkforward_results_per_pair__dl_enabled.csv
            → per-pair OOS deltas stored in experiment.wf_pairs
              These are the values that vary across runs.
    """
    manifest = load_manifest(folder)
    surface  = manifest["experiment_surface"]
    suffix   = detect_suffix(folder)

    # If we detected __baseline in an experiment folder, warn — this is likely a mistake
    if suffix == "__baseline":
        warnings.warn(
            f"Experiment folder {folder.name} uses __baseline suffix — "
            "this may indicate a misconfigured run or file overlap."
        )

    architecture   = surface["artifact_model"]
    representation = surface["behavioral_surface"]
    pair_family    = surface.get("training_pair_family", "unknown")

    state = (
        surface.get("behavioral_state")
        or surface.get("msml_regime")
        or "unknown"
    )
    feature_set       = surface.get("feature_surface", "unknown")
    sentiment_surface = surface.get("sentiment_surface", "unknown")

    population = evaluation_population(representation)

    experiment = ExperimentResult(
        name                  = folder.name,
        architecture          = architecture,
        pair_family           = pair_family,
        representation        = representation,
        state                 = state,
        feature_set           = feature_set,
        sentiment_surface     = sentiment_surface,
        evaluation_population = population,
        folder                = str(folder),
    )

    # ------------------------------------------------------------------
    # Walk-forward OOS results  ← the correct source for Section 1
    # ------------------------------------------------------------------

    wf = load_csv(folder, "walkforward_results_per_pair", suffix)

    # Load absolute experiment values for sensitivity mode
    # These are loaded once here and stored on each WalkforwardPairResult
    absolute_values = load_absolute_experiment_values(folder, suffix)

    for _, row in wf.iterrows():
        pair_name = row["Pair"]
        abs_val = absolute_values.get(pair_name, {})

        result = WalkforwardPairResult(
            pair=pair_name,
            return_delta=float(row["Return Δ"]),
            sharpe_delta=float(row["Sharpe Δ"]),
            drawdown_delta=float(row["DD Δ"]),
            folds=int(row["Folds"]),
            experiment_return=abs_val.get("return", 0.0),
            experiment_sharpe=abs_val.get("sharpe", 0.0),
            experiment_drawdown=abs_val.get("drawdown", 0.0),
        )
        experiment.wf_pairs[result.pair] = result

    # ------------------------------------------------------------------
    # Optional diagnostics
    # ------------------------------------------------------------------

    _load_optional_diagnostics(experiment, folder, suffix)

    experiment.metadata = manifest

    # Sanity: all evaluation pairs present in wf_pairs
    expected  = set(population)
    available = set(experiment.wf_pairs.keys())
    missing   = expected - available

    if missing:
        raise RuntimeError(
            f"\nMissing walk-forward pairs\n\n"
            f"{folder.name}\n\n"
            f"{sorted(missing)}"
        )

    return experiment


# ---------------------------------------------------------------------
# Shared diagnostic loader
# ---------------------------------------------------------------------

def _load_optional_diagnostics(
    experiment: ExperimentResult,
    folder: Path,
    suffix: str,
) -> None:
    """
    Load optional secondary files that are present in both baseline
    and experiment folders.
    """

    # Dynamic selector results
    try:
        df = load_csv(
            folder,
            "dynamic_selector_results_per_pair",
            suffix,
        )
        for _, row in df.iterrows():
            result = DynamicResult(
                pair         = row["Pair"],
                strategy     = row["Strategy"],
                total_return = float(row["Total Return (%)"]),
                sharpe       = float(row["Sharpe"]),
                max_drawdown = float(row["Max DD (%)"]),
                trades       = int(row["Num Trades"]),
            )
            experiment.dynamic_results[result.pair] = result
    except FileNotFoundError:
        pass

    # Baseline vs Dynamic comparison
    try:
        df = load_csv(
            folder,
            "baseline_vs_dynamic_comparison",
            suffix,
        )
        for _, row in df.iterrows():
            result = DynamicComparison(
                pair              = row["Pair"],
                baseline_return   = float(row["Baseline Return"]),
                dynamic_return    = float(row["Dynamic Return"]),
                return_delta      = float(row["Return Δ"]),
                baseline_sharpe   = float(row["Baseline Sharpe"]),
                dynamic_sharpe    = float(row["Dynamic Sharpe"]),
                sharpe_delta      = float(row["Sharpe Δ"]),
                baseline_drawdown = float(row["Baseline Max DD"]),
                dynamic_drawdown  = float(row["Dynamic Max DD"]),
                drawdown_delta    = float(row["DD Δ"]),
            )
            experiment.dynamic_comparison[result.pair] = result

        # Also expose as flat dict for report.py Section 2
        experiment.selector_metrics = {
            pair: row.to_dict()
            for pair, row in
            df.set_index("Pair").iterrows()
        }
    except FileNotFoundError:
        pass

    # Walk-forward summary
    try:
        wf = load_csv(
            folder,
            "walkforward_results_summary",
            suffix,
        )
        experiment.walkforward_metrics = wf.iloc[0].to_dict()
    except FileNotFoundError:
        experiment.walkforward_metrics = {}


# ---------------------------------------------------------------------
# Population sanity check
# ---------------------------------------------------------------------

def _check_population(
    experiment: ExperimentResult,
    population: list[str],
    folder: Path,
) -> None:
    expected  = set(population)
    available = set(experiment.pair_names)
    missing   = expected - available

    if missing:
        raise RuntimeError(
            f"\nMissing evaluation pairs\n\n"
            f"{folder.name}\n\n"
            f"{sorted(missing)}"
        )


# ---------------------------------------------------------------------
# Benchmark loader
# ---------------------------------------------------------------------

def load_benchmark(root) -> Benchmark:
    """
    Load an entire benchmark archive.

    Expected layout:

        results_archive/
            baseline/
            gen1_A__20260731T194740Z/
            gen1_A__20260730T093156Z/
            ...
    """
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(root)

    baseline_dir = root / BASELINE_FOLDER
    if not baseline_dir.exists():
        raise RuntimeError(
            "Baseline directory not found.\n\n"
            "Expected:\n\n"
            "    results_archive/baseline/"
        )

    baseline = load_baseline(baseline_dir)

    experiments = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name == BASELINE_FOLDER:
            continue
        experiment = load_experiment(folder)
        experiments.append(experiment)

    benchmark = Benchmark(
        baseline    = baseline,
        experiments = experiments,
    )

    validate_benchmark(benchmark)
    return benchmark


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_benchmark(benchmark: Benchmark) -> None:
    """
    Verify internal consistency of the benchmark archive.
    """

    baseline_pairs = set(benchmark.baseline.pair_names)

    # ------------------------------------------------------------------
    # 1. Walk-forward pair universe must match baseline pair universe
    # ------------------------------------------------------------------

    for exp in benchmark.experiments:
        current = set(exp.pair_names)
        if current != baseline_pairs:
            missing = sorted(baseline_pairs - current)
            extra   = sorted(current - baseline_pairs)
            raise RuntimeError(
                f"\nPair universe mismatch\n\n"
                f"{exp.name}\n\n"
                f"Missing: {missing}\n"
                f"Extra:   {extra}"
            )

    # ------------------------------------------------------------------
    # 2. Architecture consistency (warning only — baseline is MLP)
    # ------------------------------------------------------------------

    experiment_archs = {e.architecture for e in benchmark.experiments}
    if len(experiment_archs) > 1:
        warnings.warn(
            f"\nMultiple architectures in experiments: "
            f"{sorted(experiment_archs)}"
        )

    # ------------------------------------------------------------------
    # 3. Evaluation population must be consistent
    # ------------------------------------------------------------------

    baseline_population = set(benchmark.baseline.evaluation_population)
    for exp in benchmark.experiments:
        if set(exp.evaluation_population) != baseline_population:
            raise RuntimeError(
                f"\nEvaluation population mismatch\n\n"
                f"{exp.name}\n\n"
                f"Baseline:   {sorted(baseline_population)}\n"
                f"Experiment: {sorted(exp.evaluation_population)}"
            )

    # ------------------------------------------------------------------
    # 4. All target pairs must have walk-forward results
    # ------------------------------------------------------------------

    for exp in benchmark.experiments:
        for pair in exp.evaluation_population:
            if pair not in exp.wf_pairs:
                raise RuntimeError(
                    f"\nMissing walk-forward pair\n\n"
                    f"{exp.name}\n\n"
                    f"{pair}"
                )

    # ------------------------------------------------------------------
    # 5. Baseline must have aggregate results for all target pairs
    # ------------------------------------------------------------------

    for pair in benchmark.baseline.evaluation_population:
        if pair not in benchmark.baseline.pairs:
            raise RuntimeError(
                f"\nBaseline missing aggregate pair\n\n"
                f"{pair}"
            )


# ---------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------

def benchmark_summary(benchmark: Benchmark) -> None:
    """
    Diagnostic summary. Useful during framework development.
    """
    print()
    print("=" * 80)
    print("Benchmark Summary")
    print("=" * 80)
    print()
    print(f"Baseline arch   : {benchmark.baseline.architecture}")
    print(f"Experiment arch : {benchmark.architectures}")
    print(f"Experiments     : {len(benchmark.experiments)}")
    print(f"Pairs           : {benchmark.baseline.n_pairs}")
    print(f"Population      : {benchmark.baseline.evaluation_population}")
    print()
    print(
        f"  {'Folder':<42s}"
        f"{'Surface':<16s}"
        f"{'State':<30s}"
        f"{'Features':<18s}"
        f"{'Sentiment'}"
    )
    print("  " + "-" * 114)
    for exp in benchmark.experiments:
        print(
            f"  {exp.name:<42s}"
            f"{exp.representation:<16s}"
            f"{exp.state:<30s}"
            f"{exp.feature_set:<18s}"
            f"{exp.sentiment_surface}"
        )
    print()