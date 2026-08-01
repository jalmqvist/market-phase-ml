"""
benchmark.report

Pretty-print benchmark reports.
No benchmark logic — only rendering.
"""

from __future__ import annotations

from collections import defaultdict
from statistics  import mean

from .compare import (
    benchmark_scorecard,
    compare_to_baseline,
)


# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------

LINE = "=" * 88


# ---------------------------------------------------------------------
# Behavioral Surface display names
#
# The raw 'behavioral_surface' value comes from the MPML manifest.
# Display names are defined here and nowhere else.
#
# If the manifest naming ever changes, update only this dict.
# All rendering code below reads from this dict via
# representation_name() — never hardcodes surface strings.
#
# Domain note
# -----------
# A Behavioral Surface defines the discrete state space used to
# represent behavior within an FX pair family.  Different surfaces
# may represent the same family from different perspectives.
#
# Reactive-JPY family surfaces:
#
#   consensus_lifecycle  (manifest id: "reactive_jpy")
#       Derived from crowd consensus dynamics.
#       States: Young, Maturing, Mature, Non-Extreme.
#
#   trend_vol            (manifest id: "trend_vol")
#       Derived from market price behavior.
#       States: LVTF, HVTF, LVR, HVR.
# ---------------------------------------------------------------------

REPRESENTATION_NAMES: dict[str, str] = {
    "reactive_jpy": "Consensus Lifecycle Surface",
    "trend_vol":    "Trend / Volatility Surface",
}

#
# Short tags used in dense table columns (Section 4).
# Keep to ≤ 5 chars.
#

REPRESENTATION_TAGS: dict[str, str] = {
    "reactive_jpy": "cLife",
    "trend_vol":    "tVol",
}

#
# Grouping labels used in the family-level aggregation table.
# These are the display-facing family names.
#

REPRESENTATION_FAMILY_LABELS: dict[str, str] = {
    "reactive_jpy": "Consensus Lifecycle",
    "trend_vol":    "Trend / Volatility",
}


def representation_name(key: str) -> str:
    return REPRESENTATION_NAMES.get(key, key)


def representation_tag(key: str) -> str:
    return REPRESENTATION_TAGS.get(key, key[:5])


def representation_family_label(key: str) -> str:
    return REPRESENTATION_FAMILY_LABELS.get(key, key)


def section(title: str) -> None:
    print()
    print(LINE)
    print(title)
    print(LINE)


# ---------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------

def print_report(benchmark) -> None:

    comparisons  = compare_to_baseline(benchmark)
    architecture = benchmark.architectures[0]

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    section("MPML REFERENCE BENCHMARK")
    print()
    print(f"Architecture : {architecture}")
    print(f"Experiments  : {len(comparisons)}")
    print("Baseline     : No-DL PhaseAware (aggregate)")

    # ------------------------------------------------------------------
    # Section 1 — Architectural Uplift
    # ------------------------------------------------------------------

    section("1. Architectural Uplift — Walk-Forward OOS vs No-DL Baseline")

    for comp in comparisons:

        exp = comp.experiment
        print()
        print(
            f"  {representation_name(exp.representation)}"
            f" | {exp.state}"
            f" | {exp.feature_set}"
            f" | {exp.sentiment_surface}"
        )
        print()
        print(f"  FX Pair Family     : {exp.pair_family}")
        print(f"  Behavioral Surface : {representation_name(exp.representation)}")
        print(f"  State              : {exp.state}")
        print(f"  Feature Set        : {exp.feature_set}")
        print()

        print(
            "{:<8s}"
            "{:>10s}"
            "{:>10s}"
            "{:>10s}"
            "{:>10s}"
            "{:>10s}"
            "{:>8s}".format(
                "Pair",
                "Ret(B)",
                "ΔRet",
                "Sh(B)",
                "ΔSh",
                "DD(B)",
                "ΔDD",
            )
        )
        print("-" * 66)

        for pair in comp.target_pairs:
            print(
                "{:<8s}"
                "{:>10.2f}"
                "{:>10.2f}"
                "{:>10.3f}"
                "{:>10.3f}"
                "{:>10.2f}"
                "{:>8.2f}".format(
                    pair.pair,
                    pair.baseline.total_return,
                    pair.return_uplift,
                    pair.baseline.sharpe,
                    pair.sharpe_uplift,
                    pair.baseline.max_drawdown,
                    pair.drawdown_uplift,
                )
            )

        score = benchmark_scorecard(comp)
        print()
        print("  Target Family Summary")
        print()
        print(f"  Mean Return Uplift : {score['Target Return']:.2f}")
        print(f"  Median Return      : {score['Target Return Median']:.2f}")
        print(f"  Mean Sharpe Uplift : {score['Target Sharpe']:.3f}")
        print(f"  Median Sharpe      : {score['Target Sharpe Median']:.3f}")
        print(f"  Mean DD Uplift     : {score['Target Drawdown']:.2f}")
        print(
            f"  Positive Pairs     : "
            f"{score['Positive Target']}"
            f"/{len(comp.target_pairs)}"
        )

    # ------------------------------------------------------------------
    # Section 2 — Internal MPML Improvement
    # ------------------------------------------------------------------

    section("2. Internal MPML Improvement")
    print()
    print(
        "  Dynamic selector improvement over the "
        "static PhaseAware baseline."
    )

    for comp in comparisons:

        exp      = comp.experiment
        selector = comp.selector_metrics

        print()
        print(
            f"  {exp.state}"
            f" ({representation_name(exp.representation)})"
            f" | {exp.feature_set}"
            f" | {exp.sentiment_surface}"
        )

        if not selector:
            print("    No selector diagnostics.\n")
            continue

        print()
        print(
            "  {:<10s}"
            "{:>12s}"
            "{:>12s}"
            "{:>12s}".format("Pair", "ΔReturn", "ΔSharpe", "ΔDD")
        )
        print("  " + "-" * 48)

        def find(row, candidates):
            for key in candidates:
                if key in row:
                    return row[key]
            return None

        for pair in sorted(selector):
            row      = selector[pair]
            d_return = find(row, ["Return Δ", "Return Delta",
                                   "Return Improvement"])
            d_sharpe = find(row, ["Sharpe Δ", "Sharpe Delta",
                                   "Sharpe Improvement"])
            d_dd     = find(row, ["DD Δ", "Drawdown Δ",
                                   "Drawdown Delta", "DD Difference"])
            if d_return is None or d_sharpe is None or d_dd is None:
                continue

            print(
                "  {:<10s}"
                "{:>12.2f}"
                "{:>12.3f}"
                "{:>12.2f}".format(
                    pair,
                    float(d_return),
                    float(d_sharpe),
                    float(d_dd),
                )
            )

    # ------------------------------------------------------------------
    # Section 3 — Target Family vs Negative Controls
    # ------------------------------------------------------------------

    section("3. Target Family vs Negative Controls")
    print()
    print("  Negative Controls")
    print()

    control_pair_names = sorted({
        p.pair
        for comp in comparisons
        for p in comp.control_pairs
    })

    print(
        "  {:<10s}"
        "{:>12s}"
        "{:>12s}"
        "{:>12s}".format("Pair", "ΔReturn", "ΔSharpe", "ΔDD")
    )
    print("  " + "-" * 48)

    for pair_name in control_pair_names:
        deltas = [
            p
            for comp in comparisons
            for p in comp.control_pairs
            if p.pair == pair_name
        ]
        if not deltas:
            continue
        avg_return = mean(p.return_uplift   for p in deltas)
        avg_sharpe = mean(p.sharpe_uplift   for p in deltas)
        avg_dd     = mean(p.drawdown_uplift for p in deltas)
        print(
            "  {:<10s}"
            "{:>12.2f}"
            "{:>12.3f}"
            "{:>12.2f}".format(
                pair_name,
                avg_return,
                avg_sharpe,
                avg_dd,
            )
        )

    print()
    print("  Experiment Summary")
    print()
    print(
        "  {:<28s}"
        "{:<16s}"
        "{:>10s}"
        "{:>10s}"
        "{:>10s}".format(
            "State",
            "Feature Set",
            "Target",
            "Control",
            "Sep",
        )
    )
    print("  " + "-" * 76)

    for comp in comparisons:
        exp   = comp.experiment
        score = benchmark_scorecard(comp)
        print(
            "  {:<28s}"
            "{:<16s}"
            "{:>10.3f}"
            "{:>10.3f}"
            "{:>10.3f}".format(
                exp.state,
                exp.feature_set,
                score["Target Sharpe"],
                score["Control Sharpe"],
                score["Sharpe Separation"],
            )
        )

    # ------------------------------------------------------------------
    # Section 4 — Behavioral Family Comparison
    #
    # Compares the two Behavioral Surfaces of the Reactive-JPY family:
    #
    #   Consensus Lifecycle Surface  (reactive_jpy)
    #       4 states, all price_trend + sentiment
    #
    #   Trend / Volatility Surface   (trend_vol)
    #       4 regimes × 2 feature sets
    #       split shown as price_trend vs trend_vol_only
    #
    # Primary metric: mean walk-forward ΔSharpe per target pair.
    # ------------------------------------------------------------------

    section("4. Behavioral Family Comparison — Reactive-JPY")
    print()
    print(
        "  Compares the two Behavioral Surfaces of the Reactive-JPY family.\n"
        "  Metric: mean walk-forward ΔSharpe across surface states.\n"
        "  Trend/Volatility is split by feature set.\n"
    )

    TARGET_PAIRS = ["EURJPY", "GBPJPY", "USDJPY"]

    # Accumulate per-representation, per-feature-set, per-pair uplifts
    # data[representation][feature_set][pair] = [uplift, ...]
    data: dict = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(list)
        )
    )

    for comp in comparisons:
        exp = comp.experiment
        for pair in comp.target_pairs:
            data[exp.representation][exp.feature_set][pair.pair].append(
                pair.sharpe_uplift
            )

    # ------------------------------------------------------------------
    # Family-level summary table
    # ------------------------------------------------------------------

    col_w   = 12
    label_w = 38

    print(
        "  {:<{lw}s}".format("Surface / Feature Set", lw=label_w) +
        "".join(f"{p:>{col_w}s}" for p in TARGET_PAIRS) +
        f"{'Mean':>{col_w}s}"
    )
    print("  " + "-" * (label_w + col_w * (len(TARGET_PAIRS) + 1)))

    # Define display order explicitly
    surface_rows = [
        ("reactive_jpy", "price_trend"),
        ("trend_vol",    "price_trend"),
        ("trend_vol",    "trend_vol_only"),
    ]

    for rep, fs in surface_rows:
        pairs = data.get(rep, {}).get(fs, {})
        if not pairs:
            continue

        family_label = representation_family_label(rep)
        label        = f"{family_label}  [{fs}]"

        line   = f"  {label:<{label_w}s}"
        values = []
        for p in TARGET_PAIRS:
            v = mean(pairs[p]) if pairs.get(p) else None
            if v is not None:
                line   += f"{v:>{col_w}.3f}"
                values.append(v)
            else:
                line   += f"{'n/a':>{col_w}s}"
        row_mean = mean(values) if values else float("nan")
        line += f"{row_mean:>{col_w}.3f}"
        print(line)

    print()

    # ------------------------------------------------------------------
    # Per-experiment breakdown
    # ------------------------------------------------------------------

    print("  Per-experiment breakdown\n")

    col_w   = 10
    label_w = 54

    print(
        "  {:<{lw}s}".format("Surface  State / Feature Set", lw=label_w) +
        "".join(f"{p:>{col_w}s}" for p in TARGET_PAIRS) +
        f"{'Mean':>{col_w}s}"
    )
    print("  " + "-" * (label_w + col_w * (len(TARGET_PAIRS) + 1)))

    for comp in comparisons:
        exp        = comp.experiment
        wf_by_pair = {p.pair: p.sharpe_uplift for p in comp.target_pairs}

        tag   = representation_tag(exp.representation)
        label = f"{tag:<6s}{exp.state:<30s}{exp.feature_set}"

        line   = f"  {label:<{label_w}s}"
        values = []
        for p in TARGET_PAIRS:
            v = wf_by_pair.get(p)
            if v is not None:
                line   += f"{v:>{col_w}.3f}"
                values.append(v)
            else:
                line   += f"{'n/a':>{col_w}s}"
        row_mean = mean(values) if values else float("nan")
        line += f"{row_mean:>{col_w}.3f}"
        print(line)

    print()
    print(
        "  (ΔSh = walk-forward OOS Sharpe uplift vs no-DL baseline.\n"
        "   cLife = Consensus Lifecycle Surface,  "
        "tVol = Trend/Volatility Surface.)"
    )

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------

    print()
    print(LINE)
    print("END OF REPORT")
    print(LINE)