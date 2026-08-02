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
    TARGET_PAIRS = ["EURJPY", "GBPJPY", "USDJPY"]

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    print(f"""
# MPML REFERENCE BENCHMARK

**Architecture**: {architecture}  
**Experiments**: {len(comparisons)}  
**Baseline**: No-DL PhaseAware (aggregate)  
**Target pairs**: EURJPY, GBPJPY, USDJPY (Reactive-JPY family)  

> Δ values are walk-forward OOS deltas vs no-DL baseline.  
> `+` = positive Sharpe uplift.  
> For ΔDD: **smaller = better** (less drawdown).  
> All values rounded to 3 decimals for readability.
""")

    # ------------------------------------------------------------------
    # Section 1 — Uplift Matrix (Markdown Table)
    # ------------------------------------------------------------------

    print("## 1. Uplift Matrix — ΔRet, ΔSh, and ΔDD per State and Pair")

    # Table header: ΔRet, ΔSh, ΔDD grouped by pair
    header = (
        "| Architecture | Behavioral Surface | Feature Set | State | "
        + " | ".join(f"ΔRet {p}" for p in TARGET_PAIRS)
        + " | "
        + " | ".join(f"ΔSh {p}" for p in TARGET_PAIRS)
        + " | "
        + " | ".join(f"ΔDD {p}" for p in TARGET_PAIRS)
        + " | Mean ΔSh |"
    )
    separator = (
        "|---|---|---|---|"
        + "---|" * (len(TARGET_PAIRS) * 3)
        + "---|"
    )

    print(header)
    print(separator)

    # Sort for consistent display
    from itertools import groupby

    def group_key(comp):
        exp = comp.experiment
        return (
            representation_name(exp.representation),
            exp.feature_set,
        )

    sorted_comps = sorted(comparisons, key=group_key)

    for comp in sorted_comps:
        exp = comp.experiment

        wf = {p.pair: p for p in comp.target_pairs}
        sharpe_values = []

        row = (
            f"| {architecture} "
            f"| {representation_name(exp.representation)} "
            f"| {exp.feature_set} "
            f"| {exp.state} "
        )

        # ΔRet columns
        for p in TARGET_PAIRS:
            result = wf.get(p)
            if result is not None:
                d_ret = result.return_uplift
                row += f" | {d_ret:6.2f} "
            else:
                row += " | n/a "

        # ΔSh columns
        for p in TARGET_PAIRS:
            result = wf.get(p)
            if result is not None:
                d_sh = result.sharpe_uplift
                sharpe_values.append(d_sh)
                flag = "+" if d_sh > 0 else ""
                row += f" | {d_sh:6.3f}{flag} "
            else:
                row += " | n/a "

        # ΔDD columns (smaller = better)
        for p in TARGET_PAIRS:
            result = wf.get(p)
            if result is not None:
                d_dd = result.drawdown_uplift
                row += f" | {d_dd:6.2f} "
            else:
                row += " | n/a "

        # Mean ΔSh
        mean_sh = mean(sharpe_values) if sharpe_values else float("nan")
        row += f" | {mean_sh:6.3f} |"

        print(row)

    print("\n")

    # ------------------------------------------------------------------
    # Section 2 — Dynamic Selector Improvement
    # ------------------------------------------------------------------

    print("## 2. Internal MPML Improvement — Dynamic Selector")

    print(
        "> Dynamic selector improvement over the static PhaseAware baseline.\n"
        "> All 14 pairs shown. Target pairs: EURJPY, GBPJPY, USDJPY.\n"
    )

    for comp in comparisons:
        exp      = comp.experiment
        selector = comp.selector_metrics

        print(f"### {representation_name(exp.representation)} — {exp.state} — {exp.feature_set}")

        if not selector:
            print("No selector diagnostics.\n")
            continue

        # Table header
        header = "| Pair | ΔReturn | ΔSharpe | ΔDD |"
        separator = "|---|---|---|---|"
        print(header)
        print(separator)

        def find(row, candidates):
            for key in candidates:
                if key in row:
                    return row[key]
            return None

        target_set = set(TARGET_PAIRS)

        for pair in sorted(selector):
            row      = selector[pair]
            d_return = find(row, ["Return Δ", "Return Delta", "Return Improvement"])
            d_sharpe = find(row, ["Sharpe Δ", "Sharpe Delta", "Sharpe Improvement"])
            d_dd     = find(row, ["DD Δ", "Drawdown Δ", "Drawdown Delta", "DD Difference"])

            if d_return is None or d_sharpe is None or d_dd is None:
                continue

            marker = " *" if pair in target_set else ""
            print(
                f"| {pair}{marker} "
                f"| {float(d_return):6.2f} "
                f"| {float(d_sharpe):6.3f} "
                f"| {float(d_dd):6.2f} |"
            )

        print("\n")

    print("**(* = Reactive-JPY target pair)**\n")

    # ------------------------------------------------------------------
    # Section 3 — Target Family vs Negative Controls
    # ------------------------------------------------------------------

    print("## 3. Target Family vs Negative Controls")

    print(
        "> Control pair averages (mean across all 12 experiments)\n"
        "> Separation: mean ΔSh (target) minus mean ΔSh (controls)\n"
    )

    # Control pair averages
    control_pair_names = sorted({
        p.pair
        for comp in comparisons
        for p in comp.control_pairs
    })

    header = "| Pair | ΔReturn | ΔSharpe | ΔDD |"
    separator = "|---|---|---|---|"
    print(header)
    print(separator)

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
            f"| {pair_name} "
            f"| {avg_return:6.2f} "
            f"| {avg_sharpe:6.3f} "
            f"| {avg_dd:6.2f} |"
        )

    print("\n")

    # Separation summary
    print("### Separation Summary (mean ΔSh: target vs controls)")

    header = "| State | Feature Set | Target ΔSh | Control ΔSh | Separation |"
    separator = "|---|---|---|---|---|"
    print(header)
    print(separator)

    for comp in sorted_comps:
        exp   = comp.experiment
        score = benchmark_scorecard(comp)
        print(
            f"| {exp.state} "
            f"| {exp.feature_set} "
            f"| {score['Target Sharpe']:6.3f} "
            f"| {score['Control Sharpe']:6.3f} "
            f"| {score['Sharpe Separation']:6.3f} |"
        )

    print("\n")

    # ------------------------------------------------------------------
    # Section 4 — Behavioral Family Comparison
    # ------------------------------------------------------------------

    print("## 4. Behavioral Family Comparison — Reactive-JPY")

    print(
        "> Compares the two Behavioral Surfaces of the Reactive-JPY family.\n"
        "> Metric: mean walk-forward ΔSharpe across surface states.\n"
        "> Trend/Volatility is split by feature set.\n"
    )

    from collections import defaultdict

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

    # Family-level summary
    header = (
        "| Surface / Feature Set | "
        + " | ".join(TARGET_PAIRS)
        + " | Mean |"
    )
    separator = (
        "|---|" + "---|" * len(TARGET_PAIRS) + "---|"
    )
    print(header)
    print(separator)

    surface_rows = [
        ("reactive_jpy", "price_trend"),
        ("trend_vol",    "price_trend"),
        ("trend_vol",    "trend_vol_only"),
    ]

    for rep, fs in surface_rows:
        pairs = data.get(rep, {}).get(fs, {})
        if not pairs:
            continue
        label  = f"{representation_family_label(rep)}  [{fs}]"
        row    = f"| {label} "
        values = []
        for p in TARGET_PAIRS:
            v = mean(pairs[p]) if pairs.get(p) else None
            if v is not None:
                row   += f" | {v:6.3f} "
                values.append(v)
            else:
                row   += " | n/a "
        row_mean = mean(values) if values else float("nan")
        row += f" | {row_mean:6.3f} |"
        print(row)

    print("\n")

    # Per-experiment breakdown
    print("### Per-experiment breakdown")

    header = (
        "| Surface | State | Feature Set | "
        + " | ".join(TARGET_PAIRS)
        + " | Mean |"
    )
    separator = (
        "|---|---|---|" + "---|" * len(TARGET_PAIRS) + "---|"
    )
    print(header)
    print(separator)

    for comp in sorted_comps:
        exp        = comp.experiment
        wf_by_pair = {p.pair: p.sharpe_uplift for p in comp.target_pairs}
        tag        = representation_tag(exp.representation)
        row        = f"| {tag} | {exp.state} | {exp.feature_set} "
        values     = []
        for p in TARGET_PAIRS:
            v = wf_by_pair.get(p)
            if v is not None:
                row   += f" | {v:6.3f} "
                values.append(v)
            else:
                row   += " | n/a "
        row_mean = mean(values) if values else float("nan")
        row += f" | {row_mean:6.3f} |"
        print(row)

    print("\n")

    # Footer
    print("---")
    print("Generated by `compare_to_baseline.py` — MPML Stage 3 OOS validator.")
    print("Validated against VALIDATION_SPEC_JPY.md (frozen June 2026).")
    print("Report format: Markdown — optimized for GitHub, Jupyter, VS Code, Obsidian.")
