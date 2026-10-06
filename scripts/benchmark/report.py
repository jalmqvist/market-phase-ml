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
# Behavioral Surface display names.
#
# Persistent Commitment Lifecycle is the canonical MSML surface used by the
# current benchmark.  Keep manifest identifiers here; report rendering
# should not infer surface identity from legacy MSML regime strings.
# ---------------------------------------------------------------------

REPRESENTATION_NAMES: dict[str, str] = {
    "reactive_jpy": "Consensus Lifecycle Surface",
    "trend_vol":    "Trend / Volatility Surface",
    "persistent":   "Persistent Commitment Lifecycle Surface",
}

#
# Short tags used in dense table columns (Section 4).
# Keep to ≤ 5 chars.
#

REPRESENTATION_TAGS: dict[str, str] = {
    "reactive_jpy": "cLife",
    "trend_vol":    "tVol",
    "persistent":   "pLife",
}

#
# Grouping labels used in the family-level aggregation table.
# These are the display-facing family names.
#

REPRESENTATION_FAMILY_LABELS: dict[str, str] = {
    "reactive_jpy": "Consensus Lifecycle",
    "trend_vol":    "Trend / Volatility",
    "persistent":   "Persistent Commitment Lifecycle",
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

def print_report(benchmark, sensitivity_mode: bool = False) -> None:
    comparisons  = compare_to_baseline(benchmark, sensitivity_mode=sensitivity_mode)
    architectures = benchmark.architectures
    architecture = ", ".join(architectures)
    # The archive may contain multiple behavioral surfaces with different
    # target populations (e.g. Persistent=5 pairs, TrendVol=3 JPY pairs).
    # Use the union for matrix columns; each experiment contributes values
    # only for its own target population.
    TARGET_PAIRS = sorted({
        pair
        for experiment in benchmark.experiments
        for pair in experiment.evaluation_population
    })

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    sensitivity_note = (
        "> **Sensitivity mode:** Deltas recomputed from absolute values "
        "against current baseline.  \n"
        if sensitivity_mode else ""
    )

    print(f"""
    # MPML REFERENCE BENCHMARK

    **Architecture**: {architecture}  
    **Experiments**: {len(comparisons)}  
    **Baseline**: No-DL PhaseAware (aggregate)  
    **Target pairs**: experiment-specific; matrix columns cover the union of target populations ({", ".join(TARGET_PAIRS)})  

    {sensitivity_note}> Δ values are walk-forward OOS deltas vs no-DL baseline.  
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
        f"> All {len(benchmark.baseline.pair_names)} baseline-universe pairs shown. Target membership is defined by the experiment's FX pair family.\n"
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

        # Target membership is experiment-specific.
        target_set = set(exp.evaluation_population)

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


    # ------------------------------------------------------------------
    # Section 3 — Target Family vs Negative Controls
    # ------------------------------------------------------------------

    print("## 3. Target Family vs Negative Controls")
    print(
        "> Control statistics are summarized separately for each FX pair family.\n"
        "> Target pairs are defined by the experiment's pair family; all other "
        "baseline-universe pairs are controls.\n"
        "> Separation = mean target ΔSharpe minus mean control ΔSharpe."
    )

    families = defaultdict(list)
    for comp in comparisons:
        families[comp.experiment.pair_family].append(comp)

    for family in sorted(families):
        family_comps = families[family]
        family_label = family.replace("_", " ").title()
        print()
        print(f"### Control FX pairs — outside the {family_label} target family")
        print(
            f"> These {len(set(p.pair for comp in family_comps for p in comp.control_pairs))} FX pairs are outside the {family_label} target population and serve as negative controls.\n"
        )

        control_pair_names = sorted({
            p.pair for comp in family_comps for p in comp.control_pairs
        })
        print("| Pair | Mean ΔReturn | Mean ΔSharpe | Mean ΔDD |")
        print("|---|---:|---:|---:|")
        for pair_name in control_pair_names:
            deltas = [p for comp in family_comps for p in comp.control_pairs if p.pair == pair_name]
            if not deltas: continue
            print(f"| {pair_name} | {mean(p.return_uplift for p in deltas):6.2f} | {mean(p.sharpe_uplift for p in deltas):6.3f} | {mean(p.drawdown_uplift for p in deltas):6.2f} |")

        print()
        target_pair_names = sorted({
            p for comp in family_comps for p in comp.experiment.evaluation_population
        })
        print("#### Target vs negative-control separation")
        print(
            f"> Target ΔSh = mean across {len(target_pair_names)} target pairs ({', '.join(target_pair_names)}).  "
            f"Control ΔSh = mean across {len(control_pair_names)} control pairs ({', '.join(control_pair_names)}).  "
            "> Separation = target ΔSh minus control ΔSh."
        )
        print("| State | Behavioral Surface | Feature Set | Target ΔSh | Control ΔSh | Separation |")
        print("|---|---|---|---:|---:|---:|")
        for comp in sorted(family_comps, key=lambda c: (c.experiment.representation, c.experiment.feature_set, c.experiment.state)):
            exp=comp.experiment; score=benchmark_scorecard(comp)
            print(f"| {exp.state} | {representation_name(exp.representation)} | {exp.feature_set} | {score['Target Sharpe']:6.3f} | {score['Control Sharpe']:6.3f} | {score['Sharpe Separation']:6.3f} |")
    print("\n")

    # ------------------------------------------------------------------
    # Section 4 — Behavioral Family Comparison
    # ------------------------------------------------------------------

    print("## 4. Behavioral Surface Comparison")

    print(
        "> Compares Behavioral Surfaces within each FX pair family present in the benchmark archive.\n"
        "> Metric: mean walk-forward ΔSharpe across that family's evaluated target pairs.\n"
        "> Trend/Volatility is split by feature set.\n"
    )

    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for comp in comparisons:
        exp=comp.experiment
        for pair in comp.target_pairs:
            data[exp.pair_family][exp.representation][exp.feature_set][pair.pair].append(pair.sharpe_uplift)

    for family in sorted(data):
        family_label=family.replace("_", " ").title()
        family_pairs=sorted({
            pair for comp in comparisons if comp.experiment.pair_family == family
            for pair in comp.experiment.evaluation_population
        })
        print()
        print(f"### {family_label}")
        print("| Surface / Feature Set | " + " | ".join(family_pairs) + " | Mean |")
        print("|---|" + "---|"*len(family_pairs) + "---|")
        for rep in sorted(data[family]):
            for fs in sorted(data[family][rep]):
                pairs=data[family][rep][fs]; label=f"{representation_family_label(rep)}  [{fs}]"; row=f"| {label} "; vals=[]
                for pn in family_pairs:
                    v=mean(pairs[pn]) if pairs.get(pn) else None
                    if v is None: row += " | n/a "
                    else: row += f" | {v:6.3f} "; vals.append(v)
                print(row + f" | {mean(vals) if vals else float('nan'):6.3f} |")
        print(); print("#### Per-experiment breakdown")
        print("| Surface | State | Feature Set | " + " | ".join(family_pairs) + " | Mean |")
        print("|---|---|---|" + "---|"*len(family_pairs) + "---|")
        breakdown = []
        for comp in [c for c in sorted_comps if c.experiment.pair_family == family]:
            wf = {p.pair: p.sharpe_uplift for p in comp.target_pairs}
            vals = [wf[pn] for pn in family_pairs if pn in wf]
            row_mean = mean(vals) if vals else float("nan")
            breakdown.append((comp, wf, row_mean))

        max_by_pair = {
            pn: max(wf[pn] for _, wf, _ in breakdown if pn in wf)
            for pn in family_pairs
        }
        valid_means = [row_mean for _, _, row_mean in breakdown if row_mean == row_mean]
        max_mean = max(valid_means) if valid_means else float("nan")

        for comp, wf, row_mean in breakdown:
            exp = comp.experiment
            row = f"| {representation_tag(exp.representation)} | {exp.state} | {exp.feature_set} "
            for pn in family_pairs:
                v = wf.get(pn)
                if v is None:
                    row += " | n/a "
                else:
                    cell = f"{v:6.3f}"
                    if v == max_by_pair[pn]:
                        cell = f"**{cell.strip()}**"
                    row += f" | {cell} "
            mean_cell = f"{row_mean:6.3f}" if row_mean == row_mean else "nan"
            if row_mean == max_mean:
                mean_cell = f"**{mean_cell.strip()}**"
            row += f" | {mean_cell} |"
            print(row)

        print(
            "> **Bold** = highest ΔSharpe in that numerical column across the per-experiment rows.\n"
        )
        print("\n")

    # Footer
    print("---")
    print("Generated by `compare_to_baseline.py` — MPML Stage 3 OOS validator.")
    print("Validated against the MPML benchmark validation contract.")
    print("Report format: Markdown — optimized for GitHub, Jupyter, VS Code, Obsidian.")
