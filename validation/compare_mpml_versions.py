#!/usr/bin/env python3
"""
compare_mpml_versions.py

Regression validation tool for comparing complete MPML experiment suites
generated from two different datasets.

Purpose
-------
This tool verifies that changes to the MSML dataset construction pipeline
(e.g. leakage fixes, feature engineering changes, calibration updates or
behavioral surface revisions) do not unintentionally alter downstream MPML
behaviour.

Rather than evaluating trading performance itself, the tool compares the
outputs of two complete MPML experiment directories and summarizes how
walk-forward performance changes across all matching experiments.

The comparison is intentionally performed at the MPML level because many
dataset modifications should propagate through MSML without materially
changing MPML conclusions. This script therefore serves as a regression
validation tool for the complete behavioral learning pipeline.

Current analysis
----------------

The comparison uses the aggregate walk-forward statistics produced by each MPML experiment (walkforward_results_summary__dl_enabled.csv).

For each experiment the tool compares

    • Average Return Δ
    • Average Sharpe Δ
    • Average Maximum Drawdown Δ

and produces

    • experiment_summary.csv
    • family_summary.csv
    • aware_vs_blind.csv
    • sentiment_vs_nosentiment.csv
    • MPML_DATASET_COMPARISON.md
    • comparison.json
    • histogram plots

Typical usage
-------------

python validation/compare_mpml_versions.py \
    mpml_baseline/data/1.5.0 \
    mpml_baseline/data/1.6.0 \
    --old-label "Dataset 1.5.0 (pre-audit)" \
    --new-label "Dataset 1.6.0 (post-audit)" \
    --output-dir validation/comparison_1.5.0_vs_1.6.0

Interpretation
--------------

All reported differences are computed as

    New − Old

Positive values therefore indicate that the metric increased in the newer
dataset.

The script is intended as a lightweight regression test rather than a
replacement for detailed MPML analysis. Large deviations should be followed
by targeted investigation of walk-forward outputs, selector timelines and
other experiment artifacts.

Future extensions
-----------------

Possible future analyses include

    • sign-preservation statistics
    • pair-level stability analysis
    • fold-level comparisons
    • selector transition comparisons
    • behavioral surface comparisons
    • recommendation stability analysis

Author
------

Part of the MPML validation toolkit.

Related documentation

    docs/MPML_Architecture_Roadmap.md

"""
from __future__ import annotations
import argparse,json
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
# Expected MPML output files
###############################################################################
SUMMARY="walkforward_results_summary__dl_enabled.csv"
PAIR="walkforward_results_per_pair__dl_enabled.csv"
FOLD="walkforward_results_per_fold__dl_enabled.csv"
GUARD="vol_guard_diagnostics_summary__dl_enabled.csv"

@dataclass
class Experiment:
    name:str; family:str; aware:bool; sentiment:bool; transfer:bool
    summary:pd.DataFrame; pair:pd.DataFrame; fold:pd.DataFrame; guard:pd.DataFrame

def to_markdown_escaped(df: pd.DataFrame) -> str:
    df2 = df.copy()

    # escape pipes in column headers
    df2.columns = [str(c).replace("|", r"\|") for c in df2.columns]

    # (optional) escape pipes in string cells too
    for col in df2.columns:
        if df2[col].dtype == object:
            df2[col] = df2[col].map(lambda x: x.replace("|", r"\|") if isinstance(x, str) else x)

    return df2.to_markdown(index=False)


def top_changes(df, column, n=10):

    cols = [

        "experiment",
        "family",
        "aware",
        "sentiment",
        column,

    ]

    best = (
        df.sort_values(column, ascending=False)
        [cols]
        .head(n)
    )

    worst = (
        df.sort_values(column)
        [cols]
        .head(n)
    )

    return best, worst

def absolute_change_summary(df):
    metrics = [
        "return_delta_change",
        "sharpe_delta_change",
        "dd_delta_change",
    ]

    rows = []

    for m in metrics:
        rows.append({

            "Metric": m,

            "Mean |Δ|":
                df[m].abs().mean(),

            "Median |Δ|":
                df[m].abs().median(),

            "Max |Δ|":
                df[m].abs().max(),

        })

    return pd.DataFrame(rows)


def win_loss_summary(df):

    rows = []

    metrics = {
        "Return Δ": "return_delta_change",
        "Sharpe Δ": "sharpe_delta_change",
        "Max DD Δ": "dd_delta_change",
    }

    for name, col in metrics.items():

        rows.append({

            "Metric": name,
            "Improved": int((df[col] > 0).sum()),
            "Unchanged": int((df[col] == 0).sum()),
            "Worsened": int((df[col] < 0).sum()),

        })

    return pd.DataFrame(rows)

def discover(root:Path):
    exps={}
    for s in root.rglob(SUMMARY):
        d=s.parent
        n=d.name
        exps[n]=Experiment(
            n,
            "transfer" if "transfer" in str(d) else ("persistent" if "persistent" in n else "reactive"),
            "aware" in n,
            "nosentiment" not in n,
            "transfer" in n,
            pd.read_csv(d/SUMMARY),
            pd.read_csv(d/PAIR),
            pd.read_csv(d/FOLD),
            pd.read_csv(d/GUARD) if (d/GUARD).exists() else pd.DataFrame()
        )
    return exps

def compare(a,b):
    rows=[]
    for k in sorted(set(a)&set(b)):
        oa=a[k].summary.iloc[0]; nb=b[k].summary.iloc[0]
        rows.append({
            "experiment":k,
            "family":a[k].family,
            "aware":a[k].aware,
            "sentiment":a[k].sentiment,
            "return_delta_change":nb["Avg Return Δ"]-oa["Avg Return Δ"],
            "sharpe_delta_change":nb["Avg Sharpe Δ"]-oa["Avg Sharpe Δ"],
            "dd_delta_change":nb["Avg Max DD Δ"]-oa["Avg Max DD Δ"],
        })
    return pd.DataFrame(rows)

def report(df, out, old_dir, new_dir, old_label, new_label):

    txt = "# MPML Dataset Comparison\n\n"

    txt += "## Compared datasets\n\n"

    txt += f"**Old dataset**\n\n{old_label}\n\n"
    txt += f"Directory:\n`{old_dir}`\n\n"

    txt += f"**New dataset**\n\n{new_label}\n\n"
    txt += f"Directory:\n`{new_dir}`\n\n"

    txt += "**Comparison convention**\n\n"

    txt += "## Analysis source\n\n"

    txt += (
        "All statistics in this report are derived from\n\n"
        "`walkforward_results_per_pair__dl_enabled.csv`.\n\n"
    )

    txt += (
        "Each experiment contributes one observation consisting of\n\n"
        "- mean Return Δ across walk-forward folds\n"
        "- mean Sharpe Δ across walk-forward folds\n"
        "- mean Max Drawdown Δ across walk-forward folds\n\n"
    )

    txt += (
        "Each observation therefore corresponds to one complete "
        "MPML configuration (family × volatility regime × "
        "sentiment × awareness).\n\n"
    )

    txt += f"Experiments compared: **{len(df)}**\n\n"

    num = df.select_dtypes(include="number")

    txt += "## Overall\n\n"
    txt += num.describe().to_markdown()

    txt += "\n\n## Improvement counts\n\n"

    txt += to_markdown_escaped(win_loss_summary(df))

    txt += "\n\n"

    txt += "## Absolute change magnitude\n\n"

    txt += to_markdown_escaped(absolute_change_summary(df))

    best, worst = top_changes(df, "sharpe_delta_change")

    txt += "\n\n"

    txt += "## Largest Sharpe improvements\n\n"

    txt += best.to_markdown(index=False)

    txt += "\n\n"

    txt += "## Largest Sharpe regressions\n\n"

    txt += worst.to_markdown(index=False)

    txt += "\n\n"

    txt += "## Metric correlations\n\n"

    corr = df[
        [
            "return_delta_change",
            "sharpe_delta_change",
            "dd_delta_change",
        ]
    ].corr()

    txt += corr.to_markdown()

    txt += "\n"

    txt += "## By family\n\n"
    txt += (
        df.groupby("family")[num.columns]
        .mean()
        .to_markdown()
    )
    txt += "\n\n"

    txt += "## Blind vs Aware\n\n"
    txt += (
        df.groupby("aware")[num.columns]
        .mean()
        .to_markdown()
    )
    txt += "\n\n"

    txt += "## Sentiment vs No-sentiment\n\n"
    txt += (
        df.groupby("sentiment")[num.columns]
        .mean()
        .to_markdown()
    )

    mean_abs_return = df["return_delta_change"].abs().mean()
    mean_abs_sharpe = df["sharpe_delta_change"].abs().mean()

    pass_return = mean_abs_return < 0.05
    pass_sharpe = mean_abs_sharpe < 0.02

    txt += "# Dataset regression verdict\n\n"

    if pass_return and pass_sharpe:

        verdict = "PASS"

    else:

        verdict = "REVIEW"

    txt += f"**Overall verdict:** {verdict}\n\n"

    txt += "| Check | Result |\n"
    txt += "|------|------|\n"

    txt += (
        f"| Mean absolute Return Δ < 0.05 | "
        f"{'✓' if pass_return else '✗'} |\n"
    )

    txt += (
        f"| Mean absolute Sharpe Δ < 0.02 | "
        f"{'✓' if pass_sharpe else '✗'} |\n"
    )

    txt += "\n"

    (out / "MPML_DATASET_COMPARISON.md").write_text(txt)

def plot_hist(df, col, out, old_label, new_label):

    titles = {
        "return_delta_change":
            "Return Δ change",

        "sharpe_delta_change":
            "Sharpe Δ change",

        "dd_delta_change":
            "Max drawdown Δ change",
    }

    xlabels = {
        "return_delta_change":
            "Difference (percentage points)",

        "sharpe_delta_change":
            "Difference (Sharpe units)",

        "dd_delta_change":
            "Difference (percentage points)",
    }

    plt.figure(figsize=(6,4))

    plt.hist(df[col], bins=15)

    plt.title(
        f"{titles[col]}\n"
        f"{new_label} − {old_label}"
    )

    plt.xlabel(xlabels[col])

    plt.ylabel("Number of experiments")

    plt.tight_layout()

    plt.savefig(out / f"{col}.png")

    plt.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("old", help="Directory containing the reference (old) MPML experiments")
    ap.add_argument("new", help="Directory containing the comparison (new) MPML experiments")

    ap.add_argument(
        "--old-label",
        default=None,
        help="Human-readable label for the old dataset (defaults to directory name)"
    )

    ap.add_argument(
        "--new-label",
        default=None,
        help="Human-readable label for the new dataset (defaults to directory name)"
    )

    ap.add_argument(
        "--output-dir",
        required=True,
        help="Directory where comparison outputs will be written"
    )
    args=ap.parse_args()
    old_label = args.old_label or Path(args.old).name
    new_label = args.new_label or Path(args.new).name
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"plots").mkdir(exist_ok=True)
    old=discover(Path(args.old)); new=discover(Path(args.new))
    df=compare(old,new)
    df.to_csv(out/"experiment_summary.csv",index=False)
    df.groupby("family").mean(numeric_only=True).to_csv(out/"family_summary.csv")
    df.groupby("aware").mean(numeric_only=True).to_csv(out/"aware_vs_blind.csv")
    df.groupby("sentiment").mean(numeric_only=True).to_csv(out/"sentiment_vs_nosentiment.csv")
    (out / "comparison.json").write_text(

        json.dumps(

            {

                "old_label": old_label,
                "new_label": new_label,

                "old_directory": str(Path(args.old).resolve()),
                "new_directory": str(Path(args.new).resolve()),

                "comparison": "new - old",

                "old": len(old),
                "new": len(new),

                "missing_from_old":
                    sorted(set(new) - set(old)),

                "missing_from_new":
                    sorted(set(old) - set(new)),
            },

            indent=2,

        )

    )
    for c in [
        "return_delta_change",
        "sharpe_delta_change",
        "dd_delta_change",
    ]:
        plot_hist(
            df,
            c,
            out / "plots",
            old_label,
            new_label,
        )
    report(
        df,
        out,
        args.old,
        args.new,
        old_label,
        new_label,
    )
    print()

    print("=" * 60)
    print("MPML Dataset Comparison")
    print("=" * 60)

    print()

    print("Old dataset:")
    print(f"    {old_label}")

    print()

    print("New dataset:")
    print(f"    {new_label}")

    print()

    print("Comparison:")
    print("    New − Old")

    print()

    print(f"Matched experiments: {len(df)}")

    print()

    print(f"Output directory:\n    {out}")

    print()

if __name__=="__main__":
    main()
