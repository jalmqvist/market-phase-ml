# analysis/compare_runs.py

#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def backtest_df(summary, label):
    df = pd.DataFrame(summary["backtests"])
    if df.empty:
        return df

    df["run"] = label
    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "summaries",
        nargs="+",
    )

    parser.add_argument(
        "--output-dir",
        default="analysis/output",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dfs = []

    for path in args.summaries:
        summary = load_summary(path)

        label = Path(path).stem.replace(".summary", "")

        df = backtest_df(summary, label)

        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("No backtest data found.")
        return

    combined = pd.concat(dfs, ignore_index=True)

    comparison_csv = output_dir / "comparison.csv"
    combined.to_csv(comparison_csv, index=False)

    markdown_lines = []
    markdown_lines.append("# Run Comparison\n")

    for pair in sorted(combined["pair"].unique()):
        markdown_lines.append(f"## {pair}\n")

        pair_df = combined[combined["pair"] == pair]

        markdown_lines.append(
            pair_df[
                [
                    "run",
                    "total_return_pct",
                    "sharpe",
                    "max_drawdown_pct",
                    "win_rate_pct",
                    "n_trades",
                ]
            ].to_markdown(index=False)
        )

        markdown_lines.append("\n")

    md_path = output_dir / "comparison.md"

    md_path.write_text("\n".join(markdown_lines))

    print(f"✓ wrote {comparison_csv}")
    print(f"✓ wrote {md_path}")


if __name__ == "__main__":
    main()
