# analysis/render_run_report.py — DEPRECATED (v1 JSON-summary report renderer)
# ==============================================================================
#
# .. deprecated::
#    This script is superseded by ``analysis/pipeline.py`` (framework v2).
#    It operated on pre-generated .summary.json files and rendered only
#    backtest / ML-accuracy sections.
#
#    **Use instead:**
#
#        python analysis/pipeline.py results_archive/
#
#    The new pipeline renders a unified report covering walkforward,
#    ablation, selector uplift, vol guard, and coverage diagnostics.
#
# This script is retained for backwards compatibility but will not be
# updated further.

# analysis/render_run_report.py

#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "summaries",
        nargs="+",
    )

    parser.add_argument(
        "--output",
        default="analysis/output/report.md",
    )

    args = parser.parse_args()

    lines = []

    lines.append("# DL / MPML Experiment Report\n")

    for path in args.summaries:

        summary = load_summary(path)

        run_name = Path(path).stem.replace(".summary", "")

        lines.append(f"## {run_name}\n")

        warnings = summary.get("warnings", [])

        lines.append(f"- warnings: {len(warnings)}")

        if summary["dl_coverage"]:
            avg_cov = sum(summary["dl_coverage"].values()) / len(summary["dl_coverage"])
            lines.append(f"- avg DL coverage: {avg_cov:.2f}%")

        bt_df = pd.DataFrame(summary["backtests"])

        if not bt_df.empty:

            best = bt_df.sort_values(
                "sharpe",
                ascending=False,
            ).head(3)

            worst = bt_df.sort_values(
                "sharpe",
                ascending=True,
            ).head(3)

            lines.append("\n### Top Sharpe Pairs\n")

            lines.append(
                best[
                    [
                        "pair",
                        "total_return_pct",
                        "sharpe",
                        "win_rate_pct",
                    ]
                ].to_markdown(index=False)
            )

            lines.append("\n### Weakest Sharpe Pairs\n")

            lines.append(
                worst[
                    [
                        "pair",
                        "total_return_pct",
                        "sharpe",
                        "win_rate_pct",
                    ]
                ].to_markdown(index=False)
            )

        lines.append("\n---\n")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text("\n".join(lines))

    print(f"✓ wrote {output_path}")


if __name__ == "__main__":
    main()
