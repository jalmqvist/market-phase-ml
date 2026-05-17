#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path

ML_RESULT_RE = re.compile(
    r"^(Baseline \(No Phases\)|Phase as Feature|Separate Phase Models \(avg\))\s+"
    r"([0-9.]+)\s+"
    r"([0-9.]+)\s+"
    r"([0-9]+)$"
)

PAIR_HEADER_RE = re.compile(
    r"---\s+([A-Z]{6})\s+---"
)

DL_COVERAGE_RE = re.compile(
    r"\[DL\]\s+([A-Z]{6}): DL coverage \(any col\)=([0-9.]+)%"
)

WARNING_RE = re.compile(
    r"(⚠️|WARNING|warning|Too few training samples)"
)

BACKTEST_HEADER_RE = re.compile(
    r"ML Backtest Results Summary\s+\((.*?)\)"
)

BACKTEST_ROW_START_RE = re.compile(
    r"^(Fixed Fractional|Volatility Adjusted|Kelly Criterion|Optimal F)"
)


def parse_log(path: Path):

    text = path.read_text(errors="ignore")

    ml_results = []
    backtests = []
    diagnostics = []
    dl_coverage = {}

    current_pair = None
    inside_backtest_table = False

    lines = text.splitlines()

    for idx, line in enumerate(lines):

        # ====================================================
        # ACTIVE PAIR TRACKING
        # ====================================================

        pair_match = PAIR_HEADER_RE.search(line)

        if pair_match:
            current_pair = pair_match.group(1)

        # ====================================================
        # ML RESULT TABLES
        # ====================================================

        ml_match = ML_RESULT_RE.search(line.strip())

        if ml_match and current_pair:

            ml_results.append(
                {
                    "pair": current_pair,
                    "model": ml_match.group(1),
                    "accuracy": float(ml_match.group(2)),
                    "std": float(ml_match.group(3)),
                    "n_samples": int(ml_match.group(4)),
                }
            )

        # ====================================================
        # DL COVERAGE
        # ====================================================

        dl_match = DL_COVERAGE_RE.search(line)

        if dl_match:
            dl_coverage[dl_match.group(1)] = float(dl_match.group(2))

        # ====================================================
        # WARNINGS / DIAGNOSTICS
        # ====================================================

        if WARNING_RE.search(line):

            diagnostics.append(
                {
                    "line": idx + 1,
                    "text": line.strip(),
                }
            )

        # ====================================================
        # BACKTEST PARSING
        # ====================================================

        if "ML Backtest Results Summary" in line:
            inside_backtest_table = True
            continue

        if not inside_backtest_table:
            continue

        # stop once next major section starts
        if line.startswith("[4/5]"):
            inside_backtest_table = False
            continue

        row_text = line.strip()

        # skip headers/separators
        if (
                not row_text
                or "Pair" in row_text
                or "---" in row_text
                or "Saved to" in row_text
        ):
            continue

        # expected format:
        #
        # EURUSD 11.38 0.1161 -22.76 60.45 574

        match = re.match(
            r"^([A-Z]{6})\s+"
            r"(-?[0-9.]+)\s+"
            r"(-?[0-9.]+)\s+"
            r"(-?[0-9.]+)\s+"
            r"([0-9.]+)\s+"
            r"([0-9]+)$",
            row_text,
        )

        if not match:
            continue

        pair = match.group(1)

        backtests.append(
            {
                "pair": pair,
                "total_return_pct": float(match.group(2)),
                "sharpe": float(match.group(3)),
                "max_drawdown_pct": float(match.group(4)),
                "win_rate_pct": float(match.group(5)),
                "n_trades": int(match.group(6)),
            }
        )

    return {
        "source_log": str(path),
        "ml_results": ml_results,
        "backtests": backtests,
        "diagnostics": diagnostics,
        "dl_coverage": dl_coverage,
    }


def main():

    if len(sys.argv) != 2:
        print("usage: summarize_run.py <logfile>")
        sys.exit(1)

    log_path = Path(sys.argv[1])

    summary = parse_log(log_path)

    out_dir = Path("analysis/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{log_path.stem}.summary.json"

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ wrote {out_path}")

    print(
        f"✓ parsed "
        f"{len(summary['ml_results'])} ml rows | "
        f"{len(summary['backtests'])} backtest rows | "
        f"{len(summary['diagnostics'])} diagnostics"
    )


if __name__ == "__main__":
    main()