"""
analysis/parsers/log_parser.py
================================
Legacy log-file parser (v1 compatibility layer).

This module wraps the original ``analysis/summarize_run.py`` logic so
that old log-driven runs remain parseable.  It is used as a *fallback*
when a run directory does not contain the expected CSV outputs.

Priority order for data extraction:
1. CSV files (csv_parsers.py)  ← primary
2. Log files (this module)      ← fallback / supplement
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Patterns (same as original summarize_run.py; kept for back-compat)
# ---------------------------------------------------------------------------

_ML_RESULT_RE = re.compile(
    r"^(Baseline \(No Phases\)|Phase as Feature|Separate Phase Models \(avg\))\s+"
    r"([0-9.]+)\s+"
    r"([0-9.]+)\s+"
    r"([0-9]+)$"
)

_PAIR_HEADER_RE = re.compile(r"---\s+([A-Z]{6})\s+---")

_DL_COVERAGE_RE = re.compile(
    r"\[DL\]\s+([A-Z]{6}): DL coverage \(any col\)=([0-9.]+)%"
)

_WARNING_RE = re.compile(r"(⚠️|WARNING|warning|Too few training samples)")

_BACKTEST_ROW_RE = re.compile(
    r"^([A-Z]{6})\s+"
    r"(-?[0-9.]+)\s+"
    r"(-?[0-9.]+)\s+"
    r"(-?[0-9.]+)\s+"
    r"([0-9.]+)\s+"
    r"([0-9]+)$"
)


def parse_log(run_dir: Path) -> dict[str, Any] | None:
    """
    Find and parse the first ``.txt`` log file in *run_dir*.

    Returns None if no log file is found.

    Parameters
    ----------
    run_dir:
        Path to a single run directory.

    Returns
    -------
    dict or None
        Keys: ``source_log``, ``ml_results``, ``backtests``,
        ``diagnostics``, ``dl_coverage``.
    """
    log_files = sorted(run_dir.glob("*.txt"))
    if not log_files:
        return None

    log_path = log_files[0]
    text = log_path.read_text(errors="ignore")

    ml_results: list[dict[str, Any]] = []
    backtests: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    dl_coverage: dict[str, float] = {}

    current_pair: str | None = None
    inside_backtest_table = False

    for idx, line in enumerate(text.splitlines()):
        # Track current pair from section headers
        pair_match = _PAIR_HEADER_RE.search(line)
        if pair_match:
            current_pair = pair_match.group(1)

        # ML accuracy rows
        ml_match = _ML_RESULT_RE.search(line.strip())
        if ml_match and current_pair:
            ml_results.append({
                "pair": current_pair,
                "model": ml_match.group(1),
                "accuracy": float(ml_match.group(2)),
                "std": float(ml_match.group(3)),
                "n_samples": int(ml_match.group(4)),
            })

        # DL coverage lines: [DL] EURUSD: DL coverage (any col)=94.50%
        dl_match = _DL_COVERAGE_RE.search(line)
        if dl_match:
            dl_coverage[dl_match.group(1)] = float(dl_match.group(2))

        # Warnings / diagnostics
        if _WARNING_RE.search(line):
            diagnostics.append({"line": idx + 1, "text": line.strip()})

        # Backtest table detection
        if "ML Backtest Results Summary" in line:
            inside_backtest_table = True
            continue
        if not inside_backtest_table:
            continue
        if line.startswith("[4/5]"):
            inside_backtest_table = False
            continue

        row_text = line.strip()
        if not row_text or "Pair" in row_text or "---" in row_text or "Saved to" in row_text:
            continue

        bt_match = _BACKTEST_ROW_RE.match(row_text)
        if bt_match:
            backtests.append({
                "pair": bt_match.group(1),
                "total_return_pct": float(bt_match.group(2)),
                "sharpe": float(bt_match.group(3)),
                "max_drawdown_pct": float(bt_match.group(4)),
                "win_rate_pct": float(bt_match.group(5)),
                "n_trades": int(bt_match.group(6)),
            })

    return {
        "source_log": str(log_path.name),
        "ml_results": ml_results,
        "backtests": backtests,
        "diagnostics": diagnostics,
        "dl_coverage": dl_coverage,
    }
