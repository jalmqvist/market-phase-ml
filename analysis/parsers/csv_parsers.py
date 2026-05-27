"""
analysis/parsers/csv_parsers.py
================================
Parse all recognised MPML CSV output files in a run directory.

File name → parser mapping
----------------------------
The mapping is intentionally defined in one place (``CSV_PARSERS``) so
adding new output types only requires adding a new entry here.

The mapping uses *glob patterns* rather than exact filenames so that
both ``__baseline`` and ``__dl_enabled`` variants are captured without
duplicating parser logic.

Causal fold boundaries (v2 patch)
-----------------------------------
Walkforward fold slices are generated with strictly causal positional
boundaries::

    test_start_pos = train_end_pos + 1   # no calendar snap

This means consecutive folds are positionally adjacent, not date-snapped.
Any analysis of fold continuity or overlap should use positional indices
rather than calendar date arithmetic.

Selector uplift semantics
--------------------------
``baseline_vs_dynamic_comparison*.csv`` captures the uplift of the
XGBoost-gated *StrategySelector_Dynamic* over the rule-based
*PhaseAware* baseline on OOS walk-forward folds.  Positive Sharpe delta
means the learned router outperforms the hand-coded regime policy.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Low-level CSV reading (stdlib only — no pandas dependency at parse time)
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file and return a list of row dicts (all values as str)."""
    text = path.read_text(errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    return [dict(row) for row in reader]


def _safe_float(val: str | None) -> float | None:
    """Convert string to float; return None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val: str | None) -> int | None:
    """Convert string to int; return None on failure."""
    if val is None:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _cast_row(row: dict[str, str], float_cols: set[str], int_cols: set[str]) -> dict[str, Any]:
    """Return a copy of *row* with numeric columns cast to float/int."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k in int_cols:
            out[k] = _safe_int(v)
        elif k in float_cols:
            out[k] = _safe_float(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Individual parsers
# ---------------------------------------------------------------------------


def _parse_results_ml(path: Path) -> list[dict[str, Any]]:
    """
    Parse ``results_ml__*.csv`` — in-sample ML accuracy table.

    Columns: Model, Accuracy, Std, N Samples, Pair
    """
    float_cols = {"Accuracy", "Std"}
    int_cols = {"N Samples"}
    return [_cast_row(r, float_cols, int_cols) for r in _read_csv(path)]


def _parse_results_ml_backtest(path: Path) -> list[dict[str, Any]]:
    """
    Parse ``results_ml_backtest__*.csv`` — first-stage backtest table.

    Columns: Pair, Strategy, Total Return (%), Sharpe Ratio,
             Max Drawdown (%), Win Rate (%), Profit Factor, Total Trades
    """
    float_cols = {
        "Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)",
        "Win Rate (%)", "Profit Factor",
    }
    int_cols = {"Total Trades"}
    return [_cast_row(r, float_cols, int_cols) for r in _read_csv(path)]


def _parse_walkforward_summary(path: Path) -> list[dict[str, Any]]:
    """
    Parse ``walkforward_results_summary__*.csv`` — aggregate OOS summary.

    Columns vary by run; common ones: Pair, Sharpe_Dynamic,
    Sharpe_Baseline, Sharpe_Delta, Return_Dynamic, Return_Baseline,
    Return_Delta, MaxDD_Dynamic, MaxDD_Baseline, MaxDD_Delta, N_Folds
    """
    float_cols = {
        "Sharpe_Dynamic", "Sharpe_Baseline", "Sharpe_Delta",
        "Return_Dynamic", "Return_Baseline", "Return_Delta",
        "MaxDD_Dynamic", "MaxDD_Baseline", "MaxDD_Delta",
    }
    int_cols = {"N_Folds"}
    return [_cast_row(r, float_cols, int_cols) for r in _read_csv(path)]


def _parse_walkforward_per_pair(path: Path) -> list[dict[str, Any]]:
    """
    Parse ``walkforward_results_per_pair__*.csv`` — per-pair OOS deltas.

    Columns: Pair + same numeric columns as walkforward_summary.
    """
    return _parse_walkforward_summary(path)


def _parse_walkforward_per_fold(path: Path) -> list[dict[str, Any]]:
    """
    Parse ``walkforward_results_per_fold__*.csv`` — per-fold OOS stats.

    Columns: Pair, Fold, + Sharpe/Return/MaxDD for Dynamic and Baseline,
    plus canonical DL overlap attribution columns (``dl_overlap_pct``,
    ``dl_overlap_active``, ``dl_overlap_state``, ``dl_overlap_window``).

    Note: fold boundaries use strictly causal positional indexing
    (``test_start_pos = train_end_pos + 1``).
    """
    float_cols = {
        "Sharpe_Dynamic", "Sharpe_Baseline", "Sharpe_Delta",
        "Return_Dynamic", "Return_Baseline", "Return_Delta",
        "MaxDD_Dynamic", "MaxDD_Baseline", "MaxDD_Delta",
        "dl_overlap_pct",
    }
    bool_str_cols = {"dl_overlap_active"}
    int_cols = {"Fold"}

    rows = _read_csv(path)
    out: list[dict[str, Any]] = []
    for raw in rows:
        row: dict[str, Any] = {}
        for k, v in raw.items():
            if k in int_cols:
                row[k] = _safe_int(v)
            elif k in float_cols:
                row[k] = _safe_float(v)
            elif k in bool_str_cols:
                if v is None or v == "":
                    row[k] = None
                elif isinstance(v, str) and v.strip().lower() in ("true", "1", "yes"):
                    row[k] = True
                else:
                    row[k] = False
            else:
                row[k] = v
        out.append(row)
    return out


def _parse_selector_comparison(path: Path) -> list[dict[str, Any]]:
    """
    Parse ``baseline_vs_dynamic_comparison__*.csv`` — selector uplift.

    Selector uplift = performance of XGBoost-gated StrategySelector_Dynamic
    minus rule-based PhaseAware baseline, evaluated on OOS walk-forward folds.
    Positive Sharpe delta → ML routing outperforms hand-coded policy.
    """
    float_cols = {
        "Sharpe_Baseline", "Sharpe_Dynamic", "Sharpe_Delta",
        "Return_Baseline", "Return_Dynamic", "Return_Delta",
        "MaxDD_Baseline", "MaxDD_Dynamic", "MaxDD_Delta",
        "WinRate_Baseline", "WinRate_Dynamic",
    }
    int_cols = {"Trades_Baseline", "Trades_Dynamic"}
    return [_cast_row(r, float_cols, int_cols) for r in _read_csv(path)]


def _parse_ablation_aggregate(path: Path) -> list[dict[str, Any]]:
    """
    Parse ``ablation_summary_aggregate__*.csv`` — in-sample ablation headline.

    Ablation compares model variants (e.g. no-DL vs DL-enabled) on the
    same training data to isolate feature contribution.
    """
    float_cols = {"Accuracy", "Std", "Improvement"}
    int_cols = {"N Samples", "N_Pairs"}
    return [_cast_row(r, float_cols, int_cols) for r in _read_csv(path)]


def _parse_ablation_per_pair(path: Path) -> list[dict[str, Any]]:
    """
    Parse ``ablation_summary_per_pair__*.csv`` — per-pair ablation breakdown.
    """
    float_cols = {"Accuracy", "Std", "Improvement"}
    int_cols = {"N Samples"}
    return [_cast_row(r, float_cols, int_cols) for r in _read_csv(path)]


def _parse_vol_guard_summary(path: Path) -> list[dict[str, Any]]:
    """
    Parse ``vol_guard_diagnostics_summary__*.csv`` — vol guard aggregate.

    The vol guard suppresses mean-reversion signals when volatility
    (measured by ATR%) exceeds a per-fold quantile threshold.
    """
    float_cols = {"Guard_Rate", "ATR_Quantile", "Vol_Threshold"}
    int_cols = {"N_Folds", "N_Suppressed", "N_Total"}
    return [_cast_row(r, float_cols, int_cols) for r in _read_csv(path)]


def _parse_vol_guard_per_fold(path: Path) -> list[dict[str, Any]]:
    """
    Parse ``vol_guard_diagnostics_per_fold__*.csv`` — per-fold vol guard.
    """
    float_cols = {"Guard_Rate", "ATR_Quantile", "Vol_Threshold"}
    int_cols = {"Fold", "N_Suppressed", "N_Total"}
    return [_cast_row(r, float_cols, int_cols) for r in _read_csv(path)]


def _parse_results_summary(path: Path) -> list[dict[str, Any]]:
    """
    Parse ``results_summary__*.csv`` — top-level results (v2 format).

    This supersedes ``results_ml_backtest__*.csv`` in newer runs and
    includes DL coverage metadata alongside performance metrics.
    """
    float_cols = {
        "Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)",
        "Win Rate (%)", "Profit Factor", "DL_Coverage_Pct",
    }
    int_cols = {"Total Trades"}
    return [_cast_row(r, float_cols, int_cols) for r in _read_csv(path)]


def _parse_results_per_pair(path: Path) -> list[dict[str, Any]]:
    """
    Parse ``results_per_pair__*.csv`` — per-pair results (v2 format).
    """
    return _parse_results_summary(path)


def _parse_selector_state_timeline(path: Path) -> list[dict[str, Any]]:
    """
    Parse ``selector_state_timeline.csv`` — optional per-bar selector state export.

    This file is emitted by the runtime (when enabled) and enables timeline-
    level DL conditional analysis: true occupancy entropy, switch density,
    hold durations, and confidence collapse metrics.

    Expected columns (all optional — absent columns are passed through as-is):

    ``timestamp``             ISO-8601 bar timestamp
    ``pair``                  Instrument pair
    ``selected_strategy``     Active strategy label
    ``selector_confidence``   Float in [0, 1]
    ``phaseaware_active``     Bool — PhaseAware is the selected strategy
    ``volatility_guard_active`` Bool — vol guard is suppressing signals
    ``dl_active``             Bool — DL overlap present for this bar
    ``dl_missing``            Bool — DL unavailable / imputed
    ``imputation_state``      String label for imputation mode
    ``fallback_active``       Bool — fallback strategy is active
    ``switch_event``          Bool — strategy switch occurred at this bar
    ``previous_strategy``     Strategy label on the previous bar
    ``current_strategy``      Strategy label on the current bar
    """
    float_cols = {"selector_confidence"}
    bool_str_cols = {
        "phaseaware_active", "volatility_guard_active", "dl_active",
        "dl_missing", "fallback_active", "switch_event",
    }
    rows = _read_csv(path)
    out: list[dict[str, Any]] = []
    for raw in rows:
        row: dict[str, Any] = {}
        for k, v in raw.items():
            if k in float_cols:
                row[k] = _safe_float(v)
            elif k in bool_str_cols:
                # Normalise to Python bool; preserve None for missing values.
                if v is None or v == "":
                    row[k] = None
                elif v.strip().lower() in ("true", "1", "yes"):
                    row[k] = True
                else:
                    row[k] = False
            else:
                row[k] = v
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Mapping: glob pattern → parser function
# ---------------------------------------------------------------------------

# Each entry: (glob_pattern, summary_key, parser_function)
# Patterns are matched in order; the first match wins per file.
# Using glob patterns (not exact names) accommodates both __baseline and
# __dl_enabled variants without hardcoding exact suffixes.

CSV_PARSERS: list[tuple[str, str, Callable[[Path], list[dict[str, Any]]]]] = [
    ("results_ml__*.csv",                   "ml_accuracy",              _parse_results_ml),
    ("results_ml_backtest__*.csv",           "backtest",                 _parse_results_ml_backtest),
    ("walkforward_results_summary__*.csv",   "walkforward_summary",      _parse_walkforward_summary),
    ("walkforward_results_per_pair__*.csv",  "walkforward_per_pair",     _parse_walkforward_per_pair),
    ("walkforward_results_per_fold__*.csv",  "walkforward_per_fold",     _parse_walkforward_per_fold),
    ("baseline_vs_dynamic_comparison__*.csv","selector_comparison",      _parse_selector_comparison),
    ("ablation_summary_aggregate__*.csv",    "ablation_aggregate",       _parse_ablation_aggregate),
    ("ablation_summary_per_pair__*.csv",     "ablation_per_pair",        _parse_ablation_per_pair),
    ("vol_guard_diagnostics_summary__*.csv", "vol_guard_summary",        _parse_vol_guard_summary),
    ("vol_guard_diagnostics_per_fold__*.csv","vol_guard_per_fold",       _parse_vol_guard_per_fold),
    ("results_summary__*.csv",               "results_summary",          _parse_results_summary),
    ("results_per_pair__*.csv",              "results_per_pair",         _parse_results_per_pair),
    # Optional per-bar selector state timeline (enables timeline-level DL analysis).
    ("selector_state_timeline.csv",          "selector_state_timeline",  _parse_selector_state_timeline),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_run_csvs(run_dir: Path) -> dict[str, Any]:
    """
    Parse all recognised CSV files in *run_dir*.

    Returns a dict keyed by ``summary_key`` (see ``CSV_PARSERS``).
    Missing files result in ``None`` values; parse errors are captured
    under ``_errors`` rather than raised.

    Parameters
    ----------
    run_dir:
        Path to a single run directory.

    Returns
    -------
    dict
        Keys are the summary_key strings from ``CSV_PARSERS``.
        Values are lists of row dicts or None if the file was absent.
        ``_errors`` key lists any parse failures.
        ``_files_found`` lists matched file paths.
    """
    run_dir = Path(run_dir)
    result: dict[str, Any] = {key: None for _, key, _ in CSV_PARSERS}
    result["_errors"] = []
    result["_files_found"] = []

    # Track which files have been claimed to avoid double-counting.
    claimed: set[Path] = set()

    for pattern, key, parser in CSV_PARSERS:
        matches = sorted(run_dir.glob(pattern))
        if not matches:
            continue

        # If multiple files match (e.g. both __baseline and __dl_enabled),
        # collect all of them under the same key as a combined list.
        rows: list[dict[str, Any]] = []
        for match in matches:
            if match in claimed:
                continue
            claimed.add(match)
            result["_files_found"].append(str(match.name))
            try:
                parsed = parser(match)
                # Annotate each row with the source mode tag for downstream
                # comparison (e.g. __baseline vs __dl_enabled).
                mode_tag = _extract_mode_tag(match.name)
                for row in parsed:
                    row["_mode_tag"] = mode_tag
                rows.extend(parsed)
            except Exception as exc:  # noqa: BLE001
                result["_errors"].append(
                    {"file": match.name, "error": str(exc)}
                )

        if rows:
            # Merge with any previously parsed rows under this key.
            existing = result[key]
            result[key] = (existing or []) + rows

    return result


def _extract_mode_tag(filename: str) -> str:
    """
    Extract the mode tag from a CSV filename.

    Examples
    --------
    ``"results_ml__dl_enabled.csv"`` → ``"__dl_enabled"``
    ``"ablation_summary_aggregate__baseline.csv"`` → ``"__baseline"``
    ``"results_ml.csv"`` → ``""``
    """
    import re
    match = re.search(r"(__[a-z_]+)\.csv$", filename)
    return match.group(1) if match else ""
