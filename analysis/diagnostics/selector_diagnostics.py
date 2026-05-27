"""
analysis/diagnostics/selector_diagnostics.py
=============================================
Reusable selector diagnostics utilities for DL-conditioned analysis.

Two tiers of metrics
--------------------

**Tier 1 — fold-level (V5 artifacts)**
Computable from existing ``walkforward_per_fold`` CSV rows:

- ``selector_entropy``          : Shannon entropy over fold outcome categories
                                  (positive / negative / near-zero Sharpe delta).
                                  Proxy for routing stability across time.
- ``normalized_entropy``        : Entropy / log2(n_categories), in [0, 1].
- ``occupancy_concentration``   : 1 - normalized_entropy. Higher → more concentrated routing.
- ``fold_sharpe_std``           : Std-dev of Sharpe_Delta across folds (raw stability measure).

**Tier 2 — timeline-level (requires ``selector_state_timeline.csv``)**
Computable when the optional per-bar timeline export is present:

- ``switches_per_1000_bars``    : Strategy switch rate.
- ``mean_hold_duration``        : Mean consecutive bars on a single strategy.
- ``median_hold_duration``      : Median hold duration.
- ``confidence_collapse_count`` : Number of times confidence dropped below threshold.
- ``mean_confidence_recovery_time`` : Mean bars until confidence recovers.
- ``fallback_entry_rate``       : Rate of fallback entries per 1000 bars.
- ``fallback_exit_rate``        : Rate of fallback exits per 1000 bars.

Backwards compatibility
-----------------------
All public functions accept ``None`` or empty inputs and return diagnostics
dicts with ``None``-valued fields rather than raising.  Callers should check
for ``data_available`` flags before interpreting results.
"""

from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Fold-level diagnostics (V5 artifacts)
# ---------------------------------------------------------------------------


def compute_selector_entropy(
    fold_rows: list[dict[str, Any]],
    *,
    metric_col: str = "Sharpe_Delta",
    near_zero_threshold: float = 0.05,
) -> dict[str, Any]:
    """
    Compute Shannon entropy over fold outcome categories as a routing-stability proxy.

    The Sharpe_Delta distribution is bucketed into three categories:
    ``positive``, ``near_zero``, and ``negative``.  Shannon entropy over
    these categories measures how evenly routing outcomes are spread across
    folds.  Low entropy → consistently positive (or negative) routing.

    Parameters
    ----------
    fold_rows:
        Rows from ``walkforward_per_fold`` CSV (any pair, any fold index).
    metric_col:
        Column to use for bucketing (default: ``Sharpe_Delta``).
    near_zero_threshold:
        Absolute value below which delta is classified as ``near_zero``.

    Returns
    -------
    dict with keys:
        ``selector_entropy``        : float | None
        ``normalized_entropy``      : float | None
        ``occupancy_concentration`` : float | None
        ``fold_sharpe_std``         : float | None
        ``n_folds``                 : int
        ``outcome_counts``          : dict mapping category → count
        ``data_available``          : bool
        ``requires_timeline``       : bool (always False — fold-level is sufficient)
    """
    values: list[float] = []
    for row in (fold_rows or []):
        val = row.get(metric_col)
        if val is None:
            continue
        try:
            values.append(float(val))
        except (TypeError, ValueError):
            continue

    if not values:
        return {
            "selector_entropy": None,
            "normalized_entropy": None,
            "occupancy_concentration": None,
            "fold_sharpe_std": None,
            "n_folds": 0,
            "outcome_counts": {},
            "data_available": False,
            "requires_timeline": False,
        }

    counts: dict[str, int] = {"positive": 0, "near_zero": 0, "negative": 0}
    for v in values:
        if abs(v) <= near_zero_threshold:
            counts["near_zero"] += 1
        elif v > 0:
            counts["positive"] += 1
        else:
            counts["negative"] += 1

    total = sum(counts.values())
    entropy = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)

    n_categories = 3
    max_entropy = math.log2(n_categories)
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0

    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)

    return {
        "selector_entropy": round(entropy, 4),
        "normalized_entropy": round(normalized, 4),
        "occupancy_concentration": round(1.0 - normalized, 4),
        "fold_sharpe_std": round(std, 4),
        "n_folds": n,
        "outcome_counts": counts,
        "data_available": True,
        "requires_timeline": False,
    }


def compute_selector_entropy_per_pair(
    fold_rows: list[dict[str, Any]],
    *,
    pair_col: str = "Pair",
    metric_col: str = "Sharpe_Delta",
    near_zero_threshold: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """
    Compute selector entropy per pair from fold-level rows.

    Parameters
    ----------
    fold_rows:
        Rows from ``walkforward_per_fold`` CSV.

    Returns
    -------
    dict mapping pair → entropy dict (same structure as ``compute_selector_entropy``).
    """
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in (fold_rows or []):
        pair = row.get(pair_col) or row.get("pair") or "unknown"
        by_pair.setdefault(pair, []).append(row)
    return {
        pair: compute_selector_entropy(
            rows, metric_col=metric_col, near_zero_threshold=near_zero_threshold
        )
        for pair, rows in sorted(by_pair.items())
    }


# ---------------------------------------------------------------------------
# Timeline-level diagnostics (selector_state_timeline.csv)
# ---------------------------------------------------------------------------


def compute_switch_density(
    timeline_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute switch density metrics from per-bar selector timeline rows.

    Requires ``selector_state_timeline.csv`` (optional export).  Returns
    ``data_available=False`` when ``timeline_rows`` is empty.

    Parameters
    ----------
    timeline_rows:
        Rows from ``selector_state_timeline.csv``.  Expected columns:
        ``timestamp``, ``pair``, ``selected_strategy``, ``switch_event``,
        ``previous_strategy``, ``current_strategy``.

    Returns
    -------
    dict with keys:
        ``switches_per_1000_bars``  : float | None
        ``mean_hold_duration``      : float | None
        ``median_hold_duration``    : float | None
        ``total_bars``              : int
        ``total_switches``          : int
        ``data_available``          : bool
        ``requires_timeline``       : bool (always True)
    """
    if not timeline_rows:
        return {
            "switches_per_1000_bars": None,
            "mean_hold_duration": None,
            "median_hold_duration": None,
            "total_bars": 0,
            "total_switches": 0,
            "data_available": False,
            "requires_timeline": True,
        }

    total_bars = len(timeline_rows)
    switches = 0
    hold_durations: list[int] = []
    current_run = 0
    for row in timeline_rows:
        is_switch = _parse_bool_field(row.get("switch_event"))
        if is_switch:
            switches += 1
            if current_run > 0:
                hold_durations.append(current_run)
            current_run = 1
        else:
            current_run += 1
    if current_run > 0:
        hold_durations.append(current_run)

    rate = (switches / total_bars * 1000.0) if total_bars > 0 else None
    mean_hold: float | None = None
    median_hold: float | None = None
    if hold_durations:
        mean_hold = sum(hold_durations) / len(hold_durations)
        sorted_holds = sorted(hold_durations)
        n = len(sorted_holds)
        if n % 2 == 1:
            median_hold = float(sorted_holds[n // 2])
        else:
            median_hold = (sorted_holds[n // 2 - 1] + sorted_holds[n // 2]) / 2.0

    return {
        "switches_per_1000_bars": round(rate, 2) if rate is not None else None,
        "mean_hold_duration": round(mean_hold, 2) if mean_hold is not None else None,
        "median_hold_duration": round(median_hold, 2) if median_hold is not None else None,
        "total_bars": total_bars,
        "total_switches": switches,
        "data_available": True,
        "requires_timeline": True,
    }


def compute_switch_density_conditioned(
    timeline_rows: list[dict[str, Any]],
    *,
    dl_state_col: str = "dl_active",
) -> dict[str, dict[str, Any]]:
    """
    Compute switch density conditioned on DL state.

    Parameters
    ----------
    timeline_rows:
        Rows from ``selector_state_timeline.csv``.

    Returns
    -------
    dict with keys ``dl_active``, ``dl_missing``, and ``full``, each mapping
    to a switch density dict.
    """
    if not timeline_rows:
        _empty = compute_switch_density([])
        return {"dl_active": _empty, "dl_missing": _empty, "full": _empty}

    active_rows = [r for r in timeline_rows if _parse_bool_field(r.get(dl_state_col))]
    missing_rows = [r for r in timeline_rows if not _parse_bool_field(r.get(dl_state_col))]

    return {
        "dl_active": compute_switch_density(active_rows),
        "dl_missing": compute_switch_density(missing_rows),
        "full": compute_switch_density(timeline_rows),
    }


def compute_confidence_collapse_metrics(
    timeline_rows: list[dict[str, Any]],
    *,
    confidence_col: str = "selector_confidence",
    collapse_threshold: float = 0.40,
    fallback_col: str = "fallback_active",
) -> dict[str, Any]:
    """
    Compute confidence collapse and fallback metrics from per-bar timeline rows.

    Requires ``selector_state_timeline.csv``.

    Parameters
    ----------
    timeline_rows:
        Rows from ``selector_state_timeline.csv``.
    confidence_col:
        Column containing selector confidence scores.
    collapse_threshold:
        Confidence value below which a bar is considered a collapse event.
    fallback_col:
        Column indicating fallback strategy is active.

    Returns
    -------
    dict with keys:
        ``confidence_collapse_count``        : int | None
        ``mean_confidence_recovery_time``    : float | None
        ``fallback_entry_rate``              : float | None
        ``fallback_exit_rate``               : float | None
        ``total_bars``                       : int
        ``data_available``                   : bool
        ``requires_timeline``                : bool (always True)
    """
    if not timeline_rows:
        return {
            "confidence_collapse_count": None,
            "mean_confidence_recovery_time": None,
            "fallback_entry_rate": None,
            "fallback_exit_rate": None,
            "total_bars": 0,
            "data_available": False,
            "requires_timeline": True,
        }

    total_bars = len(timeline_rows)
    collapse_count = 0
    fallback_entries = 0
    fallback_exits = 0
    recovery_durations: list[int] = []

    in_collapse = False
    collapse_start = 0
    in_fallback = False

    for i, row in enumerate(timeline_rows):
        conf_raw = row.get(confidence_col)
        conf: float | None = None
        if conf_raw is not None:
            try:
                conf = float(conf_raw)
            except (TypeError, ValueError):
                pass

        is_fallback = _parse_bool_field(row.get(fallback_col))

        # Fallback entry/exit
        if is_fallback and not in_fallback:
            fallback_entries += 1
            in_fallback = True
        elif not is_fallback and in_fallback:
            fallback_exits += 1
            in_fallback = False

        # Confidence collapse
        if conf is not None:
            if conf < collapse_threshold and not in_collapse:
                collapse_count += 1
                in_collapse = True
                collapse_start = i
            elif conf >= collapse_threshold and in_collapse:
                recovery_durations.append(i - collapse_start)
                in_collapse = False

    mean_recovery: float | None = None
    if recovery_durations:
        mean_recovery = sum(recovery_durations) / len(recovery_durations)

    entry_rate = (fallback_entries / total_bars * 1000.0) if total_bars > 0 else None
    exit_rate = (fallback_exits / total_bars * 1000.0) if total_bars > 0 else None

    return {
        "confidence_collapse_count": collapse_count,
        "mean_confidence_recovery_time": round(mean_recovery, 2) if mean_recovery is not None else None,
        "fallback_entry_rate": round(entry_rate, 2) if entry_rate is not None else None,
        "fallback_exit_rate": round(exit_rate, 2) if exit_rate is not None else None,
        "total_bars": total_bars,
        "data_available": True,
        "requires_timeline": True,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_bool_field(val: Any) -> bool:
    """Parse a bool-ish field from a CSV row (handles str 'True'/'False'/int)."""
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return False
