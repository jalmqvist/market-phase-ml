"""
analysis/diagnostics/transition_windows.py
==========================================
DL state classification and transition window extraction for fold sequences.

DL state classification
-----------------------
Folds are classified using the DL overlap coverage fraction already present
in the ``coverage.overlap_window`` block produced by ``analysis/pipeline.py``
(specifically the ``overlap_fold_coverage_pct`` field).

Because V5 fold CSV rows do not carry explicit timestamps, state assignment
uses a **positional heuristic**: the last ``K`` folds (by index) are treated
as DL_ACTIVE, where ``K = round(N * overlap_fraction)``.  This matches the
temporal ordering assumption of walk-forward validation: later folds correspond
to more recent calendar periods, which are the periods covered by DL features.

Canonical DL state labels
--------------------------
``dl_active``
    DL overlap exists for this fold (within the estimated DL coverage window).
``dl_missing``
    DL unavailable or imputed for this fold (outside coverage window).
``dl_transition_enter``
    First fold after DL becomes active (boundary from missing → active).
``dl_transition_exit``
    First fold after DL disappears (boundary from active → missing).
    This can only occur if coverage is partial mid-sequence, which is
    unusual in practice but handled for robustness.

Backwards compatibility
-----------------------
All public functions accept ``None`` or empty inputs and return empty lists /
scaffold dicts rather than raising.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# DL state classification
# ---------------------------------------------------------------------------


def classify_folds_dl_state(
    fold_rows: list[dict[str, Any]],
    *,
    overlap_fold_coverage_pct: float | None,
    pair_col: str = "Pair",
    fold_col: str = "Fold",
) -> list[dict[str, Any]]:
    """
    Classify each fold row with a canonical DL state label.

    Preferred path (per-fold timestamp overlap)
    -------------------------------------------
    When fold rows carry a ``dl_overlap_pct`` column (emitted by the runtime
    since the overlap-attribution PR), each row is classified independently
    using its own overlap fraction:

    * ``dl_overlap_state == "active"``   → ``dl_state = "dl_active"``
    * ``dl_overlap_state == "partial"``  → ``dl_state = "dl_active"``
    * ``dl_overlap_state == "missing"``  → ``dl_state = "dl_missing"``

    Transition labels (``dl_transition_enter`` / ``dl_transition_exit``) are
    derived from state changes between consecutive folds within each pair.

    Fallback path (positional heuristic)
    ------------------------------------
    When per-fold data is unavailable, the last ``K`` folds (by index) are
    treated as DL_ACTIVE, where ``K = round(N * overlap_fraction)``.
    ``overlap_fold_coverage_pct`` is required for the fallback; when it is
    ``None`` all folds are labelled ``dl_state_unknown``.

    Parameters
    ----------
    fold_rows:
        Rows from ``walkforward_per_fold`` CSV.
    overlap_fold_coverage_pct:
        Aggregate fraction of folds in the DL overlap window, in [0, 100].
        Used only when per-fold ``dl_overlap_pct`` is absent (fallback path).
    pair_col:
        Column name for the pair identifier.
    fold_col:
        Column name for the fold index.

    Returns
    -------
    Copy of ``fold_rows`` with two additional keys:
        ``dl_state``  : str  — canonical DL state label
        ``dl_active`` : bool — True when dl_state is dl_active or dl_transition_enter
    """
    if not fold_rows:
        return []

    # ── Preferred path: per-fold dl_overlap_pct column ──────────────────────
    has_per_fold_overlap = any(
        r.get("dl_overlap_pct") is not None for r in fold_rows
    )
    if has_per_fold_overlap:
        return _classify_folds_from_per_fold_overlap(fold_rows, pair_col, fold_col)

    # ── Fallback path: positional heuristic ─────────────────────────────────
    # Group by pair so we can apply per-pair heuristics.
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in fold_rows:
        pair = row.get(pair_col) or row.get("pair") or "unknown"
        by_pair.setdefault(pair, []).append(row)

    out: list[dict[str, Any]] = []
    for pair, rows in by_pair.items():
        # Sort by fold index (ascending) so that "last K" is well-defined.
        def _fold_key(r: dict[str, Any]) -> int:
            v = r.get(fold_col) or r.get("fold") or 0
            try:
                return int(v)
            except (TypeError, ValueError):
                return 0

        sorted_rows = sorted(rows, key=_fold_key)
        n = len(sorted_rows)

        if overlap_fold_coverage_pct is None:
            labelled = [dict(r, dl_state="dl_state_unknown", dl_active=False) for r in sorted_rows]
            out.extend(labelled)
            continue

        fraction = max(0.0, min(100.0, overlap_fold_coverage_pct)) / 100.0
        k = max(0, round(n * fraction))  # number of DL_ACTIVE folds

        states: list[str] = []
        for i in range(n):
            is_active = i >= (n - k)
            if is_active:
                if i > 0 and not (i - 1 >= (n - k)):
                    states.append("dl_transition_enter")
                else:
                    states.append("dl_active")
            else:
                # Check for exit: previous fold was active (can only happen
                # in unusual partial-coverage configurations).
                if i > 0 and (i - 1) >= (n - k):
                    states.append("dl_transition_exit")
                else:
                    states.append("dl_missing")

        for row, state in zip(sorted_rows, states):
            is_active = state in ("dl_active", "dl_transition_enter")
            out.append(dict(row, dl_state=state, dl_active=is_active))

    return out


def _classify_folds_from_per_fold_overlap(
    fold_rows: list[dict[str, Any]],
    pair_col: str,
    fold_col: str,
) -> list[dict[str, Any]]:
    """
    Classify folds using per-row ``dl_overlap_pct`` / ``dl_overlap_state``
    columns emitted by the runtime overlap-attribution export.

    Derives ``dl_state`` (including transition labels) from state changes
    between consecutive folds within each pair.
    """
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in fold_rows:
        pair = row.get(pair_col) or row.get("pair") or "unknown"
        by_pair.setdefault(pair, []).append(row)

    out: list[dict[str, Any]] = []
    for pair, rows in by_pair.items():
        def _fold_key(r: dict[str, Any]) -> int:
            v = r.get(fold_col) or r.get("fold") or 0
            try:
                return int(v)
            except (TypeError, ValueError):
                return 0

        sorted_rows = sorted(rows, key=_fold_key)
        prev_is_active: bool | None = None

        for row in sorted_rows:
            raw_state = row.get("dl_overlap_state") or ""
            # Treat "active" and "partial" as dl_active; "missing" as dl_missing.
            is_active = raw_state in ("active", "partial")

            if prev_is_active is None:
                dl_state = "dl_active" if is_active else "dl_missing"
            elif is_active and not prev_is_active:
                dl_state = "dl_transition_enter"
            elif not is_active and prev_is_active:
                dl_state = "dl_transition_exit"
            elif is_active:
                dl_state = "dl_active"
            else:
                dl_state = "dl_missing"

            out.append(dict(row, dl_state=dl_state, dl_active=is_active))
            prev_is_active = is_active

    return out


def classify_timeline_dl_state(
    timeline_rows: list[dict[str, Any]],
    *,
    dl_active_col: str = "dl_active",
) -> list[dict[str, Any]]:
    """
    Enrich per-bar timeline rows with canonical DL state labels.

    If the timeline already carries a boolean ``dl_active`` column (produced
    by the runtime), this function derives ``dl_state``, ``dl_transition_enter``,
    and ``dl_transition_exit`` from it.

    Parameters
    ----------
    timeline_rows:
        Rows from ``selector_state_timeline.csv``.
    dl_active_col:
        Column containing the per-bar DL active flag.

    Returns
    -------
    Enriched rows with ``dl_state`` column added.
    """
    if not timeline_rows:
        return []

    out: list[dict[str, Any]] = []
    prev_active: bool | None = None

    for row in timeline_rows:
        raw = row.get(dl_active_col)
        is_active = _parse_bool_field(raw) if raw is not None else False

        if prev_active is None:
            # First bar.
            state = "dl_active" if is_active else "dl_missing"
        elif is_active and not prev_active:
            state = "dl_transition_enter"
        elif not is_active and prev_active:
            state = "dl_transition_exit"
        elif is_active:
            state = "dl_active"
        else:
            state = "dl_missing"

        out.append(dict(row, dl_state=state))
        prev_active = is_active

    return out


# ---------------------------------------------------------------------------
# Transition window extraction
# ---------------------------------------------------------------------------


def extract_transition_windows(
    rows: list[dict[str, Any]],
    *,
    n_before: int = 3,
    n_after: int = 3,
    state_col: str = "dl_state",
    transition_states: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Extract windows of rows surrounding DL state transitions.

    Works for both fold-level rows (from ``classify_folds_dl_state``) and
    per-bar timeline rows (from ``classify_timeline_dl_state``).

    Parameters
    ----------
    rows:
        Enriched rows with a ``dl_state`` column.
    n_before:
        Number of rows to include before each transition.
    n_after:
        Number of rows to include after each transition.
    state_col:
        Column containing the DL state label.
    transition_states:
        Set of state labels that mark a transition.  Defaults to
        ``{"dl_transition_enter", "dl_transition_exit"}``.

    Returns
    -------
    Subset of rows within the transition windows, with an additional
    ``transition_window_index`` key indicating which transition the row
    belongs to.

    Notes
    -----
    Overlapping windows are merged into a single window and rows are
    de-duplicated by their position index.
    """
    if not rows:
        return []

    if transition_states is None:
        transition_states = {"dl_transition_enter", "dl_transition_exit"}

    # Find transition indices.
    transition_indices: list[int] = [
        i for i, r in enumerate(rows) if r.get(state_col) in transition_states
    ]

    if not transition_indices:
        return []

    # Build index ranges for each transition window.
    n = len(rows)
    included: dict[int, int] = {}  # row_index → window_index (first window wins)
    for win_idx, ti in enumerate(transition_indices):
        start = max(0, ti - n_before)
        end = min(n - 1, ti + n_after)
        for ri in range(start, end + 1):
            if ri not in included:
                included[ri] = win_idx

    out: list[dict[str, Any]] = []
    for ri in sorted(included):
        out.append(dict(rows[ri], transition_window_index=included[ri]))
    return out


def summarize_transition_windows(
    transition_rows: list[dict[str, Any]],
    *,
    metric_cols: list[str] | None = None,
) -> dict[str, Any]:
    """
    Summarize metrics within extracted transition windows.

    Parameters
    ----------
    transition_rows:
        Output from ``extract_transition_windows``.
    metric_cols:
        Numeric columns to aggregate.  Defaults to common walkforward metrics.

    Returns
    -------
    dict with:
        ``n_transition_rows``  : int
        ``n_windows``          : int
        ``metrics``            : dict mapping col → {mean, std, n}
        ``data_available``     : bool
    """
    if metric_cols is None:
        metric_cols = [
            "Sharpe_Dynamic", "Sharpe_Baseline", "Sharpe_Delta",
            "Return_Dynamic", "Return_Baseline", "Return_Delta",
            "MaxDD_Dynamic", "MaxDD_Baseline", "MaxDD_Delta",
        ]

    if not transition_rows:
        return {
            "n_transition_rows": 0,
            "n_windows": 0,
            "metrics": {},
            "data_available": False,
        }

    n_windows = len({r.get("transition_window_index") for r in transition_rows})
    metrics: dict[str, dict[str, Any]] = {}
    for col in metric_cols:
        values: list[float] = []
        for row in transition_rows:
            val = row.get(col)
            if val is None:
                continue
            try:
                values.append(float(val))
            except (TypeError, ValueError):
                continue
        if values:
            import math
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            metrics[col] = {
                "mean": round(mean, 4),
                "std": round(math.sqrt(variance), 4),
                "n": len(values),
            }

    return {
        "n_transition_rows": len(transition_rows),
        "n_windows": n_windows,
        "metrics": metrics,
        "data_available": True,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_bool_field(val: Any) -> bool:
    """Parse a bool-ish field from a CSV row."""
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return False
