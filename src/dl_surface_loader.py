"""
DL surface loader for market-phase-ml.

Loads and validates a single signal surface from the consolidated DL signal
cube produced by market-sentiment-ml, returning a DataFrame keyed by
(pair, timestamp) suitable for downstream feature assembly.

Cube uniqueness contract (from market-sentiment-ml)
----------------------------------------------------
(pair, entry_time, model, target_horizon, feature_set, dl_regime)

Regime vocabulary (MSML-DL)
----------------------------
HVTF, LVTF, HVR, LVR

Optional MPML-taxonomy mapping
--------------------------------
HVR  -> HVMR  (High-Volatility Mean-Reversion)
LVR  -> LVMR  (Low-Volatility Mean-Reversion)
HVTF -> HVTF  (unchanged)
LVTF -> LVTF  (unchanged)

Usage
-----
::

    from pathlib import Path
    from src.dl_surface_loader import load_dl_surface

    surface_df = load_dl_surface(
        cube_path=Path("../market-sentiment-ml/data/output/dl_signals/dl_signals_h1_v1.parquet"),
        surface={
            "model": "lstm",
            "target_horizon": 24,
            "feature_set": "price_trend",
            "dl_regime": "HVTF",
        },
    )
    # Returns DataFrame with columns: pair, timestamp, dl_signal_strength,
    # mpml_regime_equiv, and optional dl_confidence / pred_prob_up.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

#: Columns that must be present in the cube parquet.
REQUIRED_CUBE_COLUMNS: frozenset[str] = frozenset(
    {
        "pair",
        "entry_time",
        "model",
        "target_horizon",
        "feature_set",
        "dl_regime",
        "signal_strength",
    }
)

#: Columns that may optionally be present in the cube parquet.
OPTIONAL_CUBE_COLUMNS: frozenset[str] = frozenset(
    {"dl_confidence", "pred_prob_up", "prediction_timestamp"}
)

#: Keys that must appear in the *surface* selection dict.
SURFACE_REQUIRED_KEYS: tuple[str, ...] = (
    "model",
    "target_horizon",
    "feature_set",
    "dl_regime",
)

#: Grouping grain for per-surface monotonicity validation.
#: Matches the cube uniqueness contract minus ``entry_time``.
MONOTONICITY_GROUP_COLUMNS: tuple[str, ...] = (
    "pair",
    "model",
    "target_horizon",
    "feature_set",
    "dl_regime",
)

#: Valid MSML-DL regime values.
VALID_DL_REGIMES: frozenset[str] = frozenset({"HVTF", "LVTF", "HVR", "LVR"})

#: MSML-DL -> MPML taxonomy mapping (strict bijection).
MSML_TO_MPML: dict[str, str] = {
    "HVTF": "HVTF",
    "LVTF": "LVTF",
    "HVR": "HVMR",
    "LVR": "LVMR",
}

# Columns in the output DF that must never be used as ML features.
# Kept here as documentation; enforcement is in the caller (assembler).
_LEAKAGE_GUARD_COLUMNS: frozenset[str] = frozenset(
    {"prediction_timestamp", "mpml_regime_equiv"}
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dl_surface(
    cube_path: Path,
    surface: dict,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Load and validate a single DL signal surface from the consolidated cube.

    Parameters
    ----------
    cube_path : Path
        Path to the consolidated DL signal cube parquet file produced by
        market-sentiment-ml.
    surface : dict
        Required keys: ``model``, ``target_horizon`` (int, bars), ``feature_set``,
        ``dl_regime``.  Uniquely identifies one surface in the cube.
    strict : bool, default False
        If ``True``, raise ``ValueError`` on any validation failure.
        If ``False``, emit a ``UserWarning`` and return
        :func:`empty_dl_surface_df`.

    Returns
    -------
    pd.DataFrame
        Always a DataFrame (never ``None``).
        Columns: ``pair``, ``timestamp``, ``dl_signal_strength``,
        ``mpml_regime_equiv`` plus any of
        ``dl_confidence``, ``pred_prob_up``, ``prediction_timestamp``
        that are present in the cube.
        Keyed (sorted) by ``(pair, timestamp)``.

    Raises
    ------
    ValueError
        If ``strict=True`` and any validation step fails.
    """
    # ------------------------------------------------------------------
    # 1. Validate surface dict
    # ------------------------------------------------------------------
    try:
        _validate_surface_dict(surface)
    except ValueError as exc:
        return _handle_error(str(exc), strict)

    # ------------------------------------------------------------------
    # 2. Load artifact
    # ------------------------------------------------------------------
    cube_path = Path(cube_path)
    if not cube_path.exists():
        return _handle_error(
            f"DL signal cube not found: {cube_path}", strict
        )

    try:
        cube = pd.read_parquet(cube_path)
    except (OSError, ValueError, RuntimeError) as exc:
        return _handle_error(
            f"Failed to read DL signal cube {cube_path}: {exc}", strict
        )

    # ------------------------------------------------------------------
    # 3. Validate schema
    # ------------------------------------------------------------------
    missing_cols = REQUIRED_CUBE_COLUMNS - set(cube.columns)
    if missing_cols:
        return _handle_error(
            f"DL signal cube missing required columns: {sorted(missing_cols)}",
            strict,
        )

    # ------------------------------------------------------------------
    # 4. Coerce dtypes
    # ------------------------------------------------------------------
    cube = _coerce_dtypes(cube)

    # ------------------------------------------------------------------
    # 5. Select exact surface
    # ------------------------------------------------------------------
    mask = pd.Series(True, index=cube.index)
    for key in SURFACE_REQUIRED_KEYS:
        val = surface[key]
        if key == "target_horizon":
            # Numeric comparison against nullable Int64
            mask &= cube[key] == int(val)
        else:
            mask &= cube[key] == str(val)

    surface_df = cube[mask].copy()

    if surface_df.empty:
        return _handle_error(
            f"No rows found in cube for surface: {surface}. "
            f"Available surfaces: {_available_surfaces(cube)}",
            strict,
        )

    # ------------------------------------------------------------------
    # 6. Validate hourly alignment of entry_time
    # ------------------------------------------------------------------
    bad_ts = _find_non_hourly_timestamps(surface_df["entry_time"])
    if bad_ts:
        return _handle_error(
            f"Non-hourly entry_time values detected "
            f"(expected minute=0, second=0): {bad_ts[:5]}",
            strict,
        )

    # ------------------------------------------------------------------
    # 7. Validate uniqueness — no duplicate (pair, entry_time) within surface
    # ------------------------------------------------------------------
    dup_mask = surface_df.duplicated(subset=["pair", "entry_time"], keep=False)
    if dup_mask.any():
        n_dups = int(dup_mask.sum())
        return _handle_error(
            f"Duplicate (pair, entry_time) rows within surface: {n_dups} rows. "
            "Cube uniqueness contract violated.",
            strict,
        )

    # ------------------------------------------------------------------
    # 8. Validate per-surface monotonicity
    # ------------------------------------------------------------------
    non_monotone = _find_non_monotone_groups(surface_df)
    if non_monotone:
        return _handle_error(
            f"Non-monotone entry_time detected for surface groups: "
            f"{non_monotone[:3]}",
            strict,
        )

    # ------------------------------------------------------------------
    # 9. Validate value ranges
    # ------------------------------------------------------------------
    range_errors = _validate_value_ranges(surface_df)
    if range_errors:
        return _handle_error("; ".join(range_errors), strict)

    # ------------------------------------------------------------------
    # 10. Build output DataFrame
    # ------------------------------------------------------------------
    out = surface_df[["pair", "entry_time"]].copy()

    # signal_strength -> dl_signal_strength (avoids name collisions downstream)
    out["dl_signal_strength"] = surface_df["signal_strength"].astype("float64")

    # Pass-through optional columns when present
    for col in ("dl_confidence", "pred_prob_up", "prediction_timestamp"):
        if col in surface_df.columns:
            out[col] = surface_df[col]

    # Add MPML regime equivalence (informational; must not be used as a feature)
    out["mpml_regime_equiv"] = surface_df["dl_regime"].map(MSML_TO_MPML)

    # Rename entry_time -> timestamp for internal consistency
    out = out.rename(columns={"entry_time": "timestamp"})

    # Sort by (pair, timestamp)
    out = out.sort_values(["pair", "timestamp"]).reset_index(drop=True)

    return out


def empty_dl_surface_df() -> pd.DataFrame:
    """
    Return an empty DataFrame with the expected output schema.

    Used as a safe fallback when DL signals are disabled or unavailable.
    Dtypes are consistent so that downstream code never sees type mismatches.
    """
    return pd.DataFrame(
        {
            "pair": pd.Series(dtype="object"),
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "dl_signal_strength": pd.Series(dtype="float64"),
            "dl_confidence": pd.Series(dtype="float64"),
            "pred_prob_up": pd.Series(dtype="float64"),
            "mpml_regime_equiv": pd.Series(dtype="object"),
        }
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_surface_dict(surface: dict) -> None:
    """Raise ``ValueError`` if the surface dict is missing required keys."""
    missing = sorted(set(SURFACE_REQUIRED_KEYS) - set(surface.keys()))
    if missing:
        raise ValueError(
            f"surface dict missing required keys: {missing}. "
            f"Required: {list(SURFACE_REQUIRED_KEYS)}"
        )
    regime = surface.get("dl_regime", "")
    if regime not in VALID_DL_REGIMES:
        raise ValueError(
            f"Unknown dl_regime: {regime!r}. "
            f"Valid values: {sorted(VALID_DL_REGIMES)}"
        )


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce critical columns to expected dtypes."""
    df = df.copy()

    # entry_time: tz-naive UTC datetime
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce", utc=False)
    if df["entry_time"].dt.tz is not None:
        df["entry_time"] = df["entry_time"].dt.tz_localize(None)

    # target_horizon: nullable Int64 (number of bars)
    df["target_horizon"] = pd.to_numeric(
        df["target_horizon"], errors="coerce"
    ).astype("Int64")

    # signal_strength: float64
    df["signal_strength"] = pd.to_numeric(df["signal_strength"], errors="coerce")

    return df


def _find_non_hourly_timestamps(ts: pd.Series) -> list:
    """Return list of timestamps that are not on the hour boundary."""
    bad = ts[(ts.dt.minute != 0) | (ts.dt.second != 0)]
    return bad.tolist()


def _find_non_monotone_groups(df: pd.DataFrame) -> list[str]:
    """
    Return group identifiers where ``entry_time`` is not monotone increasing.

    Grain: ``(pair, model, target_horizon, feature_set, dl_regime)``.
    """
    non_monotone: list[str] = []
    for keys, grp in df.groupby(list(MONOTONICITY_GROUP_COLUMNS)):
        if not grp["entry_time"].is_monotonic_increasing:
            non_monotone.append(str(keys))
    return non_monotone


def _validate_value_ranges(df: pd.DataFrame) -> list[str]:
    """
    Validate signal ranges.  Returns a list of error strings (empty = OK).

    - ``signal_strength`` must be in [-1, +1].
    - ``pred_prob_up`` (if present) must be in [0, 1].
    """
    errors: list[str] = []

    ss = df["signal_strength"].dropna()
    bad_ss = ss[(ss < -1) | (ss > 1)]
    if not bad_ss.empty:
        errors.append(
            f"signal_strength out of range [-1, +1]: {len(bad_ss)} values "
            f"(min={bad_ss.min():.4f}, max={bad_ss.max():.4f})"
        )

    if "pred_prob_up" in df.columns:
        probs = pd.to_numeric(df["pred_prob_up"], errors="coerce").dropna()
        bad_p = probs[(probs < 0) | (probs > 1)]
        if not bad_p.empty:
            errors.append(
                f"pred_prob_up out of range [0, 1]: {len(bad_p)} values "
                f"(min={bad_p.min():.4f}, max={bad_p.max():.4f})"
            )

    return errors


def _available_surfaces(cube: pd.DataFrame) -> list[dict]:
    """Return list of distinct surface dicts present in the cube."""
    cols = [c for c in SURFACE_REQUIRED_KEYS if c in cube.columns]
    if not cols:
        return []
    return cube[cols].drop_duplicates().to_dict("records")


def _handle_error(msg: str, strict: bool) -> pd.DataFrame:
    """Raise ``ValueError`` if *strict*, otherwise warn and return empty DF."""
    if strict:
        raise ValueError(msg)
    warnings.warn(f"dl_surface_loader: {msg}", stacklevel=3)
    return empty_dl_surface_df()
