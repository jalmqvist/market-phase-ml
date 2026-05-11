"""
D1 aggregation layer for DL signals (market-sentiment-ml integration).

Converts H1 DL signal rows into daily (D1) features suitable for joining
to the D1 regime/trading pipeline in ``main.py``.

No-leakage guarantee
--------------------
For a D1 prediction at the **start** of trading day D (timestamp 00:00 UTC),
the aggregated features use **only** H1 bars whose ``entry_time`` falls on
calendar day ``D - 1`` (i.e. entry_time < D 00:00 UTC).

Mapping:

    H1 bar at date X  →  trading_day (D1 key) = X + 1 day

This is a deterministic, leakage-free assignment: no H1 bar from day D is
ever used to compute features for the D1 prediction at the start of day D.

Produced daily feature columns
-------------------------------
- ``dl_signal_mean_24h``   — mean of dl_signal_strength over last 24 H1 bars
- ``dl_signal_std_24h``    — std  of dl_signal_strength (ddof=1; 0.0 for single bar)
- ``dl_signal_last``       — last (most recent) dl_signal_strength value
- ``dl_signal_abs_mean``   — mean of |dl_signal_strength|
- ``dl_signal_flip_count`` — number of sign changes in dl_signal_strength
                           (NaN-safe; zeros are forward-filled so they do
                           not create spurious flips)

Usage
-----
::

    from pathlib import Path
    from src.dl_daily_features import load_and_aggregate_d1

    daily_df = load_and_aggregate_d1(
        artifact_path=Path("../market-sentiment-ml/data/output/dl_predictions/run.parquet"),
        surface={
            "model": "lstm",
            "target_horizon": 24,
            "feature_set": "price_trend",
            "dl_regime": "HVTF",
        },
        strict=False,
    )
    # Returns DataFrame keyed by (pair, trading_day) with the 5 daily features.

    # Join to a D1 DataFrame keyed by (pair, timestamp):
    d1_df = d1_df.merge(
        daily_df.rename(columns={"trading_day": "timestamp"}),
        on=["pair", "timestamp"],
        how="left",
    )
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.dl_surface_loader import load_dl_surface, empty_dl_surface_df

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Column names produced by :func:`compute_d1_features`.
D1_FEATURE_COLS: tuple[str, ...] = (
    "dl_signal_mean_24h",
    "dl_signal_std_24h",
    "dl_signal_last",
    "dl_signal_abs_mean",
    "dl_signal_flip_count",
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_and_aggregate_d1(
    artifact_path: Path,
    surface: dict,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Load one DL surface parquet and aggregate to D1 daily features.

    Convenience wrapper that calls :func:`src.dl_surface_loader.load_dl_surface`
    then :func:`compute_d1_features`.

    Parameters
    ----------
    artifact_path : Path
        Path to the per-run DL prediction parquet produced by
        market-sentiment-ml.
    surface : dict
        Surface selector dict with keys ``model``, ``target_horizon``,
        ``feature_set``, ``dl_regime``.  Passed directly to
        :func:`src.dl_surface_loader.load_dl_surface`.
    strict : bool, default False
        Passed to the H1 loader.  When ``True``, validation failures raise
        ``ValueError`` instead of returning an empty DataFrame.

    Returns
    -------
    pd.DataFrame
        Daily feature DataFrame keyed by ``(pair, trading_day)``.
        Columns: ``pair``, ``trading_day`` (datetime64[ns]), and the five
        ``dl_signal_*`` features.
        Empty when the surface could not be loaded.
    """
    surface_df = load_dl_surface(artifact_path, surface, strict=strict)
    if surface_df.empty:
        return empty_d1_df()
    return compute_d1_features(surface_df)


def compute_d1_features(surface_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate H1 DL signal rows to D1 daily features.

    Parameters
    ----------
    surface_df : pd.DataFrame
        Output of :func:`src.dl_surface_loader.load_dl_surface`.
        Must contain columns ``pair``, ``timestamp``, and
        ``dl_signal_strength``.

    Returns
    -------
    pd.DataFrame
        Keyed by ``(pair, trading_day)`` where ``trading_day`` is a
        ``datetime64[ns]`` timestamp at midnight UTC corresponding to the
        D1 bar that will consume these features.

        For each group the following columns are computed:

        - ``dl_signal_mean_24h``   mean of ``dl_signal_strength``
        - ``dl_signal_std_24h``    std  of ``dl_signal_strength`` (ddof=1)
        - ``dl_signal_last``       last ``dl_signal_strength`` value
        - ``dl_signal_abs_mean``   mean of ``|dl_signal_strength|``
        - ``dl_signal_flip_count`` number of sign changes

    Raises
    ------
    ValueError
        If ``surface_df`` is missing required columns.
    """
    _validate_surface_df(surface_df)

    df = surface_df[["pair", "timestamp", "dl_signal_strength"]].copy()

    # H1 bar date: the calendar day the bar belongs to (UTC)
    df["_h1_date"] = df["timestamp"].dt.normalize()

    records = []
    for (pair, h1_date), grp in df.groupby(["pair", "_h1_date"], sort=True):
        grp = grp.sort_values("timestamp")
        signal = grp["dl_signal_strength"].to_numpy(dtype="float64")

        n = len(signal)
        mean_val = float(np.nanmean(signal))
        std_val = float(np.nanstd(signal, ddof=1)) if n > 1 else 0.0
        last_val = float(signal[-1])
        abs_mean = float(np.nanmean(np.abs(signal)))
        flip_count = _count_sign_flips(signal)

        # No-leakage: the D1 prediction that CAN USE these H1 bars is the
        # one at the START of the NEXT calendar day (h1_date + 1 day).
        trading_day = h1_date + pd.Timedelta(days=1)

        records.append(
            {
                "pair": pair,
                "trading_day": trading_day,
                "dl_signal_mean_24h": mean_val,
                "dl_signal_std_24h": std_val,
                "dl_signal_last": last_val,
                "dl_signal_abs_mean": abs_mean,
                "dl_signal_flip_count": flip_count,
            }
        )

    if not records:
        return empty_d1_df()

    result = pd.DataFrame(records)
    result["trading_day"] = pd.to_datetime(result["trading_day"])
    result = result.sort_values(["pair", "trading_day"]).reset_index(drop=True)
    return result


def empty_d1_df() -> pd.DataFrame:
    """Return an empty D1 daily features DataFrame with consistent dtypes."""
    return pd.DataFrame(
        {
            "pair": pd.Series(dtype="object"),
            "trading_day": pd.Series(dtype="datetime64[ns]"),
            "dl_signal_mean_24h": pd.Series(dtype="float64"),
            "dl_signal_std_24h": pd.Series(dtype="float64"),
            "dl_signal_last": pd.Series(dtype="float64"),
            "dl_signal_abs_mean": pd.Series(dtype="float64"),
            "dl_signal_flip_count": pd.Series(dtype="int64"),
        }
    )


def _validate_surface_df(df: pd.DataFrame) -> None:
    required = {"pair", "timestamp", "dl_signal_strength"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"surface_df is missing required columns: {sorted(missing)}. "
            "Pass the output of src.dl_surface_loader.load_dl_surface()."
        )


def _count_sign_flips(signal: np.ndarray) -> int:
    """Count sign flips between strictly positive and strictly negative values.

    NaN values are ignored entirely (removed before processing).

    Zeros do not create flips: each zero is forward-filled with the previous
    non-zero sign so that a run of zeros holds the last observed direction
    until the signal commits to a new one.  Zeros that appear before the first
    non-zero value are discarded (they carry no prior sign information).

    Returns 0 when fewer than two non-zero, non-NaN values exist.
    """
    # Drop NaN values; work only on observed finite values.
    arr = signal[~np.isnan(signal)]
    if len(arr) < 2:
        return 0

    # Map each value to +1.0 (positive), -1.0 (negative), or NaN (zero).
    # Using NaN for zero allows pd.Series.ffill() to propagate the last
    # non-zero sign through any run of zeros.
    signs = np.where(arr > 0, 1.0, np.where(arr < 0, -1.0, np.nan))

    # Forward-fill zeros (NaN) with the last non-zero sign.
    # Leading zeros (before the first non-zero value) remain NaN.
    filled = pd.Series(signs).ffill().to_numpy()

    # Drop any remaining NaN (i.e., leading zeros with no prior sign).
    valid = filled[~np.isnan(filled)]
    if len(valid) < 2:
        return 0

    return int(np.sum(valid[1:] != valid[:-1]))
