from __future__ import annotations

import warnings

import pandas as pd

from features.registry import FEATURES, DL_FEATURE_COLUMNS


def assemble_features(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """
    Select feature columns from *df* for the given feature *groups*.

    DL signal columns (those in ``DL_FEATURE_COLUMNS``) that are absent
    from *df* are filled with ``NaN`` rather than raising, so that pipelines
    without DL signals enabled can still reference the ``dl_signal`` group.
    All other missing columns raise ``KeyError``.
    """
    cols: list[str] = []

    for g in groups:
        if g not in FEATURES:
            raise ValueError(f"Unknown feature group: {g}")
        cols.extend(FEATURES[g])

    # Remove duplicates while preserving order
    cols = sorted(dict.fromkeys(cols))

    missing = [c for c in cols if c not in df.columns]
    if missing:
        dl_missing = [c for c in missing if c in DL_FEATURE_COLUMNS]
        non_dl_missing = [c for c in missing if c not in DL_FEATURE_COLUMNS]

        if non_dl_missing:
            raise KeyError(
                f"Feature columns missing from DataFrame: {non_dl_missing}"
            )

        if dl_missing:
            warnings.warn(
                f"DL signal columns not found in DataFrame; filling with NaN: "
                f"{dl_missing}. "
                "Enable DL signals and call attach_dl_signals() to populate them.",
                stacklevel=2,
            )
            df = df.copy()
            for c in dl_missing:
                df[c] = float("nan")

    return df[cols].copy()


def attach_dl_signals(
    df: pd.DataFrame,
    surface_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join DL surface signals onto the main feature DataFrame.

    Unmatched rows receive ``NaN`` for all DL signal columns so that the
    join is deterministic and missing values are explicit.

    Parameters
    ----------
    df : pd.DataFrame
        Main feature DataFrame.  Must contain ``pair`` and ``entry_time``.
    surface_df : pd.DataFrame
        Output of :func:`src.dl_surface_loader.load_dl_surface`.
        Must contain ``pair`` and ``timestamp``
        (the loader renames ``entry_time`` -> ``timestamp``).

    Returns
    -------
    pd.DataFrame
        *df* extended with DL signal columns.  Original row order is preserved.
        ``mpml_regime_equiv`` and ``prediction_timestamp`` (when present) are
        joined but must **not** be used as ML features.
    """
    dl_cols_in_group: list[str] = FEATURES.get("dl_signal", [])

    if surface_df.empty:
        out = df.copy()
        for col in dl_cols_in_group:
            if col not in out.columns:
                out[col] = float("nan")
        return out

    # Build right-hand side: keep only columns that exist in surface_df
    available = [c for c in dl_cols_in_group if c in surface_df.columns]
    # Also carry mpml_regime_equiv if present (informational, not a feature)
    extras = [
        c for c in ("mpml_regime_equiv",)
        if c in surface_df.columns and c not in available
    ]
    right_cols = ["pair", "timestamp"] + available + extras
    merge_right = surface_df[right_cols].copy()
    merge_right = merge_right.rename(columns={"timestamp": "entry_time"})

    result = df.merge(merge_right, on=["pair", "entry_time"], how="left")

    # Ensure all DL feature group columns exist (fill absent with NaN)
    for col in dl_cols_in_group:
        if col not in result.columns:
            result[col] = float("nan")

    return result
