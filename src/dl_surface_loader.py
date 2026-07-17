"""
DL surface loader for market-phase-ml.

Loads and validates a single signal surface from the DL prediction artifact
produced by market-sentiment-ml, returning a DataFrame keyed by
(pair, timestamp) suitable for downstream feature assembly.

Artifact uniqueness contract (from market-sentiment-ml)
--------------------------------------------------------
(pair, timestamp, model, target_horizon, feature_set, dl_regime)

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
    # mpml_regime_equiv, and optional dl_confidence / dl_pred_prob_up.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from pandas.api.types import is_datetime64_dtype, is_datetime64tz_dtype

from mpml.behavioral.registry import default_registry
from schemas.dl_artifact_schema import (
    DL_ARTIFACT_CREATED_COL,
    DL_AVAILABLE_TS_COL,
    DL_GENERATED_TS_COL,
    DL_PAIR_COL,
    DL_SCHEMA_VERSION,
    DL_TIMESTAMP_COL,
)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

#: Columns that must be present in the DL artifact parquet.
REQUIRED_ARTIFACT_COLUMNS: frozenset[str] = frozenset(
    {
        DL_PAIR_COL,
        DL_TIMESTAMP_COL,
        DL_AVAILABLE_TS_COL,
        DL_GENERATED_TS_COL,
        DL_ARTIFACT_CREATED_COL,
        "model",
        "target_horizon",
        "feature_set",
        "signal_strength",
    }
)

#: Columns that may optionally be present in the cube parquet.
OPTIONAL_ARTIFACT_COLUMNS: frozenset[str] = frozenset(
    {
        "dl_confidence",
        "pred_prob_up",
        "schema_version",
        "dl_regime",
        "surface_id",
        "surface_version",
        "state_id",
    }
)
# Backward-compatible aliases kept for existing imports in validation tooling.
REQUIRED_CUBE_COLUMNS = REQUIRED_ARTIFACT_COLUMNS
OPTIONAL_CUBE_COLUMNS = OPTIONAL_ARTIFACT_COLUMNS

#: Canonical identity keys required for runtime selection.
CANONICAL_SURFACE_KEYS: tuple[str, ...] = (
    "surface_id",
    "surface_version",
    "state_id",
)

#: Model keys required for runtime selection.
PREDICTION_SELECTOR_KEYS: tuple[str, ...] = (
    "model",
    "target_horizon",
    "feature_set",
)

#: Full canonical selector (surface identity + artifact model identity).
SURFACE_REQUIRED_KEYS: tuple[str, ...] = (
    *CANONICAL_SURFACE_KEYS,
    *PREDICTION_SELECTOR_KEYS,
)

#: Legacy TrendVol selector accepted via compatibility adapter.
LEGACY_SURFACE_REQUIRED_KEYS: tuple[str, ...] = (
    *PREDICTION_SELECTOR_KEYS,
    "dl_regime",
)

#: Grouping grain for per-surface monotonicity validation.
#: Matches the cube uniqueness contract minus ``entry_time``.
MONOTONICITY_GROUP_COLUMNS: tuple[str, ...] = (
    DL_PAIR_COL,
    "surface_id",
    "surface_version",
    "state_id",
    "model",
    "target_horizon",
    "feature_set",
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
    {
        "dl_prediction_available_timestamp",
        "dl_prediction_generated_timestamp",
        "dl_artifact_created_timestamp",
        "mpml_regime_equiv",
    }
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
        Legacy fallback behavior control for non-contract errors (e.g. missing
        file path). Contract violations always raise ``ValueError``.

    Returns
    -------
    pd.DataFrame
        Always a DataFrame (never ``None``).
        Columns: ``pair``, ``timestamp``, ``dl_signal_strength``,
        ``mpml_regime_equiv`` plus any of
        ``dl_confidence``, ``dl_pred_prob_up``,
        ``dl_prediction_available_timestamp``,
        ``dl_prediction_generated_timestamp``,
        ``dl_artifact_created_timestamp``
        that are present in the cube.
        Keyed (sorted) by ``(pair, timestamp)``.

    Raises
    ------
    ValueError
        If contract validation fails (always), or if ``strict=True`` and a
        non-contract load error occurs.
    """
    # ------------------------------------------------------------------
    # 1. Validate surface dict
    # ------------------------------------------------------------------
    try:
        canonical_surface = _normalize_surface_selector(surface)
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
        metadata = _read_parquet_metadata(cube_path)
    except (OSError, ValueError, RuntimeError) as exc:
        return _handle_error(
            f"Failed to read DL signal cube {cube_path}: {exc}", strict
        )

    # ------------------------------------------------------------------
    # 3. Validate DL artifact contract (fail-fast)
    # ------------------------------------------------------------------
    cube = validate_dl_artifact(cube, metadata)
    cube = _apply_behavioral_identity(cube, metadata)

    # ------------------------------------------------------------------
    # 4. Select exact surface
    # ------------------------------------------------------------------
    mask = pd.Series(True, index=cube.index)
    for key in SURFACE_REQUIRED_KEYS:
        val = canonical_surface[key]
        if key == "target_horizon":
            # Numeric comparison against nullable Int64
            mask &= cube[key] == int(val)
        else:
            mask &= cube[key] == str(val)

    surface_df = cube[mask].copy()

    if surface_df.empty:
        warnings.warn(
            f"dl_surface_loader: No rows found in cube for surface: {surface}. "
            f"Available surfaces: {_available_surfaces(cube)}",
            stacklevel=2,
        )
        return empty_dl_surface_df()

    # ------------------------------------------------------------------
    # 5. Validate value ranges on selected surface
    # ------------------------------------------------------------------
    range_errors = _validate_value_ranges(surface_df)
    if range_errors:
        raise ValueError("; ".join(range_errors))

    # ------------------------------------------------------------------
    # 6. Build output DataFrame
    # ------------------------------------------------------------------
    out = surface_df[[DL_PAIR_COL, DL_TIMESTAMP_COL]].copy()

    # signal_strength -> dl_signal_strength (avoids name collisions downstream)
    out["dl_signal_strength"] = surface_df["signal_strength"].astype("float64")

    # Pass-through optional columns when present, renaming to dl_* prefix
    if "dl_confidence" in surface_df.columns:
        out["dl_confidence"] = surface_df["dl_confidence"]
    if "pred_prob_up" in surface_df.columns:
        # Rename to dl_pred_prob_up to avoid name collisions and clarify provenance
        out["dl_pred_prob_up"] = surface_df["pred_prob_up"].astype("float64")
    if DL_AVAILABLE_TS_COL in surface_df.columns:
        out["dl_prediction_available_timestamp"] = surface_df[DL_AVAILABLE_TS_COL]
    if DL_GENERATED_TS_COL in surface_df.columns:
        out["dl_prediction_generated_timestamp"] = surface_df[DL_GENERATED_TS_COL]
    if DL_ARTIFACT_CREATED_COL in surface_df.columns:
        out["dl_artifact_created_timestamp"] = surface_df[DL_ARTIFACT_CREATED_COL]

    # Add MPML regime equivalence (informational; must not be used as a feature).
    # Canonical non-TrendVol artifacts may not contain ``dl_regime``.
    if "dl_regime" in surface_df.columns:
        out["mpml_regime_equiv"] = surface_df["dl_regime"].map(MSML_TO_MPML)
    else:
        # Non-TrendVol canonical artifacts have no dl_regime concept.
        # Keep the column for downstream schema stability, with null values.
        out["mpml_regime_equiv"] = pd.NA

    # Sort by (pair, timestamp)
    out = out.sort_values([DL_PAIR_COL, DL_TIMESTAMP_COL]).reset_index(drop=True)

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
            "dl_pred_prob_up": pd.Series(dtype="float64"),
            "dl_prediction_available_timestamp": pd.Series(dtype="datetime64[ns]"),
            "dl_prediction_generated_timestamp": pd.Series(dtype="datetime64[ns]"),
            "dl_artifact_created_timestamp": pd.Series(dtype="datetime64[ns]"),
            "mpml_regime_equiv": pd.Series(dtype="object"),
        }
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _normalize_surface_selector(surface: dict) -> dict[str, str | int]:
    """Return canonical selector dict, adapting legacy TrendVol selectors."""
    if not isinstance(surface, dict):
        raise ValueError(f"surface must be a dict; got {type(surface).__name__}")

    canonical_missing = sorted(set(SURFACE_REQUIRED_KEYS) - set(surface.keys()))
    if not canonical_missing:
        selector = {
            "surface_id": str(surface["surface_id"]).strip(),
            "surface_version": str(surface["surface_version"]).strip(),
            "state_id": str(surface["state_id"]).strip(),
            "model": str(surface["model"]).strip(),
            "target_horizon": int(surface["target_horizon"]),
            "feature_set": str(surface["feature_set"]).strip(),
        }
        empty_identity_fields = [
            key for key in ("surface_id", "surface_version", "state_id")
            if not selector[key]
        ]
        if empty_identity_fields:
            raise ValueError(
                "surface selector contains empty canonical identity fields: "
                f"{empty_identity_fields}"
            )
        if selector["surface_id"] in default_registry:
            selector["state_id"] = _canonicalize_state_id(
                selector["surface_id"],
                selector["state_id"],
                context="surface selector",
            )
        return selector

    legacy_missing = sorted(set(LEGACY_SURFACE_REQUIRED_KEYS) - set(surface.keys()))
    if not legacy_missing:
        regime = str(surface.get("dl_regime", "")).strip().upper()
        if regime not in VALID_DL_REGIMES:
            raise ValueError(
                f"Unknown dl_regime: {regime!r}. "
                f"Valid values: {sorted(VALID_DL_REGIMES)}"
            )
        trend_vol = _load_trend_vol_surface()
        state_id = _legacy_trend_vol_state_id(regime)
        return {
            "surface_id": trend_vol.surface_id,
            "surface_version": trend_vol.surface_version,
            "state_id": state_id,
            "model": str(surface["model"]).strip(),
            "target_horizon": int(surface["target_horizon"]),
            "feature_set": str(surface["feature_set"]).strip(),
        }

    raise ValueError(
        "surface dict missing required keys. "
        f"Canonical required={list(SURFACE_REQUIRED_KEYS)}. "
        f"Legacy TrendVol required={list(LEGACY_SURFACE_REQUIRED_KEYS)}."
    )


def validate_dl_artifact(
    df: pd.DataFrame,
    metadata: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Validate and normalize the DL artifact contract (schema v2).

    Raises
    ------
    ValueError
        If contract validation fails.
    """
    metadata = metadata or {}
    schema_version = _extract_schema_version(df, metadata)
    if schema_version is None:
        raise ValueError(
            "missing schema_version. "
            "Expected parquet metadata key "
            "'msml.schema_version' or row column 'schema_version'."
        )
    if not _is_compatible_schema_version(schema_version):
        raise ValueError(
            f"incompatible schema_version {schema_version!r}; "
            f"expected compatibility with {DL_SCHEMA_VERSION!r}"
        )

    missing_cols = REQUIRED_ARTIFACT_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"DL artifact missing required columns: {sorted(missing_cols)}"
        )

    tz_errors = _validate_timezone_consistency(df)
    if tz_errors:
        raise ValueError("; ".join(tz_errors))

    df = _coerce_dtypes(df)

    null_mask = df[list(REQUIRED_ARTIFACT_COLUMNS)].isna().any(axis=1)
    if null_mask.any():
        raise ValueError(
            f"null values in required contract columns: {int(null_mask.sum())} rows"
        )

    normalized_pairs = df[DL_PAIR_COL].map(_normalize_pair)
    bad_pair_mask = normalized_pairs.isna()
    if bad_pair_mask.any():
        bad_values = (
            df.loc[bad_pair_mask, DL_PAIR_COL]
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raise ValueError(f"invalid pair values for normalization: {bad_values}")
    df[DL_PAIR_COL] = normalized_pairs

    dup_mask = df.duplicated(subset=[DL_PAIR_COL, DL_TIMESTAMP_COL], keep=False)
    if dup_mask.any():
        raise ValueError(
            f"duplicate ({DL_PAIR_COL}, {DL_TIMESTAMP_COL}) rows: {int(dup_mask.sum())}"
        )

    non_monotone = _find_non_monotone_groups(df)
    if non_monotone:
        raise ValueError(
            f"non-monotone {DL_TIMESTAMP_COL} detected for surface groups: "
            f"{non_monotone[:3]}"
        )

    causal_violation = df[DL_AVAILABLE_TS_COL] > df[DL_TIMESTAMP_COL]
    if causal_violation.any():
        sample = (
            df.loc[causal_violation, [DL_PAIR_COL, DL_TIMESTAMP_COL, DL_AVAILABLE_TS_COL]]
            .head(3)
            .to_dict("records")
        )
        raise ValueError(
            f"causal contract violated: {DL_AVAILABLE_TS_COL} must be <= {DL_TIMESTAMP_COL}; "
            f"sample={sample}"
        )

    return df


def _apply_behavioral_identity(
    df: pd.DataFrame,
    metadata: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Ensure canonical Behavioral identity columns exist and are valid.

    Canonical identity:
    ``surface_id``, ``surface_version``, ``state_id``.

    Legacy TrendVol artifacts are adapted through ``dl_regime``:
    ``dl_regime`` -> ``surface_id=trend_vol``, canonical ``state_id``.
    """
    metadata = metadata or {}
    out = df.copy()

    for col in CANONICAL_SURFACE_KEYS:
        if col not in out.columns:
            out[col] = pd.NA

    metadata_surface_id = _metadata_value(metadata, "surface_id")
    metadata_surface_version = _metadata_value(metadata, "surface_version")
    metadata_state_id = _metadata_value(metadata, "state_id")

    if metadata_surface_id:
        out["surface_id"] = out["surface_id"].fillna(metadata_surface_id)
    if metadata_surface_version:
        out["surface_version"] = out["surface_version"].fillna(metadata_surface_version)
    if metadata_state_id:
        out["state_id"] = out["state_id"].fillna(metadata_state_id)

    if "dl_regime" in out.columns:
        regime_series = out["dl_regime"].map(_normalize_legacy_dl_regime)
        needs_state = out["state_id"].isna() | (out["state_id"].astype(str).str.strip() == "")
        out.loc[needs_state, "state_id"] = regime_series[needs_state]

        needs_surface = out["surface_id"].isna() | (out["surface_id"].astype(str).str.strip() == "")
        out.loc[needs_surface & regime_series.notna(), "surface_id"] = "trend_vol"

        needs_version = out["surface_version"].isna() | (
            out["surface_version"].astype(str).str.strip() == ""
        )
        trend_vol_version = _load_trend_vol_surface().surface_version
        trend_vol_rows = out["surface_id"].astype(str) == "trend_vol"
        out.loc[needs_version & trend_vol_rows, "surface_version"] = trend_vol_version

    for col in CANONICAL_SURFACE_KEYS:
        out[col] = out[col].astype("string").str.strip()
        out.loc[out[col] == "", col] = pd.NA

    missing_summary = {
        col: int(out[col].isna().sum())
        for col in CANONICAL_SURFACE_KEYS
        if out[col].isna().any()
    }
    if missing_summary:
        raise ValueError(
            "missing canonical metadata in DL artifact after compatibility adaptation: "
            f"{missing_summary}. Required keys={list(CANONICAL_SURFACE_KEYS)}."
        )

    _validate_behavioral_identity_values(out)
    return out


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce critical columns to expected dtypes."""
    df = df.copy()

    for ts_col in (
        DL_TIMESTAMP_COL,
        DL_AVAILABLE_TS_COL,
        DL_GENERATED_TS_COL,
        DL_ARTIFACT_CREATED_COL,
    ):
        if ts_col in df.columns:
            # Normalize all timestamps to UTC first, then drop timezone so MPML
            # downstream joins operate on a consistent tz-naive UTC convention.
            ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
            df[ts_col] = ts.dt.tz_localize(None)

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
    Return group identifiers where ``timestamp`` is not monotone increasing.

    Grain: ``(pair, model, target_horizon, feature_set, dl_regime)``.
    """
    non_monotone: list[str] = []
    # validate_dl_artifact() runs before _apply_behavioral_identity(), so
    # canonical identity columns may still be absent for legacy artifacts here.
    # Group only by currently-present columns (legacy validation degrades to
    # pre-canonical grouping while preserving historical compatibility).
    group_cols = [c for c in MONOTONICITY_GROUP_COLUMNS if c in df.columns]
    for keys, grp in df.groupby(group_cols):
        if not grp[DL_TIMESTAMP_COL].is_monotonic_increasing:
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


def _metadata_value(metadata: dict[str, str], key: str) -> str | None:
    candidates = (
        key,
        f"msml.{key}",
        f"behavioral.{key}",
        f"surface.{key}",
        f"mpml.{key}",
    )
    for candidate in candidates:
        value = metadata.get(candidate)
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return None


def _read_parquet_metadata(cube_path: Path) -> dict[str, str]:
    """Read file-level parquet metadata as decoded strings."""
    try:
        raw_metadata = pq.read_metadata(cube_path).metadata
    except (OSError, ValueError, RuntimeError):
        return {}
    if not raw_metadata:
        return {}
    decoded: dict[str, str] = {}
    for key, value in raw_metadata.items():
        try:
            decoded[key.decode("utf-8")] = value.decode("utf-8")
        except (AttributeError, UnicodeDecodeError):
            continue
    return decoded


def _extract_schema_version(df: pd.DataFrame, metadata: dict[str, str]) -> str | None:
    """Get schema version from parquet metadata or row column."""
    version = (
            metadata.get("msml.schema_version")
            or metadata.get("schema_version")  # backward compatibility
    )
    if version:
        return str(version)
    if "schema_version" not in df.columns:
        return None
    versions = df["schema_version"].dropna().astype(str).unique().tolist()
    if len(versions) != 1:
        raise ValueError(
            f"expected exactly one schema_version value in rows, got {versions}"
        )
    return versions[0]


def _is_compatible_schema_version(version: str) -> bool:
    """Compatibility policy: strict major-version compatibility with v2."""
    version_parts = str(version).split(".")
    expected_parts = DL_SCHEMA_VERSION.split(".")
    if len(version_parts) < 1 or len(expected_parts) < 1:
        return False
    if not version_parts[0].isdigit() or not expected_parts[0].isdigit():
        return False
    expected_major = expected_parts[0]
    major = version_parts[0]
    return major == expected_major


def _normalize_pair(pair: str) -> str | None:
    if not isinstance(pair, str):
        return None
    p = pair.strip().lower().replace("/", "-").replace("_", "-")
    while "--" in p:
        p = p.replace("--", "-")
    # Support compact FX format "XXXYYY" by converting to "xxx-yyy".
    if "-" not in p and len(p) == 6 and p.isalpha():
        p = f"{p[:3]}-{p[3:]}"
    if len(p) == 7 and p[3] == "-" and p[:3].isalpha() and p[4:].isalpha():
        return p
    return None


def _load_trend_vol_surface():
    try:
        return default_registry.load("trend_vol")
    except KeyError as exc:
        raise ValueError(
            "Legacy TrendVol compatibility adapter requires 'trend_vol' surface "
            "to be registered, but it was not available in the behavioral registry."
        ) from exc


def _legacy_trend_vol_state_id(dl_regime: str) -> str:
    try:
        return default_registry.get_state("trend_vol", dl_regime).state_id
    except KeyError as exc:
        raise ValueError(
            "Legacy TrendVol compatibility adapter failed while resolving "
            f"dl_regime={dl_regime!r}. Expected a valid TrendVol state token."
        ) from exc


def _normalize_legacy_dl_regime(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if not normalized:
        return None
    if normalized not in VALID_DL_REGIMES:
        return None
    return _legacy_trend_vol_state_id(normalized)


def _canonicalize_state_id(surface_id: str, state_id: str, *, context: str) -> str:
    try:
        return default_registry.get_state(surface_id, state_id).state_id
    except KeyError as exc:
        raise ValueError(
            f"{context}: unknown state_id {state_id!r} for surface_id {surface_id!r}"
        ) from exc


def _validate_behavioral_identity_values(df: pd.DataFrame) -> None:
    invalid: list[str] = []
    for surface_id in sorted(df["surface_id"].dropna().astype(str).unique().tolist()):
        if surface_id not in default_registry:
            continue
        surface = default_registry.load(surface_id)
        expected_version = str(surface.surface_version)
        observed_versions = (
            df.loc[df["surface_id"].astype(str) == surface_id, "surface_version"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if any(v != expected_version for v in observed_versions):
            invalid.append(
                f"surface_version mismatch for surface_id={surface_id!r}: "
                f"artifact={sorted(observed_versions)} runtime={expected_version!r}"
            )
        state_values = (
            df.loc[df["surface_id"].astype(str) == surface_id, "state_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        for state_id in state_values:
            try:
                _canonicalize_state_id(
                    surface_id,
                    state_id,
                    context="artifact canonical metadata validation",
                )
            except ValueError as exc:
                invalid.append(str(exc))
                break
    if invalid:
        raise ValueError("; ".join(invalid))


def _validate_timezone_consistency(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    modes: dict[str, str] = {}
    for col in (
        DL_TIMESTAMP_COL,
        DL_AVAILABLE_TS_COL,
        DL_GENERATED_TS_COL,
        DL_ARTIFACT_CREATED_COL,
    ):
        if col not in df.columns:
            continue
        mode = _timezone_mode(df[col])
        if mode == "mixed":
            errors.append(f"{col} has mixed timezone-awareness values")
        else:
            modes[col] = mode
    if not modes:
        return errors
    base_mode = modes[DL_TIMESTAMP_COL]
    for col, mode in modes.items():
        if mode != base_mode:
            errors.append(
                f"timezone consistency violation: {DL_TIMESTAMP_COL} is {base_mode} "
                f"but {col} is {mode}"
            )
    return errors


def _timezone_mode(series: pd.Series) -> str:
    """Return naive/aware/mixed timezone mode for a datetime-like series."""
    if is_datetime64tz_dtype(series.dtype):
        return "aware"
    try:
        converted = pd.to_datetime(series, errors="coerce", utc=False)
    except (TypeError, ValueError):
        return "mixed"
    if converted.isna().all():
        return "naive"
    if is_datetime64tz_dtype(converted.dtype):
        return "aware"
    if is_datetime64_dtype(converted.dtype):
        return "naive"
    return "mixed"


def _handle_error(msg: str, strict: bool) -> pd.DataFrame:
    """Raise ``ValueError`` if *strict*, otherwise warn and return empty DF."""
    if strict:
        raise ValueError(msg)
    warnings.warn(f"dl_surface_loader: {msg}", stacklevel=3)
    return empty_dl_surface_df()
