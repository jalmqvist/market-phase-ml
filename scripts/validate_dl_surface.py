#!/usr/bin/env python3
"""
Sanity-check script for the DL surface loader (src/dl_surface_loader.py).

Runs a suite of invariant checks against either:
  - A real cube parquet (when --cube-path points to an existing file), or
  - Synthetic in-memory test cubes that exercise every validation path.

Usage
-----
# Run all checks (synthetic cubes only — no real artifact needed)
python scripts/validate_dl_surface.py

# Also validate a real cube
python scripts/validate_dl_surface.py --cube-path ../market-sentiment-ml/data/output/dl_signals/dl_signals_h1_v1.parquet

Exit codes
----------
0  All checks passed.
1  One or more checks failed.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Ensure repo root is importable regardless of working directory
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dl_surface_loader import (  # noqa: E402
    REQUIRED_CUBE_COLUMNS,
    SURFACE_REQUIRED_KEYS,
    VALID_DL_REGIMES,
    empty_dl_surface_df,
    load_dl_surface,
)
from src.dl_config import resolve_dl_prediction_artifact_path  # noqa: E402
from src.dl_daily_features import _count_sign_flips  # noqa: E402
from schemas.dl_artifact_schema import (  # noqa: E402
    DL_ARTIFACT_CREATED_COL,
    DL_AVAILABLE_TS_COL,
    DL_GENERATED_TS_COL,
    DL_SCHEMA_VERSION,
    DL_TIMESTAMP_COL,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS = "PASS"
_FAIL = "FAIL"
_OLD_MTIME = 1
_NEW_MTIME = 2

_results: list[tuple[str, str, str]] = []  # (name, status, detail)


def _check(name: str, fn) -> bool:
    try:
        fn()
        _results.append((name, _PASS, ""))
        return True
    except AssertionError as exc:
        _results.append((name, _FAIL, str(exc)))
        return False
    except Exception as exc:  # noqa: BLE001 — catch unexpected errors in test runner
        _results.append((name, _FAIL, f"{type(exc).__name__}: {exc}"))
        return False


def _make_cube(
    n_rows: int = 10,
    pair: str = "eur-usd",
    model: str = "lstm",
    target_horizon: int = 24,
    feature_set: str = "price_trend",
    dl_regime: str = "HVTF",
    start: datetime | None = None,
) -> pd.DataFrame:
    """Build a minimal valid v2 DL artifact DataFrame for a single surface."""
    if start is None:
        start = datetime(2024, 1, 1, 0, 0, 0)
    times = [start + timedelta(hours=i) for i in range(n_rows)]
    generated_times = [t - timedelta(minutes=20) for t in times]
    available_times = [t - timedelta(minutes=5) for t in times]
    artifact_created_times = [t - timedelta(minutes=1) for t in times]
    return pd.DataFrame(
        {
            "pair": [pair] * n_rows,
            DL_TIMESTAMP_COL: pd.to_datetime(times),
            DL_AVAILABLE_TS_COL: pd.to_datetime(available_times),
            DL_GENERATED_TS_COL: pd.to_datetime(generated_times),
            DL_ARTIFACT_CREATED_COL: pd.to_datetime(artifact_created_times),
            "model": [model] * n_rows,
            "target_horizon": pd.array([target_horizon] * n_rows, dtype="Int64"),
            "feature_set": [feature_set] * n_rows,
            "dl_regime": [dl_regime] * n_rows,
            "signal_strength": [float(i) / n_rows * 2 - 1 for i in range(n_rows)],
            "pred_prob_up": [0.5 + float(i) / n_rows * 0.5 for i in range(n_rows)],
            "schema_version": [DL_SCHEMA_VERSION] * n_rows,
        }
    )


_SURFACE = {
    "model": "lstm",
    "target_horizon": 24,
    "feature_set": "price_trend",
    "dl_regime": "HVTF",
}

# ---------------------------------------------------------------------------
# Synthetic checks
# ---------------------------------------------------------------------------


def _assert_raises_value_error(fn) -> None:
    raised = False
    try:
        fn()
    except ValueError:
        raised = True
    fn_name = getattr(fn, "__name__", repr(fn))
    assert raised, f"Expected ValueError from {fn_name}"


def check_empty_df_schema():
    """empty_dl_surface_df() returns consistent dtypes."""
    df = empty_dl_surface_df()
    assert df.empty, "Expected empty DataFrame"
    assert "pair" in df.columns
    assert "timestamp" in df.columns
    assert "dl_signal_strength" in df.columns
    assert df["dl_signal_strength"].dtype == "float64"


def check_missing_cube_file():
    """Missing cube file returns empty DF (strict=False)."""
    df = load_dl_surface(
        Path("/nonexistent/path/cube.parquet"), _SURFACE, strict=False
    )
    assert df.empty, "Expected empty DF for missing file"


def check_missing_cube_file_strict():
    """Missing cube file raises ValueError when strict=True."""
    raised = False
    try:
        load_dl_surface(
            Path("/nonexistent/path/cube.parquet"), _SURFACE, strict=True
        )
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for missing file with strict=True"


def check_invalid_surface_dict_missing_key():
    """Surface dict missing a required key raises / returns empty DF."""
    bad_surface = {"model": "lstm", "target_horizon": 24, "feature_set": "price_trend"}
    df = load_dl_surface(Path("/nonexistent"), bad_surface, strict=False)
    assert df.empty, "Expected empty DF for invalid surface dict"


def check_invalid_surface_regime():
    """Unknown dl_regime returns empty DF."""
    bad_surface = dict(_SURFACE, dl_regime="UNKNOWN")
    df = load_dl_surface(Path("/nonexistent"), bad_surface, strict=False)
    assert df.empty


def check_valid_cube_roundtrip(tmp_path: Path):
    """Valid cube loads cleanly and output schema is correct."""
    cube = _make_cube()
    cube_file = tmp_path / "cube.parquet"
    cube.to_parquet(cube_file, index=False)

    df = load_dl_surface(cube_file, _SURFACE, strict=True)

    assert not df.empty, "Expected non-empty result"
    assert "pair" in df.columns
    assert "timestamp" in df.columns
    assert "dl_signal_strength" in df.columns
    assert "mpml_regime_equiv" in df.columns
    assert "signal_strength" not in df.columns, "signal_strength should be renamed"
    assert DL_TIMESTAMP_COL in df.columns
    # pred_prob_up is optional in the cube; _make_cube() always includes it, so
    # after loading it must have been renamed to dl_pred_prob_up.
    assert "pred_prob_up" not in df.columns, "pred_prob_up should be renamed to dl_pred_prob_up"
    assert "dl_pred_prob_up" in df.columns, "dl_pred_prob_up should be present (cube includes pred_prob_up)"
    assert "dl_prediction_available_timestamp" in df.columns

    # Check MPML mapping
    assert set(df["mpml_regime_equiv"].unique()) == {"HVTF"}


def check_signal_strength_range_violation(tmp_path: Path):
    """Out-of-range signal_strength fails fast."""
    cube = _make_cube()
    cube.loc[0, "signal_strength"] = 1.5  # out of range
    cube_file = tmp_path / "bad_range.parquet"
    cube.to_parquet(cube_file, index=False)

    _assert_raises_value_error(
        lambda: load_dl_surface(cube_file, _SURFACE, strict=False)
    )


def check_pred_prob_up_range_violation(tmp_path: Path):
    """Out-of-range pred_prob_up fails fast."""
    cube = _make_cube()
    cube.loc[0, "pred_prob_up"] = -0.1  # out of range
    cube_file = tmp_path / "bad_prob.parquet"
    cube.to_parquet(cube_file, index=False)

    _assert_raises_value_error(
        lambda: load_dl_surface(cube_file, _SURFACE, strict=False)
    )


def check_causal_ordering_violation(tmp_path: Path):
    """Rows where available timestamp is after bar timestamp are rejected."""
    cube = _make_cube()
    cube.loc[0, DL_AVAILABLE_TS_COL] = cube.loc[0, DL_TIMESTAMP_COL] + timedelta(minutes=1)
    cube_file = tmp_path / "causal_violation.parquet"
    cube.to_parquet(cube_file, index=False)

    _assert_raises_value_error(
        lambda: load_dl_surface(cube_file, _SURFACE, strict=False)
    )


def check_duplicate_timestamp(tmp_path: Path):
    """Duplicate (pair, timestamp) within a surface is rejected."""
    cube = _make_cube()
    cube.loc[1, DL_TIMESTAMP_COL] = cube.loc[0, DL_TIMESTAMP_COL]  # introduce duplicate
    cube_file = tmp_path / "dup.parquet"
    cube.to_parquet(cube_file, index=False)

    _assert_raises_value_error(
        lambda: load_dl_surface(cube_file, _SURFACE, strict=False)
    )


def check_non_monotone_timestamps(tmp_path: Path):
    """Non-monotone timestamp within a surface is rejected."""
    cube = _make_cube()
    # Swap first two rows to break monotonicity
    cube.loc[0, DL_TIMESTAMP_COL], cube.loc[1, DL_TIMESTAMP_COL] = (
        cube.loc[1, DL_TIMESTAMP_COL],
        cube.loc[0, DL_TIMESTAMP_COL],
    )
    cube_file = tmp_path / "non_monotone.parquet"
    cube.to_parquet(cube_file, index=False)

    _assert_raises_value_error(
        lambda: load_dl_surface(cube_file, _SURFACE, strict=False)
    )


def check_surface_selection_filters_correctly(tmp_path: Path):
    """Only rows matching the exact surface dict are returned."""
    cube_a = _make_cube(dl_regime="HVTF", n_rows=5)
    cube_b = _make_cube(dl_regime="LVTF", pair="gbp-usd", n_rows=8)
    cube = pd.concat([cube_a, cube_b], ignore_index=True)
    cube_file = tmp_path / "multi_surface.parquet"
    cube.to_parquet(cube_file, index=False)

    df = load_dl_surface(cube_file, _SURFACE, strict=True)
    assert len(df) == 5, f"Expected 5 rows; got {len(df)}"
    assert set(df["pair"].unique()) == {"eur-usd"}


def check_missing_required_column(tmp_path: Path):
    """Artifact missing a required column fails fast."""
    cube = _make_cube()
    cube = cube.drop(columns=[DL_AVAILABLE_TS_COL])
    cube_file = tmp_path / "missing_col.parquet"
    cube.to_parquet(cube_file, index=False)

    _assert_raises_value_error(
        lambda: load_dl_surface(cube_file, _SURFACE, strict=False)
    )


def check_missing_schema_version_fails(tmp_path: Path):
    """Artifact without schema_version metadata/column fails fast."""
    cube = _make_cube().drop(columns=["schema_version"])
    cube_file = tmp_path / "missing_schema_version.parquet"
    cube.to_parquet(cube_file, index=False)
    _assert_raises_value_error(
        lambda: load_dl_surface(cube_file, _SURFACE, strict=False)
    )


def check_incompatible_schema_version_fails(tmp_path: Path):
    """Artifact with incompatible schema version fails fast."""
    cube = _make_cube()
    cube["schema_version"] = "1.0.0"
    cube_file = tmp_path / "incompatible_schema_version.parquet"
    cube.to_parquet(cube_file, index=False)
    _assert_raises_value_error(
        lambda: load_dl_surface(cube_file, _SURFACE, strict=False)
    )


def check_mpml_regime_mapping(tmp_path: Path):
    """HVR dl_regime maps to HVMR in mpml_regime_equiv."""
    cube = _make_cube(dl_regime="HVR")
    cube_file = tmp_path / "hvr.parquet"
    cube.to_parquet(cube_file, index=False)

    surface = dict(_SURFACE, dl_regime="HVR")
    df = load_dl_surface(cube_file, surface, strict=True)
    assert not df.empty
    assert set(df["mpml_regime_equiv"].unique()) == {"HVMR"}


def check_target_horizon_numeric(tmp_path: Path):
    """target_horizon stored as string in parquet is coerced to Int64."""
    cube = _make_cube()
    cube["target_horizon"] = cube["target_horizon"].astype(str)
    cube_file = tmp_path / "str_horizon.parquet"
    cube.to_parquet(cube_file, index=False)

    df = load_dl_surface(cube_file, _SURFACE, strict=True)
    assert not df.empty, "Expected successful load with string target_horizon"


def check_no_surface_match_returns_empty(tmp_path: Path):
    """Surface dict matching nothing returns empty DF."""
    cube = _make_cube(dl_regime="LVTF")
    cube_file = tmp_path / "lvtf.parquet"
    cube.to_parquet(cube_file, index=False)

    df = load_dl_surface(cube_file, _SURFACE, strict=False)  # _SURFACE expects HVTF
    assert df.empty, "Expected empty DF when surface not in cube"


def check_resolve_artifact_path_file(tmp_path: Path) -> None:
    """Resolver returns file path unchanged when given a parquet file."""
    test_dir = tmp_path / "resolver_file"
    test_dir.mkdir(parents=True, exist_ok=True)
    f = test_dir / "single_surface.parquet"
    _make_cube(n_rows=1).to_parquet(f, index=False)
    resolved = resolve_dl_prediction_artifact_path(f)
    assert resolved == f


def check_resolve_artifact_path_directory_latest(tmp_path: Path) -> None:
    """Resolver picks the newest parquet when a directory is provided."""
    test_dir = tmp_path / "resolver_latest"
    test_dir.mkdir(parents=True, exist_ok=True)
    old_f = test_dir / "old.parquet"
    new_f = test_dir / "new.parquet"
    _make_cube(n_rows=1).to_parquet(old_f, index=False)
    _make_cube(n_rows=1).to_parquet(new_f, index=False)
    os.utime(old_f, (_OLD_MTIME, _OLD_MTIME))
    os.utime(new_f, (_NEW_MTIME, _NEW_MTIME))
    resolved = resolve_dl_prediction_artifact_path(test_dir)
    assert resolved == new_f


def check_resolve_artifact_path_directory_empty(tmp_path: Path) -> None:
    """Resolver returns None for a directory with no parquet artifacts."""
    test_dir = tmp_path / "resolver_empty"
    test_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_dl_prediction_artifact_path(test_dir)
    assert resolved is None


def check_resolve_artifact_path_missing_file(tmp_path: Path) -> None:
    """Resolver returns None for a missing parquet file path."""
    missing = tmp_path / "missing.parquet"
    resolved = resolve_dl_prediction_artifact_path(missing)
    assert resolved is None


# ---------------------------------------------------------------------------
# _count_sign_flips invariant checks
# ---------------------------------------------------------------------------


def check_sign_flips_basic():
    """Basic positive-to-negative transitions are counted correctly."""
    import numpy as np

    # [+, -, +, -]: 3 flips
    assert _count_sign_flips(np.array([1.0, -1.0, 1.0, -1.0])) == 3
    # [+, +, -, -]: 1 flip
    assert _count_sign_flips(np.array([0.5, 0.3, -0.2, -0.8])) == 1
    # All positive: 0 flips
    assert _count_sign_flips(np.array([0.1, 0.2, 0.3])) == 0
    # Single element: 0 flips
    assert _count_sign_flips(np.array([0.5])) == 0
    # Empty array: 0 flips
    assert _count_sign_flips(np.array([])) == 0


def check_sign_flips_nan_ignored():
    """NaN values are ignored; sign flips are counted over the remaining values."""
    import numpy as np

    # NaN between two values of the same sign: no flip
    assert _count_sign_flips(np.array([1.0, np.nan, 1.0])) == 0
    # NaN between positive and negative: 1 flip
    assert _count_sign_flips(np.array([1.0, np.nan, -1.0])) == 1
    # All NaN: 0 flips
    assert _count_sign_flips(np.array([np.nan, np.nan])) == 0
    # Leading/trailing NaN do not affect count
    assert _count_sign_flips(np.array([np.nan, 1.0, -1.0, np.nan])) == 1


def check_sign_flips_zeros_no_extra_flips():
    """Zeros are forward-filled and do not create extra flip counts."""
    import numpy as np

    # Zero between two positives: 0 flips (zero takes prior positive sign)
    assert _count_sign_flips(np.array([1.0, 0.0, 1.0])) == 0
    # Zero between positive and negative: 1 flip
    assert _count_sign_flips(np.array([1.0, 0.0, -1.0])) == 1
    # Multiple zeros between same-sign values: 0 flips
    assert _count_sign_flips(np.array([0.5, 0.0, 0.0, 0.5])) == 0
    # All zeros: 0 flips
    assert _count_sign_flips(np.array([0.0, 0.0, 0.0])) == 0
    # Leading zeros (no prior sign): discarded, not counted as flip
    assert _count_sign_flips(np.array([0.0, 0.0, 1.0, -1.0])) == 1
    # Zero run spanning a sign boundary: only one flip counted
    assert _count_sign_flips(np.array([1.0, 0.0, 0.0, -1.0, 0.0, -1.0])) == 1


# ---------------------------------------------------------------------------
# Real artifact check (optional)
# ---------------------------------------------------------------------------


def check_real_cube(cube_path: Path, surface: dict):
    """Load a real cube and validate basic invariants."""
    df = load_dl_surface(cube_path, surface, strict=True)
    assert not df.empty, "Real cube returned empty surface"
    assert df["dl_signal_strength"].between(-1, 1).all(), (
        "signal_strength out of range"
    )
    assert "timestamp" in df.columns
    assert "pair" in df.columns
    assert "mpml_regime_equiv" in df.columns
    print(f"  Real cube: {len(df):,} rows loaded for surface {surface}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_checks(cube_path: Path | None, surface: dict) -> int:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="validate_dl_"))

    synthetic = [
        ("empty_df_schema", check_empty_df_schema, []),
        ("missing_cube_file", check_missing_cube_file, []),
        ("missing_cube_strict", check_missing_cube_file_strict, []),
        ("invalid_surface_missing_key", check_invalid_surface_dict_missing_key, []),
        ("invalid_surface_regime", check_invalid_surface_regime, []),
        ("valid_roundtrip", check_valid_cube_roundtrip, [tmp]),
        ("missing_schema_version", check_missing_schema_version_fails, [tmp]),
        ("incompatible_schema_version", check_incompatible_schema_version_fails, [tmp]),
        ("signal_strength_range", check_signal_strength_range_violation, [tmp]),
        ("pred_prob_up_range", check_pred_prob_up_range_violation, [tmp]),
        ("causal_ordering_violation", check_causal_ordering_violation, [tmp]),
        ("duplicate_timestamp", check_duplicate_timestamp, [tmp]),
        ("non_monotone_timestamps", check_non_monotone_timestamps, [tmp]),
        ("surface_selection", check_surface_selection_filters_correctly, [tmp]),
        ("missing_required_column", check_missing_required_column, [tmp]),
        ("mpml_regime_mapping", check_mpml_regime_mapping, [tmp]),
        ("target_horizon_numeric", check_target_horizon_numeric, [tmp]),
        ("no_surface_match", check_no_surface_match_returns_empty, [tmp]),
        ("resolve_artifact_file", check_resolve_artifact_path_file, [tmp]),
        ("resolve_artifact_latest", check_resolve_artifact_path_directory_latest, [tmp]),
        ("resolve_artifact_empty", check_resolve_artifact_path_directory_empty, [tmp]),
        ("resolve_artifact_missing_file", check_resolve_artifact_path_missing_file, [tmp]),
        ("sign_flips_basic", check_sign_flips_basic, []),
        ("sign_flips_nan_ignored", check_sign_flips_nan_ignored, []),
        ("sign_flips_zeros_no_extra", check_sign_flips_zeros_no_extra_flips, []),
    ]

    for name, fn, args in synthetic:
        _check(name, lambda f=fn, a=args: f(*a))

    if cube_path is not None and cube_path.exists():
        _check(
            "real_cube",
            lambda: check_real_cube(cube_path, surface),
        )
    elif cube_path is not None:
        _results.append(
            ("real_cube", "SKIP", f"Cube file not found: {cube_path}")
        )

    # Print results
    max_name = max(len(r[0]) for r in _results)
    print(f"\n{'Check':<{max_name}}  Status  Detail")
    print("-" * (max_name + 30))
    n_failed = 0
    for name, status, detail in _results:
        icon = "✓" if status == _PASS else ("⚠" if status == "SKIP" else "✗")
        print(f"{name:<{max_name}}  {icon} {status:<6}  {detail}")
        if status == _FAIL:
            n_failed += 1

    print()
    if n_failed == 0:
        print(f"All {len(_results)} checks passed.")
        return 0
    else:
        print(f"{n_failed}/{len(_results)} checks FAILED.")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate DL surface loader invariants."
    )
    parser.add_argument(
        "--cube-path",
        type=Path,
        default=None,
        help="Path to a real DL signal cube parquet (optional).",
    )
    parser.add_argument("--model", default="lstm")
    parser.add_argument("--target-horizon", type=int, default=24)
    parser.add_argument("--feature-set", default="price_trend")
    parser.add_argument("--dl-regime", default="HVTF")
    args = parser.parse_args()

    surface = {
        "model": args.model,
        "target_horizon": args.target_horizon,
        "feature_set": args.feature_set,
        "dl_regime": args.dl_regime,
    }

    return run_checks(args.cube_path, surface)


if __name__ == "__main__":
    sys.exit(main())
